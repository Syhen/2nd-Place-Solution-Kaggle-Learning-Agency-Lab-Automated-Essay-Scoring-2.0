"""
@created by: heyao
@created at: 2024-04-26 17:16:08
"""
import gc
import json
import os
import random
import time
import warnings

from datasets import Dataset
from tokenizers import AddedToken

from training.utils.sampling import sample_data_by_dist

warnings.filterwarnings("ignore")

from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
import pandas as pd
import wandb
from accelerate import Accelerator
from transformers import AutoTokenizer, get_scheduler
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from aes2.utils.collators.sequence_bucket import SequenceBucketPadCollator
from aes2.utils.dataset.simple import CompetitionDataset
from aes2.utils.metrics import competition_score, OptimizedRounder
from aes2.utils.stable_training import init_weights, get_optimizer_params, differential_learning_rate, get_optimizer
from aes2.utils.meter import AverageMeter
from aes2.utils.reproduction import seed_everything
from aes2.utils.savor import save_checkpoints

from aux.models import AESRegressionModel
from aux.utils import round_target

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Trainer(object):
    def __init__(self, model, train_dataloader, val_dataloader, val_df, optimizer, accelerator: Accelerator,
                 fold, config, scheduler: torch.optim.lr_scheduler.LRScheduler = None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.val_df = val_df
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.fold = fold
        self.save_file_name = f"model_fold{fold if not config.train.fullfit else f'_full_seed{config.train.seed}'}.pth"
        self.config = config
        self.scheduler = scheduler
        self.best_loss = float("inf")
        self.best_score = 0
        self.global_epoch = 0
        self.global_steps = 0

    def train_one_epoch(self):
        self.optimizer.zero_grad()
        self.model.train()
        losses = AverageMeter()
        tqdm_obj = tqdm(self.train_dataloader, total=len(self.train_dataloader))
        for x, y, y_aux in tqdm_obj:
            self.global_steps += 1
            logits = model(x)
            loss = self.model.compute_loss(logits, [y, y_aux])
            self.accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                if self.config.train.use_wandb:
                    lr = self.scheduler.get_last_lr()
                    if isinstance(lr, list):
                        lr = lr[-1]
                    wandb.log({"scheduler/lr": lr}, step=self.global_steps)
                self.scheduler.step()
            self.optimizer.zero_grad()
            losses.update(loss.item(), y.shape[0])
            tqdm_obj.set_postfix({"loss": f"{losses.avg:.5f}"})
            if self.global_steps % self.config.train.log_steps == 0 and self.config.train.use_wandb:
                wandb.log({"train/loss": losses.avg}, step=self.global_steps)
        if self.config.train.use_wandb:
            wandb.log({"train/loss_epoch": losses.avg}, step=self.global_epoch)
        self.global_epoch += 1
        return losses.avg

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        labels = []
        predictions = []
        losses = AverageMeter()
        for x, y, y_aux in tqdm(self.val_dataloader, total=len(self.val_dataloader)):
            logits = self.model(x)
            loss = model.compute_loss(logits, [y, y_aux])
            labels.append(y.detach().cpu().numpy())
            losses.update(loss.item(), y.shape[0])
            logits = logits[0].cpu().detach().numpy()
            predictions.append(logits)

        fold = self.config.train.fold
        oof_pred = df_val[["essay_id", "score", "is_pc2"]]
        oof_pred.loc[:, "raw_pred"] = np.concatenate(predictions, axis=0)
        optimized_rounder = OptimizedRounder()
        optimized_rounder.fit(oof_pred["raw_pred"], df_val["score"])
        print(f"<<< best coef: {optimized_rounder._coef}")
        oof_pred.loc[:, "pred"] = optimized_rounder.predict(oof_pred["raw_pred"])
        print(oof_pred.head(20))
        score = competition_score(oof_pred["pred"].values, oof_pred["score"].values)

        if self.config.train.use_wandb:
            wandb.log({"val/loss": losses.avg}, step=self.global_steps)
            wandb.log({f"val/score": score}, step=self.global_steps)
            keys = ["new", "pc2"]
            for i in [0, 1]:
                temp = oof_pred[oof_pred["is_pc2"] == i]
                if temp.shape[0] == 0:
                    continue
                _score = competition_score(temp["pred"].values, temp["score"].values)
                wandb.log({f"val/score@{keys[i]}": _score}, step=self.global_steps)

        return score, losses.avg, oof_pred

    def fit(self, config):
        print("<<< start training model")
        for epoch in range(config.train.epochs):
            self.train_one_epoch()
            if self.val_dataloader is not None:
                best_score, val_loss, oof_pred = self.evaluate()
                print(f"<<< [epoch {epoch}] loss: {val_loss:.5f}")
                if best_score > self.best_score:
                    save_checkpoints(self.model, self.save_file_name, config, half_precision=True)
                    oof_pred.to_csv(
                        os.path.join(self.config.model.save_path,
                                     f"pred_df_stage{self.config.train.stage}_fold{fold}.csv"), index=False)
                    print(f"<<< improved! from {self.best_score:.5f} to {best_score:.5f}")
                    self.best_score = best_score
                else:
                    print(f"<<< no improve. score: {best_score:.5f}")
            else:
                save_checkpoints(self.model, self.save_file_name, config, half_precision=True)


def tokenize(example, tokenizer, config, max_length=1024):
    full_text = example["full_text"]
    truncation = True
    encoding = tokenizer(full_text, return_token_type_ids=False, truncation=truncation, max_length=max_length)

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": example["score"],
        "aux_labels": [float(example[col]) for col in config.multi_task.tasks]
    }


def loc_datasets(ds: Dataset, keep_columns):
    return Dataset.from_dict({col: ds[col] for col in keep_columns})


if __name__ == '__main__':
    import torch
    import torch.utils.data
    import torch.nn as nn

    t0 = time.perf_counter()
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, required=False)
    args, unknown_args = parser.parse_known_args()

    gc.enable()
    if args.config is None:
        config = OmegaConf.load("../config/exp301_debv3b.yaml")
        print("didn't pass config, DEBUG mode enable")
    else:
        config = OmegaConf.load(args.config)
        config.merge_with_dotlist(unknown_args)

    accelerator = Accelerator(cpu=config.train.device == "cpu",
                              mixed_precision=config.train.precision)
    df = pd.read_csv(config.dataset.train_file)
    folds = pd.read_csv(config.dataset.fold_file)
    df = df.merge(folds, on="essay_id", how="left")
    df_pc2 = pd.read_csv("../my_datasets/is_pc2.csv")
    df = df.merge(df_pc2, on="essay_id", how="left")
    df = df.merge(
        pd.read_csv("../my_datasets/eff_preds.csv").rename(columns={"text_id": "essay_id"}),
        on="essay_id", how="left"
    )
    for col in config.multi_task.tasks:
        df[col] = df[col].apply(lambda x: round_target(x, digits=0.5))

    tokenizer = AutoTokenizer.from_pretrained(config.model.path)
    tokenizer.add_tokens([AddedToken("\n\n", normalized=False)])
    tokenizer.add_tokens([AddedToken("\n", normalized=False)])
    if config.train.use_random_seed:
        config.train.seed = random.randint(1, 1000)
        print(f"<<< use random seed {config.train.seed}")

    fold = config.train.fold
    # ========= prepare input =========
    tokenize_function = partial(
        tokenize, tokenizer=tokenizer, max_length=config.train.max_length, config=config
    )
    validate_tokenize_function = partial(
        tokenize, tokenizer=tokenizer, max_length=config.train.validate_max_length, config=config
    )
    if not config.train.fullfit:
        df_train, df_val = df[df["kfold"] != fold], df[df["kfold"] == fold]
    else:
        df_train, df_val = df, df.sample(300, random_state=42)
    if config.dataset.external:
        df_external = pd.read_csv(config.dataset.external)
    else:
        df_external = None
    if config.train.stage == 1:
        print("<<< stage 1: only keep pc2 data in train, and KAGGLE-ONLY in eval")
        df_train = df_train[df_train["is_pc2"] == 1].reset_index(drop=True)
        if config.dataset.sampling.enable:
            if config.dataset.sampling.method == "by_dist":
                print("<<< sampling by KAGGLE-ONLY distribution")
                df_train = sample_data_by_dist(df_train)
        if df_external is not None:
            df_train = pd.concat([df_train, df_external], axis=0).reset_index(drop=True)
        if config.pl.enable:
            print("<<< enable PL")
            print(f"<<< load PL from {config.pl.path.format(fold=fold)}")
            df_pl = pd.read_csv(config.pl.path.format(fold=fold))
            df_pl = df_pl.merge(
                pd.read_csv("../my_datasets/external/external_13119_pc2.csv").drop_duplicates(subset=["essay_id"])[
                    ["essay_id", "full_text"]],
                on="essay_id", how="left"
            )
            df_pl = df_pl[["essay_id", "full_text", "raw_pred"]].rename(
                columns={"raw_pred": "score"}).reset_index(drop=True)
            df_train = pd.concat([df_train, df_pl], axis=0).reset_index(drop=True)
        print("<<< stage 1: enable save last checkpoint")
        config.train.save_last_checkpoint = True
        df_val = df_val[df_val["is_pc2"] == 0].reset_index(drop=True)
    else:
        print("<<< stage 2: only keep new data in train")
        df_train = df_train[df_train["is_pc2"] == 0].reset_index(drop=True)
        config.model.num_reinit_layers = 0
    ds_train = Dataset.from_pandas(df_train).map(tokenize_function, num_proc=config.train.num_workers)
    ds_val = Dataset.from_pandas(df_val).map(validate_tokenize_function, num_proc=config.train.num_workers)
    keep_columns = ["input_ids", "attention_mask", "labels", "aux_labels"]
    ds_train = loc_datasets(ds_train, keep_columns)
    ds_val = loc_datasets(ds_val, keep_columns)

    print(f"<<< train on {len(ds_train)} samples, validate on {len(ds_val)} samples.")

    max_length = config.train.max_length
    collate_fn = SequenceBucketPadCollator(max_length=max_length, tokenizer=tokenizer)

    train_loader = torch.utils.data.DataLoader(
        CompetitionDataset(x=ds_train), batch_size=config.train.batch_size,
        shuffle=True, num_workers=config.train.num_workers,
        collate_fn=collate_fn
    )
    collate_fn = SequenceBucketPadCollator(max_length=config.train.validate_max_length, tokenizer=tokenizer)
    val_loader = torch.utils.data.DataLoader(
        CompetitionDataset(x=ds_val), batch_size=config.train.batch_size,
        shuffle=False, num_workers=config.train.num_workers, collate_fn=collate_fn
    )
    print("<<< one sample of model output")
    print(train_loader.dataset[0]["input_ids"])
    print(tokenizer.decode(train_loader.dataset[0]["input_ids"]))
    print(train_loader.dataset[0]["aux_labels"])
    config.dataset.n_samples_in_train = int((folds.kfold != fold).sum())
    print(f"<<< train model with {AESRegressionModel.__name__}")
    print(f"<<< num samples: {config.dataset.n_samples_in_train}, "
          f"total {len(train_loader)} * {config.train.epochs} = {len(train_loader) * config.train.epochs} steps.")

    # ======= config paths ======
    name = f"{config.train.exp_name}_{config.model.path.split('/')[-1]}_fold{fold}"
    print(f"<<< train with config: {config}")
    config.model.save_path = os.path.join(config.model.save_path,
                                          f"{config.train.exp_name}_{config.model.path.split('/')[-1]}")
    save_to = config.train.get("save_to", config.model.save_path)
    os.makedirs(os.path.abspath(save_to), exist_ok=True)
    with open(os.path.join(save_to, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))
    print(f"<<< model save to {os.path.abspath(save_to)}")

    # create model
    seed_everything(config.train.seed + fold)
    model = AESRegressionModel(config)
    model.backbone.resize_token_embeddings(len(tokenizer))
    if config.train.stage == 2:
        load_path = os.path.join(config.model.save_path, f"model_stage1_fold{fold}_last.pth")
        print(f"<<< load first stage checkpoint from {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=config.train.device))

    if config.model.differential_lr.enable:
        model_parameters = differential_learning_rate(
            model,
            encoder_lr=config.optim.optimizer.lr,
            decoder_lr=config.optim.optimizer.head_lr,
            weight_decay=config.optim.optimizer.weight_decay,
            lr_factor=config.model.differential_lr.lr_factor
        )
    else:
        model_parameters = get_optimizer_params(
            model, encoder_lr=config.optim.optimizer.lr, decoder_lr=config.optim.optimizer.head_lr,
            weight_decay=config.optim.optimizer.weight_decay
        )

    # ====== config optimizer and scheduler ======
    optimizer = get_optimizer(model_parameters, config)
    total_training_steps = int(config.train.epochs * len(train_loader))
    warmup_steps = int(total_training_steps * config.optim.scheduler.get("warmup_percent", 0))
    if config.optim.scheduler.num_warmup_steps > 0:
        warmup_steps = config.optim.scheduler.num_warmup_steps
    scheduler = get_scheduler(config.optim.scheduler.name, optimizer, num_warmup_steps=warmup_steps,
                              num_training_steps=total_training_steps)

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    trainer = Trainer(model, train_loader, val_loader, df_val,
                      optimizer, accelerator, fold, config, scheduler=scheduler)
    # start training
    seed_everything(config.train.seed + fold)
    if config.train.reinit_weights and config.train.stage == 1:
        if hasattr(model.backbone.config, "initializer_range"):
            initializer_range = model.backbone.config.initializer_range
        else:
            initializer_range = 0.02
        init_weights(model.head, initializer_range=initializer_range)
        init_weights(model.customer_pooling, initializer_range=initializer_range)
    if config.train.ensure_data_order:
        print(f"<<< ensure data order of training, will set seed again.")
        seed_everything(config.train.seed + fold)  # to ensure the data order

    if bnb is not None and "bit" in config.optim.optimizer.name:
        if hasattr(model.backbone, "embeddings"):
            embs = model.backbone.embeddings
        elif hasattr(model.backbone, "shared"):
            embs = model.backbone.shared
        for emb_type in ["word", "position", "token_type"]:
            attr_name = f"{emb_type}_embeddings"

            # Note: your model type might have a different path to the embeddings
            if hasattr(embs, attr_name) and getattr(embs, attr_name) is not None:
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    getattr(embs, attr_name), 'weight', {'optim_bits': 32}
                )
        del embs
        gc.collect()
    # freeze
    if config.train.freeze_embeddings:
        model.backbone.embeddings.requires_grad_(False)
    if config.train.freeze_encoders > 0:
        model.backbone.encoder.layer[:config.train.freeze_encoders].requires_grad_(False)
    if config.train.use_wandb:
        wandb.login(key=os.environ.get("WANDB_TOKEN", ""))
        wandb.init(config=OmegaConf.to_container(config), project="AES2", name=name, dir="./logs/",
                   group=config.train.exp_name)
    try:
        trainer.fit(config)
        # trainer.evaluate()
    except (Exception, KeyboardInterrupt) as e:
        raise e
    finally:
        wandb.finish()
    if config.train.save_last_checkpoint and not config.train.fullfit:
        if config.train.stage == 1:
            save_name = f"model_stage1_fold{fold}_last.pth"
        else:
            save_name = f"model_stage2_fold{fold}_last.pth"
        torch.save(model.half().state_dict(), os.path.join(save_to, save_name))
        print(f"<<< save last checkpoint")

    print(f"<<< finished use {time.perf_counter() - t0:.1f} seconds.")
