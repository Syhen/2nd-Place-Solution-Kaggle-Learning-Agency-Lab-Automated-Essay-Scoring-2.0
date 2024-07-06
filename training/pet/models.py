"""
@created by: heyao
@created at: 2024-02-03 18:03:14
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoModel, T5EncoderModel, AutoConfig
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2OnlyMLMHead

from aes2.nn.heads.msd import MultiSampleDropout
from aes2.nn.poolers import MultiPooling
from aes2.nn.losses import RankingLoss, RMSELoss
from aes2.utils.stable_training import reinit_last_layers


class AESRegressionModel(nn.Module):
    def __init__(self, hyper_config: DictConfig):
        super(AESRegressionModel, self).__init__()
        self.hyper_config = hyper_config
        self.backbone = self.init_backbone(hyper_config.model.path)
        self.customer_pooling = MultiPooling(
            hyper_config.model.pooling,
            hidden_size=self.backbone.config.hidden_size,
            num_hidden_layers=self.backbone.config.num_hidden_layers,
            layer_start=self.hyper_config.model.get("layer_start", 4)
        )
        # feature_size = self.get_hidden_size(hyper_config.model.pooling, self.backbone.config.hidden_size)
        model_config = self.backbone.config
        if self.hyper_config.model.get("vocab_size", None):
            model_config.vocab_size = self.hyper_config.model.vocab_size
        self.head = DebertaV2OnlyMLMHead(config=model_config)
        # if self.hyper_config.model.msd:
        #     self.head = MultiSampleDropout(feature_size, 1)
        # else:
        #     self.head = nn.Linear(feature_size, 1)
        self.criterion = self.get_loss_function(hyper_config)
        # tricks
        reinit_last_layers(self.backbone, num_layers=hyper_config.model.num_reinit_layers)
        if hyper_config.train.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            self.backbone.config.use_cache = False
        self.positive_token_ids = hyper_config.model.positive_token_ids
        self.negative_token_ids = hyper_config.model.negative_token_ids

    def init_backbone(self, model_path):
        if "t5" in model_path:
            model_class = T5EncoderModel
        else:
            model_class = AutoModel
        config = AutoConfig.from_pretrained(self.hyper_config.model.path)
        config.output_hidden_states = True
        # regression model must set dropout to zero
        if self.hyper_config.model.task_type == "regression":
            config.hidden_dropout_prob = 0
            config.attention_probs_dropout_prob = 0
        return model_class.from_pretrained(model_path, config=config)

    def get_loss_function(self, config):
        if config.train.loss == "mse":
            return nn.MSELoss()
        if config.train.loss == "rmse":
            return RMSELoss()
        elif config.train.loss == "ce":
            return nn.CrossEntropyLoss(label_smoothing=config.model.get("label_smoothing", 0.0))
        elif config.train.loss == "bce":
            return nn.BCELoss()

    def get_hidden_size(self, pooling_name, hidden_size):
        def _get_hidden_size(pooling_name, hidden_size):
            if any(i in pooling_name for i in ["meanmax"]):
                return hidden_size * 2
            return hidden_size

        return sum(_get_hidden_size(name, hidden_size) for name in pooling_name.split("_"))

    def get_feature(self, input_ids, attention_mask=None, mask_token_idx=None):
        model_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(model_output, "hidden_states"):
            hidden_states = model_output.hidden_states
        else:
            hidden_states = [model_output.last_hidden_state]
        # if "weighted" in self.hyper_config.model.pooling:
        #     pooler_output = self.customer_pooling(hidden_states, attention_mask)
        # else:
        #     pooler_output = self.customer_pooling(hidden_states[-1], attention_mask)
        indices = mask_token_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hidden_states[0].shape[-1])
        pooler_output = torch.gather(hidden_states[-1], dim=1, index=indices)
        return pooler_output.squeeze(1)

    def forward(self, x):
        feature = self.get_feature(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                                   mask_token_idx=x["mask_token_indexes"])
        # print(feature.shape)
        mlm_predictions = self.head(feature)  # [B, num_vocab]
        # print(mlm_predictions.shape)
        if self.hyper_config.model.task_type == "regression":
            pos_logits = torch.mean(mlm_predictions[:, self.positive_token_ids], dim=-1)  # .unsqueeze(-1)
            neg_logits = torch.mean(mlm_predictions[:, self.negative_token_ids], dim=-1)
            logits = torch.cat([pos_logits.reshape(-1, 1), neg_logits.reshape(-1, 1)], dim=-1)
            # print(logits.shape)
            logits = F.softmax(logits, dim=-1)
            # print(logits.shape)
            logits = logits[:, 0]
        else:
            logits = mlm_predictions[:, self.positive_token_ids]
        # negative_scores = torch.mean(mlm_predictions[:, self.negative_token_ids], dim=-1).unsqueeze(-1)
        # logits = torch.cat([positive_scores, negative_scores], dim=1)
        # logits = F.softmax(logits, dim=-1)
        # logits = logits[0]
        return logits

    def compute_loss(self, logits, labels):
        if self.hyper_config.model.task_type == "regression":
            loss = self.criterion(logits.reshape(-1, ), labels.reshape(-1, ))
        else:
            loss = self.criterion(logits.reshape(-1, 6), (labels - 1).long().reshape(-1, ))
        if self.hyper_config.train.loss_factor > 0:
            ranking_loss_fn = RankingLoss()
            loss += ranking_loss_fn(logits.reshape(-1, ), labels.reshape(-1, )) * self.hyper_config.train.loss_factor
        return loss


if __name__ == '__main__':
    from omegaconf import OmegaConf

    config = OmegaConf.load("/Users/heyao/projects/kaggle-aes2/training/baseline_reg/config.yaml")
    config.model.positive_token_ids = [127, 128]
    config.model.negative_token_ids = [129, 130]
    model = AESRegressionModel(config)
    print(model)
    input_ids = torch.LongTensor([[1, 232, 123, 2312, 412, 128000, 123, 2]])
    # logits = torch.randn((4, 1))
    labels = torch.FloatTensor([3])
    logits = model({"input_ids": input_ids, "attention_mask": None})
    print(logits)
    print(model.compute_loss(logits, labels))
