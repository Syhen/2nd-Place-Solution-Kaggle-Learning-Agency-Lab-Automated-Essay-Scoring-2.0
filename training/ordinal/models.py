"""
@created by: heyao
@created at: 2024-02-03 18:03:14
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoModel, T5EncoderModel, AutoConfig

from aes2.nn.heads.msd import MultiSampleDropout
from aes2.nn.poolers import MultiPooling
from aes2.nn.losses import RankingLoss, RMSELoss
from aes2.utils.stable_training import reinit_last_layers
from aes2.nn.heads.mix_of_expert import SparseMoeBlock
from training.ordinal.deberta_v2 import DebertaV2Model

MAX_RATING = 6


class OrdinalRegressionLoss(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

    def forward(self, logits, labels):
        sets = []
        for i in range(MAX_RATING - 1):
            label_mask = labels > i - 1 + 1
            label_tensor = (labels[label_mask] > i + 1).to(torch.int64)
            sets.append([label_tensor, label_mask])

        loss = 0
        for i, (label_tensor, label_mask) in enumerate(sets):
            if len(label_tensor) < 1:
                continue
            logit = logits[label_mask, i]
            loss += -torch.sum(
                F.logsigmoid(logit) * label_tensor + (F.logsigmoid(logit) - logit) * (1 - label_tensor)
            )
        loss /= MAX_RATING
        return loss


@torch.no_grad()
def to_labels(labels: torch.Tensor, num_labels):
    labels = labels.clone()
    labels = labels - 1
    B = labels.shape[0]
    outputs = torch.zeros((B, num_labels - 1), dtype=torch.float32, device=labels.device)
    cols = torch.arange(outputs.shape[1], device=labels.device)
    mask = cols < labels.unsqueeze(1)
    outputs[mask] = 1
    return outputs


class AESRegressionModel(nn.Module):
    """
    [1, 2, 3, 4, 5, 6] -> [
      [0, 0, 0, 0, 0], 1
      [1, 0, 0, 0, 0], 2
      [1, 1, 0, 0, 0], 3
      [1, 1, 1, 0, 0], 4
      [1, 1, 1, 1, 0], 5
      [1, 1, 1, 1, 1], 6
    ]

    """

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
        feature_size = self.get_hidden_size(hyper_config.model.pooling, self.backbone.config.hidden_size)
        self.n_labels = self.hyper_config.model.n_labels
        self.hyper_config.model.use_moe = self.hyper_config.model.get("use_moe", False)
        self.hyper_config.model.n_experts = self.hyper_config.model.get("n_experts", 4)
        if self.hyper_config.model.use_moe:
            self.moe_module = SparseMoeBlock(
                self.backbone.config.hidden_size, self.backbone.config.hidden_size,
                self.hyper_config.model.n_experts
            )
        if self.hyper_config.model.msd:
            self.head = MultiSampleDropout(feature_size, self.n_labels - 1)
        else:
            self.head = nn.Linear(feature_size, self.n_labels - 1)
        if self.hyper_config.train.hyper_loss_factor:
            self.head_clf = nn.Linear(feature_size, self.n_labels)
        self.criterion = self.get_loss_function(hyper_config)
        # tricks
        reinit_last_layers(self.backbone, num_layers=hyper_config.model.num_reinit_layers)
        if hyper_config.train.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            self.backbone.config.use_cache = False

    def init_backbone(self, model_path):
        if "t5" in model_path:
            model_class = T5EncoderModel
        elif self.hyper_config.train.use_cope:
            model_class = DebertaV2Model
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
        return nn.BCEWithLogitsLoss(reduction="none")

    def get_hidden_size(self, pooling_name, hidden_size):
        def _get_hidden_size(pooling_name, hidden_size):
            if any(i in pooling_name for i in ["meanmax"]):
                return hidden_size * 2
            return hidden_size

        return sum(_get_hidden_size(name, hidden_size) for name in pooling_name.split("_"))

    def get_feature(self, input_ids, attention_mask=None):
        model_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(model_output, "hidden_states"):
            hidden_states = model_output.hidden_states
        else:
            hidden_states = [model_output.last_hidden_state]
        if "weighted" in self.hyper_config.model.pooling:
            pooler_output = self.customer_pooling(hidden_states, attention_mask)
        else:
            if self.hyper_config.model.use_moe:
                pooler_output = self.customer_pooling(self.moe_module(hidden_states[-1]), attention_mask)
            else:
                pooler_output = self.customer_pooling(hidden_states[-1], attention_mask)
        return pooler_output

    def forward(self, x):
        feature = self.get_feature(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        logits = self.head(feature)
        if self.hyper_config.train.hyper_loss_factor:
            return logits, self.head_clf(feature)
        return logits

    def compute_loss(self, logits, labels, is_pc2=None):
        if self.hyper_config.train.hyper_loss_factor:
            logits, logits_clf = logits
        new_labels = to_labels(labels, self.n_labels)
        # print(logits)
        # print(labels)
        # print(new_labels)
        loss = self.criterion(logits.reshape(-1, ), new_labels.reshape(-1, ))
        # print(loss)
        # if is_pc2 is not None:
        #     with torch.no_grad():
        #         weights = torch.where(is_pc2 == 1, self.hyper_config.train.pc2_loss_factor,
        #                               self.hyper_config.train.new_loss_factor)
        #     loss = weights.reshape(-1, ) * loss.reshape(-1, )
        loss = loss.mean()
        # print(loss)
        if self.hyper_config.train.loss_factor > 0:
            ranking_loss_fn = RankingLoss()
            if self.hyper_config.model.task_type == "regression":
                loss += ranking_loss_fn(logits.sigmoid().sum(dim=-1).reshape(-1, ) + 1,
                                        labels.reshape(-1, )) * self.hyper_config.train.loss_factor
            else:
                loss += ranking_loss_fn(logits.argmax(-1).reshape(-1, ) + 1,
                                        labels.reshape(-1, )) * self.hyper_config.train.loss_factor
        # print(loss)
        if self.hyper_config.train.hyper_loss_factor:
            loss_clf = nn.CrossEntropyLoss()(logits_clf.reshape(-1, self.hyper_config.model.n_labels),
                                             (labels - 1).long().reshape(-1, ))
            loss = loss + self.hyper_config.train.hyper_loss_factor * loss_clf
        return loss


if __name__ == '__main__':
    from omegaconf import OmegaConf

    config = OmegaConf.load("/Users/heyao/projects/kaggle-aes2/training/baseline_reg/config.yaml")
    config.model.task_type = "regression"
    config.model.n_labels = 6
    model = AESRegressionModel(config)
    print(model)
    logits = torch.randn((4, 5))
    labels = torch.FloatTensor([1, 2, 3, 4])
    print(logits)
    print(labels)
    print(model.compute_loss(logits, labels))
