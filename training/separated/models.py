"""
@created by: heyao
@created at: 2024-02-03 18:03:14
"""
from abc import ABC

import torch
import torch.nn as nn
from torch.autograd import Function
from omegaconf import DictConfig
from transformers import AutoModel, T5EncoderModel, AutoConfig

from aes2.nn.heads.msd import MultiSampleDropout
from aes2.nn.poolers import MultiPooling
from aes2.nn.losses import RankingLoss, RMSELoss
from aes2.utils.stable_training import reinit_last_layers


class SeparatedHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(SeparatedHead, self).__init__()
        # Define the two heads as separate linear layers
        self.head_pc2 = nn.Linear(input_size, output_size)
        self.head_ko = nn.Linear(input_size, output_size)

    def forward(self, x, is_pc2):
        # x is the input features
        # is_pc2 is a list or tensor indicating which head to use for each sample
        # Convert is_pc2 to a tensor if it's not already
        is_pc2 = (is_pc2 == 1)
        # Output tensor
        # Apply head1 to samples where is_pc2 is 1
        ko_logits = self.head_ko(x[~is_pc2])
        output = torch.zeros((x.size(0), self.head_ko.out_features), dtype=ko_logits.dtype, device=x.device)
        output[is_pc2] = self.head_pc2(x[is_pc2])
        # Apply head2 to samples where is_pc2 is 0
        output[~is_pc2] = ko_logits
        return output


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
        feature_size = self.get_hidden_size(hyper_config.model.pooling, self.backbone.config.hidden_size)
        self.n_labels = self.hyper_config.model.n_labels
        if self.hyper_config.model.msd:
            self.head = MultiSampleDropout(feature_size, self.n_labels)
        else:
            self.head = SeparatedHead(feature_size, self.n_labels)
        self.criterion = self.get_loss_function(hyper_config)
        # tricks
        reinit_last_layers(self.backbone, num_layers=hyper_config.model.num_reinit_layers)
        if hyper_config.train.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            self.backbone.config.use_cache = False

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
            return nn.MSELoss(reduction="none")
        if config.train.loss == "rmse":
            return RMSELoss(reduction="none")
        if config.train.loss == "ce":
            return nn.CrossEntropyLoss(reduction="none", label_smoothing=config.model.get("label_smoothing", 0.0))

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
            pooler_output = self.customer_pooling(hidden_states[-1], attention_mask)
        return pooler_output

    def forward(self, x):
        feature = self.get_feature(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        logits = self.head(feature, x["is_pc2"])
        return logits

    def compute_loss_for_adv(self, logits, labels, is_pc2):
        assert isinstance(logits, tuple)
        logits, domain_logits = logits
        loss = self.compute_loss(logits, labels)
        criterion = nn.BCEWithLogitsLoss()
        domain_loss = criterion(domain_logits.reshape(-1, ), is_pc2.float().reshape(-1, ))
        return loss + domain_loss

    def compute_loss(self, logits, labels, is_pc2=None):
        if self.hyper_config.train.loss in ("mse", "rmse"):
            loss = self.criterion(logits.reshape(-1, ), labels.reshape(-1, ))
        else:
            loss = self.criterion(logits.reshape(-1, self.hyper_config.model.n_labels),
                                  (labels - 1).long().reshape(-1, ))
        if is_pc2 is not None:
            with torch.no_grad():
                weights = torch.where(is_pc2 == 1, self.hyper_config.train.pc2_loss_factor,
                                      self.hyper_config.train.new_loss_factor)
            loss = weights.reshape(-1, ) * loss.reshape(-1, )
        loss = loss.mean()
        if self.hyper_config.train.loss_factor > 0:
            ranking_loss_fn = RankingLoss()
            if self.hyper_config.model.task_type == "regression":
                loss += ranking_loss_fn(logits.reshape(-1, ),
                                        labels.reshape(-1, )) * self.hyper_config.train.loss_factor
            else:
                loss += ranking_loss_fn(logits.argmax(-1).reshape(-1, ) + 1,
                                        labels.reshape(-1, )) * self.hyper_config.train.loss_factor
        return loss


if __name__ == '__main__':
    from omegaconf import OmegaConf

    config = OmegaConf.load("/Users/heyao/projects/kaggle-aes2/training/baseline_reg/config.yaml")
    model = AESRegressionModel(config)
    print(model)
    logits = torch.randn((4, 1))
    labels = torch.FloatTensor([1, 2, 3, 4])
    print(model.compute_loss(logits, labels))
