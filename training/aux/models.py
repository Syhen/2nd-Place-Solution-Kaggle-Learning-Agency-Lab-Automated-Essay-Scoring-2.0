"""
@created by: heyao
@created at: 2024-02-03 18:03:14
"""
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel, T5EncoderModel, AutoConfig

from aes2.nn.heads.msd import MultiSampleDropout
from aes2.nn.poolers import MultiPooling
from aes2.nn.losses import RankingLoss
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
        feature_size = self.get_hidden_size(hyper_config.model.pooling, self.backbone.config.hidden_size)
        if self.hyper_config.model.msd:
            self.head = MultiSampleDropout(feature_size, 1)
        else:
            self.head = nn.Linear(feature_size, 1)
        self.aux_head = nn.Linear(feature_size, len(self.hyper_config.multi_task.tasks))
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
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        return model_class.from_pretrained(model_path, config=config)

    def get_loss_function(self, config):
        if config.train.loss == "mse":
            return nn.MSELoss()
        if config.train.loss == "rmse":
            return lambda x, y: torch.sqrt(nn.MSELoss()(x, y))

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

    def forward(self, x, do_aux=True):
        feature = self.get_feature(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        logits = self.head(feature)
        if do_aux:
            aux_logits = self.aux_head(feature)
            return logits, aux_logits
        return logits

    def compute_loss(self, logits, labels):
        logits, aux_logits = logits
        labels, aux_labels = labels
        loss = self.criterion(logits.reshape(-1, ), labels.reshape(-1, ))
        if self.hyper_config.train.loss_factor > 0:
            ranking_loss_fn = RankingLoss()
            loss += ranking_loss_fn(logits.reshape(-1, ), labels.reshape(-1, )) * self.hyper_config.train.loss_factor
        aux_loss = 0
        for i in range(logits.shape[1]):
            aux_loss += self.criterion(aux_logits[:, i].reshape(-1, ), aux_labels[:, i].reshape(-1, ))
        aux_loss /= logits.shape[1]
        loss += aux_loss * self.hyper_config.multi_task.weight
        return loss


if __name__ == '__main__':
    from omegaconf import OmegaConf

    config = OmegaConf.load("/Users/heyao/projects/kaggle-aes2/training/baseline_reg/config.yaml")
    model = AESRegressionModel(config)
    print(model)
    logits = torch.randn((4, 1))
    labels = torch.FloatTensor([1, 2, 3, 4])
    print(model.compute_loss(logits, labels))
