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
from aes2.nn.poolers import MeanPooling
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


class MultiScaleHead(nn.Module):
    """
    extract features from backbone outputs
        - multi-head attention mechanism
        - weighted average of top transformer layers
        - multi-scale essay representations
    """

    def __init__(self, config, hidden_size):
        super(MultiScaleHead, self).__init__()

        self.config = config
        self.num_features = hidden_size
        self.num_layers = config.model.num_layers

        ###################################################################################
        ###### Common-Block ###############################################################
        ###################################################################################

        init_amax = 5
        weight_data = torch.linspace(-init_amax, init_amax, self.num_layers)
        weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weights = nn.Parameter(weight_data, requires_grad=True)

        ###################################################################################
        ###### Pooling ####################################################################
        ###################################################################################
        self.pool = MeanPooling()

        ###################################################################################
        ###### Classifiers ################################################################
        ###################################################################################
        # self.context_pool_classifier = nn.Linear(self.num_features, 1)
        self.mean_pool_classifier = nn.Linear(self.num_features, 5)
        # self.paragraph_pool_classifier = nn.Linear(self.num_features, 1)
        # self.sentence_pool_classifier = nn.Linear(self.num_features, 1)

    def forward(
            self,
            backbone_outputs,
            attention_mask,
            paragraph_head_idxs,
            paragraph_tail_idxs,
            paragraph_attention_mask,
            sentence_head_idxs,
            sentence_tail_idxs,
            sentence_attention_mask,
            is_pc2=None
    ):

        ###################################################################################
        ###### Scorer 2: Mean Pool ########################################################
        ###################################################################################

        # x = torch.stack(backbone_outputs.hidden_states[-self.num_layers:])
        # w = F.softmax(self.weights, dim=0)
        # encoder_layer = (w * x).sum(dim=0)  # (bs, max_len, hidden_size)
        encoder_layer = backbone_outputs
        # encoder_layer = self.mha_layer_norm(encoder_layer)

        paragraph_reps = []
        bs = backbone_outputs.shape[0]
        for i in range(bs):
            # print("head:", sentence_head_idxs[i])
            # print(sentence_tail_idxs[i])
            # print(paragraph_head_idxs[i])
            # print(paragraph_tail_idxs[i])
            sentence_reps = []
            for head, tail in zip(paragraph_head_idxs[i], paragraph_tail_idxs[i]):
                if tail - head <= 2:
                    continue
                idx = (head <= sentence_head_idxs[i]) & (sentence_head_idxs[i] <= tail)
                selected_head_idxs: torch.Tensor = sentence_head_idxs[i][idx]
                selected_tail_idxs: torch.Tensor = sentence_tail_idxs[i][idx]
                if not selected_head_idxs.any():
                    continue
                # print("select:", selected_head_idxs, selected_tail_idxs)
                # print(idx, head, tail)

                # sentence_rep = encoder_layer[i, selected_head_idxs[0]: selected_tail_idxs[-1] + 1, :]
                # print(selected_head_idxs)
                # print(encoder_layer[[i], selected_head_idxs, :].shape)
                # print(encoder_layer[[i], :, :].shape)
                # print("=" * 60)
                sentence_reps.append(torch.mean(encoder_layer[[i], selected_head_idxs, :].unsqueeze(dim=0), dim=1).unsqueeze(dim=1))
            paragraph_reps.append(torch.mean(torch.cat(sentence_reps, dim=1), dim=1))
        # print(paragraph_reps[0].shape, paragraph_reps[1].shape)
        final_features = torch.cat(paragraph_reps, dim=0)
        # print(final_features.shape)
        # print(f"{final_features = }")
        # print(f"{final_features.shape = }")

        # extended_attention_mask = attention_mask[:, None, None, :]
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # encoder_layer = self.attention(encoder_layer, extended_attention_mask)[0]
        # mean_pool_features = self.pool(encoder_layer, attention_mask)
        # mean_pool_logits = self.context_pool_classifier(mean_pool_features)

        ###################################################################################
        ###### Scorer 3: Paragraph Scale ##################################################
        ###################################################################################
        # print(self.mean_pool_classifier(final_features).shape)
        return self.mean_pool_classifier(final_features)

        # paragraph_feature_vector = []
        # for i in range(bs):
        #     span_vec_i = []
        #     for head, tail in zip(paragraph_head_idxs[i], paragraph_tail_idxs[i]):
        #         tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
        #         span_vec_i.append(tmp)
        #     span_vec_i = torch.stack(span_vec_i)  # (num_spans, h)
        #     paragraph_feature_vector.append(span_vec_i)
        #
        # paragraph_feature_vector = torch.stack(paragraph_feature_vector)  # (bs, num_spans, h)
        # paragraph_feature_vector = self.paragraph_layer_norm(paragraph_feature_vector)  # (bs, num_spans, h)
        #
        # self.paragraph_lstm_layer.flatten_parameters()
        # paragraph_feature_vector = self.paragraph_lstm_layer(paragraph_feature_vector)[0]
        #
        # extended_attention_mask = paragraph_attention_mask[:, None, None, :]
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # paragraph_feature_vector = self.paragraph_attention(paragraph_feature_vector, extended_attention_mask)[0]
        #
        # paragraph_feature_vector = self.pool(paragraph_feature_vector, paragraph_attention_mask)
        # paragraph_logits = self.paragraph_pool_classifier(paragraph_feature_vector)
        #
        # ###################################################################################
        # ###### Scorer 4: Sentence Scale ###################################################
        # ###################################################################################
        #
        # bs = encoder_layer.shape[0]
        # sentence_feature_vector = []
        #
        # for i in range(bs):
        #     span_vec_i = []
        #     for head, tail in zip(sentence_head_idxs[i], sentence_tail_idxs[i]):
        #         tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
        #         span_vec_i.append(tmp)
        #     span_vec_i = torch.stack(span_vec_i)  # (num_spans, h)
        #     sentence_feature_vector.append(span_vec_i)
        #
        # sentence_feature_vector = torch.stack(sentence_feature_vector)  # (bs, num_spans, h)
        # sentence_feature_vector = self.sentence_layer_norm(sentence_feature_vector)  # (bs, num_spans, h)
        #
        # self.sentence_lstm_layer.flatten_parameters()
        # sentence_feature_vector = self.sentence_lstm_layer(sentence_feature_vector)[0]
        #
        # extended_attention_mask = sentence_attention_mask[:, None, None, :]
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # sentence_feature_vector = self.sentence_attention(sentence_feature_vector, extended_attention_mask)[0]
        #
        # sentence_feature_vector = self.pool(sentence_feature_vector, sentence_attention_mask)
        # sentence_logits = self.sentence_pool_classifier(sentence_feature_vector)
        #
        # ###################################################################################
        # ###### Multi-Scale Ensemble #######################################################
        # ###################################################################################
        # logits = mean_pool_logits + paragraph_logits + sentence_logits
        # logits = logits.reshape(bs, 1)
        # return logits


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
    Text -> paragraphs -> sentences
    2 x mean pooling
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
        # if self.hyper_config.model.msd:
        #     self.head = MultiSampleDropout(feature_size, self.n_labels - 1)
        # else:
        #     self.head = nn.Linear(feature_size, self.n_labels - 1)
        self.head = MultiScaleHead(self.hyper_config, feature_size)
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
        return model_output.last_hidden_state
        # if hasattr(model_output, "hidden_states"):
        #     hidden_states = model_output.hidden_states
        # else:
        #     hidden_states = [model_output.last_hidden_state]
        # if "weighted" in self.hyper_config.model.pooling:
        #     pooler_output = self.customer_pooling(hidden_states, attention_mask)
        # else:
            # if self.hyper_config.model.use_moe:
            #     pooler_output = self.customer_pooling(self.moe_module(hidden_states[-1]), attention_mask)
            # else:
            #     pooler_output = self.customer_pooling(hidden_states[-1], attention_mask)
        # return pooler_output

    def forward(self, x):
        # print(x)
        feature = self.get_feature(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        logits = self.head(
            feature, x["attention_mask"], x["span_head_idxs"], x["span_tail_idxs"],
            x["span_attention_mask"], x["sentence_head_idxs"], x["sentence_tail_idxs"],
            x["sentence_attention_mask"]
        )
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
