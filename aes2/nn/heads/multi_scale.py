"""
@created by: heyao
@created at: 2024-05-30 16:48:51
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertAttention

from aes2.nn.poolers import MeanPooling


class ContextPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        context_token = hidden_states[:, 0]
        pooled_output = self.dense(context_token)
        pooled_output = F.relu(pooled_output)  # TODO: check which activation to use
        return pooled_output


class MultiScaleHead(nn.Module):
    """
    extract features from backbone outputs
        - multi-head attention mechanism
        - weighted average of top transformer layers
        - multi-scale essay representations
    """

    def __init__(self, config):
        super(MultiScaleHead, self).__init__()

        self.config = config
        self.num_features = config["hidden_size"]
        self.num_layers = config["num_layers"]

        ###################################################################################
        ###### Common-Block ###############################################################
        ###################################################################################

        init_amax = 5
        weight_data = torch.linspace(-init_amax, init_amax, self.num_layers)
        weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weights = nn.Parameter(weight_data, requires_grad=True)

        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": 4,
                "hidden_size": self.num_features,
                "attention_probs_dropout_prob": 0.0,
                "hidden_dropout_prob": 0.0,
                "is_decoder": False,
            }
        )
        self.mha_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
        self.attention = BertAttention(attention_config, position_embedding_type="absolute")

        ###################################################################################
        ###### Paragraph Scale ############################################################
        ###################################################################################

        paragraph_attention_config = BertConfig()
        paragraph_attention_config.update(
            {
                "num_attention_heads": 4,  # 4,
                "hidden_size": self.num_features,
                "attention_probs_dropout_prob": 0.0,
                "hidden_dropout_prob": 0.0,
                "is_decoder": False,
            }
        )
        self.paragraph_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
        self.paragraph_attention = BertAttention(paragraph_attention_config, position_embedding_type="absolute")

        self.paragraph_lstm_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
        self.paragraph_lstm_layer = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.num_features // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        ###################################################################################
        ###### Sentence Scale #############################################################
        ###################################################################################

        sentence_attention_config = BertConfig()
        sentence_attention_config.update(
            {
                "num_attention_heads": 4,  # 4,
                "hidden_size": self.num_features,
                "attention_probs_dropout_prob": 0.0,
                "hidden_dropout_prob": 0.0,
                "is_decoder": False,
            }
        )
        self.sentence_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
        self.sentence_attention = BertAttention(sentence_attention_config, position_embedding_type="absolute")

        self.sentence_lstm_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
        self.sentence_lstm_layer = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.num_features // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        ###################################################################################
        ###### Pooling ####################################################################
        ###################################################################################
        self.pool = MeanPooling()
        self.context_pool = ContextPooler(self.num_features)

        ###################################################################################
        ###### Classifiers ################################################################
        ###################################################################################
        self.context_pool_classifier = nn.Linear(self.num_features, 1)
        self.mean_pool_classifier = nn.Linear(self.num_features, 1)
        self.paragraph_pool_classifier = nn.Linear(self.num_features, 1)
        self.sentence_pool_classifier = nn.Linear(self.num_features, 1)

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
    ):

        ###################################################################################
        ###### Scorer 1: Context Pool #####################################################
        ###################################################################################
        context_features = backbone_outputs[0]
        context_features = self.context_pool(context_features)
        context_logits = self.context_pool_classifier(context_features)

        ###################################################################################
        ###### Scorer 2: Mean Pool ########################################################
        ###################################################################################

        x = torch.stack(backbone_outputs.hidden_states[-self.num_layers:])
        w = F.softmax(self.weights, dim=0)
        encoder_layer = (w * x).sum(dim=0)  # (bs, max_len, hidden_size)

        encoder_layer = self.mha_layer_norm(encoder_layer)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_layer = self.attention(encoder_layer, extended_attention_mask)[0]

        mean_pool_features = self.pool(encoder_layer, attention_mask)
        mean_pool_logits = self.context_pool_classifier(mean_pool_features)

        ###################################################################################
        ###### Scorer 3: Paragraph Scale ##################################################
        ###################################################################################

        bs = encoder_layer.shape[0]
        paragraph_feature_vector = []

        for i in range(bs):
            span_vec_i = []
            for head, tail in zip(paragraph_head_idxs[i], paragraph_tail_idxs[i]):
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_spans, h)
            paragraph_feature_vector.append(span_vec_i)

        paragraph_feature_vector = torch.stack(paragraph_feature_vector)  # (bs, num_spans, h)
        paragraph_feature_vector = self.paragraph_layer_norm(paragraph_feature_vector)  # (bs, num_spans, h)

        self.paragraph_lstm_layer.flatten_parameters()
        paragraph_feature_vector = self.paragraph_lstm_layer(paragraph_feature_vector)[0]

        extended_attention_mask = paragraph_attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        paragraph_feature_vector = self.paragraph_attention(paragraph_feature_vector, extended_attention_mask)[0]

        paragraph_feature_vector = self.pool(paragraph_feature_vector, paragraph_attention_mask)
        paragraph_logits = self.paragraph_pool_classifier(paragraph_feature_vector)

        ###################################################################################
        ###### Scorer 4: Sentence Scale ###################################################
        ###################################################################################

        bs = encoder_layer.shape[0]
        sentence_feature_vector = []

        for i in range(bs):
            span_vec_i = []
            for head, tail in zip(sentence_head_idxs[i], sentence_tail_idxs[i]):
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_spans, h)
            sentence_feature_vector.append(span_vec_i)

        sentence_feature_vector = torch.stack(sentence_feature_vector)  # (bs, num_spans, h)
        sentence_feature_vector = self.sentence_layer_norm(sentence_feature_vector)  # (bs, num_spans, h)

        self.sentence_lstm_layer.flatten_parameters()
        sentence_feature_vector = self.sentence_lstm_layer(sentence_feature_vector)[0]

        extended_attention_mask = sentence_attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        sentence_feature_vector = self.sentence_attention(sentence_feature_vector, extended_attention_mask)[0]

        sentence_feature_vector = self.pool(sentence_feature_vector, sentence_attention_mask)
        sentence_logits = self.sentence_pool_classifier(sentence_feature_vector)

        ###################################################################################
        ###### Multi-Scale Ensemble #######################################################
        ###################################################################################
        logits = context_logits + mean_pool_logits + paragraph_logits + sentence_logits
        logits = logits.reshape(bs, 1)
        return logits
