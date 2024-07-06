"""
@created by: heyao
@created at: 2022-07-08 23:18:35
"""
import torch
import torch.nn as nn

from ..poolers.base import BasePooling


class WeightedLayerPooling(BasePooling):
    def __init__(self, *, num_hidden_layers, layer_start: int = 4, layer_weights=None, **kwargs):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float))

    def forward(self, x, mask=None, word_ids=None):
        x = torch.cat([i.unsqueeze(0) for i in x])
        all_layer_embedding = x[self.layer_start:, ...]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average


if __name__ == '__main__':
    hidden_states = [torch.rand((12, 128, 768)) for _ in range(13)]
    # hidden_states = torch.cat([i.unsqueeze(0) for i in hidden_states])
    pooling = WeightedLayerPooling(num_hidden_layers=12, layer_start=8)
    print(pooling(hidden_states).shape)
