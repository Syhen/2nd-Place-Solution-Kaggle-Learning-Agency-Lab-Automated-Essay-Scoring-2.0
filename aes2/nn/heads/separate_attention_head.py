"""
@created by: heyao
@created at: 2022-09-21 23:09:45
"""
import torch
import torch.nn as nn

from ..poolers import MultiPooling


class SeparatedHead(nn.Module):
    def __init__(self, in_features, out_features, pooling="attention"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pooler = nn.ModuleList([
            nn.Sequential(
                MultiPooling(pooling, hidden_size=in_features),
                nn.Linear(in_features, 1)
            ) for _ in range(out_features)
        ])

    def forward(self, x, mask=None):
        outs = []
        for layer in self.pooler:
            outs.append(layer(x))
        return torch.cat(outs, dim=-1)


class ResidualSeparatedHead(nn.Module):
    def __init__(self, in_features, out_features, pooling="attention"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pooler = nn.ModuleList([
            nn.Sequential(
                MultiPooling(pooling, hidden_size=in_features),
                nn.Linear(in_features, 1)
            ) for _ in range(out_features)
        ])

    def forward(self, x, mask=None):
        outs = []
        for i, layer in enumerate(self.pooler):
            if i == 0:
                o_0 = layer(x)
            else:
                o_0 += layer(x)
            o = o_0 / (i + 1)
            outs.append(o)
        return torch.cat(outs, dim=-1)


if __name__ == '__main__':
    x = torch.rand((2, 768, 768))
    module = ResidualSeparatedHead(768, 6)
    print(module(x).shape)
