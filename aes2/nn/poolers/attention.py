"""
@created by: heyao
@created at: 2022-06-23 13:02:06
"""
import torch
import torch.nn as nn

from ..heads import GAU
from ..poolers.base import BasePooling


class AttentionPooling(BasePooling):
    def __init__(self, *, hidden_size, **kwargs):
        super(AttentionPooling, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, mask=None, word_ids=None):
        w = self.attention(x).float()
        if mask is not None:
            w[mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        x = torch.sum(w * x, dim=1)
        return x


class GAUPooling(BasePooling):
    def __init__(self, *, hidden_size, **kwargs):
        super(GAUPooling, self).__init__()
        self.gau = GAU(dim=hidden_size)

    def forward(self, x, mask=None, word_ids=None):
        return self.gau(x)


if __name__ == '__main__':
    x = torch.rand((4, 128, 768))
    pooling = GAUPooling(hidden_size=768)
    print(pooling(x).shape)
