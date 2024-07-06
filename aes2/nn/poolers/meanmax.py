"""
@created by: heyao
@created at: 2022-09-04 13:50:38
"""
import torch
import torch.nn as nn

from ..poolers.base import BasePooling


class MeanPooling(BasePooling):
    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__()

    def forward(self, x, mask=None, word_ids=None):
        if mask is not None:
            input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
            sum_embeddings = torch.sum(x * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
        else:
            mean_embeddings = torch.mean(x, dim=1)
        return mean_embeddings


class MaxPooling(BasePooling):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, mask=None, word_ids=None):
        if mask is None:
            return torch.max(x, dim=1)[0]
        input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        max_embeddings = torch.max(x * input_mask_expanded, dim=1)[0]
        return max_embeddings


class MeanMaxPooling(BasePooling):
    def __init__(self, **kwargs):
        super().__init__()
        self.mean_pooling = MeanPooling()
        self.max_pooling = MaxPooling()

    def forward(self, x, mask=None, word_ids=None):
        return torch.cat([self.mean_pooling(x, mask), self.max_pooling(x, mask)], dim=1)


if __name__ == '__main__':
    x = torch.rand((4, 128, 768))
    pooling = MeanMaxPooling()
    print(pooling(x, mask=torch.LongTensor([[1] * 78 + [0] * 50] * 4)).shape)
