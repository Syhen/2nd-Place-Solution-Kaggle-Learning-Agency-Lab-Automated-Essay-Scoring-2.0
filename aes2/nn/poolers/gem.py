"""
@created by: heyao
@created at: 2024-06-20 01:34:50
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from aes2.nn.poolers import BasePooling
from aes2.nn.poolers.meanmax import MeanPooling


class GeMPooling(BasePooling):
    def __init__(self, p=0.2, eps=1e-6, **kwargs):
        super(GeMPooling, self).__init__()
        # self.p = nn.Parameter(torch.ones(1) * p)
        self.p = p
        self.eps = eps
        self.mean_pool = MeanPooling(**kwargs)

    def forward(self, x, mask=None, word_ids=None):
        return self.gem(x, p=self.p, eps=self.eps, mask=None)

    def gem(self, x, p=3, eps=1e-6, mask=None):
        # return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
        return (torch.mean(x.pow(p), dim=1) + eps).pow(1 / p)

    def __repr__(self):
        return self.__class__.__name__ + \
               '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
               ', ' + 'eps=' + str(self.eps) + ')'
