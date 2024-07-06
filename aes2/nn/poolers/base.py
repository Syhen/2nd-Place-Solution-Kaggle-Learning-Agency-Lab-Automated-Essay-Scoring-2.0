"""
@created by: heyao
@created at: 2022-09-04 19:43:06
"""
import torch.nn as nn


class BasePooling(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, mask=None, word_ids=None):
        raise NotImplementedError()


class IdentityPooling(BasePooling):
    def __init__(self, **kwargs):
        super(IdentityPooling, self).__init__()

    def forward(self, x, mask=None, word_ids=None):
        return x
