"""
@created by: heyao
@created at: 2022-09-02 20:15:07
"""
import numpy as np
import torch

from aes2.utils.aug.mask import regular_mask_aug_without_pad


class CompetitionDataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None, mask_ratio=0, tokenizer=None):
        self.x = x
        self.y = y
        self.mask_ratio = mask_ratio
        self.tokenizer = tokenizer
        if mask_ratio != 0 and tokenizer is None:
            raise RuntimeError("mask aug must input tokenizer")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        if self.mask_ratio:
            x["input_ids"] = regular_mask_aug_without_pad(np.array(x["input_ids"]),
                                                          self.tokenizer, mask_ratio=self.mask_ratio).tolist()
        if self.y is None:
            return x
        return x, self.y[item]
