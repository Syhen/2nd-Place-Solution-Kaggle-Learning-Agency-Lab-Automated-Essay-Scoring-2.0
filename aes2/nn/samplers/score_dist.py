"""
@created by: heyao
@created at: 2024-05-17 01:49:43
"""
import random

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class BalancedSampler(Sampler):
    def __init__(self, data_source: pd.DataFrame, batch_size):
        super().__init__(data_source)
        data_source["index_num"] = data_source.reset_index(drop=True).index
        self.data_source = data_source
        self.batch_size = batch_size
        assert self.batch_size % 4 == 0, "batch_size much multiple of 4"
        self.pc2_batch_size = batch_size // 4 * 3
        self.ko_batch_size = batch_size // 4 * 1
        self._init_pool()

    def _init_pool(self):
        self.pc2 = self.data_source[self.data_source["is_pc2"] == 1].to_dict("records")
        self.ko = self.data_source[self.data_source["is_pc2"] == 0].to_dict("records")

    def _to_yield(self, ls):
        return pd.DataFrame(ls).index_num.to_list()

    def _sample(self, candidates, n):
        sampled = []
        for _ in range(n):
            if not len(candidates):
                break
            sampled.append(candidates.pop(random.randint(0, len(candidates) - 1)))
        return sampled

    def __iter__(self):
        while self.pc2 or self.ko:
            pc2 = self._sample(self.pc2, self.pc2_batch_size)
            ko = self._sample(self.ko, self.ko_batch_size)
            yield self._to_yield(pc2 + ko)
        self._init_pool()

    def __len__(self):
        return int(np.ceil(self.data_source.shape[0] / self.batch_size))
