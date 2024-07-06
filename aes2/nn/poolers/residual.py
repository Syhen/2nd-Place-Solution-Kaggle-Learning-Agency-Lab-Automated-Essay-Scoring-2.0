"""
@created by: heyao
@created at: 2022-09-22 02:24:26
"""
from ..poolers import BasePooling
from ..poolers.meanmax import MeanPooling


class ResidualLastLayersMeanPooling(BasePooling):
    def __init__(self, *, layer_start=16, **kwargs):
        super().__init__(**kwargs)
        self.layer_start = layer_start
        self.pooler = MeanPooling()

    def forward(self, x, mask=None, word_ids=None):
        out = self.pooler(x[self.layer_start], mask=mask)
        for hidden_state in x[self.layer_start + 1:]:
            out += self.pooler(hidden_state, mask=mask)
        return out / (len(x) - self.layer_start)
