"""
@created by: heyao
@created at: 2022-09-04 13:56:28
"""
import torch
import torch.nn as nn

from ..poolers.base import BasePooling


class LSTMPooling(BasePooling):
    def __init__(self, *, hidden_size, double_hidden_size=False, **kwargs):
        super().__init__()
        hidden_cells = hidden_size if double_hidden_size else hidden_size // 2
        self.lstm = nn.LSTM(hidden_size, hidden_cells, bidirectional=True, batch_first=True)

    def forward(self, x, mask=None, word_ids=None):
        feature, _ = self.lstm(x)
        return feature


class GRUPooling(BasePooling):
    def __init__(self, *, hidden_size, double_hidden_size=False, **kwargs):
        super().__init__()
        hidden_cells = hidden_size if double_hidden_size else hidden_size // 2
        self.lstm = nn.GRU(hidden_size, hidden_cells, bidirectional=True, batch_first=True)

    def forward(self, x, mask=None, word_ids=None):
        feature, _ = self.lstm(x)
        return feature


class WordLevelLSTM(BasePooling):
    def __init__(self, hidden_size, double_hidden_size=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.double_hidden_size = double_hidden_size
        hidden_cells = hidden_size if double_hidden_size else hidden_size // 2
        self.lstm = nn.LSTM(hidden_size, hidden_cells, bidirectional=True, batch_first=True)
        self.max_words = kwargs.get("max_words", 2560)

    def unsorted_segment_sum(self, data, segment_ids, num_segments, x=None):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """

        segment_ids = torch.repeat_interleave(segment_ids.unsqueeze(-1),
                                              repeats=data.shape[-1],
                                              dim=-1)
        shape = [data.shape[0], num_segments] + list(data.shape[2:])
        if x is None:
            x = torch.zeros(*shape, dtype=data.dtype, device=data.device)
        x.scatter_add_(1, segment_ids, data)
        return x

    def unsorted_segment_mean(self, data, segment_ids, num_segments):
        """
          Computes the mean along segments of a tensor. Analogous to tf.unsorted_segment_mean.

          :param data: A tensor whose segments are to be summed.
          :param segment_ids: The segment indices tensor.
          :param num_segments: The number of segments.
          :return: A tensor of same data type as the data argument.
          """
        tensor = self.unsorted_segment_sum(data, segment_ids, num_segments)
        base = self.unsorted_segment_sum(torch.ones_like(data), segment_ids, num_segments)
        ## base + 1e-5 lb 693, clamp 690 not sure if by randomness
        base = torch.clamp(base, min=1.)
        # base += 1e-5
        tensor = tensor / base
        return tensor

    def unsorted_segment_reduce(self, data, segment_ids, num_segments, combiner='sum'):
        if combiner == 'sum':
            return self.unsorted_segment_sum(data, segment_ids, num_segments)
        elif combiner == 'mean' or combiner == 'avg':
            return self.unsorted_segment_mean(data, segment_ids, num_segments)

    def groupby(self, logits, word_ids, combiner='sum'):
        word_ids_ = word_ids + 1
        word_ids_ *= (word_ids_ < self.max_words).long()
        logits = self.unsorted_segment_reduce(logits, word_ids_.long(), self.max_words + 1, combiner=combiner)
        return logits[:, 1:]

    def forward(self, x, mask=None, word_ids=None):
        x = self.groupby(x, word_ids=word_ids, combiner="mean")
        feature, _ = self.lstm(x)
        return feature


if __name__ == '__main__':
    x = torch.rand((4, 128, 768))
    pooling = LSTMPooling(hidden_size=768)
    print(pooling(x).shape)
