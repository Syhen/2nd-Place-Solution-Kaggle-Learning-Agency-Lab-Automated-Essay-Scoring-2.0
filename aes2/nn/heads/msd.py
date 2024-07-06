"""
@created by: heyao
@created at: 2022-09-04 20:22:16
"""
import torch.nn as nn


class MultiSampleDropout(nn.Module):
    def __init__(self, feature_size, out_size):
        super(MultiSampleDropout, self).__init__()
        self.dropout0 = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.dropout4 = nn.Dropout(0.5)
        self.head = nn.Linear(feature_size, out_size)

    def forward(self, x):
        x1 = self.head(self.dropout0(x))
        x2 = self.head(self.dropout1(x))
        x3 = self.head(self.dropout2(x))
        x4 = self.head(self.dropout3(x))
        x5 = self.head(self.dropout4(x))
        return (x1 + x2 + x3 + x4 + x5) / 5


if __name__ == '__main__':
    import torch

    x = torch.rand((12, 768))
    head = MultiSampleDropout(768, 10)
    print(head(x).shape)
