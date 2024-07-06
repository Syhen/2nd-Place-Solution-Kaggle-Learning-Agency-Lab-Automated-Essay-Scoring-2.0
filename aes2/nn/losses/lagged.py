"""
@created by: heyao
@created at: 2024-05-24 02:37:32
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LaggedCE(nn.Module):
    def __init__(self, n_classes=6):
        super(LaggedCE, self).__init__()
        self.n_classes = n_classes

    def forward(self, y_pred, y_true):
        criterion = nn.CrossEntropyLoss()

        loss = criterion(y_pred.reshape(-1, self.n_classes), y_true.reshape(-1, ))

        for lag, w in [(1, 0.5), (2, 0.3), (3, 0.2)]:
            # negative lag loss
            # if target < 0, target = 0
            neg_lag_target = F.relu(y_true.reshape(-1) - lag)
            neg_lag_target = neg_lag_target.long()
            neg_lag_loss = criterion(y_pred.reshape(-1, self.n_classes), neg_lag_target)

            # positive lag loss
            # if target > 949, target = 949
            pos_lag_target = self.n_classes - 1 - F.relu((self.n_classes - 1 - (y_true.reshape(-1) + lag)))
            pos_lag_target = pos_lag_target.long()
            pos_lag_loss = criterion(y_pred.reshape(-1, self.n_classes), pos_lag_target)
            loss += (neg_lag_loss + pos_lag_loss) * w
        return loss


if __name__ == '__main__':
    criterion = LaggedCE()
    labels = torch.FloatTensor([1, 2, 3, 4, 5, 6]).long() - 1
    logits = torch.randn((6, 6))
    labels = logits.argmax(-1)
    print(logits.softmax(-1))
    print(labels)
    print(criterion(logits, labels))
