"""
@created by: heyao
@created at: 2022-09-05 20:54:33
"""
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CumsumMeter(object):
    def __init__(self, score_function):
        self.reset()
        self.score_function = score_function

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, target, pred):
        self.targets += target.tolist()
        self.predictions += pred.tolist()

    @property
    def score(self):
        target = np.array(self.targets)
        predictions = np.array(self.predictions)
        target = target
        predictions = predictions
        return self.score_function(target, predictions)
