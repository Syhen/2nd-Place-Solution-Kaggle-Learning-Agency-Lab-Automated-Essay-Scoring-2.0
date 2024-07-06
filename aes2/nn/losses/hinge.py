"""
@created by: heyao
@created at: 2022-09-19 16:13:47
from: https://discuss.pytorch.org/t/non-linear-regression-methods/62060
"""
import torch


def hinge_loss(outputVal, dataOutput, model):
    loss1 = torch.sum(torch.clamp(1 - torch.matmul(outputVal.t(), dataOutput), min=0))
    loss2 = torch.sum(model.head.weight ** 2)  # l2 penalty
    totalLoss = loss1 + loss2
    return totalLoss / (outputVal.shape[0] * 6)
