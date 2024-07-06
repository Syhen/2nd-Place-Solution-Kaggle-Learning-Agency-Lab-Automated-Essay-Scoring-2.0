"""
@created by: heyao
@created at: 2022-09-04 21:57:39
"""
import torch


def mcrmse(outputs, targets):
    colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
    loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
    return loss
