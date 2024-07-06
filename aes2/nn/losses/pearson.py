"""
@created by: heyao
@created at: 2024-06-28 16:24:30
"""
import torch

import torch.nn as nn


def pearson_correlation_loss(pred, target):
    # 计算均值

    mean_pred = pred.mean(dim=0)

    mean_target = target.mean(dim=0)

    # 计算去均值后的向量

    std_pred = pred - mean_pred

    std_target = target - mean_target

    # 计算协方差

    covariance = (std_pred * std_target).sum(dim=0) / (pred.size(0) - 1)

    # 计算标准差

    std_pred = std_pred.norm(dim=0) / torch.sqrt(torch.tensor(pred.size(0) - 1))

    std_target = std_target.norm(dim=0) / torch.sqrt(torch.tensor(pred.size(0) - 1))

    # 计算皮尔逊相关系数

    correlation_coefficient = covariance / (std_pred * std_target)

    # 皮尔逊相关系数的损失函数通常是1 - 相关系数，因为我们要最大化相关性

    loss = 1 - correlation_coefficient.mean()

    return loss


if __name__ == '__main__':
    pred = torch.randn((4, 1))
    labels = torch.tensor([1, 2, 1, 6], dtype=torch.float32)
    print(pearson_correlation_loss(pred, labels))
