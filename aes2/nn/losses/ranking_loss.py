"""
@created by: heyao
@created at: 2023-08-01 22:31:02
from Raja
"""
import torch
import torch.nn as nn


def get_ranking_loss(logits, labels, margin=0.0):
    labels1 = labels.unsqueeze(1)
    labels2 = labels.unsqueeze(0)

    logits1 = logits.unsqueeze(1)
    logits2 = logits.unsqueeze(0)

    y_ij = labels1 - labels2
    r_ij = (logits1 >= logits2).float() - (logits1 < logits2).float()

    residual = torch.clamp(-r_ij * y_ij + margin, min=0.0)
    loss = residual.mean()
    return loss


class RankingLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(RankingLoss, self).__init__()
        self.margin = margin

    def forward(self, logits, labels):
        return get_ranking_loss(logits, labels, self.margin)


import torch

import torch.nn.functional as F


class MultiPositiveRankingLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(MultiPositiveRankingLoss, self).__init__()

        self.margin = margin

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)

        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for positive and negative pairs

        labels = labels.unsqueeze(1)

        positive_mask = labels == labels.T

        negative_mask = ~positive_mask

        # Remove diagonal from positive mask to avoid comparing the sample with itself

        positive_mask.fill_diagonal_(False)

        # Extract positive and negative distances

        positive_distances = distances * positive_mask

        negative_distances = distances * negative_mask

        # Compute the loss

        positive_distances = positive_distances[positive_mask]

        negative_distances = negative_distances[negative_mask].view(batch_size, -1)

        # Expand positive distances to match the number of negatives for broadcasting

        positive_distances = positive_distances.unsqueeze(1).expand(-1, negative_distances.size(1))

        # Compute hinge loss

        loss = F.relu(self.margin + positive_distances - negative_distances).mean()

        return loss


class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, logits, labels):
        # 计算MSE损失
        mse = self.mse_loss(logits, labels)

        # 计算Pairwise Ranking Loss
        ranking_loss = 0
        num_pairs = 0

        for i in range(len(logits)):
            for j in range(i + 1, len(logits)):
                if labels[i] > labels[j]:
                    ranking_loss += torch.clamp(1 - (logits[i] - logits[j]), min=0)
                    num_pairs += 1

                elif labels[i] < labels[j]:
                    ranking_loss += torch.clamp(1 - (logits[j] - logits[i]), min=0)
                    num_pairs += 1

        if num_pairs > 0:
            ranking_loss /= num_pairs
        # 总损失
        total_loss = self.alpha * mse + (1 - self.alpha) * ranking_loss
        return total_loss


if __name__ == '__main__':
    labels = torch.FloatTensor([[0.1, 0.3], [-0.1, -0.5], [0.5, 0.1]])
    logits = torch.FloatTensor([[0.3, 0.5], [-0.2, -0.4], [0.2, -0.1]])
    print(get_ranking_loss(logits, labels))
