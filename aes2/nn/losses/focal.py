"""
@created by: heyao
@created at: 2022-08-25 01:17:06
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, reduction='none', alpha=1, gamma=2):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1. - pt) ** self.gamma * bce_loss
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class MultiFocalLoss(nn.Module):
    def __init__(self, reduction='none', alpha=1, gamma=2):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets: torch.FloatTensor):
        bce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=-100)
        pt = torch.exp(-bce_loss)
        # print(((1. - pt) ** self.gamma * bce_loss).shape)
        mask = targets == -100
        targets[mask] = 0
        alpha = self.alpha.gather(dim=0, index=targets)
        targets[mask] = -100
        loss = alpha * (1. - pt) ** self.gamma * bce_loss
        loss = loss[targets != -100]
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class SmoothFocalLoss(nn.Module):
    def __init__(self, reduction='none', alpha=1, gamma=2, smoothing=0.0):
        super().__init__()
        self.reduction = reduction
        self.focal_loss = FocalLoss(reduction='none', alpha=alpha, gamma=gamma)
        self.smoothing = smoothing

    @staticmethod
    def _smooth(targets: torch.Tensor, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothFocalLoss._smooth(targets, self.smoothing)
        loss = self.focal_loss(inputs, targets)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        """
        :param alpha: (tensor) 1D tensor containing the class weights, size [num_classes]
        :param gamma: (float) focusing parameter for modulating factor (1 - p_t)
        :param ignore_index: (int) index that indicates ignored tokens
        :param reduction: (str) reduction method to apply to the output: 'none', 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        :param logits: (tensor) model outputs, size [batch_size, seq_length, num_classes]
        :param targets: (tensor) ground truth labels, size [batch_size, seq_length]
        """
        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probabilities with respect to target class
        with torch.no_grad():
            new_targets = deepcopy(targets)
            mask = new_targets == -100
            new_targets[mask] = 0
        log_probs = log_probs.gather(dim=-1, index=new_targets.unsqueeze(-1)).squeeze(-1)

        # Calculate the probabilities of the targets
        probs = log_probs.exp()

        # Calculate focal loss
        focal_loss = -(1 - probs) ** self.gamma * log_probs

        # Apply weights if provided
        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            # Gather weights with respect to target class
            alpha = self.alpha.gather(dim=0, index=new_targets)
            focal_loss = alpha * focal_loss

        # Handle ignore_index by setting loss to zero for those tokens
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            focal_loss = focal_loss * mask

        # Apply reduction
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss


if __name__ == '__main__':
    # Example usage:
    batch_size = 10
    seq_length = 50
    num_classes = 5

    # Example alpha for class imbalance (make sure to normalize it so that sum(alpha) = 1)
    alpha = torch.tensor([0.25, 0.75, 0.75, 0.5, 0.5])

    # Example logits and targets
    logits = torch.randn(batch_size, seq_length, num_classes)
    targets = torch.randint(0, num_classes, (batch_size, seq_length))

    # Create the FocalLoss object
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0, ignore_index=-100, reduction='mean')

    # Calculate the loss
    loss = focal_loss(logits.view(-1, num_classes).to("mps"), targets.view(-1, ).to("mps"))
    print(loss)
    print(nn.CrossEntropyLoss()(logits.view(-1, num_classes), targets.view(-1, )))
