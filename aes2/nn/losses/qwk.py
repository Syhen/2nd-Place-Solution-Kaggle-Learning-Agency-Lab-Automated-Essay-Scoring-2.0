"""
@created by: heyao
@created at: 2024-05-08 17:33:39
"""
import torch
import torch.nn.functional as F


def qwk_loss(output, target, n_classes):
    """
    Differentiable approximation of Quadratic Weighted Kappa loss.
    """

    # Ensure the output is of size (batch_size, n_classes)
    output = output.view(-1, n_classes)
    target = target.view(-1)
    # Convert output to probabilities (softmax) and target to one-hot encoding
    output_prob = F.softmax(output, dim=1)
    target_one_hot = F.one_hot(target, num_classes=n_classes).float()

    # Weight matrix calculation
    w = torch.zeros((n_classes, n_classes), device=output.device)
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = (i - j) ** 2 / (n_classes - 1) ** 2
    w += 1

    # # Outer product of probabilities and one-hot targets
    # output_prob_ext = output_prob.unsqueeze(2)
    # target_one_hot_ext = target_one_hot.unsqueeze(1)
    # hist_rater_a = output_prob_ext * target_one_hot_ext
    # hist_rater_b = target_one_hot_ext * output_prob_ext

    # Confusion matrix
    print(output_prob)
    print(target_one_hot)
    conf_mat = torch.matmul(output_prob.transpose(1, 0), target_one_hot)

    # Normalized confusion matrix
    conf_mat = conf_mat  # / conf_mat.sum()

    # Compute the QWK loss
    print(f"{conf_mat = }")
    numerator = torch.sum(w * conf_mat)
    # expected_probs = torch.matmul(output_prob.transpose(1, 0), output_prob)
    expected_probs = torch.matmul(target_one_hot.transpose(1, 0), target_one_hot)
    print(f"{expected_probs = }")
    print(f"{conf_mat - expected_probs = }")
    print(w)
    denominator = torch.sum(w * expected_probs)
    return numerator / denominator


import torch

import torch.nn as nn

import torch.nn.functional as F


class WeightedKappaLoss(nn.Module):
    def __init__(self, num_classes, weight_type='quadratic'):

        super(WeightedKappaLoss, self).__init__()

        self.num_classes = num_classes

        self.weight_type = weight_type

        # Create weight matrix

        self.weight_matrix = self.create_weight_matrix()

    def create_weight_matrix(self):

        weight_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)

        for i in range(self.num_classes):

            for j in range(self.num_classes):

                if self.weight_type == 'linear':

                    weight_matrix[i, j] = abs(i - j)

                elif self.weight_type == 'quadratic':

                    weight_matrix[i, j] = (i - j) ** 2

                else:

                    raise ValueError("weight_type must be 'linear' or 'quadratic'")

        return weight_matrix

    def forward(self, y_pred, y_true):

        y_true = y_true.view(-1).long()

        y_pred = F.softmax(y_pred, dim=1)

        # Calculate the confusion matrix

        hist_true = torch.histc(y_true.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

        hist_pred = torch.histc(y_pred.argmax(dim=1).float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

        confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)

        for t, p in zip(y_true, y_pred.argmax(dim=1)):
            confusion_matrix[t, p] += 1

        # Normalize the confusion matrix

        confusion_matrix = confusion_matrix / confusion_matrix.sum()

        # Calculate expected matrix

        expected_matrix = torch.outer(hist_true, hist_pred)

        expected_matrix = expected_matrix / expected_matrix.sum()

        # Calculate the weighted kappa

        weight_matrix = self.weight_matrix.to(y_pred.device)

        kappa = 1 - (weight_matrix * confusion_matrix).sum() / (weight_matrix * expected_matrix).sum()
        # kappa = torch.abs((weight_matrix * confusion_matrix).sum() / (weight_matrix * expected_matrix).sum())

        return kappa


if __name__ == '__main__':
    # Example usage
    n_classes = 5
    output = torch.randn(10, n_classes, requires_grad=True)  # Example logits from a model
    target = torch.randint(0, n_classes, (10,))  # Example target labels
    print(output.softmax(-1).argmax(-1))
    target = output.softmax(-1).argmax(-1)
    print(target)
    qwk_loss = WeightedKappaLoss(n_classes)
    loss = qwk_loss(output, target)
    print("QWK Loss:", loss.item())
