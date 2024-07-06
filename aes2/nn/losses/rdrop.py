"""
@created by: heyao
@created at: 2022-08-25 01:18:08
"""
import torch.nn as nn
import torch.nn.functional as F


class RDropLoss(nn.Module):
    """from: https://github.com/dropreg/R-Drop"""

    def __init__(self, alpha=0.5, reduction="mean"):
        super(RDropLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.reduction = reduction

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        if self.reduction == "sum":
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()
        elif self.reduction == "mean":
            p_loss = p_loss.mean()
            q_loss = q_loss.mean()
        elif self.reduction == "none":
            raise ValueError("reduction should be sum or mean")

        loss = (p_loss + q_loss) / 2
        return loss

    def forward(self, logits1, logits2, labels):
        ce_loss = 0.5 * (self.cross_entropy(logits1, labels) + self.cross_entropy(logits2, labels))
        kl_loss = self.compute_kl_loss(logits1, logits2)
        loss = ce_loss + self.alpha * kl_loss
        return loss
