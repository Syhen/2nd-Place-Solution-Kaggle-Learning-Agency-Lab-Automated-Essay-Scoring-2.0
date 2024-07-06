"""
@created by: heyao
@created at: 2024-05-03 18:21:11
"""
import torch


class AWP(object):
    def __init__(self, model, optimizer, accelerator, adv_param="weight", adv_lr=1.0, adv_eps=0.01, start_epoch=1,
                 adv_step=1):
        """[Adversarial Weight Perturbation Helps Robust Generalization](https://arxiv.org/abs/2004.05884)

        Usage:
        ```python
        # ============ add awp object ============
        awp = AWP(model, optimizer)

        ...
        torch.nn.utils.clip_grad_norm_(
            parameters=self.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()

        # ============ run after normal training ============
        adv_loss = awp.attack_backward(x, y, epoch=current_epoch)
        adv_loss.backward()
        awp._restore()
        optimizer.step()

        # ============ update scheduler ============
        scheduler.step()
        ...
        ```
        :param model: `nn.Module`.
        :param optimizer:
        :param adv_param: str.
        :param adv_lr: float. default 1.0 for some large model, freeze some weight and use 1.0 is great. May also need
            differential learning rate etc.
        :param adv_eps: float. default 0.01. if validation loss keep big, reduce it.
        :param start_epoch: int. default 1. also, we may need a start score.
        :param adv_step: int. default 1.
        """
        self.model = model
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, x, label, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for _ in range(self.adv_step):
            self._attack_step()
            logits = self.model(x)
            adv_loss = self.model.compute_loss(logits, label)
            self.accelerator.backward(adv_loss)
            self._restore()
            self.optimizer.step()
        return

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
