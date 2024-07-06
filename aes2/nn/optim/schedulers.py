"""
@created by: heyao
@created at: 2023-08-08 18:19:40
"""
import math
import torch.optim.lr_scheduler as lr_scheduler


class CustomLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_lr, warmup_steps, total_steps, stop_lr, last_epoch=-1):
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.stop_lr = stop_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = self.max_lr * (self.last_epoch / max(1, self.warmup_steps))
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.stop_lr + 0.5 * (self.max_lr - self.stop_lr) * (1 + math.cos(math.pi * progress))
        return [lr for _ in self.base_lrs]


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.head = torch.nn.Linear(3, 1)

        def forward(self, x):
            return x
    # 使用示例
    # 定义模型和优化器
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    # 定义学习率调度器
    max_lr = 1e-5
    warmup_steps = 50
    total_steps = 5000
    stop_lr = 6e-6
    num_epochs = 5
    scheduler = CustomLRScheduler(optimizer, max_lr, warmup_steps, total_steps, stop_lr)

    lrs = []
    # 在训练循环中使用调度器
    for _ in range(total_steps):
        scheduler.step()
        lrs.append(scheduler.get_last_lr())
    plt.plot(lrs)
    plt.show()
