"""
@created by: heyao
@created at: 2022-08-25 00:45:27
"""


class EMA(object):
    def __init__(self, model, decay=0.999):
        """
        ```python
        # 初始化
        ema = EMA(model, 0.999)
        ema.register()  # register is to add model weights to the RAM

        # 训练过程中，更新完参数后，同步update shadow weights
        def train():
            optimizer.step()
            ema.update()

        # eval前，apply shadow weights；eval之后，恢复原来模型的参数
        def evaluate():
            ema.apply_shadow()
            # evaluate
            ema.restore()
        ```
        :param model:
        :param decay:
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
