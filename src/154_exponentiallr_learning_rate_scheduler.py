class ExponentialLRScheduler:
    def __init__(self, initial_lr: float, gamma: float) -> None:
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * self.gamma**epoch


scheduler = ExponentialLRScheduler(initial_lr=0.1, gamma=0.9)
print(f"{scheduler.get_lr(epoch=0):.4f}")  # 0.1000
