import numpy as np


def momentum_optimizer(
    parameter: float, grad: float, velocity: float, learning_rate: float = 0.01, momentum: float = 0.9
) -> tuple[float, float]:
    velocity = momentum * velocity + learning_rate * grad
    parameter -= velocity
    return round(parameter, 5), round(velocity, 5)


print(momentum_optimizer(1.0, 0.1, 0.5, 0.01, 0.9))  # (0.549, 0.451)
