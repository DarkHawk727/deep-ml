import numpy as np


def adagrad_optimizer(
    parameter: np.ndarray, grad: np.ndarray, G: np.ndarray, learning_rate=0.1, epsilon=1e-8
) -> tuple[np.ndarray, np.ndarray]:
    updated_G = G + grad**2
    updated_parameter = parameter - learning_rate / (updated_G + epsilon) ** 0.5 * grad

    return np.round(updated_parameter, 5), np.round(updated_G, 5)


print(adagrad_optimizer(1.0, 0.5, 1.0, 0.01, 1e-8))  # (0.99553, 1.25)
