import numpy as np


def adadelta_optimizer(
    parameter: np.ndarray, grad: np.ndarray, u: np.ndarray, v: np.ndarray, rho=0.95, epsilon=1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = rho * u + (1 - rho) * grad**2
    delta = np.sqrt(v + epsilon) / np.sqrt(u + epsilon) * grad
    v = rho * v + (1 - rho) * delta**2
    parameter = parameter - delta

    return np.round(parameter, 5), np.round(u, 5), np.round(v, 5)


print(adadelta_optimizer(1.0, 0.5, 1.0, 1.0, 0.95, 1e-6))  # (0.49035, 0.9625, 0.96299)
