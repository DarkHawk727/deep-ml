import numpy as np


def adamax_optimizer(
    parameter: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    u: np.ndarray,
    t: int,
    learning_rate=0.002,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta1 * m + (1.0 - beta1) * grad
    u = np.maximum(beta2 * u, np.abs(grad) + epsilon)

    m_hat = m / (1.0 - beta1**t)

    parameter -= learning_rate / u * m_hat

    return np.round(parameter, 5), np.round(m, 5), np.round(u, 5)


print(adamax_optimizer(1.0, 0.1, 1.0, 1.0, 1, 0.002, 0.9, 0.999, 1e-8))  # (0.98178, 0.91, 0.999)
