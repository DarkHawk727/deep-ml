from typing import Callable

import numpy as np


def adam_optimizer(
    f: Callable[[np.ndarray], np.ndarray],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    num_iterations=10,
):
    theta = np.array(x0, dtype=float, copy=True)
    m_t = np.zeros_like(x0)
    v_t = np.zeros_like(x0)
    for t in range(1, num_iterations + 1):
        g = np.asarray(grad(theta), dtype=theta.dtype)

        m_t = beta1 * m_t + (1 - beta1) * g
        v_t = beta2 * v_t + (1 - beta2) * g**2

        m_hat = m_t / (1 - beta1**t)
        v_hat = v_t / (1 - beta2**t)

        theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return theta


def gradient(x: np.ndarray) -> np.ndarray:
    return np.array([2 * x[0], 2 * x[1]])


print(adam_optimizer(lambda x: x[0] ** 2 + x[1] ** 2, gradient, np.array([1.0, 1.0])))  # [0.99000325 0.99000325]
