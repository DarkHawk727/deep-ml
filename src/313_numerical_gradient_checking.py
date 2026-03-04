from typing import Callable

import numpy as np


def numerical_gradient_check(
    f: Callable[[np.ndarray], float], x: np.ndarray, analytical_grad: np.ndarray, epsilon: float = 1e-7
) -> tuple[np.ndarray, float]:
    e_i = np.zeros_like(x)
    num_grad = np.zeros_like(x)

    for i in range(len(x)):
        e_i[i] = epsilon
        f_plus = f(x + e_i)
        f_minus = f(x - e_i)
        num_grad[i] = (f_plus - f_minus) / (2 * epsilon)
        e_i[i] = 0.0

    rel_err = np.linalg.norm(num_grad - analytical_grad) / (np.linalg.norm(num_grad) + np.linalg.norm(analytical_grad))

    return num_grad, float(rel_err)


num_grad, rel_err = numerical_gradient_check(lambda x: np.sum(x**2), np.array([3.0]), np.array([6.0]))
print(([float(g) for g in np.round(num_grad, 4)], bool(rel_err < 1e-5)))
