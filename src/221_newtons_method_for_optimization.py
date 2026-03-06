from typing import Callable

import numpy as np


def newtons_method_optimization(
    gradient_func: Callable[[list[float]], list[float]],
    hessian_func: Callable[[list[float]], list[list[float]]],
    x0: list[float],
    tol: float = 1e-6,
    max_iter: int = 100,
) -> list[float]:
    x = np.array(x0, dtype=float)

    for _ in range(max_iter):
        g = np.array(gradient_func(x.tolist()), dtype=float)
        H = np.array(hessian_func(x.tolist()), dtype=float)

        if np.linalg.norm(g) < tol:
            break

        step = np.linalg.solve(H, g)

        x -= step

    return x.tolist()


print([round(v, 4) for v in newtons_method_optimization(lambda x: [2 * x[0]], lambda x: [[2.0]], [5.0])])  # [0.0]
