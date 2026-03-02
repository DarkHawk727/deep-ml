from typing import Callable


def jacobian_matrix(f: Callable[[list[float]], list[float]], x: list[float], h: float = 1e-5) -> list[list[float]]:
    n = len(x)
    fx = f(x)
    m = len(fx)

    J = [[0.0] * n for _ in range(m)]

    for i in range(n):
        x_step = x.copy()
        x_step[i] += h
        f_step = f(x_step)

        for j in range(m):
            J[j][i] = (f_step[j] - fx[j]) / h

    return J


print(
    [[round(val, 4) for val in row] for row in jacobian_matrix(lambda x: [2 * x[0] + 3 * x[1], x[0] - x[1]], [1, 2])]
)  # [[2.0, 3.0], [1.0, -1.0]]
