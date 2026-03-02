from typing import Callable


def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
    n = len(point)
    x = list(point)

    fx = f(x)
    H = [[0.0] * n for _ in range(n)]

    for i in range(n):
        x[i] += h
        f_up = f(x)
        x[i] -= 2 * h
        f_dn = f(x)
        x[i] += h
        H[i][i] = (f_up - 2.0 * fx + f_dn) / (h * h)

    inv_4h2 = 1.0 / (4.0 * h * h)
    for i in range(n):
        for j in range(i + 1, n):
            x[i] += h
            x[j] += h
            f_pp = f(x)
            x[j] -= 2 * h
            f_pm = f(x)
            x[i] -= 2 * h
            x[j] += 2 * h
            f_mp = f(x)
            x[j] -= 2 * h
            f_mm = f(x)
            x[i] += h
            x[j] += h

            val = (f_pp - f_pm - f_mp + f_mm) * inv_4h2
            H[i][j] = val
            H[j][i] = val

    return H


print(
    [[round(v, 4) for v in row] for row in compute_hessian(lambda p: p[0] ** 2 + p[1] ** 2, [0.0, 0.0])]
)  # [[2.0, 0.0], [0.0, 2.0]]
