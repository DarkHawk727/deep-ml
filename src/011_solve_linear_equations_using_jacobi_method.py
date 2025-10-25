import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros_like(b, dtype=float)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for _ in range(n):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < 1.0e-10:
            return x_new.tolist()
        x = x_new

    return np.round(x, 4).tolist()


print(solve_jacobi(A=np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]]), b=np.array([-1, 2, 3]), n=2))
