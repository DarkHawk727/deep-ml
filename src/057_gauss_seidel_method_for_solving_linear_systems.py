from typing import Optional

import numpy as np


def gauss_seidel(A: np.ndarray, b: np.ndarray, n: int, x0: Optional[np.ndarray] = None) -> np.ndarray:
    if not x0:
        x0 = np.zeros(len(b))

    L, U = np.tril(A), np.triu(A, k=1)
    D = np.diag(np.diag(A))

    x = np.copy(x0)
    for _ in range(n):
        x = np.linalg.inv(L) @ (b - U @ x)

    return x


print(
    gauss_seidel(np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=float), np.array([4, 7, 3], dtype=float), 5)
)  # [0.5008, 0.99968, 0.49984]
