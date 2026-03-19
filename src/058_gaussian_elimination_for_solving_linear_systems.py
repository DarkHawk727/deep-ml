import numpy as np


def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).eps
    n = A.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j][i] / (A[i][i] + eps)
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1 :], x[i + 1 :])) / (A[i][i] + eps)

    return x


print(
    gaussian_elimination(np.array([[2, 8, 4], [2, 5, 1], [4, 10, -1]], dtype=float), np.array([2, 5, 1], dtype=float))
)  # [11.0, -4.0, 3.0]
