import numpy as np


def rref(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(float).copy()

    m, n = matrix.shape
    row = 0

    for col in range(n):
        if row >= m:
            break

        pivot = row
        while pivot < m and matrix[pivot, col] == 0:
            pivot += 1

        if pivot == m:
            continue

        matrix[[row, pivot]] = matrix[[pivot, row]]
        matrix[row] /= matrix[row, col]

        for r in range(m):
            if r != row:
                matrix[r] -= matrix[row] * matrix[r, col]

        row += 1

    matrix[np.abs(matrix) < 1e-12] = 0.0
    return matrix


print(
    rref(np.array([[1, 2, -1, -4], [2, 3, -1, -11], [-2, 0, -3, 22]]))
)  # [[1. 0. 0. -8.], [0. 1. 0. 1.], [0. 0. 1. -2.]]
