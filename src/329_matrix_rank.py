import numpy as np


def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    A = A.astype(float).copy()

    m, n = A.shape
    row = 0

    for col in range(n):
        if row >= m:
            break

        pivot = row
        while pivot < m and A[pivot, col] == 0:
            pivot += 1

        if pivot == m:
            continue

        A[[row, pivot]] = A[[pivot, row]]
        A[row] /= A[row, col]

        for r in range(m):
            if r != row:
                A[r] -= A[row] * A[r, col]

        row += 1

    A[np.abs(A) < tol] = 0.0

    return row


print(matrix_rank(np.array([[1, 2], [3, 4]])))  # 2
