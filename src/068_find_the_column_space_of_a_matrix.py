import numpy as np


def matrix_image(A: np.ndarray) -> np.ndarray:
    m, n = A.shape
    basis_columns = []
    current = np.empty((m, 0), dtype=A.dtype)
    current_rank = 0

    for j in range(n):
        candidate = np.column_stack((current, A[:, j]))
        new_rank = np.linalg.matrix_rank(candidate, tol=1e-10)
        if new_rank > current_rank:
            basis_columns.append(A[:, j])
            current = candidate
            current_rank = new_rank

    return np.column_stack(basis_columns) if basis_columns else np.empty((m, 0), dtype=A.dtype)


print(matrix_image(np.array([[1, 0], [0, 1]])))  # [[1, 0], [0, 1]]
