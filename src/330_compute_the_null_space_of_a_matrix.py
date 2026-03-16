import numpy as np


def compute_null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    rank = np.linalg.matrix_rank(A, tol=tol)
    nullity = A.shape[1] - rank
    if nullity == 0:
        return np.zeros((A.shape[1], 0))
    U, S, Vt = np.linalg.svd(A)
    null_space = Vt[rank:].T
    return null_space


print(compute_null_space(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).shape)
