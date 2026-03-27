import numpy as np


def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    A_np = np.array(A, dtype=float)
    m, n = A_np.shape
    Q = np.zeros((m, m))
    R = np.zeros((m, n))

    for j in range(n):
        v = A_np[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A_np[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]

    return Q.tolist(), R.tolist()


Q, R = qr_decomposition([[1, 0], [0, 1]])
print([[round(x, 4) for x in row] for row in Q])  # [[1.0, 0.0], [0.0, 1.0]]
