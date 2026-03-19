import numpy as np


def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    basis: list[np.ndarray] = []

    for v in vectors:
        w = np.array(v, dtype=float).copy()
        for u in basis:
            w -= np.dot(w, u) * u

        norm = np.linalg.norm(w)
        if norm > tol:
            basis.append(w / norm)

    return basis


print([b.round(4) for b in orthonormal_basis([[1, 0], [1, 1]])])  # [array([1., 0.]), array([0., 1.])]
