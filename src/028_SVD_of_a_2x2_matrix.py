import numpy as np


def svd_2x2(A: np.ndarray) -> tuple:
    B = A.T @ A

    theta = 0.5 * np.arctan2(2.0 * B[0, 1], B[0, 0] - B[1, 1])

    c, s = np.cos(theta), np.sin(theta)
    V = np.array([[c, -s], [s, c]])

    D = V.T @ B @ V

    lam = np.diag(D)
    lam = np.clip(lam, 0, None)
    sigma = np.sqrt(lam)

    idx = np.argsort(-sigma)
    sigma = sigma[idx]
    V = V[:, idx]

    Sigma = np.diag(sigma)

    inv = np.zeros_like(sigma)
    inv[sigma > 1e-12] = 1.0 / sigma[sigma > 1e-12]
    Sigma_inv = np.diag(inv)

    U = A @ V @ Sigma_inv

    u0 = U[:, 0]
    u0 /= np.linalg.norm(u0) if np.linalg.norm(u0) else 1.0
    if sigma[1] > 1e-12:
        u1 = U[:, 1]
        u1 /= np.linalg.norm(u1) if np.linalg.norm(u1) else 1.0
    else:
        u1 = np.array([-u0[1], u0[0]])
    U = np.column_stack([u0, u1])

    return U, np.diagonal(Sigma), V.T


print(svd_2x2(np.array([[-10, 8], [10, -1]])))
"""
(array([[-0.8,  0.6],
       [ 0.6,  0.8]]), array([15.65247584,  4.47213595]), array([[ 0.89442719, -0.4472136 ],
       [ 0.4472136 ,  0.89442719]]))
"""
