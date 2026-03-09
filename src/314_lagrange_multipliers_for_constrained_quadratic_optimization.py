import numpy as np


def lagrange_optimize(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> dict:
    A = a.reshape(1, 2)

    KKT = np.block([[Q, -a[:, None]], [a[None, :], np.zeros((1, 1))]])
    rhs = np.concatenate([-c, np.array([b])])

    sol = np.linalg.solve(KKT, rhs)

    return {
        "x": np.round(sol[:2], 4).tolist(),
        "lambda": float(np.round(sol[2], 4)),
        "objective": float(np.round(0.5 * sol[:2].T @ Q @ sol[:2] + c.T @ sol[:2], 4)),
    }


print(
    lagrange_optimize(np.array([[2, 0], [0, 2]]), np.array([0, 0]), np.array([1, 1]), 2)
)  # {'x': [1.0, 1.0], 'lambda': 2.0, 'objective': 2.0}
