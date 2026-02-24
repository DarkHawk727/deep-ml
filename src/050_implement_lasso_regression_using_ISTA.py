import numpy as np


def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(w) * np.maximum(np.absolute(w) - threshold, 0.0)


def l1_regularization_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    for _ in range(max_iter):
        w_old = w.copy()
        b_old = b

        r = (X @ w + b) - y

        grad_w = (X.T @ r) / n
        grad_b = np.mean(r)

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        w = soft_threshold(w, learning_rate * alpha)

        if np.linalg.norm(w - w_old) + abs(b - b_old) < tol:
            break

    return w, b


print(
    l1_regularization_gradient_descent(
        np.array([[1, 0.01], [2, 0.02], [3, 0.03], [4, 0.04], [5, 0.05]]), np.array([2, 4, 6, 8, 10])
    )
)  # (array([1.93151749, 0.        ]), np.float64(0.21672773121470018))
