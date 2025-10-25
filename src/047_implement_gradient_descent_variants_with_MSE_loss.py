import numpy as np


def gradient_descent(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iterations: int, batch_size: int = 1, method="batch"):
    m, d = X.shape
    if method == "batch":
        for _ in range(n_iterations):
            weights -= learning_rate * 2.0 / m * (X.T @ (X @ weights - y))
    elif method == "mini_batch":
        b = batch_size
        for _ in range(n_iterations):
            for start in range(0, m, batch_size):
                end = start + batch_size
                Xb, yb = X[start:end], y[start:end]
                b = Xb.shape[0]
                weights -= learning_rate * 2.0 / b * (Xb.T @ (Xb @ weights - yb))
    elif method == "stochastic":
        for _ in range(n_iterations):
            for i in range(m):
                weights -= learning_rate * 2.0 * (weights @ X[i] - y[i]) * X[i]

    return weights


for method in ["batch", "mini_batch", "stochastic"]:
    print(
        gradient_descent(
            X=np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
            y=np.array([2, 3, 4, 5]),
            weights=np.zeros(2),
            learning_rate=0.01,
            n_iterations=1000,
            batch_size=2,
            method=method,
        )
    )
