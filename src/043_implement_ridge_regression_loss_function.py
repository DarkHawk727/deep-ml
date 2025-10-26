import numpy as np


def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    n = X.shape[0]
    y_pred = X @ w
    mse = 1.0 / n * np.sum((y_pred - y_true) ** 2)
    regularization_term = alpha * np.sum(w**2)

    return round(mse + regularization_term, 3)


print(
    ridge_loss(
        X=np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
        w=np.array([0.2, 2]),
        y_true=np.array([2, 3, 4, 5]),
        alpha=0.1,
    )
)  # 2.204
