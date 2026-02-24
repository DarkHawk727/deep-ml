import numpy as np


def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    z = X @ weights + bias

    arr = 1.0 / (1.0 + np.clip(np.exp(-z), -500, 500))

    return np.where(arr >= 0.5, 1.0, 0.0).astype(int)


print(predict_logistic(np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]), np.array([1, 1]), 0))  # [1, 1, 0, 0]
