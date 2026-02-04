import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.round(np.sqrt(1.0 / y_true.size * np.sum((y_true - y_pred) ** 2)), 3)


print(rmse(np.array([3, -0.5, 2, 7]), np.array([2.5, 0.0, 2, 8])))  # 0.612
