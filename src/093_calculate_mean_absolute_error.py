import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.abs(y_true - y_pred).sum() / y_true.size


print(round(mae(np.array([3, -0.5, 2, 7]), np.array([2.5, 0.0, 2, 8])), 3))  # 0.5
