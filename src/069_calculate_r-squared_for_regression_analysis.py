import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    SSR = np.sum((y_true - y_pred) ** 2)
    SST = np.sum((y_true - np.mean(y_true)) ** 2)

    return round(1.0 - SSR / SST, 3)


print(r_squared(np.array([1, 2, 3, 4, 5]), np.array([1.1, 2.1, 2.9, 4.2, 4.8])))  # 0.989
