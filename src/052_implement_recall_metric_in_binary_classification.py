import numpy as np


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    TP = np.sum(y_pred & y_true)
    FN = np.sum(~y_pred & y_true)

    return round(TP / (TP + FN), 3) if TP + FN != 0 else 0.0


print(recall(y_true=np.array([1, 0, 1, 1, 0, 1]), y_pred=np.array([1, 0, 1, 0, 0, 1])))  # 0.75
