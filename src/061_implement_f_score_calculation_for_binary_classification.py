import numpy as np


def f_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float) -> float:
    tp: int = int(np.count_nonzero(y_true & y_pred))
    fp: int = int(np.count_nonzero((~y_true) & y_pred))
    fn: int = int(np.count_nonzero(y_true & (~y_pred)))

    precision: float = tp / (tp + fp)
    recall: float = tp / (tp + fn)

    return round((1.0 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall), 3)


print(f_score(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 1, 0, 0, 1]), 1.0))
