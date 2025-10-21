import numpy as np


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp: int = int(np.count_nonzero(y_true & y_pred))
    fp: int = int(np.count_nonzero((~y_true) & y_pred))

    return tp / (tp + fp)
