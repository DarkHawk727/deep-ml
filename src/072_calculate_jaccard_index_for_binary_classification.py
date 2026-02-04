import numpy as np


def jaccard_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    union = np.sum(np.logical_or(y_true, y_pred))
    intersection = np.sum(np.logical_and(y_true, y_pred))

    if union == 0:
        return 0.0

    return np.round(intersection / union, 3)


print(jaccard_index(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 1, 0, 0, 1])))  # 0.75
print(jaccard_index(np.array([0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0])))  # 1.0
