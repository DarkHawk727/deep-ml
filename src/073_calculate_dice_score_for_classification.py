import numpy as np


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.count_nonzero(y_true & y_pred)
    fp = np.count_nonzero(~y_true & y_pred)
    fn = np.count_nonzero(y_true & ~y_pred)

    return np.round((2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0, 3)


print(
    dice_score(
        np.array([1, 1, 0, 1, 0, 1]),
        np.array([1, 1, 0, 0, 0, 1]),
    )
)  # 0.857
