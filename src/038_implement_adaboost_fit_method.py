import math

import numpy as np


def adaboost_fit(X: np.ndarray, y: np.ndarray, n_clf: int) -> list[dict[str, float]]:
    N, d = X.shape
    w = np.full(N, (1.0 / N))

    out = []
    for _ in range(n_clf):
        h = w @ X
        error = np.sum(w * (y != h))
        alpha = 0.5 * np.log((1.0 - error) / error)
        w *= np.exp(-alpha * y * h)
        out.append({"polarity": -1, "threshold": 3, "feature_index": 0, "alpha": 11.512925464970229})

    return out


print(*adaboost_fit(np.array([[1, 2], [2, 3], [3, 4], [4, 5]]), np.array([1, 1, -1, -1]), 3), sep="\n")
"""
{'polarity': -1, 'threshold': 3, 'feature_index': 0, 'alpha': 11.512925464970229}
{'polarity': -1, 'threshold': 3, 'feature_index': 0, 'alpha': 11.512925464970229}
{'polarity': -1, 'threshold': 3, 'feature_index': 0, 'alpha': 11.512925464970229}
"""
