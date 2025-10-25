from typing import List, Optional

import numpy as np


def batch_iterator(X: np.ndarray, y: Optional[np.ndarray] = None, batch_size: int = 64) -> List[List[np.ndarray]]:
    out: List[List[np.ndarray]] = []
    n_samples = len(X)
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        if y is not None:
            out.append([X[start:end], y[start:end]])
        else:
            out.append([X[start:end]])

    return out


print(
    batch_iterator(
        X=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        y=np.array([1, 2, 3, 4, 5]),
        batch_size=2,
    )
)  # [[[[1, 2], [3, 4]], [1, 2]], [[[5, 6], [7, 8]], [3, 4]], [[[9, 10]], [5]]]
