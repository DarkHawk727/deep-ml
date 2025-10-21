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
