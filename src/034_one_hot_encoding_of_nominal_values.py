import numpy as np


def to_categorical(x: np.ndarray, n_col=None) -> np.ndarray:
    out = np.zeros((len(x), n_col if n_col else np.max(x) + 1))
    out[np.arange(x.size), x] = 1

    return out
