import numpy as np


def kernel_function(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.dot(x1, x2)
