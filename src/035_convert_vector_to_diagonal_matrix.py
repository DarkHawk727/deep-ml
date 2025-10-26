import numpy as np


def make_diagonal(x: np.ndarray) -> np.ndarray:
    return np.diag(x)


print(
    make_diagonal(x=np.array([1, 2, 3]))
)  # np.array([[1. 0. 0.], [0. 2. 0.], [0. 0. 3.]])
