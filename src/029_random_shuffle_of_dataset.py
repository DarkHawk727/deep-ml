import numpy as np


def shuffle_data(X: np.ndarray, y: np.ndarray, seed=None) -> tuple[np.ndarray, np.ndarray]:
    if seed:
        np.random.seed(seed)
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], y[permutation]


print(
    shuffle_data(X=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), y=np.array([1, 2, 3, 4]))
)  # (np.array([[5, 6], [1, 2], [7, 8],[3, 4]]), np.array([3, 1, 4, 2]))
