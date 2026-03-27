import numpy as np


def calculate_correlation_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    if Y is None:
        Y = X
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    covariance_matrix = X_centered.T @ Y_centered / (X.shape[0] - 1)
    std_X = np.std(X, axis=0, ddof=1)
    std_Y = np.std(Y, axis=0, ddof=1)
    correlation_matrix = covariance_matrix / np.outer(std_X, std_Y)

    return correlation_matrix


print(calculate_correlation_matrix(np.array([[1, 2], [3, 4], [5, 6]])))  # [[1., 1.], [1., 1.]]
