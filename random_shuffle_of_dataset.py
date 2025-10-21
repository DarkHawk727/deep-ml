import numpy as np


def shuffle_data(X: np.ndarray, y: np.ndarray, seed=None) -> tuple[np.ndarray, np.ndarray]:
	if seed:
		np.random.seed(seed)
	permutation = np.random.permutation(X.shape[0])
	return X[permutation], y[permutation]
