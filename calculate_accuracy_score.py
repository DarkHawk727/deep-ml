import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
	correct = np.sum(np.where(y_true-y_pred == 0, 1, 0))
	
	return correct / y_true.shape[0]
