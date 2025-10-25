import numpy as np


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    return np.cov(np.array(vectors)).tolist()


print(calculate_covariance_matrix(vectors=[[1, 2, 3], [4, 5, 6]]))  # [[1.0 1.0], [1.0 1.0]]
