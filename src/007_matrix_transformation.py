import numpy as np


def transform_matrix(A: list[list[int | float]], T: list[list[int | float]], S: list[list[int | float]]) -> list[list[int | float]] | int:
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        return -1
    else:
        return np.matmul(np.matmul(np.linalg.inv(T), A), S)


print(transform_matrix(A=[[1, 2], [3, 4]], T=[[2, 0], [0, 2]], S=[[1, 1], [0, 1]]))  # [[0.5,1.5],[1.5,3.5]]
