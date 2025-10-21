import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	mean = []
    if mode == "row":
        for row in matrix:
            mean.append(sum(row) / len(row))
    elif mode == "column":
        for col in zip(*matrix):
            mean.append(sum(col) / len(col))

    return mean
