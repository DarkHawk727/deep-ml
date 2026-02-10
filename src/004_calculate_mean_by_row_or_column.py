import numpy as np


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    mean = []
    if mode == "row":
        for row in matrix:
            mean.append(sum(row) / len(row))
    elif mode == "column":
        for col in zip(*matrix):
            mean.append(sum(col) / len(col))
    else:
        raise ValueError(f"unrecognized mode: {mode}")

    return mean


print(calculate_matrix_mean(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode="column"))  # [4.0, 5.0, 6.0]
