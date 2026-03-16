import numpy as np


def is_linearly_independent(vectors: list[list[float]]) -> bool:
    return np.linalg.matrix_rank(np.array(vectors)) == len(vectors)


print(is_linearly_independent([[1, 0], [0, 1]]))  # True
