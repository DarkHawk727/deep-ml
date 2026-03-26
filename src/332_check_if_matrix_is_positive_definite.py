import numpy as np


def check_positive_definite(matrix: list[list[float]]) -> dict:
    A = np.array(matrix)
    eigenvalues = np.linalg.eigvals(A)
    is_positive_definite = np.all(eigenvalues > 0)
    return {"is_positive_definite": is_positive_definite, "eigenvalues": sorted(np.round(eigenvalues, 4).tolist())}


print(check_positive_definite([[2, 1], [1, 2]]))  # {'is_positive_definite': True, 'eigenvalues': [1.0, 3.0]}
