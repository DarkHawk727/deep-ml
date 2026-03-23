import numpy as np


def cramers_rule(A: list[list[float]], b: list[float]) -> list[float] | int:
    n = len(A)
    det_A = np.linalg.det(A)

    if det_A == 0:
        return -1

    x = []
    for i in range(n):
        A_i = np.copy(A)
        A_i[:, i] = b
        det_A_i = np.linalg.det(A_i)
        x.append(det_A_i / det_A)

    return x


print(np.round(cramers_rule([[2, -1, 3], [4, 2, 1], [-6, 1, -2]], [5, 10, -3]), 4))  # [0.1667, 3.3333, 2.6667]
