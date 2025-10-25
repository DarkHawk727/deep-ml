import numpy as np


def transform_basis(B: list[list[int | float]], C: list[list[int | float]]) -> list[list[float]]:
    return (np.linalg.inv(np.array(C)) @ np.array(B)).tolist()


print(
    transform_basis(
        B=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], C=[[1, 2.3, 3], [4.4, 25, 6], [7.4, 8, 9]]
    )
) # [[-0.6772, -0.0126, 0.2342], [-0.0184, 0.0505, -0.0275], [0.5732, -0.0345, -0.0569]]
