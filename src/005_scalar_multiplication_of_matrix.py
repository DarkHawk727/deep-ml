def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] *= scalar

    return matrix


print(scalar_multiply(matrix=[[1, 2], [3, 4]], scalar=2))  # [[2, 4], [6, 8]]
