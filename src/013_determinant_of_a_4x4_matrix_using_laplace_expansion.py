def minor(matrix: list[list[int | float]], i: int, j: int) -> list[list[int | float]]:
    return [row[:j] + row[j + 1 :] for k, row in enumerate(matrix) if k != i]


def determinant_2x2(matrix: list[list[int | float]]) -> float:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def determinant_3x3(matrix: list[list[int | float]]) -> float:
    det = 0
    for col in range(3):
        cofactor = ((-1) ** col) * matrix[0][col] * determinant_2x2(minor(matrix, 0, col))
        det += cofactor
    return det


def determinant_4x4(matrix: list[list[int | float]]) -> float:
    det = 0
    for col in range(4):
        cofactor = ((-1) ** col) * matrix[0][col] * determinant_3x3(minor(matrix, 0, col))
        det += cofactor

    return det


print(determinant_4x4([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))  # 0
