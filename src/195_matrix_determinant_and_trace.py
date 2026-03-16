def calculate_determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0.0
        for j in range(len(matrix)):
            submatrix = [row[:j] + row[j + 1 :] for row in matrix[1:]]
            det += ((-1) ** j) * matrix[0][j] * calculate_determinant(submatrix)
        return det


def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
    trace = 0.0
    for i in range(len(matrix)):
        trace += matrix[i][i]

    return calculate_determinant(matrix), trace


print(matrix_determinant_and_trace([[1, 0], [0, 1]]))  # Output: (1, 2.0)
