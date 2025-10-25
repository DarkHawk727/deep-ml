def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    det = lambda A: A[0][0] * A[1][1] - A[0][1] * A[1][0]

    if det(matrix) == 0:
        return None
    else:
        a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        return [
            [1.0 / det(matrix) * d, 1.0 / det(matrix) * -b],
            [1.0 / det(matrix) * -c, 1.0 / det(matrix) * a],
        ]


print(inverse_2x2(matrix=[[4, 7], [2, 6]]))
