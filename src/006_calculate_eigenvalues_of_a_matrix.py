def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    # We can just hardcode since matrix.shape = (2,2)
    det = lambda A: A[0][0] * A[1][1] - A[0][1] * A[1][0]
    trace = lambda A: A[0][0] + A[1][1]

    # We can transform the given equation from the prompt with the quadratic equation
    # We have a=1, b=trace(matrix), and c=det(matrix)
    lambda_1 = (trace(matrix) - (trace(matrix) ** 2 - 4 * 1 * det(matrix)) ** 0.5) / (2.0 * 1.0)
    lambda_2 = (trace(matrix) + (trace(matrix) ** 2 - 4 * 1 * det(matrix)) ** 0.5) / (2.0 * 1.0)

    return sorted([lambda_1, lambda_2], reverse=True)


print(calculate_eigenvalues(matrix=[[2, 1], [1, 2]]))  # [3.0, 1.0]
