def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    return [list(i) for i in zip(*a)]


print(transpose_matrix([[1, 2, 3], [4, 5, 6]]))
