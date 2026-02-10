def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float] | int:
    if len(a) != len(b):
        return -1
    else:
        out = []
        for row in a:
            s = 0.0
            for elem1, elem2 in zip(row, b):
                s += elem1 * elem2
            out.append(s)
        return out


print(matrix_dot_vector(a=[[1, 2], [2, 4]], b=[1, 2]))  # [5, 10]
