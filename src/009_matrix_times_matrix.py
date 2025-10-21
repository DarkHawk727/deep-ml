def matrixmul(a: list[list[int|float]], b: list[list[int|float]]) -> list[list[int|float]]:
    if len(a[0]) != len(b): 
        return -1
    else:
        out: list[list[int|float]] = [[0 for j in range(len(b[0]))] for i in range(len(a))]
        for i in range(len(a)):
            for j in range(len(b[0])):
                for k in range(len(b)):
                    out[i][j] += a[i][k] * b[k][j]
        return out
