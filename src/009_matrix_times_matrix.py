def matrixmul(a: list[list[int|float]], b: list[list[int|float]]) -> list[list[int|float]] | int:
    if len(a[0]) != len(b): 
        return -1
    else:
        out: list[list[int|float]] = [[0 for j in range(len(b[0]))] for i in range(len(a))]
        for i in range(len(a)):
            for j in range(len(b[0])):
                for k in range(len(b)):
                    out[i][j] += a[i][k] * b[k][j]
        return out


print(matrixmul(a=[[1,2],[2,4]], b=[[2,1],[3,4]]))  # [[ 8,  9],[16, 18]]
