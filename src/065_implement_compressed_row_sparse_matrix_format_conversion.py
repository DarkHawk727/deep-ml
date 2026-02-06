def compressed_row_sparse_matrix(
    dense_matrix: list[list[float]],
) -> tuple[list[float], list[int], list[int]]:
    vals: list[int | float] = []
    indices: list[int] = []
    indptr: list[int] = [0]
    nnz = 0
    for row in dense_matrix:
        for c, val in enumerate(row):
            if val != 0:
                vals.append(val)
                indices.append(c)
                nnz += 1
        indptr.append(nnz)

    return vals, indices, indptr


print(
    *compressed_row_sparse_matrix(
        [[1, 0, 0, 0], [0, 2, 0, 0], [3, 0, 4, 0], [1, 0, 0, 5]]
    ),
    sep="\n",
)
"""
Values array:         [1, 2, 3, 4, 1, 5]
Column indices array: [0, 1, 0, 2, 0, 3]
Row pointer array:    [0, 1, 2, 4, 6]
"""
