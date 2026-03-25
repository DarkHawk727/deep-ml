def compressed_col_sparse_matrix(dense_matrix: list[list[int | float]]) -> tuple[list[float], list[int], list[int]]:
    vals: list[int | float] = []
    row_idx: list[int] = []
    col_ptr: list[int] = [0]
    nnz = 0
    for col in zip(*dense_matrix):
        for c, val in enumerate(col):
            if val != 0:
                vals.append(val)
                row_idx.append(c)
                nnz += 1
        col_ptr.append(nnz)

    return vals, row_idx, col_ptr



vals, row_idx, col_ptr = compressed_col_sparse_matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
print(vals)
print(row_idx)
print(col_ptr)
