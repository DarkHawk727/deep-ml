def vector_sum(a: list[int | float], b: list[int | float]) -> int | list[int | float]:
    if len(a) != len(b):
        return -1
    else:
        out = []
        for a_i, b_i in zip(a, b):
            out.append(a_i + b_i)

        return out


print(vector_sum([1, 2, 3], [4, 5, 6]))  # [5, 7, 9]
