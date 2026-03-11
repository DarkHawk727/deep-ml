def orthogonal_projection(v: list[float], L: list[float]) -> list[float]:
    dot_product = lambda x, y: sum([x_i * y_i for x_i, y_i in zip(x, y)])

    return [round(dot_product(v, L) / dot_product(L, L) * L_i, 3) for L_i in L]


print(orthogonal_projection([3, 4], [1, 0]))  # [3.0, 0.0]
