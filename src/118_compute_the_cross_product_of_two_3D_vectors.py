def cross_product(a: list[float], b: list[float]) -> list[float]:
    a1, a2, a3 = a
    b1, b2, b3 = b

    return [a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1]


print(cross_product([1, 0, 0], [0, 1, 0]))  # [0, 0, 1]
