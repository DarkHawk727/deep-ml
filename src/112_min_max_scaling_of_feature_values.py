def min_max(x: list[float]) -> list[float]:
    x_min, x_max = min(x), max(x)

    return [(x_i - x_min) / (x_max - x_min) for x_i in x]


print(min_max([1, 2, 3, 4, 5]))  # [0.0, 0.25, 0.5, 0.75, 1.0]
