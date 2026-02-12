import math


def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    PARTIAL_DERIVATIVES: dict[str, list] = {
        "poly2d": [lambda x, y: y * (2 * x + y), lambda x, y: x * (2 * y + x)],
        "exp_sum": [lambda x, y: math.exp(x + y), lambda x, y: math.exp(x + y)],
        "product_sin": [lambda x, y: math.sin(y), lambda x, y: x * math.cos(y)],
        "poly3d": [lambda x, y, z: 2 * x * y, lambda x, y, z: x**2 + z**2, lambda x, y, z: 2 * y * z],
        "squared_error": [lambda x, y: 2 * (x - y), lambda x, y: 2 * (y - x)],
    }

    return tuple(pd(*point) for pd in PARTIAL_DERIVATIVES[func_name])


print(compute_partial_derivatives("poly2d", (2.0, 3.0)))  # (21.0, 16.0)
