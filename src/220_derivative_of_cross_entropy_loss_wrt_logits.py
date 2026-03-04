import math


def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
    n = len(logits)

    y = [0.0] * n
    y[target] = 1.0

    exps = [math.exp(l) for l in logits]
    Z = sum(exps)
    p = [e / Z for e in exps]

    return [p_i - y_i for p_i, y_i in zip(p, y)]


print([round(g, 4) for g in cross_entropy_derivative([1.0, 2.0, 3.0], 0)])  # [-0.91, 0.2447, 0.6652]
