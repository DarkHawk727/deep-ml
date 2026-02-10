import math


def binomial_probability(n: int, k: int, p: float) -> float:
    probability = math.comb(n, k) * math.pow(p, k) * math.pow(1.0 - p, n - k)

    return round(probability, 5)
