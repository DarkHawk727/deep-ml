import math


def poisson_probability(k: int, lam: float) -> float:
    val = math.pow(lam, k) * math.exp(-lam) / math.factorial(k)

    return round(val, 5)
