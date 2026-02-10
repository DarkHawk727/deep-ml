import math


def normal_pdf(x: float, mean: float, std_dev: float) -> float:

    val = (
        1.0
        / math.sqrt(2.0 * math.pi * math.pow(std_dev, 2))
        * math.exp(-math.pow(x - mean, 2) / (2.0 * math.pow(std_dev, 2)))
    )
    return round(val, 5)
