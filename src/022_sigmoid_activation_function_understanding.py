import math


def sigmoid(z: float) -> float:
    result = 1.0 / (1.0 + math.exp(-z))
    return round(result, 4)


print(sigmoid(z=0))  # 0.5
