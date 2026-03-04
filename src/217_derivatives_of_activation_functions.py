import math


def activation_derivatives(x: float) -> dict[str, float]:
    sigmoid = lambda z: 1.0 / (1.0 + math.exp(-x))

    return {"sigmoid": sigmoid(x) * (1.0 - sigmoid(x)), "tanh": 1.0 - math.tanh(x) ** 2, "relu": 0.0 if x <= 0 else 1.0}


print({k: round(v, 4) for k, v in activation_derivatives(0.0).items()})
