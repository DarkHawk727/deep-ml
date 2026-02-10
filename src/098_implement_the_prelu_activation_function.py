def prelu(x: float, alpha: float = 0.25) -> float:
    return max(alpha * x, x)
