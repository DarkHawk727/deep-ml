import math
from typing import Literal

import numpy as np  # need to keep this for tests that use np.pi


def taylor_approximation(func_name: Literal["exp", "sin", "cos"], x: float, n_terms: int) -> float:
    res = 0.0
    if func_name == "exp":
        for i in range(n_terms):
            res += x**i / math.factorial(i)
    elif func_name == "sin":
        for i in range(n_terms):
            res += (-1) ** i * x ** (2 * i + 1) / math.factorial(2 * i + 1)
    elif func_name == "cos":
        for i in range(n_terms):
            res += (-1) ** i * x ** (2 * i) / math.factorial(2 * i)

    return res


print(round(taylor_approximation("exp", 1.0, 10), 6))  # 2.718282
