from typing import Literal

import numpy as np


def compute_norm(arr: np.ndarray, norm_type: Literal["l1", "l2", "frobenius"]) -> float:
    if norm_type == "l1":
        return np.sum(np.abs(arr))
    elif norm_type == "l2":
        return np.sqrt(np.sum(np.abs(arr) ** 2))
    elif norm_type == "frobenius":
        return np.sqrt(np.sum(np.abs(arr) ** 2))
    else:
        raise ValueError("Invalid norm type")


print(compute_norm(np.array([1, -2, 3]), 'l1')) # 6.0