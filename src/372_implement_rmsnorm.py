import numpy as np


def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)

    return x / rms * g


print(
    np.round(rmsnorm(np.array([[1.0, 2.0, 3.0]]), np.array([1.0, 1.0, 1.0])), 4).tolist()
)  # [[0.4629, 0.9258, 1.3887]]
