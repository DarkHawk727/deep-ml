import numpy as np


def _rmsnorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)

    return x / rms


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def engram_context_gating(
    h: np.ndarray, e: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    T, d = h.shape
    k = e @ W_K
    v = e @ W_V

    h_norm = _rmsnorm(h, eps)
    k_norm = _rmsnorm(k, eps)

    score = np.sum(h_norm * k_norm, axis=-1, keepdims=True) / np.sqrt(d)
    alpha_t = _sigmoid(score)

    return alpha_t * v


print(
    engram_context_gating(
        np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]),
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]),
        np.array([[0.5, 0.4, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1], [0.3, 0.2, 0.1, 0.0]]),
    ).shape
)  # (2, 4)
