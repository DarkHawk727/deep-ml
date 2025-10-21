import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_sim: float = np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))
    return np.round(cos_sim, decimals=3)
