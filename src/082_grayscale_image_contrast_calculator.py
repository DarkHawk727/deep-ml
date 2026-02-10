import numpy as np


def calculate_contrast(img: np.ndarray) -> int:
    return np.max(img) - np.min(img)


print(calculate_contrast(np.array([[0, 50], [200, 255]])))  # 255
