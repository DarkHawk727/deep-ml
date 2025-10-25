from typing import Tuple

import numpy as np


def feature_scaling(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    standardized_data = (data - data.mean(axis=0)) / data.std(axis=0)
    normalized_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    return np.round(standardized_data, 4), np.round(normalized_data, 4)


print(
    feature_scaling(data=np.array([[1, 2], [3, 4], [5, 6]]))
)  # ([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
