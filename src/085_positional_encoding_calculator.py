import numpy as np


def pos_encoding(position: int, d_model: int) -> int | np.ndarray:
    if position == 0 or d_model <= 0:
        return -1

    positions = np.arange(position)[:, np.newaxis]

    div_term = np.exp(-np.log(10_000.0) * (np.arange(0, d_model, 2) / d_model))

    PE = np.zeros((position, d_model))
    PE[:, 0::2] = np.sin(positions * div_term)
    PE[:, 1::2] = np.cos(positions * div_term)

    return PE.astype(np.float16)

print(pos_encoding(2, 8))
"""
[[0.      1.      0.      1.      0.      1.      0.      1.     ]
 [0.8413  0.5405  0.09985 0.995   0.01    1.      0.001   1.     ]]
"""
