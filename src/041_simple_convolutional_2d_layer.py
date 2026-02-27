import math

import numpy as np


def simple_conv2d(X: np.ndarray, K: np.ndarray, p: int, s: int) -> np.ndarray:
    H, W = X.shape
    kH, kW = K.shape

    if p > 0:
        X = np.pad(X, p, mode="constant")

    H_p, W_p = X.shape

    out_H = (H_p - kH) // s + 1
    out_W = (W_p - kW) // s + 1

    Y = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            start_i = i * s
            start_j = j * s

            patch = X[start_i : start_i + kH, start_j : start_j + kW]
            Y[i, j] = np.sum(patch * K)

    return Y


print(
    simple_conv2d(
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0],
            ]
        ),
        np.array(
            [
                [1.0, 2.0],
                [3.0, -1.0],
            ]
        ),
        0,
        1,
    )
)
"""
np.array(
    [
        [16.0, 21.0, 26.0, 31.0],
        [41.0, 46.0, 51.0, 56.0],
        [66.0, 71.0, 76.0, 81.0],
        [91.0, 96.0, 101.0, 106.0],
    ]
)
"""
