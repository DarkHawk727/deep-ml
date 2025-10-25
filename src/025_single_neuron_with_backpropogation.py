from typing import List, Tuple

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train_neuron(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: float, lr: float, n_epochs: int) -> Tuple[np.ndarray, float, List[float]]:
    mses = []
    n = X.shape[0]
    for _ in range(n_epochs):
        z = X @ W + b
        y_pred = sigmoid(z)
        mses.append(round(1.0 / n * np.sum((y_pred - y) ** 2), 4))

        grad = 2.0 / n * ((y_pred - y) * y_pred * (1.0 - y_pred))

        W -= lr * X.T @ grad
        b -= lr * np.sum(grad)

    return np.round(W, 4), round(b, 4), [round(mse, 4) for mse in mses]


print(
    train_neuron(
        X=np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]),
        y=np.array([1, 0, 0]),
        W=np.array([0.1, -0.2]),
        b=0.0,
        lr=0.1,
        n_epochs=2,
    )
)  # (np.array([0.1036, -0.1425]), -0.0167, [0.3033, 0.2942])
