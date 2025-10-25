import numpy as np


def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X_np = np.array(X)
    y_np = np.array(y)
    return (np.linalg.inv(X_np.T @ X_np) @ X_np.T @ y_np).tolist()


print(linear_regression_normal_equation(X=[[1, 1], [1, 2], [1, 3]], y=[1, 2, 3]))  # [0.0, 1.0]
