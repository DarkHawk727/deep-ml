import numpy as np


def find_best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    best_feature = -1
    best_threshold = float("inf")
    best_gini = float("inf")

    n, d = X.shape

    for feature in range(d):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_indices = X[:, feature] <= threshold
            right_indices = X[:, feature] > threshold

            if len(left_indices) == 0 or len(right_indices) == 0:
                continue

            gini_left = 1 - np.sum((np.bincount(y[left_indices]) / len(y[left_indices])) ** 2)
            gini_right = 1 - np.sum((np.bincount(y[right_indices]) / len(y[right_indices])) ** 2)

            gini_split = (len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right) / n

            if gini_split < best_gini:
                best_gini = gini_split
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


f1, t1 = find_best_split(np.array([[2.5], [3.5], [1.0], [4.0]]), np.array([0, 1, 0, 1]))
print(f1, round(t1, 4))
