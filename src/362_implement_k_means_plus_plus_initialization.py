import numpy as np


def kmeans_plus_plus_init(X: np.ndarray, k: int, seed: int = None) -> np.ndarray:
    np.random.seed(seed)

    centroids = np.zeros((k, X.shape[1]))
    centroids[0] = X[np.random.choice(X.shape[0], 1)]
    distances = np.full(X.shape[0], np.inf)
    for i in range(1, k):
        for j in range(i):
            dist = np.linalg.norm(X - centroids[j], axis=1) ** 2
            distances = np.minimum(distances, dist)

        probabilities = distances / np.sum(distances)
        next_centroid_index = np.random.choice(X.shape[0], p=probabilities)
        centroids[i] = X[next_centroid_index]

    return centroids


print(
    kmeans_plus_plus_init(
        np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]), 2, seed=42
    ).tolist()
)  # [[0.0, 10.0], [10.0, 10.0]]
