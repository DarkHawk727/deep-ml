import numpy as np


def pca(data: np.ndarray, k: int) -> np.ndarray:
    normalized = (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)

    covariance = np.cov(normalized, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    for i in range(eigenvectors.shape[1]):
        for j in range(eigenvectors.shape[0]):
            if np.abs(eigenvectors[j, i]) > 1e-10:
                if eigenvectors[j, i] < 0:
                    eigenvectors[:, i] *= -1
                break

    top_k_eigenvectors = eigenvectors[:, sorted_indices[:k]]

    return np.round(top_k_eigenvectors, 4)


print(pca(np.array([[1, 2], [3, 4], [5, 6]]), k=1).tolist())  # [[0.7071], [0.7071]]
