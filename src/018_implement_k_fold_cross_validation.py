import numpy as np


def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> list[tuple[list[int], list[int]]]:
    if n_samples <= 0 or k <= 1:
        return []

    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop].tolist()
        train_indices = np.concatenate((indices[:start], indices[stop:])).tolist()
        folds.append((train_indices, test_indices))
        current = stop

    return folds


print(k_fold_cross_validation(10, 5, False))
