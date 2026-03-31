from collections import Counter


def empirical_pmf(samples: list[float]) -> list[tuple[float, float]]:
    if not samples:
        return []

    count = Counter(samples)
    total = len(samples)
    return [(value, freq / total) for value, freq in count.items()]


print(empirical_pmf([1, 2, 2, 3, 3, 3]))  # [(1, 0.16666666666666666), (2, 0.3333333333333333), (3, 0.5)]
