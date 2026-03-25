import math


def _euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    x1, y1, x2, y2 = *p1, *p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def k_nearest_neighbors(
    points: list[tuple[float, float]], query_point: tuple[float, float], k: int
) -> list[tuple[float, float]]:
    return list(sorted(points, key=lambda p: _euclidean_distance(p, query_point)))[:k]


print(k_nearest_neighbors([(1, 2), (3, 4), (1, 1), (5, 6), (2, 3)], (2, 2), 3))  # [(1, 2), (2, 3), (1, 1)]
