def _calculate_distance(p1: tuple[float, ...], p2: tuple[float, ...]) -> float:
    distance = 0.0
    for x, y in zip(p1, p2):
        distance += (x - y) ** 2

    return distance**0.5


def _mean_points(centroid_points: list[tuple[float, ...]]) -> tuple[float, ...]:
    sums = [0.0] * len(centroid_points[0])
    for point in centroid_points:
        for i, x in enumerate(point):
            sums[i] += x
    n = len(centroid_points)

    return tuple(s / n for s in sums)


def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int,
) -> list[tuple[float, ...]]:
    centroids = list(initial_centroids)
    for _ in range(max_iterations):
        # Assign points
        assigned_centroids: list[int] = []
        for point in points:
            best_i, best_distance = 0, float("inf")
            for i, centroid in enumerate(centroids):
                dist = _calculate_distance(point, centroid)
                if dist < best_distance:
                    best_i = i
                    best_distance = dist
            assigned_centroids.append(best_i)

        # Bucket points by centroid
        buckets: list[list[tuple[float, ...]]] = [[] for _ in range(k)]
        for point, centroid_id in zip(points, assigned_centroids):
            buckets[centroid_id].append(point)

        # Update centroids to means
        new_centroids: list[tuple[float, ...]] = []
        for j in range(k):
            if buckets[j]:
                new_centroids.append(_mean_points(buckets[j]))
            else:
                new_centroids.append(centroids[j])

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return centroids


print(
    k_means_clustering([(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], 2, [(1, 1), (10, 1)], 10)
)  # [(1.0, 2.0), (10.0, 2.0)]
