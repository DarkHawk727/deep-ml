import numpy as np


def descriptive_statistics(data: np.ndarray) -> dict[str, float]:
    mean = np.mean(data)
    median = np.median(data)
    u, c = np.unique(data, return_counts=True)
    mode = u[np.argmax(c)]
    variance = np.var(data)
    std_dev = np.std(data)
    percentiles = np.percentile(data, [25, 50, 75])
    iqr = percentiles[2] - percentiles[0]
    stats_dict = {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": np.round(variance, 4),
        "standard_deviation": np.round(std_dev, 4),
        "25th_percentile": np.round(percentiles[0], 4),
        "50th_percentile": np.round(percentiles[1], 4),
        "75th_percentile": np.round(percentiles[2], 4),
        "interquartile_range": np.round(iqr, 4),
    }
    return stats_dict
