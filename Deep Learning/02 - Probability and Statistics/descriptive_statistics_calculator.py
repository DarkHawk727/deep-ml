import numpy as np


def descriptive_statistics(data):
    # Your code here
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
        "25th_percentile": percentiles[0],
        "50th_percentile": percentiles[1],
        "75th_percentile": percentiles[2],
        "interquartile_range": iqr,
    }
    return stats_dict
