import numpy as np


def calculate_latency_percentiles(latencies: list[float]) -> dict[str, np.floating]:
    if not latencies:
        return {"P50": 0.0, "P95": 0.0, "P99": 0.0}

    return {
        "P50": round(np.percentile(latencies, 50), 4),
        "P95": round(np.percentile(latencies, 95), 4),
        "P99": round(np.percentile(latencies, 99), 4),
    }


print(
    calculate_latency_percentiles([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
)  # {'P50': 55.0, 'P95': 95.5, 'P99': 99.1}
