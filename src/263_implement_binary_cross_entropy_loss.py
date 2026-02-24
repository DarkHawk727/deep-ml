import math


def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
    clip = lambda x: max(min(x, 1 - epsilon), epsilon)
    y_true_clipped = [clip(y) for y in y_true]
    y_pred_clipped = [clip(y) for y in y_pred]

    BCE = 0.0
    for yt, yp in zip(y_true_clipped, y_pred_clipped):
        BCE += -yt * math.log(yp) - (1 - yt) * math.log(1 - yp)

    return BCE / len(y_true)


print(binary_cross_entropy([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]))  # 0.164252033486018
