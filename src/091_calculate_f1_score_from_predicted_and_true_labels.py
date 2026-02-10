def calculate_f1_score(y_true: list[int], y_pred: list[int]) -> float:
    tp, fp, fn = 0, 0, 0
    for truth, prediction in zip(y_true, y_pred):
        if truth & prediction:
            tp += 1
        elif ~truth & prediction:
            fp += 1
        elif truth & ~prediction:
            fn += 1

    precision: float = tp / (tp + fp) if tp + fp != 0.0 else 0.0
    recall: float = tp / (tp + fn) if tp + fn != 0.0 else 0.0

    f1_score = 2.0 * (precision * recall) / ((1.0 * precision) + recall) if precision + recall != 0 else 0.0
    return round(f1_score, 3)


print(calculate_f1_score([1, 0, 1, 1, 0], [1, 0, 0, 1, 1]))
