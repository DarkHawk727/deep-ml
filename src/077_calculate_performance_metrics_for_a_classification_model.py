def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    tp, fp, tn, fn = 0, 0, 0, 0
    for y_true, y_pred in zip(actual, predicted):
        if y_true and y_pred:
            tp += 1
        elif y_true and not y_pred:
            fn += 1
        elif not y_true and y_pred:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2.0 * (precision * recall) / (precision + recall)

    return (
        [[tp, fn], [fp, tn]],
        round((tp + tn) / (tp + tn + fp + fn), 3),
        round(f1, 3),
        round(tn / (tn + fp), 3),
        round(tn / (tn + fn), 3),
    )


print(performance_metrics([1, 0, 1, 0, 1], [1, 0, 0, 1, 1]))  # ([[2, 1], [1, 1]], 0.6, 0.667, 0.5, 0.5)
