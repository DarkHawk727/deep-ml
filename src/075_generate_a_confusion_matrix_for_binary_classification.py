def confusion_matrix(data: list[list[int]]):
    tp, fp, tn, fn = 0, 0, 0, 0
    for y_true, y_pred in data:
        if y_true and not y_pred:
            fn += 1
        elif not y_true and y_pred:
            fp += 1
        elif y_true and y_pred:
            tp += 1
        else:
            tn += 1

    return [[tp, fn], [fp, tn]]


print(confusion_matrix([[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]))
