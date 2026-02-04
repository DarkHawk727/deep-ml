from collections import Counter


def gini_impurity(y: list[int]) -> float:
    counts = Counter(y)
    n = len(y)
    impurity = 1.0

    for count in counts.values():
        impurity -= (count / n) ** 2

    return round(impurity, 3)


print(gini_impurity([0, 1, 1, 1, 0]))  # 0.48
