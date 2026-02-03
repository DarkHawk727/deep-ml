# Just the Levenshtein Distance
def OSA(source: str, target: str) -> int:
    head = lambda s: s[0]
    tail = lambda s: s[1:]

    if len(target) == 0:
        return len(source)
    elif len(source) == 0:
        return len(target)
    elif head(source) == head(target):
        return OSA(tail(source), tail(target))
    else:
        return 1 + min(
            OSA(tail(source), target),
            OSA(source, tail(target)),
            OSA(tail(source), tail(target)),
        )


print(OSA("butterfly", "dragonfly"))  # 6
