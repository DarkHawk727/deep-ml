def calculate_brightness(img: list[list[int]]) -> int | float:
    if not img or not img[0]:
        return -1

    width = len(img[0])

    total, count = 0, 0
    for row in img:
        if len(row) != width:
            return -1
        for pixel in row:
            if not (0 <= pixel <= 255):
                return -1
            total += pixel
        count += width

    return total / count


print(calculate_brightness([[100, 200], [50, 150]]))  # 125.0
