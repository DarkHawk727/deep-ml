def translate_object(points: list[list[float]], tx: float, ty: float) -> list[list[float]]:
    translated_points: list[list[float]] = []
    for x, y in points:
        translated_x = float(x + tx)
        translated_y = float(y + ty)
        translated_points.append([translated_x, translated_y])
    return translated_points


print(translate_object([[0, 0], [1, 0], [0.5, 1]], 2, 3))  # [[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]
