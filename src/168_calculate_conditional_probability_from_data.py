def conditional_probability(data: list[tuple[str, str]], x: str, y: str) -> float:
    count_x, count_y_given_x = 0.0, 0.0
    for x_val, y_val in data:
        if x_val == x:
            count_x += 1
            if y_val == y:
                count_y_given_x += 1

    if count_x == 0:
        return 0.0

    return count_y_given_x / count_x


print(
    conditional_probability(
        [("sunny", "walk"), ("rainy", "read"), ("sunny", "run"), ("cloudy", "read"), ("sunny", "walk")], "sunny", "walk"
    )
)
