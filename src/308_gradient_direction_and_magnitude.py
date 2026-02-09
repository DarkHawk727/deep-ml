def gradient_direction_magnitude(gradient: list[float]) -> dict:

    if all(gradient):
        l2_norm = sum(elem**2 for elem in gradient) ** 0.5

        return {
            "magnitude": l2_norm,
            "direction": [elem / l2_norm for elem in gradient],
            "descent_direction": [-elem / l2_norm for elem in gradient],
        }
    else:
        return {
            "magnitude": 0.0,
            "direction": [0.0 for _ in range(len(gradient))],
            "descent_direction": [0.0 for _ in range(len(gradient))],
        }


print(
    gradient_direction_magnitude([0.0, 4.0])
)  # {'magnitude': 5.0, 'direction': [0.6, 0.8], 'descent_direction': [-0.6, -0.8]}
