def law_of_total_probability(priors: dict[str, float], conditionals: dict[str, float]) -> float:
    total_probability = 0.0
    for event, prior in priors.items():
        conditional = conditionals.get(event, 0)
        total_probability += prior * conditional
    return total_probability


print(law_of_total_probability({"B1": 0.3, "B2": 0.7}, {"B1": 0.2, "B2": 0.5}))  # Output: 0.41
