import math


def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    activations: list[float] = []
    for row in features:
        s = 0.0
        for elem1, elem2 in zip(row, weights):
            s += elem1 * elem2
        activations.append(s + bias)

    probabilities: list[float] = []
    for activation in activations:
        probabilities.append(1.0 / (1.0 + math.exp(-activation)))

    mse = 0.0
    for predicted, actual in zip(probabilities, labels):         
        mse += (predicted - actual) ** 2
    mse /= len(labels)

    return [round(probability, 4) for probability in probabilities], round(mse, 4)

# ([0.4626, 0.4134, 0.6682], 0.3349)
print(
    single_neuron_model(
        features=[[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]],
        labels=[0, 1, 0],
        weights=[0.7, -0.4],
        bias=-0.1,
    )
)
