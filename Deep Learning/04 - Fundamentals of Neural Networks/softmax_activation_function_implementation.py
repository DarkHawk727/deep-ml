import math

def softmax(scores: list[float]) -> list[float]:
    deno = sum([math.exp(score) for score in scores])
    probabilities: list[float] = [math.exp(score) / deno for score in scores]

    return [round(prob, 4) for prob in probabilities]
