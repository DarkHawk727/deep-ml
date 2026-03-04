import math


def softmax_derivative(x: list[float]) -> list[list[float]]:
    n = len(x)

    exps = [math.exp(v) for v in x]
    Z = sum(exps)
    s = [e / Z for e in exps]

    J = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                J[i][j] = s[i] * (1.0 - s[i])
            else:
                J[i][j] = -s[i] * s[j]

    return J


print(
    [[round(v, 4) for v in row] for row in softmax_derivative([1.0, 2.0, 3.0])]
)  # [[0.0819, -0.022, -0.0599], [-0.022, 0.1848, -0.1628], [-0.0599, -0.1628, 0.2227]]
