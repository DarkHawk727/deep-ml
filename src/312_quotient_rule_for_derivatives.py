def evaluate(coeffs: list[float], x: float) -> float:
    res = 0.0
    for i, coef in enumerate(reversed(coeffs)):
        res += coef * x**i

    return res


def derivative(coeffs: list[float]) -> list[float]:
    deriv = [(len(coeffs) - 1 - k) * coeffs[k] for k in range(len(coeffs) - 1)]
    return deriv if deriv else [0.0]


def quotient_rule_derivative(g_coeffs: list[float], h_coeffs: list[float], x: float) -> float:
    g_deriv: list[float] = derivative(g_coeffs)
    h_deriv: list[float] = derivative(h_coeffs)

    g_at_x: float = evaluate(g_coeffs, x)
    h_at_x: float = evaluate(h_coeffs, x)
    g_deriv_at_x: float = evaluate(g_deriv, x)
    h_deriv_at_x: float = evaluate(h_deriv, x)

    return (g_deriv_at_x * h_at_x - g_at_x * h_deriv_at_x) / h_at_x**2


# Coefficients given in DESCENDING order of powers
print(round(quotient_rule_derivative([1, 0, 1], [1, 2], 2.0), 4))
