def product_rule_derivative(
    f_coeffs: list[float], g_coeffs: list[float]
) -> list[float]:
    m, n = len(f_coeffs), len(g_coeffs)
    prod = [0.0] * (m + n - 1)
    for i in range(m):
        for j in range(n):
            prod[i + j] += f_coeffs[i] * g_coeffs[j]

    return [k * prod[k] for k in range(1, len(prod))]


print(product_rule_derivative([1, 2], [3, 4]))  # [10.0, 16.0]
