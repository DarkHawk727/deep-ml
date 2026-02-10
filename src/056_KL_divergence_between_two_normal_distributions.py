import math


def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    return math.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2.0 * sigma_q**2) - 0.5


print(kl_divergence_normal(0.0, 1.0, 1.0, 1.0))  # 0.5
