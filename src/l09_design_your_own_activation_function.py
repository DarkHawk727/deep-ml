import numpy as np


# ELU
def activation(x):
    alpha = 1.0
    result = np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0))

    return result
