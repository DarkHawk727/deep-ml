from typing import Callable

import numpy as np


def nag_optimizer(
    parameter: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    velocity: np.ndarray,
    learning_rate=0.01,
    momentum=0.9,
) -> tuple[np.ndarray, np.ndarray]:

    lookahead_parameter = parameter - momentum * velocity
    gradient = grad_fn(lookahead_parameter)
    velocity = momentum * velocity + learning_rate * gradient
    parameter = parameter - velocity
    
    return np.round(parameter, 5), np.round(velocity, 5)


def gradient_function(x):
    if isinstance(x, np.ndarray):
        n = len(x)
        return x - np.arange(n)
    else:
        return x - 0


print(nag_optimizer(1.0, gradient_function, 0.5, 0.01, 0.9))  # (0.5445, 0.4555)
