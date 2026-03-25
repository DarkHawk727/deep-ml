import numpy as np


class DropoutLayer:
    def __init__(self, p: float) -> None:
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
        self.p = p
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape).astype(x.dtype)
            return x * self.mask / (1 - self.p)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask / (1 - self.p)


np.random.seed(42)

x = np.array([[1.0, 2.0], [3.0, 4.0]])
grad = np.array([[0.5, 0.2], [1.0, 2.0]])

dropout = DropoutLayer(0.2)

print(dropout.forward(x, training=True), dropout.forward(x, training=False), dropout.backward(grad))
"""
(array([[1.25, 0.  ], [3.75, 5.  ]]), array([[1., 2.], [3., 4.]]), array([[0.625, 0.   ], [1.25 , 2.5  ]]))
"""
