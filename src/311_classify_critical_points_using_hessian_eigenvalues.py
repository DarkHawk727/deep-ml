import numpy as np


def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10) -> int | None:
    eigenvalues = np.linalg.eigvals(hessian)
    if (eigenvalues > 0).all():
        return -1
    elif (eigenvalues < 0).all():
        return 1
    else:
        if np.isclose(eigenvalues, 0.0, tol).any():
            return None
        else:
            return 0


print(classify_critical_point(np.array([[2, 0], [0, 2]])))  # -1
