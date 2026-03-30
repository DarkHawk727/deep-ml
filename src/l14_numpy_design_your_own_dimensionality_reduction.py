import numpy as np


class MyReducer:
    """
    Implement your own dimensionality reduction to 10 dimensions.

    Your goal: Project high-dimensional data to 10 dimensions while
    preserving structure for classification.

    A k-NN classifier will be trained on your reduced data to evaluate quality.
    """

    def __init__(self):
        self.n_components = 10
        # Add any attributes you need to store learned parameters
        self.S = np.zeros(self.n_components)
        self.V = np.zeros((self.n_components, 0))

    def fit(self, X):
        """
        Learn the reduction from training data.

        Args:
            X: Training data, shape (n_samples, n_features)

        Returns:
            self
        """
        # TODO: Analyze X and store what you need for transform()
        X_centered = X - np.mean(X, axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.S = S[: self.n_components]
        self.V = Vt[: self.n_components, :]
        return self

    def transform(self, X):
        """
        Apply the learned reduction to data.

        Args:
            X: Data to transform, shape (n_samples, n_features)

        Returns:
            X_reduced: shape (n_samples, 10)
        """
        # TODO: Project X to 10 dimensions using parameters from fit()
        X_centered = X - np.mean(X, axis=0)
        return X_centered @ self.V.T

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
