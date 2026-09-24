from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression as _SklearnLogReg


class BaseClassifier(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X, y) -> "BaseClassifier":
        ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Returns P(at_risk = 1) for each row."""
        ...

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class ScratchLogisticRegression(BaseClassifier):
    """Multivariate logistic regression fit with batch gradient descent.

    hypothesis: p = sigmoid(theta_0 + theta_1*x_1 + ... + theta_k*x_k)
    cost:       binary cross-entropy
    """

    name = "From scratch (logistic)"

    def __init__(self, learning_rate: float = 0.1, iterations: int = 2000,
                 class_weight: str | None = "balanced"):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.class_weight = class_weight
        self.theta_0: float = 0.0
        self.theta: np.ndarray | None = None
        self.cost_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight != "balanced":
            return np.ones_like(y, dtype=float)
        n_pos = max(1, int(y.sum()))
        n_neg = max(1, len(y) - n_pos)
        w_pos = len(y) / (2.0 * n_pos)
        w_neg = len(y) / (2.0 * n_neg)
        return np.where(y == 1, w_pos, w_neg)

    def fit(self, X, y) -> "ScratchLogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape

        self.theta_0 = 0.0
        self.theta = np.zeros(n_features)
        weights = self._sample_weights(y)
        self.cost_history = []

        for _ in range(self.iterations):
            z = self.theta_0 + X @ self.theta
            p = self._sigmoid(z)
            errors = (p - y) * weights

            grad_0 = np.mean(errors)
            grad = (X.T @ errors) / n_samples

            self.theta_0 -= self.learning_rate * grad_0
            self.theta -= self.learning_rate * grad

            eps = 1e-12
            cost = -np.mean(weights * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
            if not np.isfinite(cost):
                raise ValueError("Training diverged - lower the learning rate.")
            self.cost_history.append(float(cost))

        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self._sigmoid(self.theta_0 + X @ self.theta)


class SklearnLogisticRegression(BaseClassifier):
    name = "Scikit-learn (logistic)"

    def __init__(self, class_weight: str | None = "balanced", max_iter: int = 1000):
        self._model = _SklearnLogReg(class_weight=class_weight, max_iter=max_iter)

    def fit(self, X, y) -> "SklearnLogisticRegression":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


def build_model(kind: str, learning_rate: float = 0.1, iterations: int = 2000,
                 class_weight: str | None = "balanced") -> BaseClassifier:
    """Config-driven factory: build_model('scratch' | 'sklearn')."""
    if kind == "scratch":
        return ScratchLogisticRegression(learning_rate=learning_rate, iterations=iterations,
                                          class_weight=class_weight)
    if kind == "sklearn":
        return SklearnLogisticRegression(class_weight=class_weight)
    raise ValueError(f"Unknown model kind: {kind!r}")
