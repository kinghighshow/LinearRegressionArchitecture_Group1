"""
model.py
--------
Two univariate linear regression implementations behind one shared
interface (BaseRegressionModel), so evaluation.py and the demo notebook
can call .fit()/.predict() on either one identically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression as _SklearnLR


class BaseRegressionModel(ABC):
    """Common interface implemented by both models below."""

    name: str = "base"

    @abstractmethod
    def fit(self, X, y) -> "BaseRegressionModel":
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        ...


class ScratchLinearRegression(BaseRegressionModel):
    """Univariate linear regression fit with batch gradient descent (NumPy only).

    hypothesis: y_hat = theta_0 + theta_1 * x
    cost:       MSE = mean((y_hat - y) ** 2)
    """

    name = "From scratch"

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta_0: float = 0.0
        self.theta_1: float = 0.0
        self.cost_history: list[float] = []

    @staticmethod
    def _predict(x, theta_0: float, theta_1: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        return theta_0 + theta_1 * x

    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def fit(self, X, y) -> "ScratchLinearRegression":
        x = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        self.theta_0, self.theta_1 = 0.0, 0.0
        self.cost_history = [self._mse(y, self._predict(x, self.theta_0, self.theta_1))]

        for _ in range(self.iterations):
            preds = self._predict(x, self.theta_0, self.theta_1)
            errors = preds - y

            gradient_0 = 2.0 * np.mean(errors)
            gradient_1 = 2.0 * np.mean(errors * x)

            self.theta_0 -= self.learning_rate * gradient_0
            self.theta_1 -= self.learning_rate * gradient_1

            cost = self._mse(y, self._predict(x, self.theta_0, self.theta_1))
            if not np.isfinite(cost):
                raise ValueError("Training diverged - lower the learning rate.")
            self.cost_history.append(cost)

        return self

    def predict(self, X) -> np.ndarray:
        return self._predict(X, self.theta_0, self.theta_1)


class SklearnLinearRegression(BaseRegressionModel):
    """Thin, interface-matching wrapper around sklearn's LinearRegression."""

    name = "Scikit-learn"

    def __init__(self):
        self._model = _SklearnLR()

    def fit(self, X, y) -> "SklearnLinearRegression":
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    @property
    def intercept_(self) -> float:
        return float(self._model.intercept_)

    @property
    def coef_(self) -> float:
        return float(self._model.coef_[0])


def build_model(kind: str, learning_rate: float = 0.01,
                 iterations: int = 1000) -> BaseRegressionModel:
    """Config-driven factory: build_model('scratch' | 'sklearn')."""
    if kind == "scratch":
        return ScratchLinearRegression(learning_rate=learning_rate, iterations=iterations)
    if kind == "sklearn":
        return SklearnLinearRegression()
    raise ValueError(f"Unknown model kind: {kind!r}")
