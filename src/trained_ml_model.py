from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .model_selection import BaseClassifier


@dataclass
class TrainedMLModel:
    model: BaseClassifier
    scaler: Any
    feature_names: list[str]
    threshold: float = 0.5
    metadata: dict = field(default_factory=dict)

    def predict_with_confidence(self, X_raw: pd.DataFrame) -> list[dict]:
        """X_raw: DataFrame containing (at least) self.feature_names columns,
        already engineered the same way DataPreparation.run() produced them."""
        X = self.scaler.transform(X_raw[self.feature_names])
        proba = self.model.predict_proba(X)
        labels = (proba >= self.threshold).astype(int)
        return [
            {"at_risk": int(label), "confidence": float(p)}
            for label, p in zip(labels, proba)
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TrainedMLModel":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No trained model artifact at: {path}")
        return joblib.load(path)
