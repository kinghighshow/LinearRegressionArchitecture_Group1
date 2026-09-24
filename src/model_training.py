"""
model_training.py
------------------
Stage 4 of the Robot PM MLOps pipeline.

Input:  SplitData from DataPreparation + config
Output: {model_name: fitted BaseClassifier}
Next:   -> ModelEvaluationValidation
"""
from __future__ import annotations

from typing import Optional

from .data_preparation import SplitData
from .model_selection import BaseClassifier, build_model

try:
    from .config import ExperimentConfig
except ImportError:
    ExperimentConfig = None  # type: ignore


class ModelTraining:
    def __init__(self, learning_rate: float = 0.1, iterations: int = 2000,
                 class_weight: str | None = "balanced",
                 candidates: Optional[list[str]] = None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.class_weight = class_weight
        self.candidates = candidates or ["scratch", "sklearn"]

    @classmethod
    def from_config(cls, config: "ExperimentConfig") -> "ModelTraining":
        t = config.robot_training
        m = config.model
        return cls(
            learning_rate=t.get("learning_rate", 0.1),
            iterations=t.get("iterations", 2000),
            class_weight=m.get("class_weight", "balanced"),
            candidates=m.get("candidates", ["scratch", "sklearn"]),
        )

    def train(self, split: SplitData) -> dict[str, BaseClassifier]:
        fitted: dict[str, BaseClassifier] = {}
        for kind in self.candidates:
            model = build_model(
                kind,
                learning_rate=self.learning_rate,
                iterations=self.iterations,
                class_weight=self.class_weight,
            )
            model.fit(split.X_train, split.y_train)
            fitted[model.name] = model
        return fitted
