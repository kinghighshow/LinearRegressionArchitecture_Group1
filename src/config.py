from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    raw: dict = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(raw=data, source_path=path)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # ==== Housing linear-regression pipeline ============================
    @property
    def csv_paths(self) -> dict:
        return self.get("data", "csv", default={})

    @property
    def api(self) -> dict:
        return self.get("api", default={})

    @property
    def database(self) -> dict:
        return self.get("database", default={})

    @property
    def feature(self) -> str:
        return self.get("modeling", "feature", default="median_income")

    @property
    def target(self) -> str:
        return self.get("modeling", "target", default="median_house_value")

    @property
    def training(self) -> dict:
        """Housing pipeline's hyperparameters (test_size, random_state,
        learning_rate, iterations). Kept separate from robot_training so
        the two models never share a learning rate."""
        return self.get("training", default={})

    @property
    def test_size(self) -> float:
        return self.get("training", "test_size", default=0.2)

    @property
    def random_state(self) -> int:
        return self.get("training", "random_state", default=42)

    @property
    def learning_rate(self) -> float:
        return self.get("training", "learning_rate", default=0.01)

    @property
    def iterations(self) -> int:
        return self.get("training", "iterations", default=1000)

    @property
    def results_path(self) -> str:
        """Housing pipeline's experiment log."""
        return self.get("experiment_tracking", "results_path",
                         default="experiments/housing_results.csv")

    # ==== Robot failure-prediction pipeline ==============================
    @property
    def data(self) -> dict:
        return self.get("data", default={})

    @property
    def features(self) -> list[str]:
        return self.get("features", "predictors",
                         default=[f"Axis #{i}" for i in range(1, 9)])

    @property
    def labeling(self) -> dict:
        return self.get("labeling", default={})

    @property
    def robot_training(self) -> dict:
        """Robot pipeline's hyperparameters - deliberately separate from
        `training` (the housing pipeline's), since logistic regression on
        this feature set needs a different learning rate/iteration count
        than the housing linear regression does."""
        return self.get("robot_training", default={})

    @property
    def model(self) -> dict:
        return self.get("model", default={})

    @property
    def artifact_path(self) -> str:
        return self.get("model", "artifact_path",
                         default="models/robot_failure_model.joblib")

    @property
    def robot_results_path(self) -> str:
        """Robot pipeline's experiment log - a different file from the
        housing pipeline's results_path, since the two log different
        metric columns (RMSE/MAE/R2 vs. Accuracy/Precision/ROC-AUC)."""
        return self.get("experiment_tracking", "robot_results_path",
                         default="experiments/robot_results.csv")

"""

"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    raw: dict = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(raw=data, source_path=path)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # ==== Housing linear-regression pipeline ============================
    @property
    def csv_paths(self) -> dict:
        return self.get("data", "csv", default={})

    @property
    def api(self) -> dict:
        return self.get("api", default={})

    @property
    def database(self) -> dict:
        return self.get("database", default={})

    @property
    def feature(self) -> str:
        return self.get("modeling", "feature", default="median_income")

    @property
    def target(self) -> str:
        return self.get("modeling", "target", default="median_house_value")

    @property
    def training(self) -> dict:
        """Housing pipeline's hyperparameters (test_size, random_state,
        learning_rate, iterations). Kept separate from robot_training so
        the two models never share a learning rate."""
        return self.get("training", default={})

    @property
    def test_size(self) -> float:
        return self.get("training", "test_size", default=0.2)

    @property
    def random_state(self) -> int:
        return self.get("training", "random_state", default=42)

    @property
    def learning_rate(self) -> float:
        return self.get("training", "learning_rate", default=0.01)

    @property
    def iterations(self) -> int:
        return self.get("training", "iterations", default=1000)

    @property
    def results_path(self) -> str:
        """Housing pipeline's experiment log."""
        return self.get("experiment_tracking", "results_path",
                         default="experiments/housing_results.csv")

    # ==== Robot failure-prediction pipeline ==============================
    @property
    def data(self) -> dict:
        return self.get("data", default={})

    @property
    def features(self) -> list[str]:
        return self.get("features", "predictors",
                         default=[f"Axis #{i}" for i in range(1, 9)])

    @property
    def labeling(self) -> dict:
        return self.get("labeling", default={})

    @property
    def robot_training(self) -> dict:
        """Robot pipeline's hyperparameters - deliberately separate from
        `training` (the housing pipeline's), since logistic regression on
        this feature set needs a different learning rate/iteration count
        than the housing linear regression does."""
        return self.get("robot_training", default={})

    @property
    def model(self) -> dict:
        return self.get("model", default={})

    @property
    def artifact_path(self) -> str:
        return self.get("model", "artifact_path",
                         default="models/robot_failure_model.joblib")

    @property
    def robot_results_path(self) -> str:
        """Robot pipeline's experiment log - a different file from the
        housing pipeline's results_path, since the two log different
        metric columns (RMSE/MAE/R2 vs. Accuracy/Precision/ROC-AUC)."""
        return self.get("experiment_tracking", "robot_results_path",
                         default="experiments/robot_results.csv")

