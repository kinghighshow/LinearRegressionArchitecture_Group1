from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """Thin wrapper around the parsed YAML dict with convenience accessors."""

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
        """Safe nested lookup, e.g. config.get('training', 'learning_rate')."""
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # ---- Convenience sections used across the pipeline -----------------
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
    def training(self) -> dict:
        return self.get("training", default={})

    @property
    def results_path(self) -> str:
        return self.get("experiment_tracking", "results_path",
                         default="experiments/results.csv")
