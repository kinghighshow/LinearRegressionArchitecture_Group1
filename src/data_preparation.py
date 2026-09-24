"""
data_preparation.py
--------------------
Stage 2 of the Robot PM MLOps pipeline.

Input:  the validated DataFrame from DataExtractionAnalysis
Output: scaled train/test feature matrices + a binary "at_risk" label
Next:   -> ModelSelection / ModelTraining

Label definition (an explicit design decision - documented here rather
than left implicit): Final_Project_Group_1's analysis notebook already
computed a per-axis baseline (first `baseline_fraction` of readings while
RUNNING) and raised WARNING (1 abnormal axis) / MAINTENANCE (2+ abnormal
axes) alerts. We reuse that exact logic and collapse it into a binary
target: at_risk = 1 when Alert is WARNING or MAINTENANCE, else 0. There
are no ground-truth failure records in the source data, so this is a
proxy label, not a confirmed failure outcome - the model learns to
recognize the same abnormal-current pattern the rule-based alert flags,
but from a learned probability instead of a hand-set threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .config import ExperimentConfig
except ImportError:
    ExperimentConfig = None  # type: ignore


@dataclass
class SplitData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame


class DataPreparation:
    def __init__(
        self,
        axis_columns: Optional[list[str]] = None,
        running_threshold: float = 0.5,
        baseline_fraction: float = 0.4,
        rolling_window: int = 5,
        test_size: float = 0.2,
    ):
        self.axis_columns = axis_columns or [f"Axis #{i}" for i in range(1, 9)]
        self.running_threshold = running_threshold
        self.baseline_fraction = baseline_fraction
        self.rolling_window = rolling_window
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.axis_limits_: Optional[pd.Series] = None

    @classmethod
    def from_config(cls, config: "ExperimentConfig") -> "DataPreparation":
        lbl = config.labeling
        return cls(
            running_threshold=lbl.get("running_threshold", 0.5),
            baseline_fraction=lbl.get("baseline_fraction", 0.4),
            rolling_window=lbl.get("rolling_window", 5),
            test_size=config.robot_training.get("test_size", 0.2),
        )

    # ---- Steps, each independently callable --------------------------
    def compute_state(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Total"] = df[self.axis_columns].sum(axis=1)
        df["State"] = np.where(df["Total"] > self.running_threshold, "RUNNING", "STOPPED")
        return df

    def compute_alert(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reproduces Antonio's baseline-limit anomaly/alert logic from
        Final_Project_Group_1.ipynb, and stores the fitted limits so the
        same rule can be replayed on new streamed data."""
        df = df.copy()
        split = int(len(df) * self.baseline_fraction)
        baseline = df.iloc[:split]
        limits = baseline.loc[baseline["State"] == "RUNNING", self.axis_columns].max()
        self.axis_limits_ = limits

        flags = df[self.axis_columns].gt(limits)
        flags.loc[df["State"] == "STOPPED"] = False
        df["Abnormal_Axes"] = flags.sum(axis=1)
        df["Alert"] = np.select(
            [df["Abnormal_Axes"] >= 2, df["Abnormal_Axes"] == 1],
            ["MAINTENANCE", "WARNING"],
            default="NORMAL",
        )
        return df

    def add_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["at_risk"] = (df["Alert"] != "NORMAL").astype(int)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds rolling mean/std per axis so the model can learn from a
        short trend rather than only the single instantaneous reading
        the hand-set threshold rule uses."""
        df = df.copy()
        for col in self.axis_columns:
            df[f"{col}_roll_mean"] = (
                df[col].rolling(self.rolling_window, min_periods=1).mean()
            )
            df[f"{col}_roll_std"] = (
                df[col].rolling(self.rolling_window, min_periods=1).std().fillna(0.0)
            )
        return df

    def feature_columns(self, df: pd.DataFrame) -> list[str]:
        engineered = [c for c in df.columns if c.endswith("_roll_mean") or c.endswith("_roll_std")]
        return self.axis_columns + ["Total"] + engineered

    def chronological_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Time-ordered split (not random) - the right choice for streaming
        sensor data, so the test set is always later in time than train."""
        split_idx = int(len(df) * (1 - self.test_size))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def run(self, raw_df: pd.DataFrame) -> SplitData:
        """Full pipeline: state -> alert -> label -> features -> split -> scale."""
        df = self.compute_state(raw_df)
        df = self.compute_alert(df)
        df = self.add_label(df)
        df = self.engineer_features(df)

        feature_names = self.feature_columns(df)
        train_df, test_df = self.chronological_split(df)

        X_train = self.scaler.fit_transform(train_df[feature_names])
        X_test = self.scaler.transform(test_df[feature_names])

        return SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=train_df["at_risk"].to_numpy(),
            y_test=test_df["at_risk"].to_numpy(),
            feature_names=feature_names,
            train_df=train_df,
            test_df=test_df,
        )
