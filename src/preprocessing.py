"""
preprocessing.py
-----------------
Turns a raw DataFrame plus a chosen feature/target pair into scaled
train/test arrays, matching the steps validated in Part 2's notebook:
select columns -> drop missing -> split 80/20 -> standardize (fit on
train only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .config import ExperimentConfig
except ImportError:
    ExperimentConfig = None  # type: ignore


@dataclass
class SplitData:
    """Container returned by Preprocessor.run(), passed straight into model.py."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame


class Preprocessor:
    """Feature selection, missing-value handling, split and scaling."""

    def __init__(
        self,
        feature: str,
        target: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.feature = feature
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    @classmethod
    def from_config(cls, config: "ExperimentConfig") -> "Preprocessor":
        return cls(
            feature=config.feature,
            target=config.target,
            test_size=config.test_size,
            random_state=config.random_state,
        )

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = {self.feature, self.target} - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        return df[[self.feature, self.target]].copy()

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        cleaned = df.dropna(subset=[self.feature, self.target])
        dropped = before - len(cleaned)
        if dropped:
            print(f"Dropped {dropped} row(s) with missing {self.feature}/{self.target}.")
        return cleaned

    def split_and_scale(self, df: pd.DataFrame) -> SplitData:
        X = df[[self.feature]]
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return SplitData(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            X_train_raw=X_train,
            X_test_raw=X_test,
        )

    def run(self, df: pd.DataFrame) -> SplitData:
        """Full pipeline: select -> clean -> split -> scale."""
        df = self.select_columns(df)
        df = self.handle_missing(df)
        return self.split_and_scale(df)
