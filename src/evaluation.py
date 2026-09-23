"""
evaluation.py
-------------
RMSE / MAE / R^2 scoring, a side-by-side comparison table, regression
plots, and a save_results() helper that appends rows to
experiments/results.csv (the CSV-based experiment tracking Part 4 needs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluator:
    """Stateless helper: every method takes what it needs and returns a value."""

    def compute_metrics(self, model_name: str, y_true, y_pred) -> dict:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"Model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}

    def compare(self, results: Iterable[dict]) -> pd.DataFrame:
        """Turn a list of compute_metrics() dicts into one comparison table."""
        return pd.DataFrame(list(results))

    def plot_regression(
        self,
        X_raw: pd.Series,
        y_true,
        fitted_models: dict,
        feature_name: str,
        target_name: str,
        scaler=None,
    ) -> plt.Figure:
        """One panel per model: observed test points + that model's fitted line.

        fitted_models: {display_name: model} where each model has .predict().
        """
        grid_raw = pd.DataFrame({feature_name: np.linspace(X_raw.min(), X_raw.max(), 300)})
        grid_input = scaler.transform(grid_raw) if scaler is not None else grid_raw

        n = len(fitted_models)
        fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        for ax, (name, model) in zip(axes, fitted_models.items()):
            ax.scatter(X_raw, y_true, s=14, alpha=0.2, color="#4C72B0", label="Observed")
            ax.plot(grid_raw[feature_name], model.predict(grid_input),
                    linewidth=2.5, color="#C44E52", label="Fitted line")
            ax.set_title(name)
            ax.set_xlabel(feature_name)
            ax.legend(loc="upper left")
            ax.grid(alpha=0.2)

        axes[0].set_ylabel(target_name)
        fig.tight_layout()
        return fig

    def save_results(self, rows: Iterable[dict], path: str | Path,
                      append: bool = True) -> pd.DataFrame:
        """Append (or overwrite) experiment rows to a results CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_rows = pd.DataFrame(list(rows))

        if append and path.is_file():
            existing = pd.read_csv(path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows

        combined.to_csv(path, index=False)
        return combined
