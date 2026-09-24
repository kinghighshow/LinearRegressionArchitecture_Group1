from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluationValidation:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def compute_metrics(self, model_name: str, y_true, y_proba) -> dict:
        y_pred = (np.asarray(y_proba) >= self.threshold).astype(int)
        return {
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
            "Mean_Confidence": float(np.mean(y_proba)),
            "Threshold": self.threshold,
        }

    def compare(self, results: Iterable[dict]) -> pd.DataFrame:
        return pd.DataFrame(list(results))

    def confusion(self, y_true, y_proba) -> np.ndarray:
        y_pred = (np.asarray(y_proba) >= self.threshold).astype(int)
        return confusion_matrix(y_true, y_pred, labels=[0, 1])

    def plot_roc(self, fitted_probas: dict, y_true) -> plt.Figure:
        """fitted_probas: {model_name: y_proba array}"""
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, proba in fitted_probas.items():
            if len(set(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc = roc_auc_score(y_true, proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - at_risk classifier")
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig

    def plot_confusion(self, y_true, y_proba, model_name: str) -> plt.Figure:
        cm = self.confusion(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
        ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        ax.set_title(f"Confusion Matrix - {model_name}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        return fig

    def plot_confidence_distribution(self, fitted_probas: dict, y_true) -> plt.Figure:
        n = len(fitted_probas)
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4), sharey=True)
        axes = np.atleast_1d(axes)
        y_true = np.asarray(y_true)
        for ax, (name, proba) in zip(axes, fitted_probas.items()):
            proba = np.asarray(proba)
            ax.hist(proba[y_true == 0], bins=20, alpha=0.6, label="Actual: normal")
            ax.hist(proba[y_true == 1], bins=20, alpha=0.6, label="Actual: at_risk")
            ax.axvline(self.threshold, color="black", linestyle="--", linewidth=1)
            ax.set_title(name)
            ax.set_xlabel("Predicted confidence (P(at_risk))")
            ax.legend()
        axes[0].set_ylabel("Count")
        fig.tight_layout()
        return fig

    def save_results(self, rows: Iterable[dict], path: str | Path,
                      append: bool = True) -> pd.DataFrame:
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
