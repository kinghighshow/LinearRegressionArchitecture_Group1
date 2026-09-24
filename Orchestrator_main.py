#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import ExperimentConfig
from src.data_extraction_analysis import DataExtractionAnalysis
from src.data_preparation import DataPreparation
from src.model_evaluation_validation import ModelEvaluationValidation
from src.model_training import ModelTraining
from src.trained_ml_model import TrainedMLModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Orchestrator")


def run_train(config: ExperimentConfig, csv_override: str | None) -> None:
    extractor = DataExtractionAnalysis(config, logger=logger)
    raw_df = extractor.extract(csv_override)

    preparer = DataPreparation.from_config(config)
    split = preparer.run(raw_df)
    logger.info("Prepared %d train / %d test rows, %d at_risk in test set",
                len(split.y_train), len(split.y_test), int(split.y_test.sum()))

    trainer = ModelTraining.from_config(config)
    fitted_models = trainer.train(split)

    evaluator = ModelEvaluationValidation(threshold=config.model.get("threshold", 0.5))
    results = []
    best_name, best_model, best_auc = None, None, -1.0
    for name, model in fitted_models.items():
        proba = model.predict_proba(split.X_test)
        metrics = evaluator.compute_metrics(name, split.y_test, proba)
        results.append(metrics)
        logger.info("%s -> %s", name, {k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in metrics.items()})
        if metrics["ROC_AUC"] > best_auc:
            best_name, best_model, best_auc = name, model, metrics["ROC_AUC"]

    evaluator.save_results(results, config.robot_results_path, append=True)

    artifact = TrainedMLModel(
        model=best_model,
        scaler=preparer.scaler,
        feature_names=split.feature_names,
        threshold=config.model.get("threshold", 0.5),
        metadata={"selected_model": best_name, "roc_auc": best_auc},
    )
    artifact.save(config.artifact_path)
    logger.info("Selected '%s' (ROC-AUC=%.4f), saved to %s", best_name, best_auc, config.artifact_path)


def run_predict(config: ExperimentConfig, csv_override: str | None) -> None:
    artifact = TrainedMLModel.load(config.artifact_path)

    extractor = DataExtractionAnalysis(config, logger=logger)
    raw_df = extractor.extract(csv_override)

    preparer = DataPreparation.from_config(config)
    # Reuse the pipeline's feature engineering, but scoring only needs the
    # engineered columns - not a fresh label/scaler fit.
    df = preparer.compute_state(raw_df)
    df = preparer.compute_alert(df)
    df = preparer.engineer_features(df)

    predictions = artifact.predict_with_confidence(df)
    for row, pred in zip(df.to_dict(orient="records")[:20], predictions[:20]):
        logger.info("%s -> at_risk=%d confidence=%.3f", row["Time"], pred["at_risk"], pred["confidence"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot failure prediction pipeline")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--data-csv", default=None, help="Override data.csv_path from the config")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    if args.mode == "train":
        run_train(config, args.data_csv)
    else:
        run_predict(config, args.data_csv)


if __name__ == "__main__":
    main()
