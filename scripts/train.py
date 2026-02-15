#!/usr/bin/env python
"""Training script for adaptive gradient boosting model."""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_gradient_boosting_with_dynamic_feature_synthesis.data.loader import (
    load_dataset,
)
from adaptive_gradient_boosting_with_dynamic_feature_synthesis.training.trainer import (
    AdaptiveGBMTrainer,
)
from adaptive_gradient_boosting_with_dynamic_feature_synthesis.utils.config import (
    load_config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train adaptive gradient boosting model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Set random seed
    seed = config.get("random_state", 42)
    set_seed(seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path(config.get("paths", {}).get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = Path(config.get("paths", {}).get("checkpoints_dir", "checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading dataset")
    data_config = config.get("data", {})
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(
        dataset_name=data_config.get("dataset_name", "synthetic_classification"),
        test_size=data_config.get("test_size", 0.2),
        val_size=data_config.get("val_size", 0.1),
        random_state=seed,
    )

    logger.info(
        f"Data loaded - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    # Initialize trainer
    model_config = config.get("model", {})
    training_config = config.get("training", {})

    trainer = AdaptiveGBMTrainer(
        model_config=model_config,
        checkpoint_dir=str(checkpoints_dir),
        patience=training_config.get("patience", 20),
        min_delta=training_config.get("min_delta", 0.0001),
        random_state=seed,
    )

    # Train model
    logger.info("Starting training")
    start_time = time.time()

    try:
        # Wrap MLflow in try/except as server may not be available
        try:
            import mlflow

            mlflow.set_experiment("adaptive-gbm")
            mlflow.start_run()
            mlflow.log_params(model_config)
            logger.info("MLflow tracking enabled")
            mlflow_enabled = True
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            mlflow_enabled = False

        model, history = trainer.train(X_train, y_train, X_val, y_val)

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Log to MLflow if available
        if mlflow_enabled:
            try:
                mlflow.log_metric("training_time", training_time)
                if len(history["val_auc"]) > 0:
                    mlflow.log_metric("best_val_auc", max(history["val_auc"]))
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    # Save training history
    history_path = output_dir / "training_history.json"
    trainer.save_history(str(history_path))

    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_preds_proba = model.predict_proba(trainer.preprocessor.transform(X_test))[:, 1]
    test_preds = (test_preds_proba >= 0.5).astype(int)

    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    test_metrics = {
        "test_auc": float(roc_auc_score(y_test, test_preds_proba)),
        "test_accuracy": float(accuracy_score(y_test, test_preds)),
        "test_f1": float(f1_score(y_test, test_preds)),
        "training_time": float(training_time),
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "n_test_samples": len(X_test),
        "config_file": args.config,
    }

    logger.info(f"Test metrics: {test_metrics}")

    # Save test metrics
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(f"Results saved to {output_dir}")

    # Save final model
    final_model_path = models_dir / "final_model.pkl"
    import joblib

    joblib.dump(model, final_model_path)
    joblib.dump(trainer.preprocessor, models_dir / "preprocessor.pkl")
    logger.info(f"Model saved to {final_model_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Training time: {training_time:.2f}s")
    print(f"Test AUC: {test_metrics['test_auc']:.4f}")
    print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"Test F1: {test_metrics['test_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
