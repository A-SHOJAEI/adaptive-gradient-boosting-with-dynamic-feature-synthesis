#!/usr/bin/env python
"""Evaluation script for adaptive gradient boosting model."""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_gradient_boosting_with_dynamic_feature_synthesis.data.loader import (
    load_dataset,
)
from adaptive_gradient_boosting_with_dynamic_feature_synthesis.evaluation.metrics import (
    compute_classification_metrics,
)
from adaptive_gradient_boosting_with_dynamic_feature_synthesis.evaluation.analysis import (
    plot_feature_importance,
    plot_roc_curve,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive gradient boosting model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/final_model.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=str,
        default="models/preprocessor.pkl",
        help="Path to preprocessor",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic_classification",
        help="Dataset name to evaluate on",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and preprocessor
    logger.info(f"Loading model from {args.model_path}")
    model = joblib.load(args.model_path)

    logger.info(f"Loading preprocessor from {args.preprocessor_path}")
    preprocessor = joblib.load(args.preprocessor_path)

    # Load test data
    logger.info(f"Loading dataset: {args.dataset}")
    _, _, X_test, _, _, y_test = load_dataset(
        dataset_name=args.dataset, test_size=0.2, val_size=0.1, random_state=42
    )

    logger.info(f"Test set size: {len(X_test)}")

    # Preprocess test data
    X_test_processed = preprocessor.transform(X_test)

    # Make predictions
    logger.info("Making predictions")
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Compute metrics
    logger.info("Computing metrics")
    metrics = compute_classification_metrics(
        y_test.values, y_pred, y_pred_proba
    )

    # Add per-class analysis
    from sklearn.metrics import classification_report, confusion_matrix

    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Combine all results
    results = {
        "metrics": metrics,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "dataset": args.dataset,
        "n_samples": len(X_test),
    }

    # Save results to JSON
    results_json_path = output_dir / "evaluation_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_json_path}")

    # Save results to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = output_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved metrics to {metrics_csv_path}")

    # Generate visualizations
    logger.info("Generating visualizations")

    # ROC curve
    plot_roc_curve(
        y_test.values,
        y_pred_proba,
        output_path=str(output_dir / "roc_curve.png"),
        title="ROC Curve - Adaptive Gradient Boosting",
    )

    # Feature importance
    feature_importance = model.get_feature_importance(top_n=20)
    if feature_importance:
        plot_feature_importance(
            feature_importance,
            output_path=str(output_dir / "feature_importance.png"),
            top_n=20,
        )

    # Print summary table
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Test samples: {len(X_test)}")
    print("-" * 60)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 60)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:<20} {metric_value:>10.4f}")
    print("-" * 60)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("=" * 60)
    print(f"\nDetailed results saved to {output_dir}")


if __name__ == "__main__":
    main()
