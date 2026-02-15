#!/usr/bin/env python
"""Prediction script for adaptive gradient boosting model."""

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
        description="Make predictions with adaptive gradient boosting model"
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
        "--input-file",
        type=str,
        required=True,
        help="Path to input CSV file with features",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="predictions.csv",
        help="Path to save predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    """Main prediction function."""
    args = parse_args()

    # Load model and preprocessor
    logger.info(f"Loading model from {args.model_path}")
    try:
        model = joblib.load(args.model_path)
    except FileNotFoundError:
        logger.error(f"Model not found at {args.model_path}")
        logger.error("Please train a model first using: python scripts/train.py")
        sys.exit(1)

    logger.info(f"Loading preprocessor from {args.preprocessor_path}")
    try:
        preprocessor = joblib.load(args.preprocessor_path)
    except FileNotFoundError:
        logger.error(f"Preprocessor not found at {args.preprocessor_path}")
        logger.error("Please train a model first using: python scripts/train.py")
        sys.exit(1)

    # Load input data
    logger.info(f"Loading input data from {args.input_file}")
    try:
        X_input = pd.read_csv(args.input_file)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        sys.exit(1)

    logger.info(f"Input data shape: {X_input.shape}")

    # Preprocess input
    logger.info("Preprocessing input data")
    try:
        X_processed = preprocessor.transform(X_input)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        logger.error("Make sure input data has the same features as training data")
        sys.exit(1)

    # Make predictions
    logger.info("Making predictions")
    try:
        y_pred_proba = model.predict_proba(X_processed)[:, 1]
        y_pred = (y_pred_proba >= args.threshold).astype(int)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

    # Prepare output
    output_df = X_input.copy()
    output_df["predicted_class"] = y_pred
    output_df["probability_positive"] = y_pred_proba
    output_df["probability_negative"] = 1 - y_pred_proba
    output_df["confidence"] = np.maximum(y_pred_proba, 1 - y_pred_proba)

    # Save predictions
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Input samples: {len(X_input)}")
    print(f"Threshold: {args.threshold}")
    print("-" * 60)
    print(f"Predicted class 0: {(y_pred == 0).sum()} samples")
    print(f"Predicted class 1: {(y_pred == 1).sum()} samples")
    print("-" * 60)
    print(f"Average confidence: {output_df['confidence'].mean():.4f}")
    print(f"Min confidence: {output_df['confidence'].min():.4f}")
    print(f"Max confidence: {output_df['confidence'].max():.4f}")
    print("=" * 60)
    print(f"\nPredictions saved to {output_path}")

    # Show first few predictions
    print("\nFirst 5 predictions:")
    print(output_df[["predicted_class", "probability_positive", "confidence"]].head())


if __name__ == "__main__":
    main()
