"""Results analysis and visualization utilities."""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)


def plot_feature_importance(
    feature_importance: Dict[str, float],
    output_path: str,
    top_n: int = 20,
) -> None:
    """Plot feature importance scores.

    Args:
        feature_importance: Dictionary mapping feature names to importance scores.
        output_path: Path to save the plot.
        top_n: Number of top features to display.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Sort and select top N features
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )[:top_n]

    if len(sorted_features) == 0:
        logger.warning("No features to plot")
        return

    features, importances = zip(*sorted_features)

    # Create plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved feature importance plot to {output_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: str,
    title: str = "ROC Curve",
) -> None:
    """Plot ROC curve.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities for positive class.
        output_path: Path to save the plot.
        title: Plot title.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved ROC curve to {output_path}")
