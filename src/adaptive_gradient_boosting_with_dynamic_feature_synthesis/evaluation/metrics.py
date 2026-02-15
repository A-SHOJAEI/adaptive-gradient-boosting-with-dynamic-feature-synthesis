"""Evaluation metrics for classification tasks."""

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels (binary).
        y_pred_proba: Predicted probabilities for positive class.

    Returns:
        Dictionary with metric names and values.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "auc_pr": average_precision_score(y_true, y_pred_proba),
    }

    logger.debug(f"Computed metrics: {metrics}")
    return metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    feature_count_original: int,
    feature_count_final: int,
    training_time: float,
    baseline_time: float = 1.0,
) -> Dict[str, float]:
    """Compute all evaluation metrics including custom research metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities for positive class.
        feature_count_original: Number of original features.
        feature_count_final: Number of features after synthesis.
        training_time: Training time in seconds.
        baseline_time: Baseline training time for efficiency comparison.

    Returns:
        Dictionary with all metrics.
    """
    # Classification metrics
    classification_metrics = compute_classification_metrics(y_true, y_pred, y_pred_proba)

    # Research-specific metrics
    feature_reduction_ratio = 1.0 - (feature_count_final / max(feature_count_original, 1))
    training_efficiency_gain = baseline_time / max(training_time, 0.001)

    all_metrics = {
        **classification_metrics,
        "feature_reduction_ratio": feature_reduction_ratio,
        "training_efficiency_gain": training_efficiency_gain,
        "feature_count_original": feature_count_original,
        "feature_count_final": feature_count_final,
        "training_time": training_time,
    }

    logger.info(f"Computed all metrics: {all_metrics}")
    return all_metrics
