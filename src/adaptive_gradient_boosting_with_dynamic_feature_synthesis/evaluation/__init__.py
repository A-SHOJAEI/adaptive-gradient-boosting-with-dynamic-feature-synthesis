"""Evaluation metrics and analysis modules."""

from adaptive_gradient_boosting_with_dynamic_feature_synthesis.evaluation.metrics import (
    compute_metrics,
    compute_classification_metrics,
)
from adaptive_gradient_boosting_with_dynamic_feature_synthesis.evaluation.analysis import (
    plot_feature_importance,
    plot_roc_curve,
)

__all__ = [
    "compute_metrics",
    "compute_classification_metrics",
    "plot_feature_importance",
    "plot_roc_curve",
]
