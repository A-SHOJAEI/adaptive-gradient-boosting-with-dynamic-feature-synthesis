"""Adaptive Gradient Boosting with Dynamic Feature Synthesis.

A novel gradient boosting framework that dynamically synthesizes features
during training based on residual error patterns.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from adaptive_gradient_boosting_with_dynamic_feature_synthesis.models.model import (
    AdaptiveGradientBoostingModel,
)
from adaptive_gradient_boosting_with_dynamic_feature_synthesis.training.trainer import (
    AdaptiveGBMTrainer,
)

__all__ = [
    "AdaptiveGradientBoostingModel",
    "AdaptiveGBMTrainer",
]
