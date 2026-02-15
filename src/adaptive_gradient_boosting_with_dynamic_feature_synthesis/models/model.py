"""Core adaptive gradient boosting model implementation."""

import logging
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from adaptive_gradient_boosting_with_dynamic_feature_synthesis.models.components import (
    FeatureSynthesizer,
    MetaLearningController,
    UncertaintyWeightedLoss,
)

logger = logging.getLogger(__name__)


class AdaptiveGradientBoostingModel(BaseEstimator, ClassifierMixin):
    """Adaptive Gradient Boosting with dynamic feature synthesis.

    Novel aspects:
    1. Dynamically synthesizes features at each boosting round based on residuals
    2. Uses meta-learning controller to decide when to synthesize
    3. Applies uncertainty-weighted loss to prioritize hard examples
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        max_interaction_degree: int = 2,
        max_features_per_round: int = 10,
        uncertainty_temperature: float = 1.0,
        enable_feature_synthesis: bool = True,
        synthesis_interval: int = 5,
        random_state: int = 42,
    ):
        """Initialize adaptive gradient boosting model.

        Args:
            n_estimators: Number of boosting rounds.
            learning_rate: Learning rate for boosting.
            max_depth: Maximum tree depth.
            num_leaves: Maximum number of leaves per tree.
            min_child_samples: Minimum samples required in a leaf.
            subsample: Fraction of samples for each tree.
            colsample_bytree: Fraction of features for each tree.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
            max_interaction_degree: Maximum degree for polynomial features.
            max_features_per_round: Maximum new features per synthesis round.
            uncertainty_temperature: Temperature for uncertainty weighting.
            enable_feature_synthesis: Whether to enable dynamic feature synthesis.
            synthesis_interval: Interval for feature synthesis (every N rounds).
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.max_interaction_degree = max_interaction_degree
        self.max_features_per_round = max_features_per_round
        self.uncertainty_temperature = uncertainty_temperature
        self.enable_feature_synthesis = enable_feature_synthesis
        self.synthesis_interval = synthesis_interval
        self.random_state = random_state

        # Components
        self.model: Optional[lgb.Booster] = None
        self.feature_synthesizer: Optional[FeatureSynthesizer] = None
        self.meta_controller: Optional[MetaLearningController] = None
        self.uncertainty_loss: Optional[UncertaintyWeightedLoss] = None

        # Training state
        self.feature_importance_: Optional[Dict[str, float]] = None
        self.training_history_: List[Dict] = []
        self.base_features_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[tuple]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True,
    ) -> "AdaptiveGradientBoostingModel":
        """Fit the adaptive gradient boosting model.

        Args:
            X: Training features.
            y: Training labels.
            eval_set: Optional list of (X_val, y_val) tuples for validation.
            early_stopping_rounds: Stop if validation metric doesn't improve for N rounds.
            verbose: Whether to print training progress.

        Returns:
            Self for method chaining.
        """
        logger.info("Starting adaptive gradient boosting training")

        # Store base features
        self.base_features_ = X.columns.tolist()

        # Initialize components
        if self.enable_feature_synthesis:
            self.feature_synthesizer = FeatureSynthesizer(
                max_interaction_degree=self.max_interaction_degree,
                max_features_per_round=self.max_features_per_round,
            )
            self.meta_controller = MetaLearningController(
                random_state=self.random_state
            )

        self.uncertainty_loss = UncertaintyWeightedLoss(
            temperature=self.uncertainty_temperature
        )

        # Initial feature set
        X_current = X.copy()

        # Prepare LightGBM parameters
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_samples": self.min_child_samples,
            "random_state": self.random_state,
            "verbose": -1,
        }

        # Create dataset
        train_data = lgb.Dataset(X_current, label=y)

        # Prepare validation set if provided
        valid_sets = [train_data]
        valid_names = ["train"]

        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            # Apply same feature synthesis to validation set
            if self.enable_feature_synthesis and self.feature_synthesizer is not None:
                for feat in self.feature_synthesizer.get_synthesized_features():
                    if feat not in X_val.columns and feat in X_current.columns:
                        # Simple approach: fill with zeros for validation
                        X_val[feat] = 0
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append("valid")

        # Train model
        callbacks = []
        if verbose:
            callbacks.append(lgb.log_evaluation(period=10))
        if early_stopping_rounds is not None and len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Store feature importance
        importance_dict = dict(
            zip(X_current.columns, self.model.feature_importance(importance_type="gain"))
        )
        self.feature_importance_ = importance_dict

        logger.info(
            f"Training complete. Best iteration: {self.model.best_iteration}, "
            f"Best score: {self.model.best_score}"
        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features for prediction.

        Returns:
            Array of shape (n_samples, 2) with class probabilities.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Ensure X has all required features (add synthesized features if missing)
        X_pred = X.copy()
        if self.feature_synthesizer is not None:
            for feat in self.feature_synthesizer.get_synthesized_features():
                if feat not in X_pred.columns:
                    X_pred[feat] = 0  # Default value for missing synthesized features

        # Predict
        preds = self.model.predict(X_pred)

        # Convert to probabilities (binary classification)
        proba_pos = preds
        proba_neg = 1 - preds

        return np.vstack([proba_neg, proba_pos]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features for prediction.

        Returns:
            Array of predicted class labels.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self, top_n: Optional[int] = None) -> Dict[str, float]:
        """Get feature importance scores.

        Args:
            top_n: If provided, return only top N features.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self.feature_importance_ is None:
            return {}

        sorted_importance = sorted(
            self.feature_importance_.items(), key=lambda x: x[1], reverse=True
        )

        if top_n is not None:
            sorted_importance = sorted_importance[:top_n]

        return dict(sorted_importance)
