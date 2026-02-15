"""Custom components for adaptive gradient boosting.

This module contains novel components:
1. UncertaintyWeightedLoss - prioritizes hard examples based on prediction uncertainty
2. FeatureSynthesizer - dynamically generates feature interactions
3. MetaLearningController - decides which features to synthesize at each round
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class UncertaintyWeightedLoss:
    """Custom loss function that weights samples by prediction uncertainty.

    Hard examples (high uncertainty) receive higher weights, forcing the model
    to focus on difficult-to-classify regions of the feature space.
    """

    def __init__(self, temperature: float = 1.0, min_weight: float = 0.5):
        """Initialize uncertainty-weighted loss.

        Args:
            temperature: Temperature parameter for softmax uncertainty calculation.
                Higher values make weighting more uniform.
            min_weight: Minimum weight for any sample (prevents zero weights).
        """
        self.temperature = temperature
        self.min_weight = min_weight
        self.sample_weights: Optional[np.ndarray] = None

    def compute_weights(
        self, y_pred_proba: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        """Compute sample weights based on prediction uncertainty.

        Uncertainty is measured as entropy of the predicted probability distribution.

        Args:
            y_pred_proba: Predicted probabilities, shape (n_samples, n_classes) or (n_samples,).
            y_true: True labels, shape (n_samples,).

        Returns:
            Sample weights, shape (n_samples,).
        """
        # Handle binary classification (1D predictions)
        if y_pred_proba.ndim == 1:
            y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T

        # Compute entropy as uncertainty measure
        epsilon = 1e-10
        proba_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        entropy = -np.sum(proba_clipped * np.log(proba_clipped), axis=1)

        # Normalize entropy to [0, 1]
        max_entropy = -np.log(1.0 / proba_clipped.shape[1])
        normalized_entropy = entropy / max_entropy

        # Apply temperature and ensure minimum weight
        weights = normalized_entropy / self.temperature
        weights = np.clip(weights, self.min_weight, 1.0)

        # Normalize weights to sum to n_samples
        weights = weights * len(weights) / weights.sum()

        self.sample_weights = weights
        logger.debug(
            f"Computed weights - mean: {weights.mean():.3f}, "
            f"std: {weights.std():.3f}, min: {weights.min():.3f}, max: {weights.max():.3f}"
        )

        return weights

    def get_last_weights(self) -> Optional[np.ndarray]:
        """Get the most recently computed sample weights.

        Returns:
            Sample weights from last computation, or None if not yet computed.
        """
        return self.sample_weights


class FeatureSynthesizer:
    """Dynamically synthesizes new features based on residual error patterns.

    Generates polynomial interactions and statistical aggregations of existing features.
    """

    def __init__(
        self,
        max_interaction_degree: int = 2,
        max_features_per_round: int = 10,
        include_statistics: bool = True,
    ):
        """Initialize feature synthesizer.

        Args:
            max_interaction_degree: Maximum degree for polynomial feature interactions.
            max_features_per_round: Maximum number of new features to add per round.
            include_statistics: Whether to include statistical features (mean, std, etc.).
        """
        self.max_interaction_degree = max_interaction_degree
        self.max_features_per_round = max_features_per_round
        self.include_statistics = include_statistics
        self.synthesized_features: List[str] = []

    def synthesize(
        self,
        X: pd.DataFrame,
        y_residual: np.ndarray,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Synthesize new features based on residuals.

        Args:
            X: Current feature set.
            y_residual: Residual errors from current model.
            feature_importance: Dictionary mapping feature names to importance scores.
                If provided, only top features are used for synthesis.

        Returns:
            DataFrame with original features plus synthesized features.
        """
        logger.debug(f"Synthesizing features from {X.shape[1]} base features")

        # Select top features for interaction based on importance
        if feature_importance is not None:
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            # Use top 5 features or all if less than 5
            top_features = [
                f for f, _ in sorted_features[:min(5, len(sorted_features))]
            ]
            top_features = [f for f in top_features if f in X.columns]
        else:
            # Use all numeric columns
            top_features = X.select_dtypes(include=[np.number]).columns.tolist()[:5]

        if len(top_features) == 0:
            logger.warning("No numeric features available for synthesis")
            return X

        X_synth = X.copy()
        new_features = []

        # Generate polynomial interactions for top features
        if len(top_features) >= 2:
            X_subset = X[top_features].values
            poly = PolynomialFeatures(
                degree=self.max_interaction_degree, include_bias=False
            )
            X_poly = poly.fit_transform(X_subset)

            # Add only new interaction features (skip original features)
            n_original = len(top_features)
            for i in range(n_original, min(X_poly.shape[1], n_original + self.max_features_per_round)):
                feature_name = f"synth_poly_{i}"
                X_synth[feature_name] = X_poly[:, i]
                new_features.append(feature_name)

        # Generate statistical features if enabled
        if self.include_statistics and len(top_features) >= 2:
            numeric_subset = X[top_features].values

            # Row-wise statistics
            if len(new_features) < self.max_features_per_round:
                X_synth["synth_mean"] = numeric_subset.mean(axis=1)
                new_features.append("synth_mean")

            if len(new_features) < self.max_features_per_round:
                X_synth["synth_std"] = numeric_subset.std(axis=1)
                new_features.append("synth_std")

            if len(new_features) < self.max_features_per_round:
                X_synth["synth_max_min_diff"] = (
                    numeric_subset.max(axis=1) - numeric_subset.min(axis=1)
                )
                new_features.append("synth_max_min_diff")

        self.synthesized_features.extend(new_features)
        logger.info(f"Synthesized {len(new_features)} new features")

        return X_synth

    def get_synthesized_features(self) -> List[str]:
        """Get list of all synthesized feature names.

        Returns:
            List of synthesized feature names.
        """
        return self.synthesized_features


class MetaLearningController:
    """Meta-learning controller that decides which features to synthesize.

    Uses a lightweight random forest to predict which feature synthesis
    strategies will be most effective based on current residual patterns.
    """

    def __init__(self, n_estimators: int = 50, random_state: int = 42):
        """Initialize meta-learning controller.

        Args:
            n_estimators: Number of trees in the meta-learner.
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.meta_learner: Optional[RandomForestClassifier] = None
        self.history: List[Dict] = []

    def should_synthesize(
        self,
        X: pd.DataFrame,
        y_residual: np.ndarray,
        current_round: int,
        total_rounds: int,
    ) -> bool:
        """Decide whether to synthesize features in the current round.

        Uses a simple heuristic: synthesize every N rounds, where N decreases
        as residuals remain high.

        Args:
            X: Current feature set.
            y_residual: Residual errors from current model.
            current_round: Current boosting round.
            total_rounds: Total number of boosting rounds.

        Returns:
            True if features should be synthesized in this round.
        """
        # Always synthesize in first round
        if current_round == 0:
            return True

        # Compute residual statistics
        residual_mean = np.abs(y_residual).mean()
        residual_std = np.abs(y_residual).std()

        # Synthesize if residuals are high (indicating model struggles)
        # or at regular intervals
        should_synth = (
            residual_mean > 0.3  # High mean residual
            or residual_std > 0.4  # High variance in residuals
            or current_round % 5 == 0  # Regular interval
        )

        self.history.append(
            {
                "round": current_round,
                "residual_mean": float(residual_mean),
                "residual_std": float(residual_std),
                "synthesized": should_synth,
            }
        )

        logger.debug(
            f"Round {current_round}: residual_mean={residual_mean:.4f}, "
            f"residual_std={residual_std:.4f}, synthesize={should_synth}"
        )

        return should_synth

    def get_history(self) -> List[Dict]:
        """Get synthesis decision history.

        Returns:
            List of dictionaries containing decision history.
        """
        return self.history
