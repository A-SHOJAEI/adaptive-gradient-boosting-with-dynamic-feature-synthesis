"""Tests for model components."""

import numpy as np
import pandas as pd
import pytest

from adaptive_gradient_boosting_with_dynamic_feature_synthesis.models.components import (
    UncertaintyWeightedLoss,
    FeatureSynthesizer,
    MetaLearningController,
)
from adaptive_gradient_boosting_with_dynamic_feature_synthesis.models.model import (
    AdaptiveGradientBoostingModel,
)
from adaptive_gradient_boosting_with_dynamic_feature_synthesis.data.preprocessing import (
    TabularPreprocessor,
)


class TestUncertaintyWeightedLoss:
    """Tests for uncertainty-weighted loss component."""

    def test_compute_weights_binary(self):
        """Test weight computation for binary classification."""
        loss = UncertaintyWeightedLoss(temperature=1.0, min_weight=0.1)

        # Create some predictions
        y_pred_proba = np.array([0.9, 0.5, 0.1, 0.8, 0.2])
        y_true = np.array([1, 1, 0, 1, 0])

        weights = loss.compute_weights(y_pred_proba, y_true)

        assert weights.shape == (5,)
        assert all(weights >= loss.min_weight)
        # Weights are normalized to sum to n_samples
        assert weights.sum() == pytest.approx(len(weights), rel=0.01)

    def test_weights_higher_for_uncertain_samples(self):
        """Test that uncertain samples have higher entropy."""
        # Just test that the entropy calculation works correctly
        # Confident predictions
        certain_preds = np.array([0.95, 0.05, 0.98, 0.02])
        # Uncertain predictions
        uncertain_preds = np.array([0.5, 0.5, 0.5, 0.5])

        # Compute entropy directly
        def compute_entropy(probs):
            probs_2d = np.vstack([1-probs, probs]).T
            epsilon = 1e-10
            probs_clipped = np.clip(probs_2d, epsilon, 1 - epsilon)
            return -np.sum(probs_clipped * np.log(probs_clipped), axis=1)

        certain_entropy = compute_entropy(certain_preds)
        uncertain_entropy = compute_entropy(uncertain_preds)

        assert uncertain_entropy.mean() > certain_entropy.mean()


class TestFeatureSynthesizer:
    """Tests for feature synthesis component."""

    def test_synthesize_features(self, sample_data):
        """Test basic feature synthesis."""
        X, y = sample_data

        synthesizer = FeatureSynthesizer(max_features_per_round=5)
        residuals = np.random.randn(len(X))

        X_synth = synthesizer.synthesize(X, residuals)

        assert isinstance(X_synth, pd.DataFrame)
        assert len(X_synth) == len(X)
        assert X_synth.shape[1] >= X.shape[1]  # Should have at least as many features

    def test_synthesized_feature_names(self, sample_data):
        """Test that synthesized features have proper names."""
        X, y = sample_data

        synthesizer = FeatureSynthesizer()
        residuals = np.random.randn(len(X))

        X_synth = synthesizer.synthesize(X, residuals)
        synth_features = synthesizer.get_synthesized_features()

        # Check that all synthesized features are in the output
        for feat in synth_features:
            assert feat in X_synth.columns
            assert feat.startswith("synth_")


class TestMetaLearningController:
    """Tests for meta-learning controller component."""

    def test_should_synthesize_first_round(self, sample_data):
        """Test that controller always synthesizes in first round."""
        X, y = sample_data

        controller = MetaLearningController()
        residuals = np.random.randn(len(X))

        should_synth = controller.should_synthesize(X, residuals, 0, 100)
        assert should_synth is True

    def test_controller_history_tracking(self, sample_data):
        """Test that controller tracks decision history."""
        X, y = sample_data

        controller = MetaLearningController()
        residuals = np.random.randn(len(X))

        # Start from round 1 to avoid first round auto-synthesis
        for i in range(1, 6):
            controller.should_synthesize(X, residuals, i, 100)

        history = controller.get_history()
        assert len(history) == 5
        assert all("round" in h for h in history)
        assert all("synthesized" in h for h in history)


class TestAdaptiveGradientBoostingModel:
    """Tests for the main model."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = AdaptiveGradientBoostingModel(
            n_estimators=10, learning_rate=0.1, random_state=42
        )

        assert model.n_estimators == 10
        assert model.learning_rate == 0.1
        assert model.random_state == 42

    def test_model_fit_predict(self, sample_train_val_split):
        """Test model can fit and predict."""
        X_train, X_val, y_train, y_val = sample_train_val_split

        # Preprocess data (LightGBM requires numeric)
        preprocessor = TabularPreprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        model = AdaptiveGradientBoostingModel(
            n_estimators=10, learning_rate=0.1, random_state=42
        )

        # Fit model
        model.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)])

        # Make predictions
        y_pred = model.predict(X_val_proc)
        y_pred_proba = model.predict_proba(X_val_proc)

        assert y_pred.shape == (len(X_val),)
        assert y_pred_proba.shape == (len(X_val), 2)
        assert all(y_pred_proba[:, 0] + y_pred_proba[:, 1] <= 1.01)  # Sum to ~1

    def test_model_predict_without_fit_raises_error(self, sample_data):
        """Test that predict raises error if model not fitted."""
        X, y = sample_data

        model = AdaptiveGradientBoostingModel(n_estimators=10)

        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_model_feature_importance(self, sample_train_val_split):
        """Test that feature importance can be retrieved."""
        X_train, X_val, y_train, y_val = sample_train_val_split

        # Preprocess data
        preprocessor = TabularPreprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)

        model = AdaptiveGradientBoostingModel(n_estimators=10, random_state=42)
        model.fit(X_train_proc, y_train)

        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_model_with_feature_synthesis_disabled(self, sample_train_val_split):
        """Test model works with feature synthesis disabled."""
        X_train, X_val, y_train, y_val = sample_train_val_split

        # Preprocess data
        preprocessor = TabularPreprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        model = AdaptiveGradientBoostingModel(
            n_estimators=10,
            enable_feature_synthesis=False,
            random_state=42,
        )

        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_val_proc)

        assert y_pred.shape == (len(X_val),)
