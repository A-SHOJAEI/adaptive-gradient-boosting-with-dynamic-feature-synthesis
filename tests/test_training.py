"""Tests for training utilities."""

import tempfile
from pathlib import Path

import pytest

from adaptive_gradient_boosting_with_dynamic_feature_synthesis.training.trainer import (
    AdaptiveGBMTrainer,
)


class TestAdaptiveGBMTrainer:
    """Tests for the trainer class."""

    def test_trainer_initialization(self, sample_config):
        """Test trainer can be initialized."""
        model_config = sample_config["model"]
        training_config = sample_config["training"]

        trainer = AdaptiveGBMTrainer(
            model_config=model_config,
            patience=training_config["patience"],
            random_state=42,
        )

        assert trainer.patience == training_config["patience"]
        assert trainer.random_state == 42

    def test_trainer_train(self, sample_train_val_split, sample_config):
        """Test trainer can train a model."""
        X_train, X_val, y_train, y_val = sample_train_val_split
        model_config = sample_config["model"]

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AdaptiveGBMTrainer(
                model_config=model_config,
                checkpoint_dir=tmpdir,
                patience=5,
                random_state=42,
            )

            model, history = trainer.train(X_train, y_train, X_val, y_val)

            assert model is not None
            assert isinstance(history, dict)
            assert "train_auc" in history
            assert "val_auc" in history

    def test_trainer_saves_checkpoint(self, sample_train_val_split, sample_config):
        """Test trainer saves checkpoints."""
        X_train, X_val, y_train, y_val = sample_train_val_split
        model_config = sample_config["model"]

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AdaptiveGBMTrainer(
                model_config=model_config,
                checkpoint_dir=tmpdir,
                patience=5,
                random_state=42,
            )

            trainer.train(X_train, y_train, X_val, y_val)

            # Check that checkpoint was saved
            checkpoint_path = Path(tmpdir) / "best_model.pkl"
            assert checkpoint_path.exists()

    def test_trainer_load_checkpoint(self, sample_train_val_split, sample_config):
        """Test trainer can load checkpoints."""
        X_train, X_val, y_train, y_val = sample_train_val_split
        model_config = sample_config["model"]

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AdaptiveGBMTrainer(
                model_config=model_config,
                checkpoint_dir=tmpdir,
                patience=5,
                random_state=42,
            )

            trainer.train(X_train, y_train, X_val, y_val)

            # Load checkpoint
            loaded_model = trainer.load_checkpoint("best_model")
            loaded_preprocessor = trainer.preprocessor
            assert loaded_model is not None

            # Make prediction with loaded model (must preprocess first)
            X_val_proc = loaded_preprocessor.transform(X_val)
            y_pred = loaded_model.predict(X_val_proc)
            assert len(y_pred) == len(X_val)

    def test_trainer_save_history(self, sample_train_val_split, sample_config):
        """Test trainer can save training history."""
        X_train, X_val, y_train, y_val = sample_train_val_split
        model_config = sample_config["model"]

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AdaptiveGBMTrainer(
                model_config=model_config,
                checkpoint_dir=tmpdir,
                patience=5,
                random_state=42,
            )

            trainer.train(X_train, y_train, X_val, y_val)

            history_path = Path(tmpdir) / "history.json"
            trainer.save_history(str(history_path))

            assert history_path.exists()
