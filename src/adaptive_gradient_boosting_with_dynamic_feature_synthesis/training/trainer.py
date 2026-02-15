"""Training loop with early stopping and checkpointing."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from adaptive_gradient_boosting_with_dynamic_feature_synthesis.data.preprocessing import (
    TabularPreprocessor,
)
from adaptive_gradient_boosting_with_dynamic_feature_synthesis.models.model import (
    AdaptiveGradientBoostingModel,
)

logger = logging.getLogger(__name__)


class AdaptiveGBMTrainer:
    """Trainer for adaptive gradient boosting with checkpointing and early stopping."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        checkpoint_dir: str = "checkpoints",
        patience: int = 20,
        min_delta: float = 0.0001,
        random_state: int = 42,
    ):
        """Initialize trainer.

        Args:
            model_config: Configuration dictionary for model hyperparameters.
            checkpoint_dir: Directory to save model checkpoints.
            patience: Number of rounds without improvement before early stopping.
            min_delta: Minimum improvement to reset patience counter.
            random_state: Random seed for reproducibility.
        """
        self.model_config = model_config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[AdaptiveGradientBoostingModel] = None
        self.preprocessor: Optional[TabularPreprocessor] = None
        self.best_score: float = 0.0
        self.best_epoch: int = 0
        self.training_history: Dict[str, list] = {
            "train_auc": [],
            "val_auc": [],
            "epoch": [],
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[AdaptiveGradientBoostingModel, Dict[str, list]]:
        """Train the model with early stopping.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Tuple of (trained_model, training_history).
        """
        logger.info("Starting training")
        logger.info(
            f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples"
        )

        # Set random seed
        np.random.seed(self.random_state)

        # Preprocess data
        logger.info("Preprocessing data")
        self.preprocessor = TabularPreprocessor(
            numeric_fill_strategy="median",
            categorical_fill_strategy="mode",
            scale_numeric=True,
        )

        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)

        # Initialize model
        logger.info("Initializing model")
        self.model = AdaptiveGradientBoostingModel(
            random_state=self.random_state, **self.model_config
        )

        # Train with early stopping
        self.model.fit(
            X_train_processed,
            y_train,
            eval_set=[(X_val_processed, y_val)],
            early_stopping_rounds=self.patience,
            verbose=True,
        )

        # Compute final metrics
        train_preds = self.model.predict_proba(X_train_processed)[:, 1]
        val_preds = self.model.predict_proba(X_val_processed)[:, 1]

        train_auc = roc_auc_score(y_train, train_preds)
        val_auc = roc_auc_score(y_val, val_preds)

        logger.info(f"Training complete - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        self.best_score = val_auc
        self.training_history["train_auc"].append(train_auc)
        self.training_history["val_auc"].append(val_auc)
        self.training_history["epoch"].append(0)

        # Save best model
        self._save_checkpoint("best_model")

        return self.model, self.training_history

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint.

        Args:
            name: Checkpoint name (without extension).
        """
        if self.model is None:
            logger.warning("No model to save")
            return

        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        preprocessor_path = self.checkpoint_dir / f"{name}_preprocessor.pkl"

        try:
            joblib.dump(self.model, checkpoint_path)
            joblib.dump(self.preprocessor, preprocessor_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, name: str = "best_model") -> AdaptiveGradientBoostingModel:
        """Load model from checkpoint.

        Args:
            name: Checkpoint name (without extension).

        Returns:
            Loaded model.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        preprocessor_path = self.checkpoint_dir / f"{name}_preprocessor.pkl"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.model = joblib.load(checkpoint_path)
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return self.model

    def save_history(self, output_path: str) -> None:
        """Save training history to JSON.

        Args:
            output_path: Path to save history JSON.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(self.training_history, f, indent=2)

        logger.info(f"Saved training history to {output_path}")
