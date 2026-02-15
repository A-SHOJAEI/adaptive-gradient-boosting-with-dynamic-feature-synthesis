"""Data preprocessing for tabular data."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class TabularPreprocessor:
    """Preprocessor for tabular data with mixed numeric and categorical features.

    Handles:
    - Missing value imputation
    - Categorical encoding
    - Numeric feature scaling
    - Feature type detection
    """

    def __init__(
        self,
        numeric_fill_strategy: str = "median",
        categorical_fill_strategy: str = "mode",
        scale_numeric: bool = True,
    ):
        """Initialize the preprocessor.

        Args:
            numeric_fill_strategy: Strategy for filling numeric NaNs ('mean', 'median', 'zero').
            categorical_fill_strategy: Strategy for filling categorical NaNs ('mode', 'missing').
            scale_numeric: Whether to scale numeric features to zero mean and unit variance.
        """
        self.numeric_fill_strategy = numeric_fill_strategy
        self.categorical_fill_strategy = categorical_fill_strategy
        self.scale_numeric = scale_numeric

        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.label_encoders: dict = {}
        self.scaler: Optional[StandardScaler] = None
        self.numeric_fill_values: dict = {}
        self.categorical_fill_values: dict = {}
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        """Fit the preprocessor on training data.

        Args:
            X: Training features.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting preprocessor")

        # Detect feature types
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        logger.info(
            f"Detected {len(self.numeric_cols)} numeric and "
            f"{len(self.categorical_cols)} categorical features"
        )

        # Fit numeric imputation
        for col in self.numeric_cols:
            if self.numeric_fill_strategy == "mean":
                self.numeric_fill_values[col] = X[col].mean()
            elif self.numeric_fill_strategy == "median":
                self.numeric_fill_values[col] = X[col].median()
            else:  # zero
                self.numeric_fill_values[col] = 0.0

        # Fit categorical imputation and encoding
        for col in self.categorical_cols:
            if self.categorical_fill_strategy == "mode":
                mode_val = X[col].mode()
                self.categorical_fill_values[col] = mode_val[0] if len(mode_val) > 0 else "missing"
            else:
                self.categorical_fill_values[col] = "missing"

            # Fit label encoder
            le = LabelEncoder()
            col_filled = X[col].fillna(self.categorical_fill_values[col])
            le.fit(col_filled)
            self.label_encoders[col] = le

        # Fit scaler if needed
        if self.scale_numeric and len(self.numeric_cols) > 0:
            X_numeric = X[self.numeric_cols].copy()
            for col in self.numeric_cols:
                X_numeric[col] = X_numeric[col].fillna(self.numeric_fill_values[col])
            self.scaler = StandardScaler()
            self.scaler.fit(X_numeric)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor.

        Args:
            X: Features to transform.

        Returns:
            Transformed features as DataFrame with all numeric columns.

        Raises:
            RuntimeError: If preprocessor hasn't been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        X_transformed = X.copy()

        # Handle numeric features
        for col in self.numeric_cols:
            X_transformed[col] = X_transformed[col].fillna(self.numeric_fill_values[col])

        # Handle categorical features
        encoded_cols = {}
        for col in self.categorical_cols:
            col_filled = X_transformed[col].fillna(self.categorical_fill_values[col])
            # Handle unseen categories
            le = self.label_encoders[col]
            encoded = []
            for val in col_filled:
                if val in le.classes_:
                    encoded.append(le.transform([val])[0])
                else:
                    # Assign to first class for unseen categories
                    encoded.append(0)
            encoded_cols[col] = encoded

        # Replace categorical columns with encoded versions (avoid dict-modified-during-iteration)
        for col in list(encoded_cols.keys()):
            X_transformed[col] = encoded_cols[col]

        # Scale numeric features
        if self.scale_numeric and self.scaler is not None and len(self.numeric_cols) > 0:
            X_transformed[self.numeric_cols] = self.scaler.transform(
                X_transformed[self.numeric_cols]
            )

        logger.debug(f"Transformed data shape: {X_transformed.shape}")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Features to fit and transform.

        Returns:
            Transformed features.
        """
        return self.fit(X).transform(X)
