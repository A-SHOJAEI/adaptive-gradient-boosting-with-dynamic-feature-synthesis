"""Tests for data loading and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from adaptive_gradient_boosting_with_dynamic_feature_synthesis.data.loader import (
    load_dataset,
    get_available_datasets,
)
from adaptive_gradient_boosting_with_dynamic_feature_synthesis.data.preprocessing import (
    TabularPreprocessor,
)


class TestDataLoader:
    """Tests for data loading utilities."""

    def test_get_available_datasets(self):
        """Test that available datasets list is returned."""
        datasets = get_available_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert "synthetic_classification" in datasets

    def test_load_synthetic_dataset(self):
        """Test loading synthetic classification dataset."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(
            dataset_name="synthetic_classification",
            test_size=0.2,
            val_size=0.1,
            random_state=42,
        )

        # Check shapes
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert len(X_train) > 0
        assert len(X_train) == len(y_train)
        assert len(X_val) > 0
        assert len(X_test) > 0

        # Check that labels are binary
        assert set(y_train.unique()).issubset({0, 1})

    def test_load_dataset_split_proportions(self):
        """Test that dataset split proportions are correct."""
        test_size = 0.2
        val_size = 0.1

        X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(
            dataset_name="synthetic_classification",
            test_size=test_size,
            val_size=val_size,
            random_state=42,
        )

        total_samples = len(X_train) + len(X_val) + len(X_test)
        test_ratio = len(X_test) / total_samples
        val_ratio = len(X_val) / total_samples

        # Allow some tolerance due to rounding
        assert abs(test_ratio - test_size) < 0.05
        assert abs(val_ratio - val_size) < 0.05


class TestTabularPreprocessor:
    """Tests for tabular data preprocessing."""

    def test_preprocessor_fit_transform(self, sample_data):
        """Test basic fit and transform."""
        X, y = sample_data

        preprocessor = TabularPreprocessor()
        X_transformed = preprocessor.fit_transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed) == len(X)
        assert preprocessor.is_fitted

    def test_preprocessor_handles_missing_values(self):
        """Test that preprocessor handles missing values."""
        X = pd.DataFrame(
            {
                "num_1": [1.0, 2.0, np.nan, 4.0],
                "num_2": [5.0, np.nan, 7.0, 8.0],
                "cat_1": ["A", "B", None, "A"],
            }
        )

        preprocessor = TabularPreprocessor()
        X_transformed = preprocessor.fit_transform(X)

        # Check no missing values remain
        assert not X_transformed.isnull().any().any()

    def test_preprocessor_separate_fit_transform(self, sample_data):
        """Test separate fit and transform calls."""
        X, y = sample_data

        # Split data
        split_idx = len(X) // 2
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]

        preprocessor = TabularPreprocessor()
        preprocessor.fit(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        assert isinstance(X_test_transformed, pd.DataFrame)
        assert len(X_test_transformed) == len(X_test)

    def test_preprocessor_raises_error_if_not_fitted(self, sample_data):
        """Test that transform raises error if not fitted."""
        X, y = sample_data

        preprocessor = TabularPreprocessor()

        with pytest.raises(RuntimeError):
            preprocessor.transform(X)

    def test_preprocessor_scaling(self, sample_data):
        """Test that numeric scaling works."""
        X, y = sample_data

        preprocessor = TabularPreprocessor(scale_numeric=True)
        X_transformed = preprocessor.fit_transform(X)

        # Check that numeric columns are roughly standardized
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                mean = X_transformed[col].mean()
                std = X_transformed[col].std()
                assert abs(mean) < 0.1  # Close to 0
                assert abs(std - 1.0) < 0.1  # Close to 1
