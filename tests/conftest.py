"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Generate sample tabular data for testing.

    Returns:
        Tuple of (X, y) with features and labels.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = pd.DataFrame(
        {
            "num_1": np.random.randn(n_samples),
            "num_2": np.random.randn(n_samples),
            "num_3": np.random.randn(n_samples),
            "cat_1": np.random.choice(["A", "B", "C"], n_samples),
            "cat_2": np.random.choice(["X", "Y"], n_samples),
        }
    )

    y = pd.Series(np.random.randint(0, 2, n_samples), name="target")

    return X, y


@pytest.fixture
def sample_train_val_split(sample_data):
    """Split sample data into train and validation sets.

    Args:
        sample_data: Fixture providing sample data.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val).
    """
    from sklearn.model_selection import train_test_split

    X, y = sample_data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val


@pytest.fixture
def sample_config():
    """Sample configuration dictionary.

    Returns:
        Dictionary with configuration parameters.
    """
    return {
        "random_state": 42,
        "data": {
            "dataset_name": "synthetic_classification",
            "test_size": 0.2,
            "val_size": 0.1,
        },
        "model": {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "max_depth": 3,
            "num_leaves": 7,
            "enable_feature_synthesis": True,
        },
        "training": {
            "patience": 5,
            "min_delta": 0.001,
            "checkpoint_dir": "test_checkpoints",
        },
    }
