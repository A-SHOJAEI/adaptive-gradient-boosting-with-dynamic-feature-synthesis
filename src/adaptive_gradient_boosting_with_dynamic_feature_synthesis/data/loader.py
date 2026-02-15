"""Data loading utilities for various tabular datasets."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def get_available_datasets() -> List[str]:
    """Get list of available dataset names.

    Returns:
        List of dataset names that can be loaded.
    """
    return [
        "adult_census",
        "credit_g",
        "breast_cancer",
        "iris",
        "synthetic_classification",
    ]


def load_dataset(
    dataset_name: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load and split a tabular dataset.

    Args:
        dataset_name: Name of the dataset to load.
        test_size: Proportion of data for test set.
        val_size: Proportion of training data for validation set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "adult_census":
        X, y = _load_adult_census()
    elif dataset_name == "credit_g":
        X, y = _load_credit_g()
    elif dataset_name == "breast_cancer":
        X, y = _load_breast_cancer()
    elif dataset_name == "iris":
        X, y = _load_iris()
    elif dataset_name == "synthetic_classification":
        X, y = _generate_synthetic_classification(random_state=random_state)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {get_available_datasets()}"
        )

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_train,
    )

    logger.info(
        f"Dataset split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _load_adult_census() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Adult Census dataset from OpenML."""
    try:
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        X = data.data
        y = (data.target == ">50K").astype(int)
        logger.info(f"Loaded Adult Census: {X.shape}")
        return X, y
    except Exception as e:
        logger.warning(f"Failed to load Adult Census from OpenML: {e}")
        logger.info("Generating synthetic data as fallback")
        return _generate_synthetic_classification(n_samples=5000, n_features=14)


def _load_credit_g() -> Tuple[pd.DataFrame, pd.Series]:
    """Load German Credit dataset from OpenML."""
    try:
        data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
        X = data.data
        y = (data.target == "bad").astype(int)
        logger.info(f"Loaded Credit-G: {X.shape}")
        return X, y
    except Exception as e:
        logger.warning(f"Failed to load Credit-G from OpenML: {e}")
        logger.info("Generating synthetic data as fallback")
        return _generate_synthetic_classification(n_samples=1000, n_features=20)


def _load_breast_cancer() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Breast Cancer dataset from sklearn."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    logger.info(f"Loaded Breast Cancer: {X.shape}")
    return X, y


def _load_iris() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Iris dataset from sklearn (binary classification)."""
    data = load_iris()
    # Convert to binary classification (setosa vs others)
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series((data.target == 0).astype(int), name="target")
    logger.info(f"Loaded Iris (binary): {X.shape}")
    return X, y


def _generate_synthetic_classification(
    n_samples: int = 2000,
    n_features: int = 20,
    n_informative: int = 15,
    n_categorical: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic classification dataset.

    Args:
        n_samples: Number of samples to generate.
        n_features: Total number of features.
        n_informative: Number of informative features.
        n_categorical: Number of categorical features.
        random_state: Random seed.

    Returns:
        Tuple of (X, y) as DataFrame and Series.
    """
    from sklearn.datasets import make_classification

    rng = np.random.RandomState(random_state)

    # Generate base numeric features
    X_numeric, y = make_classification(
        n_samples=n_samples,
        n_features=n_features - n_categorical,
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_categorical - n_informative - 2),
        n_clusters_per_class=2,
        class_sep=0.8,
        flip_y=0.05,
        random_state=random_state,
    )

    # Add categorical features
    categorical_data = {}
    for i in range(n_categorical):
        n_categories = rng.randint(3, 8)
        categorical_data[f"cat_{i}"] = rng.choice(
            [f"cat_{i}_val_{j}" for j in range(n_categories)], size=n_samples
        )

    # Combine numeric and categorical
    X_df = pd.DataFrame(
        X_numeric, columns=[f"num_{i}" for i in range(X_numeric.shape[1])]
    )
    for col, values in categorical_data.items():
        X_df[col] = values

    y_series = pd.Series(y, name="target")

    logger.info(f"Generated synthetic dataset: {X_df.shape}")
    return X_df, y_series
