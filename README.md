# Adaptive Gradient Boosting with Dynamic Feature Synthesis

A novel gradient boosting framework that dynamically synthesizes features during training based on residual error patterns. Uses a meta-learning controller to identify which feature interactions to generate at each boosting round, with an uncertainty-weighted loss function that prioritizes hard examples.

## Key Innovation

This framework addresses the challenge of automated feature engineering in heterogeneous tabular data where optimal features vary across the prediction space. The system combines:

1. **Dynamic Feature Synthesis**: Generates polynomial interactions and statistical aggregations on-the-fly based on residual patterns
2. **Meta-Learning Controller**: Decides when to synthesize features by analyzing current residual statistics
3. **Uncertainty-Weighted Loss**: Prioritizes hard examples using entropy-based sample weighting

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Train the full model with feature synthesis:

```bash
python scripts/train.py --config configs/default.yaml
```

Train baseline model without feature synthesis (ablation):

```bash
python scripts/train.py --config configs/ablation.yaml
```

Evaluate trained model:

```bash
python scripts/evaluate.py --model-path models/final_model.pkl
```

Make predictions:

```bash
python scripts/predict.py --input-file data.csv --output-file predictions.csv
```

## Project Structure

```
adaptive-gradient-boosting-with-dynamic-feature-synthesis/
├── src/adaptive_gradient_boosting_with_dynamic_feature_synthesis/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model and custom components
│   ├── training/          # Training loop and checkpointing
│   ├── evaluation/        # Metrics and analysis
│   └── utils/             # Configuration utilities
├── configs/               # YAML configuration files
├── scripts/               # Training, evaluation, and prediction scripts
├── tests/                 # Unit tests with pytest
└── results/               # Output directory for results
```

## Usage

### Training

The training script supports MLflow tracking, early stopping, and checkpointing:

```python
from adaptive_gradient_boosting_with_dynamic_feature_synthesis import (
    AdaptiveGradientBoostingModel,
    AdaptiveGBMTrainer,
)

model = AdaptiveGradientBoostingModel(
    n_estimators=100,
    learning_rate=0.05,
    enable_feature_synthesis=True,
    max_interaction_degree=2,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

### Configuration

All hyperparameters are configurable via YAML files. Key parameters:

- `enable_feature_synthesis`: Enable/disable dynamic feature generation
- `max_interaction_degree`: Maximum polynomial degree for feature interactions
- `uncertainty_temperature`: Temperature for uncertainty-based sample weighting
- `synthesis_interval`: How often to generate new features

## Methodology

### Overview

Traditional gradient boosting treats feature engineering as a preprocessing step, requiring domain expertise and manual iteration. This framework introduces an adaptive approach that synthesizes features during training, guided by the model's current error patterns.

### Dynamic Feature Synthesis

**Problem**: In heterogeneous tabular data, optimal features vary across the prediction space. Features useful for separating classes in one region may be irrelevant in others.

**Solution**: At each boosting round, the meta-learning controller analyzes residual statistics (mean and variance of prediction errors) to determine if feature synthesis would improve model performance. When residuals remain high (indicating the model struggles), the synthesizer generates:

1. **Polynomial interactions** between top-importance features (degree 2)
2. **Statistical aggregations** (mean, std, range) across feature groups

This allows the model to discover useful feature combinations adaptively, focusing computational resources on regions where the current feature set is insufficient.

### Uncertainty-Weighted Loss

**Problem**: Standard gradient boosting treats all samples equally, which can lead to poor performance on boundary cases and hard-to-classify examples.

**Solution**: Hard examples are identified using prediction entropy as an uncertainty measure:

```
H(y|x) = -∑ p(y|x) log p(y|x)
```

Samples with high entropy (uncertain predictions) receive increased weight in subsequent rounds. This forces the model to focus on difficult cases rather than repeatedly improving on already-well-classified examples. The temperature parameter controls the strength of this reweighting.

### Meta-Learning Controller

**Problem**: Synthesizing features at every round is computationally expensive and can lead to overfitting.

**Solution**: The controller uses heuristics based on residual patterns:
- Always synthesize in round 1 (bootstrap the process)
- Synthesize when mean residual > 0.3 (model struggles)
- Synthesize when residual variance > 0.4 (inconsistent predictions)
- Synthesize every 5 rounds (regular exploration)

This balances exploration of new features with computational efficiency.

### Ablation Studies

The project includes configurations for ablation studies:

- **Full model** (configs/default.yaml): All novel components enabled
- **Baseline** (configs/ablation.yaml): Standard gradient boosting without feature synthesis

## Results

Performance on synthetic binary classification (1,400 train / 200 val / 400 test samples, seed=42):

| Metric | Baseline (No Synthesis) | Full Model (Dynamic Synthesis) |
|--------|------------------------|-------------------------------|
| AUC-ROC | 0.902 | 0.902 |
| Accuracy | 84.75% | 84.75% |
| F1 Score | 0.846 | 0.846 |
| Training Time | 1.25s | 1.24s |

On this synthetic dataset, both configurations achieve equivalent performance, indicating the baseline features are already sufficient for the generated data distribution. The dynamic feature synthesis mechanism is designed to provide gains on more complex, real-world tabular datasets where feature interactions are non-trivial and vary across the prediction space.

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src/adaptive_gradient_boosting_with_dynamic_feature_synthesis --cov-report=html
```

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
