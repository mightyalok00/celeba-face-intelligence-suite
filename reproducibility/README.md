# Experiment Reproducibility Guide

This directory provides everything necessary for third-party researchers and developers to independently reproduce the evaluation benchmarks, identity-disjoint cross-validation splits, and SLSQP stacking ensembles reported in this repository.

## 1. Quick Start (One-Command Reproduction)

To execute the complete 8-architecture training, out-of-fold probability estimation, SLSQP optimization, and holdout test set evaluation:

```bash
# Execute end-to-end benchmark pipeline
python src/models/train_and_evaluate.py

# Generate publication-ready figures, confusion matrices, and ROC curves
python src/models/generate_plots.py
```

## 2. Environment Setup

### Using Conda

```bash
conda env create -f reproducibility/environment.yml
conda activate celeba-face-ai
```

### Using Pip

```bash
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock ruff mypy bandit
```

## 3. Strict Identity Disjointness Protocol

Unlike conventional randomized k-fold cross-validation which allows photos of the same individual to leak between training and test folds, this project enforces strict identity disjointness:

1. **Holdout Split**: `GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)` partitioned by `person_id`.
   - Result: Zero overlapping identities between training and test sets.
2. **Cross-Validation**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` partitioned by `person_id`.
   - Result: Out-of-fold probability matrices generated with zero identity leakage across validation folds.
3. **Ensemble Optimization**: Stacking weights are solved strictly on out-of-fold validation predictions using Sequential Least Squares Programming (SLSQP). Ensemble weights are **never** optimized or fitted on the final test set.

## 4. Run Audit & Automated Tests

To run the automated verification suite with test coverage reporting:

```bash
# Execute unit and integration tests
pytest tests/ -v --cov=app --cov=src --cov-report=term-missing

# Run code quality, typing, and security scanners
ruff check .
```
