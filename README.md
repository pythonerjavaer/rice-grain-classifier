# Rice Grain Classifier

A reproducible model-comparison toolkit for classifying rice varieties from grain morphology. It packages preprocessing, cross-validation, hyperparameter search and final evaluation behind one command instead of leaving the workflow tied to a notebook.

## Models

- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbours
- Decision Tree
- AdaBoost
- Gradient Boosting
- Random Forest
- Support Vector Machine

Every candidate receives the same mean-imputation and min-max scaling pipeline. Hyperparameters are selected only on stratified training folds; the held-out test split is used once for the final accuracy and macro-F1 comparison.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
rice-classifier --quick
```

With no input file, the command generates deterministic morphology data so the complete pipeline can be verified immediately. To use a real dataset, place the two-class label in the final CSV column:

```bash
rice-classifier --input /path/to/rice.csv --output-dir artifacts/experiment
```

The output directory contains `metrics.json` and the best fitted pipeline as `best_model.joblib`.

## Expected feature shape

The reference workflow uses seven grain measurements: area, perimeter, major-axis length, minor-axis length, eccentricity, convex area and extent. Column names may differ because all columns except the last are treated as numeric inputs.

## Development

```bash
pytest
```

Tests cover deterministic class encoding and a complete cross-validated comparison.

