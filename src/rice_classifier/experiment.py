"""Cross-validated classifier comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data import split_features_and_target


@dataclass(frozen=True)
class ClassifierResult:
    cv_accuracy: float
    test_accuracy: float
    macro_f1: float
    parameters: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _specifications(random_state: int) -> dict[str, tuple[object, dict[str, list[object]]]]:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2500, random_state=random_state),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "naive_bayes": (GaussianNB(), {"model__var_smoothing": [1e-9, 1e-8]}),
        "knn": (KNeighborsClassifier(), {"model__n_neighbors": [5, 11], "model__weights": ["uniform", "distance"]}),
        "decision_tree": (
            DecisionTreeClassifier(random_state=random_state),
            {"model__max_depth": [3, 6, None], "model__min_samples_leaf": [1, 3]},
        ),
        "adaboost": (
            AdaBoostClassifier(random_state=random_state),
            {"model__n_estimators": [50, 120], "model__learning_rate": [0.2, 1.0]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            {"model__n_estimators": [80, 160], "model__max_leaf_nodes": [6, 12, None]},
        ),
        "svm": (SVC(random_state=random_state), {"model__C": [1, 10], "model__kernel": ["linear", "rbf"]}),
    }


def compare_models(
    df: pd.DataFrame,
    *,
    model_names: list[str] | None = None,
    folds: int = 5,
    random_state: int = 0,
) -> tuple[dict[str, ClassifierResult], object]:
    """Tune and compare classifiers using consistent splits and preprocessing."""

    X, y = split_features_and_target(df)
    if len(df) < 40:
        raise ValueError("At least 40 rows are required for a reliable comparison")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    specifications = _specifications(random_state)
    selected = model_names or list(specifications)
    unknown = sorted(set(selected) - set(specifications))
    if unknown:
        raise ValueError(f"Unknown classifiers: {', '.join(unknown)}")

    results: dict[str, ClassifierResult] = {}
    fitted: dict[str, object] = {}
    for name in selected:
        estimator, grid = specifications[name]
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", MinMaxScaler()),
                ("model", estimator),
            ]
        )
        # A single worker also runs reliably in restricted containers and CI.
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="accuracy", n_jobs=1)
        search.fit(X_train, y_train)
        predictions = search.predict(X_test)
        results[name] = ClassifierResult(
            cv_accuracy=float(search.best_score_),
            test_accuracy=float(accuracy_score(y_test, predictions)),
            macro_f1=float(f1_score(y_test, predictions, average="macro")),
            parameters={key.removeprefix("model__"): value for key, value in search.best_params_.items()},
        )
        fitted[name] = search.best_estimator_

    best_name = max(results, key=lambda name: (results[name].macro_f1, results[name].test_accuracy))
    return results, fitted[best_name]
