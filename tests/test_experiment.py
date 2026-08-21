from rice_classifier.data import make_demo_data, split_features_and_target
from rice_classifier.experiment import compare_models


def test_target_encoding_is_deterministic() -> None:
    X, y = split_features_and_target(make_demo_data(rows=60))
    assert X.shape == (60, 7)
    assert set(y) == {0, 1}


def test_quick_model_comparison() -> None:
    results, model = compare_models(
        make_demo_data(rows=180),
        model_names=["logistic_regression", "random_forest"],
        folds=3,
    )
    assert set(results) == {"logistic_regression", "random_forest"}
    assert max(result.macro_f1 for result in results.values()) > 0.85
    assert hasattr(model, "predict")

