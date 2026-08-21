"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import make_demo_data
from .experiment import compare_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classifiers for rice-grain morphology data")
    parser.add_argument("--input", help="CSV file; the final column must contain the class label")
    parser.add_argument("--output-dir", default="artifacts/run")
    parser.add_argument("--quick", action="store_true", help="compare only logistic regression and random forest")
    args = parser.parse_args()

    df = pd.read_csv(args.input) if args.input else make_demo_data()
    names = ["logistic_regression", "random_forest"] if args.quick else None
    results, best_model = compare_models(df, model_names=names)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {name: result.to_dict() for name, result in results.items()}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    joblib.dump(best_model, output / "best_model.joblib")
    print(json.dumps(payload, indent=2))

