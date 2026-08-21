"""Dataset loading and deterministic demonstration data."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent",
]


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain feature columns and a final class column")
    working = df.replace("?", np.nan).copy()
    X = working.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    raw_target = working.iloc[:, -1].astype(str).str.strip()
    labels = sorted(raw_target.unique())
    if len(labels) != 2:
        raise ValueError(f"Expected two rice classes, found {len(labels)}")
    mapping = {label: index for index, label in enumerate(labels)}
    return X, raw_target.map(mapping).astype(int)


def make_demo_data(rows: int = 420, seed: int = 7) -> pd.DataFrame:
    """Create two separable, rice-like morphology populations."""

    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, rows)
    area = rng.normal(12200 + target * 2500, 900, rows)
    major = rng.normal(185 + target * 24, 11, rows)
    minor = rng.normal(83 + target * 8, 6, rows)
    perimeter = rng.normal(440 + target * 48, 20, rows)
    eccentricity = np.sqrt(np.maximum(0, 1 - (minor / major) ** 2))
    convex = area * rng.normal(1.035, 0.012, rows)
    extent = rng.normal(0.72 - target * 0.055, 0.035, rows)
    frame = pd.DataFrame(
        np.column_stack([area, perimeter, major, minor, eccentricity, convex, extent]),
        columns=FEATURES,
    )
    frame["Class"] = np.where(target == 0, "class1", "class2")
    return frame

