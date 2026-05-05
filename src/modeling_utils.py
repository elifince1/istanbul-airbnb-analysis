#!/usr/bin/env python3
"""Reusable helpers for the Phase 4 price-modeling workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "processed" / "listings_enriched.csv"
TARGET_COLUMN = "price"

IDENTIFIER_COLUMNS = {
    "id",
    "host_id",
    "scrape_id",
    "listing_url",
    "host_url",
    "picture_url",
    "host_thumbnail_url",
    "host_picture_url",
}

TEXT_COLUMNS = {
    "name",
    "description",
    "neighborhood_overview",
    "host_name",
    "host_location",
    "host_about",
    "host_verifications",
    "amenities",
    "bathrooms_text",
}

DATE_COLUMNS = {
    "last_scraped",
    "host_since",
    "calendar_last_scraped",
    "first_review",
    "last_review",
    "first_review_computed",
    "last_review_computed",
}

REDUNDANT_COLUMNS = {
    "source",
    "neighbourhood",
    "license",
}

LEAKAGE_COLUMNS = {
    "estimated_revenue_l365d",
    "estimated_occupancy_l365d",
}

EXCLUDED_FEATURE_COLUMNS = (
    IDENTIFIER_COLUMNS | TEXT_COLUMNS | DATE_COLUMNS | REDUNDANT_COLUMNS | LEAKAGE_COLUMNS
)


def load_modeling_dataset(csv_path: Path | str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the processed listing-level dataset used for modeling."""
    return pd.read_csv(csv_path, low_memory=False)


def split_features_and_target(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a model-ready feature frame and the numeric target series."""
    if target_column not in df.columns:
        raise KeyError(f"Missing target column: {target_column}")

    y = pd.to_numeric(df[target_column], errors="coerce")
    excluded = EXCLUDED_FEATURE_COLUMNS | {target_column}
    feature_columns = [col for col in df.columns if col not in excluded]
    X = df.loc[:, feature_columns].copy()
    return X, y


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical groups for preprocessing."""
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]
    return numeric_columns, categorical_columns


def log_transform_target(y: pd.Series | Iterable[float]) -> pd.Series:
    """Apply log1p to the price target for more stable regression."""
    y_series = pd.Series(y, copy=False, dtype="float64")
    return np.log1p(y_series)


def inverse_log_predictions(predictions: pd.Series | np.ndarray | Iterable[float]):
    """Map log-price predictions back to the original TRY scale."""
    restored = np.expm1(np.asarray(predictions, dtype="float64"))
    restored = np.clip(restored, a_min=0.0, a_max=None)
    restored = np.round(restored, 10)

    if isinstance(predictions, pd.Series):
        return pd.Series(restored, index=predictions.index, name=predictions.name)
    return restored


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray | Iterable[float],
    y_pred: pd.Series | np.ndarray | Iterable[float],
) -> dict[str, float]:
    """Compute regression metrics on the original price scale."""
    y_true_arr = np.asarray(y_true, dtype="float64")
    y_pred_arr = np.asarray(y_pred, dtype="float64")
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }
