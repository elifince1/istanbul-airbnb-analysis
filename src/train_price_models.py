#!/usr/bin/env python3
"""Train and compare course-aligned price prediction models."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from modeling_utils import (
    DEFAULT_DATASET_PATH,
    TARGET_COLUMN,
    compute_regression_metrics,
    infer_feature_types,
    inverse_log_predictions,
    load_modeling_dataset,
    log_transform_target,
    split_features_and_target,
)


BASE_DIR = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def _fit_and_predict_log(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame
) -> np.ndarray:
    """Fit one pipeline and return log-price predictions with runtime checks."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        pipeline.fit(X_train, log_transform_target(y_train))
        predictions = pipeline.predict(X_eval)

    predictions = np.asarray(predictions, dtype="float64")
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite predictions encountered during model evaluation.")
    return predictions


def build_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str], scale_numeric: bool
) -> ColumnTransformer:
    """Create a shared preprocessing graph for mixed-type Airbnb features."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_model_registry(
    numeric_columns: list[str], categorical_columns: list[str]
) -> dict[str, Pipeline]:
    """Return the regression models selected from the course material."""
    scaled_preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        scale_numeric=True,
    )
    tree_preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        scale_numeric=False,
    )

    return {
        "ridge": Pipeline(
            steps=[
                ("preprocessor", scaled_preprocessor),
                ("model", Ridge(alpha=10.0, solver="lsqr")),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("preprocessor", scaled_preprocessor),
                ("model", KNeighborsRegressor(n_neighbors=5, weights="distance")),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=12,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=16,
                        min_samples_leaf=5,
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_single_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = 5,
) -> dict[str, object]:
    """Evaluate one model with train-only CV and a final held-out test set."""
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_records = []

    y_train = pd.Series(y_train, copy=False).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)

    for train_idx, valid_idx in kfold.split(X_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        fitted_fold = clone(pipeline)
        fold_predictions_log = _fit_and_predict_log(
            fitted_fold, X_fold_train, y_fold_train, X_fold_valid
        )
        fold_predictions = inverse_log_predictions(fold_predictions_log)
        cv_records.append(compute_regression_metrics(y_fold_valid, fold_predictions))

    fitted_pipeline = clone(pipeline)
    test_predictions_log = _fit_and_predict_log(
        fitted_pipeline, X_train, y_train, X_test
    )
    test_predictions = inverse_log_predictions(test_predictions_log)
    test_metrics = compute_regression_metrics(y_test, test_predictions)

    cv_metrics = pd.DataFrame(cv_records).mean().to_dict()
    return {
        "model_name": model_name,
        "cv_mae_mean": float(cv_metrics["mae"]),
        "cv_rmse_mean": float(cv_metrics["rmse"]),
        "cv_r2_mean": float(cv_metrics["r2"]),
        "test_mae": float(test_metrics["mae"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_r2": float(test_metrics["r2"]),
        "fitted_pipeline": fitted_pipeline,
        "test_predictions": pd.Series(test_predictions, index=y_test.index, name=model_name),
    }


def run_training_workflow(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    test_size: float = 0.2,
    cv_splits: int = 3,
) -> dict[str, object]:
    """Train every selected model and return a structured results payload."""
    df = load_modeling_dataset(dataset_path)
    X, y = split_features_and_target(df, target_column=TARGET_COLUMN)
    numeric_columns, categorical_columns = infer_feature_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    registry = build_model_registry(numeric_columns, categorical_columns)
    results = []
    fitted_models: dict[str, Pipeline] = {}
    test_predictions: dict[str, pd.Series] = {}

    for model_name, pipeline in registry.items():
        evaluation = evaluate_single_model(
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv_splits=cv_splits,
        )
        fitted_models[model_name] = evaluation.pop("fitted_pipeline")
        test_predictions[model_name] = evaluation.pop("test_predictions")
        results.append(evaluation)

    results_df = pd.DataFrame(results).sort_values(
        by=["test_rmse", "test_mae"], ascending=True
    ).reset_index(drop=True)
    best_model_name = results_df.loc[0, "model_name"]

    return {
        "results": results_df,
        "best_model_name": best_model_name,
        "fitted_models": fitted_models,
        "test_predictions": test_predictions,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def _clean_feature_name(feature_name: str) -> str:
    """Drop transformer prefixes to keep notebook tables readable."""
    for prefix in ("numeric__", "categorical__"):
        if feature_name.startswith(prefix):
            return feature_name[len(prefix) :]
    return feature_name


def get_feature_name_table(fitted_pipeline: Pipeline) -> pd.Index:
    """Return transformed feature names from the preprocessing step."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    return pd.Index([_clean_feature_name(name) for name in feature_names], name="feature")


def get_feature_importance_table(
    fitted_pipeline: Pipeline, top_n: int = 15
) -> pd.DataFrame:
    """Return a sorted feature-importance table for tree-based models."""
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame(
        {
            "feature": get_feature_name_table(fitted_pipeline),
            "importance": model.feature_importances_,
        }
    )
    return importance_df.sort_values("importance", ascending=False).head(top_n)


def get_linear_coefficient_table(
    fitted_pipeline: Pipeline, top_n: int = 15
) -> pd.DataFrame:
    """Return the strongest absolute coefficients from the Ridge baseline."""
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    coefficient_df = pd.DataFrame(
        {
            "feature": get_feature_name_table(fitted_pipeline),
            "coefficient": model.coef_,
        }
    )
    coefficient_df["abs_coefficient"] = coefficient_df["coefficient"].abs()
    return coefficient_df.sort_values("abs_coefficient", ascending=False).head(top_n)


def main() -> None:
    """Run the training workflow and print a compact summary table."""
    payload = run_training_workflow()
    results = payload["results"]
    print("Model comparison (sorted by test RMSE):")
    print(results.to_string(index=False))
    print()
    print(f"Best model: {payload['best_model_name']}")


if __name__ == "__main__":
    main()
