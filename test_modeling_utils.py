from pathlib import Path
import math
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modeling_utils import (  # noqa: E402
    compute_regression_metrics,
    inverse_log_predictions,
    log_transform_target,
    split_features_and_target,
)
from create_modeling_notebook import NOTEBOOK_PATH, build_notebook  # noqa: E402
from train_price_models import build_model_registry, evaluate_single_model  # noqa: E402


def test_split_features_and_target_excludes_identifier_text_and_leakage_columns():
    df = pd.DataFrame(
        {
            "price": [100.0, 200.0],
            "id": [1, 2],
            "host_id": [10, 20],
            "listing_url": ["a", "b"],
            "name": ["flat", "loft"],
            "description": ["nice", "great"],
            "amenities": ["wifi", "wifi,kitchen"],
            "estimated_revenue_l365d": [10000.0, 20000.0],
            "accommodates": [2, 4],
            "room_type": ["Private room", "Entire home/apt"],
            "host_response_rate": [90.0, 100.0],
        }
    )

    X, y = split_features_and_target(df)

    assert y.tolist() == [100.0, 200.0]
    assert set(X.columns) == {"accommodates", "room_type", "host_response_rate"}


def test_log_target_round_trip_recovers_original_prices():
    y = pd.Series([80.0, 2535.0, 100000.0], name="price")

    transformed = log_transform_target(y)
    restored = inverse_log_predictions(transformed)

    assert restored.tolist() == y.tolist()


def test_compute_regression_metrics_returns_expected_keys_and_values():
    y_true = pd.Series([100.0, 200.0, 300.0], name="price")
    y_pred = pd.Series([110.0, 190.0, 310.0], name="prediction")

    metrics = compute_regression_metrics(y_true, y_pred)

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert math.isclose(metrics["mae"], 10.0)
    assert math.isclose(metrics["rmse"], 10.0)
    assert math.isclose(metrics["r2"], 0.985)


def test_build_model_registry_contains_course_aligned_regressors():
    registry = build_model_registry(
        numeric_columns=["accommodates", "host_response_rate"],
        categorical_columns=["room_type"],
    )

    assert set(registry) == {"ridge", "knn", "decision_tree", "random_forest"}


def test_model_pipelines_can_fit_a_small_mixed_type_dataset():
    X = pd.DataFrame(
        {
            "accommodates": [1, 2, 3, 4, 2, 5],
            "host_response_rate": [50.0, 80.0, 90.0, 100.0, 75.0, 60.0],
            "room_type": [
                "Private room",
                "Entire home/apt",
                "Entire home/apt",
                "Hotel room",
                "Private room",
                "Entire home/apt",
            ],
        }
    )
    y = pd.Series([100.0, 220.0, 300.0, 450.0, 180.0, 380.0], name="price")

    registry = build_model_registry(
        numeric_columns=["accommodates", "host_response_rate"],
        categorical_columns=["room_type"],
    )

    for pipeline in registry.values():
        pipeline.fit(X, log_transform_target(y))
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)


def test_evaluate_single_model_returns_cv_and_test_metrics():
    X = pd.DataFrame(
        {
            "accommodates": [1, 2, 3, 4, 2, 5, 3, 1, 4, 5],
            "host_response_rate": [50.0, 80.0, 90.0, 100.0, 75.0, 60.0, 85.0, 40.0, 95.0, 70.0],
            "room_type": [
                "Private room",
                "Entire home/apt",
                "Entire home/apt",
                "Hotel room",
                "Private room",
                "Entire home/apt",
                "Private room",
                "Shared room",
                "Hotel room",
                "Entire home/apt",
            ],
        }
    )
    y = pd.Series([100.0, 220.0, 300.0, 450.0, 180.0, 380.0, 240.0, 90.0, 420.0, 360.0], name="price")

    registry = build_model_registry(
        numeric_columns=["accommodates", "host_response_rate"],
        categorical_columns=["room_type"],
    )

    result = evaluate_single_model(
        model_name="ridge",
        pipeline=registry["ridge"],
        X_train=X.iloc[:8].reset_index(drop=True),
        y_train=y.iloc[:8].reset_index(drop=True),
        X_test=X.iloc[8:].reset_index(drop=True),
        y_test=y.iloc[8:].reset_index(drop=True),
        cv_splits=2,
    )

    assert result["model_name"] == "ridge"
    assert {"cv_mae_mean", "cv_rmse_mean", "cv_r2_mean", "test_mae", "test_rmse", "test_r2"} <= set(result)


def test_build_notebook_contains_expected_phase4_sections():
    notebook = build_notebook()
    sources = "\n".join(cell["source"] for cell in notebook.cells)

    assert NOTEBOOK_PATH.name == "03_modeling.ipynb"
    assert "## 0. Setup" in sources
    assert "## 1. Load Data And Train Models" in sources
    assert "## 2. Compare Model Performance" in sources
    assert "## 3. Inspect The Best Model" in sources
    assert "## 4. Summary" in sources
