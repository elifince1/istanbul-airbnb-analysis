#!/usr/bin/env python3
"""Generate the Phase 4 modeling notebook as a reproducible .ipynb file."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "03_modeling.ipynb"


def build_notebook():
    """Build the modeling notebook object."""
    cells = []

    def md(source: str) -> None:
        cells.append(new_markdown_cell(source))

    def code(source: str) -> None:
        cells.append(new_code_cell(source))

    md(
        """# Istanbul Airbnb Analysis — Machine Learning

**DSA 210 — Spring 2025–2026**  
**Student:** Elif İnce

This notebook applies course-aligned regression models to the enriched Istanbul Airbnb dataset in order to predict listing price and compare model families covered in class."""
    )

    md("## 0. Setup")
    code(
        """from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str((PROJECT_ROOT / 'src').resolve()))

from train_price_models import (
    get_feature_importance_table,
    get_linear_coefficient_table,
    run_training_workflow,
)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

FIGURES_DIR = PROJECT_ROOT / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

def save_fig(name, fig=None):
    path = FIGURES_DIR / f'{name}.png'
    if fig is not None:
        fig.savefig(path, bbox_inches='tight', facecolor='white')
    else:
        plt.savefig(path, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')"""
    )

    md("## 1. Load Data And Train Models")
    code(
        """payload = run_training_workflow()
results = payload['results']
best_model_name = payload['best_model_name']
fitted_models = payload['fitted_models']
test_predictions = payload['test_predictions']
y_test = payload['y_test']

print(f'Best model: {best_model_name}')
print()
results"""
    )

    md("## 2. Compare Model Performance")
    code(
        """comparison = results[['model_name', 'cv_rmse_mean', 'test_rmse', 'test_mae', 'test_r2']].copy()
comparison"""
    )
    code(
        """fig, ax = plt.subplots(figsize=(10, 6))
plot_df = results.sort_values('test_rmse')
sns.barplot(data=plot_df, x='model_name', y='test_rmse', palette='muted', ax=ax)
ax.set_title('Test RMSE by Model')
ax.set_xlabel('Model')
ax.set_ylabel('RMSE (TRY)')
plt.xticks(rotation=15)
plt.tight_layout()
save_fig('15_model_comparison')
plt.show()"""
    )

    md("## 3. Inspect The Best Model")
    code(
        """best_predictions = test_predictions[best_model_name].sort_index()
aligned_y_test = y_test.loc[best_predictions.index].sort_index()

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(aligned_y_test, best_predictions, alpha=0.35, s=18, color='#4C72B0')
line_min = min(aligned_y_test.min(), best_predictions.min())
line_max = max(aligned_y_test.max(), best_predictions.max())
ax.plot([line_min, line_max], [line_min, line_max], color='red', linestyle='--')
ax.set_title(f'Actual vs Predicted Price ({best_model_name})')
ax.set_xlabel('Actual Price (TRY)')
ax.set_ylabel('Predicted Price (TRY)')
plt.tight_layout()
save_fig('16_actual_vs_predicted')
plt.show()"""
    )
    code(
        """residuals = aligned_y_test - best_predictions

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(best_predictions, residuals, alpha=0.35, s=18, color='#55A868')
ax.axhline(0, color='red', linestyle='--')
ax.set_title(f'Residual Plot ({best_model_name})')
ax.set_xlabel('Predicted Price (TRY)')
ax.set_ylabel('Residual (Actual - Predicted)')
plt.tight_layout()
save_fig('17_residuals_best_model')
plt.show()"""
    )
    code(
        """importance_model_name = best_model_name if best_model_name in {'decision_tree', 'random_forest'} else 'random_forest'
importance_df = get_feature_importance_table(fitted_models[importance_model_name], top_n=15)
importance_df"""
    )
    code(
        """if not importance_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='crest', ax=ax)
    ax.set_title(f'Top Feature Importances ({importance_model_name})')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    save_fig('18_random_forest_feature_importance')
    plt.show()"""
    )
    code(
        """ridge_coefficients = get_linear_coefficient_table(fitted_models['ridge'], top_n=15)
ridge_coefficients"""
    )

    md("## 4. Summary")
    code(
        """best_row = results.loc[results['model_name'] == best_model_name].iloc[0]
print('Best model summary')
print(f\"Model: {best_model_name}\")
print(f\"Test MAE:  {best_row['test_mae']:.2f} TRY\")
print(f\"Test RMSE: {best_row['test_rmse']:.2f} TRY\")
print(f\"Test R²:   {best_row['test_r2']:.3f}\")

print('\\nInterpretation notes:')
print('- Tree-based models can capture non-linear price patterns more flexibly than the linear baseline.')
print('- Results remain predictive, not causal; important variables are associated with price, not proven causes of price.')"""
    )

    notebook = new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = cells
    return notebook


def write_notebook(notebook=None, output_path: Path = NOTEBOOK_PATH) -> Path:
    """Persist the generated notebook to disk."""
    if notebook is None:
        notebook = build_notebook()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return output_path


def main() -> None:
    """Create or refresh the notebook file."""
    notebook_path = write_notebook()
    print(f"Created notebook: {notebook_path}")


if __name__ == "__main__":
    main()
