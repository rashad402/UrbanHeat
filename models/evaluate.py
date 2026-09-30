"""Evaluation: accuracy, physics consistency, baseline comparison (plan §14).

Metrics:
  - LST accuracy: R^2, RMSE, MAE on the held-out split.
  - Physics consistency: mean/max SEB residual magnitude (esp. under interventions).
  - Baseline comparison: PINN vs RF/XGB/CNN on the above -> the paper's comparison table.
Produces figures into docs/figures/ (pred-vs-obs scatter, loss curves, residual maps).
"""


def regression_metrics(y_true, y_pred):
    """Return {'r2':..., 'rmse':..., 'mae':...}."""
    raise NotImplementedError


def physics_consistency(model, features):
    """Return {'seb_residual_mean':..., 'seb_residual_max':...} over the given samples."""
    raise NotImplementedError


def comparison_table(results_by_model):
    """Assemble the PINN-vs-baselines table for the report."""
    raise NotImplementedError
