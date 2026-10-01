"""Evaluation: accuracy, physics consistency, baseline comparison (plan §14).

Metrics:
  * LST accuracy: R^2, RMSE, MAE on the held-out split (Kelvin == degC for RMSE/MAE).
  * Physics consistency: SEB residual magnitude (filled in once models/sebal.py is settled).
  * Baseline comparison: PINN vs RF/XGB/MLP -> the report's comparison table.
"""

import numpy as np


def regression_metrics(y_true, y_pred):
    """Return {'r2', 'rmse', 'mae'}. RMSE/MAE are in Kelvin (== degC differences)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }


def physics_consistency(seb_residual_values):
    """Summarize SEB residual magnitude [W/m^2] (used once the PINN/SEBAL module is ready)."""
    v = np.abs(np.asarray(seb_residual_values, dtype="float64"))
    return {"seb_residual_mean": float(v.mean()), "seb_residual_max": float(v.max())}


def comparison_table(results_by_model, title="LST prediction (held-out spatial blocks)"):
    """Render the model-comparison table as plain text for the report/logs."""
    rows = [f"{title}", f"{'model':<18}{'R2':>9}{'RMSE (K)':>11}{'MAE (K)':>10}"]
    rows.append("-" * 48)
    for name, m in results_by_model.items():
        rows.append(f"{name:<18}{m['r2']:>9.3f}{m['rmse']:>11.3f}{m['mae']:>10.3f}")
    return "\n".join(rows)
