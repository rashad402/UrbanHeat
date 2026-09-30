"""Counterfactual "what-if" intervention engine (plan §10).

Given ward pixels and an intervention, modify surface features and re-infer LST with the
trained PINN, then recompute SEB to confirm the predicted delta-T is energy-consistent.

Interventions:
  - green_roof / urban_greening: NDVI += delta (0.2..0.4), clamp <= ~0.85; adjust emissivity.
  - cool_roof: albedo := target (0.15 -> 0.50).
  - tree_canopy: NDVI increase (+ optional roughness/wind change).

Validation (no ground truth): cross-check delta-T against published tropical field ranges
(cool roofs 1.5-4.0 C, greening 0.5-3.0 C); flag out-of-range outputs.
"""

INTERVENTIONS = {
    "green_roof": {"ndvi_delta": (0.2, 0.4)},
    "cool_roof": {"albedo_set": 0.50},
    "urban_greening": {"ndvi_delta": (0.2, 0.4)},
    "tree_canopy": {"ndvi_delta": (0.2, 0.4)},
}


def apply_intervention(features, intervention):
    """Return modified feature array for the given intervention spec."""
    raise NotImplementedError


def predict_delta_t(model, features_base, intervention, scaler):
    """Return per-pixel/per-ward delta_t = T_new - T_base."""
    raise NotImplementedError


def plausibility_check(delta_t, intervention_name):
    """True if delta_t falls within published tropical cooling ranges for this intervention."""
    raise NotImplementedError
