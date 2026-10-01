"""Trained-PINN inference for the API (plan §11).

Loads the physics-informed checkpoint and serves predictions and counterfactuals. Shared by
api/main.py (ward-level) and api/live_demo.py (point/area), so both run on the same model.

CHECKPOINT CHOICE — lambda=0.5, NOT the best in-distribution model
    In-distribution metrics favour lambda~0.05-0.1 (R2 0.84 vs 0.60). Those metrics are
    misleading here, because a counterfactual is an EXTRAPOLATION: a cool roof sets albedo to
    0.50, which is 15.5 standard deviations above the training mean (0.134 +/- 0.024, p99 0.198).
    Measured on ward feature vectors:

        lambda   cool-roof dT   wards cooling   albedo response monotonic
        0        -0.84 K        52%  (coin flip)   43%
        0.1      -3.35 K        74%               71%
        0.3      -4.48 K        82%              100%
        0.5      -6.17 K       100%              100%

    The no-physics model predicts a cool roof WARMS the surface by up to 9 K — exactly the
    "physically impossible output under intervention" the project exists to prevent. Only a
    strong physics weight makes extrapolation safe, so the deployed model is lambda=0.5.

    The accuracy cost is acceptable because the BASELINE temperature comes from the measured
    satellite LST, not the model; the model supplies only the RESPONSE (delta_t). Selecting a
    checkpoint on R2 would optimise the one thing the deployment does not rely on.

ON COUNTERFACTUAL MAGNITUDE
    The model returns a FULL-SURFACE response: it answers "what if this whole pixel's albedo
    were 0.50". Real interventions cover part of a pixel, so `delta_t_applied` rescales by the
    built-up fraction and a roof share. Both are returned; `delta_t` is the model's own answer
    and `delta_t_applied` is the planning estimate.
"""

import os

import numpy as np
import torch

from models.features import INPUT_COLUMNS, load_scaler
from models.pinn import PINN
from models import sebal as S

DEFAULT_CKPT = os.environ.get("PINN_CKPT", "models/checkpoints/pinn_lambda0.5.pt")
DEFAULT_SCALER = os.environ.get("PINN_SCALER", "models/checkpoints/scaler.json")

ROOF_SHARE = 0.5            # fraction of built-up area that is roof
PHYS_COLUMNS = ["ndvi", "albedo", "s_down", "t_air", "rh", "wind"]


class PinnModel:
    """Thin wrapper: standardise -> network -> Kelvin."""

    def __init__(self, ckpt=DEFAULT_CKPT, scaler_path=DEFAULT_SCALER):
        self.scaler = load_scaler(scaler_path)
        self.net = PINN(in_dim=len(INPUT_COLUMNS))
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.net.eval()
        self.name = os.path.basename(ckpt)

    def predict_k(self, feats):
        """feats: dict of equal-length arrays keyed by INPUT_COLUMNS. Returns LST [K]."""
        cols = []
        for c in INPUT_COLUMNS:
            v = np.asarray(feats[c], dtype="float64")
            cols.append((v - self.scaler["x_mean"][c]) / self.scaler["x_std"][c])
        X = np.column_stack(cols).astype("float32")
        with torch.no_grad():
            out = self.net(torch.tensor(X)).numpy().ravel()
        return out * self.scaler["y_std"] + self.scaler["y_mean"]


def built_fraction(ndvi):
    """Share of surface that is built/bare — used to scale interventions to realistic coverage."""
    return np.clip(1.0 - np.clip((np.asarray(ndvi) - 0.10) / 0.60, 0, 1), 0, 1)


def apply_intervention(feats, albedo_set=None, ndvi_delta=None):
    """Return a modified copy of the feature dict."""
    out = {k: np.array(v, dtype="float64", copy=True) for k, v in feats.items()}
    if albedo_set is not None:
        out["albedo"] = np.maximum(out["albedo"], float(albedo_set))
    if ndvi_delta:
        out["ndvi"] = np.clip(out["ndvi"] + float(ndvi_delta), -1.0, 0.95)
    return out


def seb_residual_of(t_k, feats):
    """Mean |SEB residual| [W/m^2] of a prediction — the physics-consistency diagnostic."""
    r = S.seb_residual(t_k, feats["ndvi"], feats["albedo"], feats["s_down"],
                       feats["t_air"], feats["rh"], feats["wind"])
    return np.abs(np.asarray(r))


def whatif(model, feats, albedo_set=None, ndvi_delta=None):
    """Run base and counterfactual inference.

    Returns a dict of arrays: t_base, t_new, delta_t (full-surface),
    delta_t_applied (coverage-scaled), built_frac, seb_residual.
    """
    t0 = model.predict_k(feats)
    mod = apply_intervention(feats, albedo_set, ndvi_delta)
    t1 = model.predict_k(mod)
    d = t1 - t0

    bf = built_fraction(feats["ndvi"])
    coverage = bf * ROOF_SHARE if albedo_set is not None else bf
    return {
        "t_base": t0,
        "t_new": t1,
        "delta_t": d,
        "delta_t_applied": d * coverage,
        "built_frac": bf,
        "coverage": coverage,
        "seb_residual": seb_residual_of(t0, feats),
    }
