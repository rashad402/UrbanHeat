"""Trained-PINN inference for the API (plan §11).

Loads the physics-informed checkpoint and serves predictions and counterfactuals. Shared by
api/main.py (ward-level) and api/live_demo.py (point/area), so both run on the same model.

CHECKPOINT CHOICE — lambda=0.5, NOT the best in-distribution model
    In-distribution metrics favour lambda<=0.1 (R2 0.77-0.80 against 0.52). Those metrics are
    misleading here, because a counterfactual is an EXTRAPOLATION: a cool roof sets albedo to
    0.50, which is 15.6 standard deviations above the training mean (0.133 +/- 0.024).
    Measured on the 74 ward-mean feature vectors of the rebuilt table (2026-10-02):

        lambda  cool-roof dT  worst ward  wards cooling  albedo steps in the WRONG direction
        0       -2.07 K        +5.20 K     74%           25%
        0.1     -2.59 K        +4.05 K     77%           19%
        0.5     -2.27 K        +0.14 K     99%           12%
        2       -6.11 K        -2.38 K    100%            7%

    "Worst ward" is the least-cooling ward: a positive value means the model predicts that
    painting that ward's roofs white HEATS it — exactly the "physically impossible output under
    intervention" this project exists to prevent. Without a strong physics weight a quarter of
    the city gets that answer.

    lambda=2 is the only checkpoint with no heating ward at all, and it was considered. It was
    not chosen because it flattens the vegetation response to -0.47 K against lambda=0.5's
    -0.62 K and lambda=0.1's -1.37 K. The platform's purpose is COMPARING interventions, and a
    model that systematically understates greening against cool roofs would bias that comparison
    in a policy-relevant direction. lambda=0.5 keeps every ward cooling (its one outlier is
    +0.14 K, neutral within noise) while leaving the strategies comparable.

    NOTE: these numbers replace an earlier table measured on the pre-ERA5-fix dataset, which
    covered only 50 wards. The conclusion survived the rebuild; the magnitudes did not.

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

# Fraction of built-up area that is roof. This multiplies every reported delta_t, so it is an
# ASSUMPTION WITH TEETH, not a detail. scripts/build_roof_share.py measures it per ward from
# Open Buildings footprints and api/planner.py uses those values; this constant is the fallback
# for callers that have no ward context (api/main.py, api/live_demo.py) and when that file has
# not been generated. Override with the ROOF_SHARE environment variable.
ROOF_SHARE = float(os.environ.get("ROOF_SHARE", 0.5))
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


def whatif(model, feats, albedo_set=None, ndvi_delta=None, roof_share=None):
    """Run base and counterfactual inference.

    Returns a dict of arrays: t_base, t_new, delta_t (full-surface),
    delta_t_applied (coverage-scaled), built_frac, seb_residual.

    `roof_share` may be a scalar or a per-pixel array, letting api/planner.py pass the MEASURED
    per-ward value instead of the module default.
    """
    t0 = model.predict_k(feats)
    mod = apply_intervention(feats, albedo_set, ndvi_delta)
    t1 = model.predict_k(mod)
    d = t1 - t0

    share = ROOF_SHARE if roof_share is None else np.asarray(roof_share, dtype="float64")
    bf = built_fraction(feats["ndvi"])
    coverage = bf * share if albedo_set is not None else bf
    return {
        "t_base": t0,
        "t_new": t1,
        "delta_t": d,
        "delta_t_applied": d * coverage,
        "built_frac": bf,
        "coverage": coverage,
        "seb_residual": seb_residual_of(t0, feats),
    }
