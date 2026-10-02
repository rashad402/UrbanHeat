"""The decisive figure: what happens when models are asked to extrapolate (plan §14).

A cool roof sets albedo to ~0.50, which is ~15 standard deviations above the training mean.
In-distribution test metrics never probe this, so a model can look excellent (R2 0.84) and still
predict that a cool roof HEATS the city. This plots the albedo response curve for each physics
weight against the SEB physics reference.

Run:  python scripts/plot_extrapolation.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.inference import PinnModel          # noqa: E402
from models.features import INPUT_COLUMNS    # noqa: E402
from models import sebal as S                # noqa: E402

PARQUET = "data/processed/kochi_samples.parquet"
OUT = "docs/figures/extrapolation.png"

# These must be the lambdas run_pinn.py actually trained on the CURRENT dataset. Loading a
# checkpoint from an earlier training run against the current scaler.json produces a plausible
# looking but meaningless curve, which is why the pre-fix checkpoints were moved out of this
# directory (see models/checkpoints/pre_era5fix/README.md).
CKPTS = [("mlp_lambda0", "λ=0 (no physics)", "#c02a26"),
         ("pinn_lambda0.1", "λ=0.1", "#b45309"),
         ("pinn_lambda0.5", "λ=0.5 (deployed)", "#0f766e")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_parquet(PARQUET)
    w = df[df.ward_id.notna()].groupby("ward_id")[INPUT_COLUMNS].mean()
    base = {c: float(w[c].mean()) for c in INPUT_COLUMNS}
    a_mean, a_std = df.albedo.mean(), df.albedo.std()
    p99 = df.albedo.quantile(0.99)

    alb = np.linspace(0.08, 0.55, 40)
    feats = {c: np.full(len(alb), base[c]) for c in INPUT_COLUMNS}
    feats["albedo"] = alb

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # --- left: albedo response curves ---
    for ck, lab, col in CKPTS:
        t = PinnModel(f"models/checkpoints/{ck}.pt").predict_k(feats) - 273.15
        ax[0].plot(alb, t, "-", color=col, lw=2.2, label=lab)
    seb = S.equilibrium_temperature(feats["ndvi"], alb, feats["s_down"],
                                    feats["t_air"], feats["rh"], feats["wind"]) - 273.15
    ax[0].plot(alb, seb, "k--", lw=1.6, label="SEB physics (reference)")

    lo, hi = 29.0, 41.5                                  # explicit limits: annotations stay inside
    ax[0].set_ylim(lo, hi)
    ax[0].set_xlim(alb.min(), alb.max())
    ax[0].axvspan(alb.min(), p99, color="#2dd4bf", alpha=.10)
    ax[0].axvline(p99, color="#0f766e", ls=":", lw=1.4)
    ax[0].text(0.095, hi - 0.9, "training data\n(99% below here)", fontsize=8.5, color="#0f766e")
    ax[0].axvline(0.50, color="#c02a26", ls=":", lw=1.4)
    ax[0].text(0.432, hi - 0.9, "cool roof\n(+15σ)", fontsize=8.5, color="#c02a26")
    ax[0].set_xlabel("surface albedo")
    ax[0].set_ylabel("predicted LST (°C)")
    ax[0].set_title("Raising albedo must COOL the surface.\nWithout physics, the model says it heats it.")
    ax[0].legend(fontsize=9, loc="lower left")
    ax[0].grid(alpha=.25)

    # --- right: how often the direction is right, per ward ---
    wf = {c: w[c].to_numpy() for c in INPUT_COLUMNS}
    labs, pct = [], []
    for ck, lab, col in [("mlp_lambda0", "λ=0", None), ("pinn_lambda0.1", "λ=0.1", None),
                         ("pinn_lambda0.5", "λ=0.5", None), ("pinn_lambda2", "λ=2", None)]:
        m = PinnModel(f"models/checkpoints/{ck}.pt")
        t0 = m.predict_k(wf)
        mod = dict(wf); mod["albedo"] = np.maximum(wf["albedo"], 0.50)
        pct.append(100.0 * np.mean(m.predict_k(mod) - t0 < 0))
        labs.append(lab)
    cols = ["#c02a26" if p < 90 else "#0f766e" for p in pct]
    ax[1].bar(labs, pct, color=cols)
    ax[1].axhline(100, color="#0f766e", ls="--", lw=1.2)
    ax[1].axhline(50, color="#888", ls=":", lw=1)
    ax[1].text(4.3, 44, "coin flip", fontsize=8.5, color="#777")
    ax[1].set_ylabel("% of wards cooled by a cool roof")
    ax[1].set_xlabel("physics weight")
    ax[1].set_ylim(0, 108)
    ax[1].set_title("Only a strong physics term makes the\ncounterfactual reliable everywhere")
    for i, p in enumerate(pct):
        ax[1].text(i, p + 2, f"{p:.0f}%", ha="center", fontsize=9)
    ax[1].grid(alpha=.25, axis="y")

    fig.suptitle("Counterfactuals are an extrapolation problem — this is what the physics loss is for",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"Saved {OUT}  (albedo mean {a_mean:.3f} +/- {a_std:.3f}, p99 {p99:.3f})")


if __name__ == "__main__":
    main()
