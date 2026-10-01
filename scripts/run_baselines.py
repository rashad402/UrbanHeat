"""Train and evaluate the baseline LST models — the data sanity gate (plan §7).

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --split random      # leakage demo for the report

Trains Random Forest, XGBoost and the PINN-shaped MLP on the same features, evaluates on a
held-out set of spatial blocks, prints the comparison table and saves:

    models/checkpoints/scaler.json
    models/checkpoints/mlp_baseline.pt
    docs/figures/baselines.png
    docs/figures/baseline_metrics.json

GATE: if the best model can't reach R^2 ~0.7, fix the data/features before building the PINN.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import features as F              # noqa: E402
from models.baselines import (                # noqa: E402
    train_random_forest, train_xgboost, train_mlp, mlp_predict)
from models.evaluate import regression_metrics, comparison_table  # noqa: E402

PARQUET = "data/processed/kochi_samples.parquet"
CKPT = "models/checkpoints"
FIGS = "docs/figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="spatial_block", choices=["spatial_block", "random"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    df = F.load_samples(PARQUET)
    print(f"Loaded {len(df):,} pixels, {df.date.nunique()} dates")

    tr, va, te = F.make_splits(df, policy=args.split, seed=args.seed)
    print(f"Split '{args.split}': train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    scaler = F.fit_scaler(tr)
    F.save_scaler(scaler, f"{CKPT}/scaler.json")

    Xtr, ytr = F.transform(tr, scaler)          # y in Kelvin for tree models
    Xva, yva = F.transform(va, scaler)
    Xte, yte = F.transform(te, scaler)
    # standardized target for the NN
    _, ytr_s = F.transform(tr, scaler, scale_y=True)
    _, yva_s = F.transform(va, scaler, scale_y=True)

    results = {}

    print("\n[1/3] Random Forest...")
    rf = train_random_forest(Xtr, ytr, seed=args.seed)
    results["RandomForest"] = regression_metrics(yte, rf.predict(Xte))

    print("[2/3] XGBoost...")
    xgb = train_xgboost(Xtr, ytr, Xva, yva, seed=args.seed)
    results["XGBoost"] = regression_metrics(yte, xgb.predict(Xte))

    print("[3/3] MLP (PINN architecture, data loss only)...")
    mlp = train_mlp(Xtr, ytr_s, Xva, yva_s, seed=args.seed)
    pred_mlp = F.inverse_y(mlp_predict(mlp, Xte), scaler)
    results["MLP (no physics)"] = regression_metrics(yte, pred_mlp)
    torch.save(mlp.state_dict(), f"{CKPT}/mlp_baseline.pt")

    # Trivial reference: always predict the training mean.
    results["(mean baseline)"] = regression_metrics(yte, np.full_like(yte, ytr.mean()))

    print("\n" + comparison_table(results,
          f"LST prediction — held-out {'spatial blocks' if args.split=='spatial_block' else 'random pixels'}"))

    best = max((m["r2"] for k, m in results.items() if not k.startswith("(")))
    print(f"\nGATE: best R2 = {best:.3f} -> " +
          ("PASS, features predict LST; proceed to the PINN." if best >= 0.7
           else "BELOW 0.7 — inspect features/data before building the PINN."))

    with open(f"{FIGS}/baseline_metrics_{args.split}.json", "w") as f:
        json.dump({"split": args.split, "n_train": len(tr), "n_test": len(te),
                   "results": results}, f, indent=2)

    _plot(yte, {"RandomForest": rf.predict(Xte), "XGBoost": xgb.predict(Xte),
                "MLP (no physics)": pred_mlp}, results, args.split)


def _plot(y_true, preds, results, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    lo, hi = float(np.min(y_true)) - 1, float(np.max(y_true)) + 1
    for ax, (name, p) in zip(axes, preds.items()):
        ax.hexbin(y_true, p, gridsize=45, cmap="inferno", mincnt=1)
        ax.plot([lo, hi], [lo, hi], "c--", lw=1.4)
        m = results[name]
        ax.set_title(f"{name}\nR²={m['r2']:.3f}  RMSE={m['rmse']:.2f} K")
        ax.set_xlabel("Observed LST (K)")
        ax.set_ylabel("Predicted LST (K)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    fig.suptitle(f"Baseline LST models — held-out {split} split", fontsize=13)
    fig.tight_layout()
    out = f"{FIGS}/baselines_{split}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
