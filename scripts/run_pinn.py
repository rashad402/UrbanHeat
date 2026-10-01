"""Train the PINN across physics weights and quantify what the SEB loss buys (plan §9, §14).

Usage:
    python scripts/run_pinn.py                 # lambda sweep + comparison
    python scripts/run_pinn.py --lambdas 0 0.5 # custom sweep

For each lambda it reports three things on the held-out spatial blocks:
  1. ACCURACY        R^2 / RMSE / MAE                      (expected: flat — already saturated)
  2. PHYSICS         mean |SEB residual| of the prediction (expected: falls sharply)
  3. COUNTERFACTUAL  dT for a cool roof and for greening   (expected: the sign gets FIXED)

(3) is the decisive one. In the raw data albedo correlates POSITIVELY with LST (land-cover
confounding), so a purely data-driven model learns "brighter = hotter" and predicts that cool
roofs WARM the city — physically backwards. The SEB term should correct that sign.

lambda = 0 reproduces the plain MLP baseline exactly, so the sweep is a clean ablation.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import features as F                      # noqa: E402
from models import sebal as S                         # noqa: E402
from models.train import train_pinn, predict          # noqa: E402
from models.evaluate import regression_metrics        # noqa: E402

PARQUET = "data/processed/kochi_samples.parquet"
CKPT, FIGS = "models/checkpoints", "docs/figures"
COOL_ALBEDO, GREEN_NDVI = 0.50, 0.30


def counterfactual(predict_fn, df, scaler, kind):
    """Return per-pixel dT [K] for an intervention, using each model's own predictions."""
    base = predict_fn(F.transform(df, scaler)[0])
    mod = df.copy()
    if kind == "cool_roof":
        mod["albedo"] = np.maximum(mod["albedo"].to_numpy(), COOL_ALBEDO)
    else:
        mod["ndvi"] = np.clip(mod["ndvi"].to_numpy() + GREEN_NDVI, -1, 0.95)
    return predict_fn(F.transform(mod, scaler)[0]) - base


def seb_abs_residual(t_pred_k, df):
    r = S.seb_residual(t_pred_k, df.ndvi.values, df.albedo.values, df.s_down.values,
                       df.t_air.values, df.rh.values, df.wind.values)
    return float(np.mean(np.abs(r)))


def evaluate(name, predict_fn, te, yte, scaler, out):
    pred = predict_fn(F.transform(te, scaler)[0])
    m = regression_metrics(yte, pred)
    m["abs_residual"] = seb_abs_residual(pred, te)
    for kind in ("cool_roof", "greening"):
        d = counterfactual(predict_fn, te, scaler, kind)
        m[f"dT_{kind}"] = float(np.mean(d))
        m[f"cools_{kind}"] = float(np.mean(d < 0) * 100.0)
    out[name] = m
    print(f"  {name:<16} R2 {m['r2']:.3f}  RMSE {m['rmse']:.2f}K  "
          f"|res| {m['abs_residual']:6.1f} W/m2  "
          f"coolroof {m['dT_cool_roof']:+.2f}K ({m['cools_cool_roof']:.0f}% cool)  "
          f"green {m['dT_greening']:+.2f}K ({m['cools_greening']:.0f}% cool)")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 2.0])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    df = F.load_samples(PARQUET)
    tr, va, te = F.make_splits(df, policy="spatial_block", seed=args.seed)
    scaler = F.fit_scaler(tr)
    F.save_scaler(scaler, f"{CKPT}/scaler.json")
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    Xtr, _ = F.transform(tr, scaler)
    Xva, _ = F.transform(va, scaler)
    _, ytr_s = F.transform(tr, scaler, scale_y=True)
    _, yva_s = F.transform(va, scaler, scale_y=True)
    yte = te[F.TARGET_COLUMN].to_numpy(dtype="float64")

    results = {}

    # Reference: the strongest tabular baseline, with no physics at all.
    print("\nXGBoost reference:")
    from models.baselines import train_xgboost
    ytr = tr[F.TARGET_COLUMN].to_numpy(dtype="float64")
    yva = va[F.TARGET_COLUMN].to_numpy(dtype="float64")
    xgb = train_xgboost(Xtr, ytr, Xva, yva, seed=args.seed)
    evaluate("XGBoost", lambda X: xgb.predict(X), te, yte, scaler, results)

    # Physics-weight sweep. lambda = 0 is the plain MLP (same architecture and seed).
    print("\nPINN sweep (lambda = 0 is the no-physics MLP):")
    for lam in args.lambdas:
        print(f"\n  training lambda = {lam} ...")
        model, hist = train_pinn(Xtr, ytr_s, tr, Xva, yva_s, va, scaler,
                                 lambda_max=lam, epochs=args.epochs, seed=args.seed,
                                 verbose=True)
        name = "MLP (no physics)" if lam == 0 else f"PINN (lam={lam:g})"
        evaluate(name, lambda X: predict(model, X, scaler), te, yte, scaler, results)
        torch.save(model.state_dict(),
                   f"{CKPT}/{'mlp_lambda0' if lam == 0 else f'pinn_lambda{lam:g}'}.pt")
        results[name]["lambda"] = lam

    with open(f"{FIGS}/pinn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    _table(results)
    _plot(results)


def _table(r):
    print("\n" + "=" * 104)
    print(f"{'model':<18}{'R2':>7}{'RMSE(K)':>9}{'MAE(K)':>8}{'|SEB res|':>11}"
          f"{'coolroof dT':>13}{'% cool':>8}{'greening dT':>13}{'% cool':>8}")
    print("-" * 104)
    for k, m in r.items():
        print(f"{k:<18}{m['r2']:>7.3f}{m['rmse']:>9.2f}{m['mae']:>8.2f}{m['abs_residual']:>11.1f}"
              f"{m['dT_cool_roof']:>13.2f}{m['cools_cool_roof']:>8.0f}"
              f"{m['dT_greening']:>13.2f}{m['cools_greening']:>8.0f}")
    print("=" * 104)


def _plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pinns = {k: v for k, v in r.items() if "lambda" in v}
    lams = [v["lambda"] for v in pinns.values()]
    order = np.argsort(lams)
    lams = np.array(lams)[order]
    rmse = np.array([v["rmse"] for v in pinns.values()])[order]
    res = np.array([v["abs_residual"] for v in pinns.values()])[order]
    dcool = np.array([v["dT_cool_roof"] for v in pinns.values()])[order]
    x = np.arange(len(lams))

    pcool = np.array([v["cools_cool_roof"] for v in pinns.values()])[order]
    pgreen = np.array([v["cools_greening"] for v in pinns.values()])[order]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].plot(x, rmse, "o-", color="#c02a26")
    ax[0].set_ylabel("Test RMSE (K)")
    ax[0].set_title("Cost: accuracy degrades\n(cheap to λ≈0.1, then steep)")
    ax[1].plot(x, res, "o-", color="#0f766e")
    ax[1].set_ylabel("mean |SEB residual| (W/m²)")
    ax[1].set_title("Gain: physics consistency\n(falls monotonically)")
    ax[2].plot(x, pcool, "o-", color="#1f7a5a", label="cool roof")
    ax[2].plot(x, pgreen, "s--", color="#b45309", label="greening")
    ax[2].set_ylabel("% of pixels predicted to cool")
    ax[2].set_title("Gain: counterfactuals become\nconsistent (sign was already right)")
    ax[2].legend(loc="lower right", fontsize=9)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([f"λ={l:g}" for l in lams]); a.set_xlabel("physics weight")
        a.grid(alpha=.25)
        a.axvline(float(np.argmin(np.abs(lams - 0.1))), color="#888", ls=":", lw=1.2)
    fig.suptitle("The SEB physics loss trades accuracy for physical consistency "
                 "(dotted line = recommended λ=0.1)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/pinn_ablation.png", dpi=110, bbox_inches="tight")
    print(f"Saved {FIGS}/pinn_ablation.png")


if __name__ == "__main__":
    main()
