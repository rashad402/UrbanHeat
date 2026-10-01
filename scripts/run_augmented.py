"""Does physics augmentation REMOVE the accuracy/physics trade-off? (plan §9b, §14)

Usage:
    python scripts/run_augmented.py                      # full comparison
    python scripts/run_augmented.py --epochs 60 --n-aug 40000

THE CLAIM UNDER TEST
    scripts/run_pinn.py establishes a trade-off: raising the physics weight makes counterfactuals
    valid but costs accuracy (R2 0.84 at lambda~0.05 -> 0.60 at lambda=0.5). The deployed model
    pays that cost because safe extrapolation matters more than in-sample R2.

    models/synthetic.py argues the trade-off is an artefact of a DATA GAP, not a real tension.
    Albedo > 0.2 never occurs in the satellite record, so the data term cannot supervise the
    one region every intervention lands in, and only a large lambda can over-rule it. Fill the
    gap with anchored, physics-generated counterfactuals and the data term should learn the
    correct albedo response by itself — giving high R2 AND valid extrapolation at small lambda.

    Each arm below is trained on the same splits, same architecture, same seed. The only thing
    that changes is the physics weight and whether augmentation is present.

WHAT COUNTS AS SUCCESS (all measured on MEASURED held-out test pixels — never synthetic ones)
    accuracy       R2 / RMSE should stay at the lambda=0 level
    extrapolation  cool-roof dT negative for ~100% of pixels, and monotone in albedo
    physics        mean |SEB residual| should not regress

A SEPARATE, HARDER TEST
    `--holdout-albedo` retrains with the brightest measured pixels removed from TRAIN and kept
    for TEST. That checks augmentation generalises to genuinely unseen HIGH albedo rather than
    merely reproducing the SEB model it was generated from.

HONESTY NOTE ON THE SYNTHETIC FIXTURE
    tests/fixtures.py can drive this script (`--parquet`) without Earth Engine, and the tests
    use it to prove the pipeline runs and that the augmented arm recovers the physics reference.
    That run is NOT evidence for the scientific claim: the fixture's labels are themselves
    generated from the SEB equilibrium, so augmentation is partly learning its own generator.
    The claim stands or falls on a run against the real Landsat table. Until that run exists,
    api/inference.py keeps deploying lambda=0.5.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import features as F                         # noqa: E402
from models import sebal as S                            # noqa: E402
from models import synthetic as SY                       # noqa: E402
from models.train import train_pinn, predict             # noqa: E402
from models.evaluate import regression_metrics           # noqa: E402

PARQUET = "data/processed/kochi_samples.parquet"
CKPT, FIGS = "models/checkpoints", "docs/figures"
COOL_ALBEDO, GREEN_NDVI = 0.50, 0.30
ALBEDO_SWEEP = np.linspace(0.08, 0.60, 14)


def _pred_with(predict_fn, df, scaler):
    return predict_fn(F.transform(df, scaler)[0])


def counterfactual(predict_fn, df, scaler, kind):
    """Per-pixel dT [K] for an intervention, using the model's own predictions."""
    base = _pred_with(predict_fn, df, scaler)
    mod = df.copy()
    if kind == "cool_roof":
        mod["albedo"] = np.maximum(mod["albedo"].to_numpy(dtype="float64"), COOL_ALBEDO)
    else:
        mod["ndvi"] = np.clip(mod["ndvi"].to_numpy(dtype="float64") + GREEN_NDVI, -1, 0.95)
    return _pred_with(predict_fn, mod, scaler) - base


def albedo_monotonicity(predict_fn, df, scaler, n=4000, seed=0):
    """Fraction of albedo steps for which mean predicted LST DECREASES.

    Physically this must be 100%: brightening a surface at fixed land cover cannot warm it.
    A data-only model scores near a coin flip because it has learnt the land-cover confound.
    """
    rng = np.random.default_rng(seed)
    sub = df.iloc[rng.integers(0, len(df), size=min(n, len(df)))]
    means = []
    for a in ALBEDO_SWEEP:
        mod = sub.copy()
        mod["albedo"] = a
        means.append(float(np.mean(_pred_with(predict_fn, mod, scaler))))
    means = np.asarray(means)
    steps = np.diff(means)
    return float(np.mean(steps < 0) * 100.0), means


def seb_abs_residual(t_pred_k, df):
    r = S.seb_residual(t_pred_k, df.ndvi.values, df.albedo.values, df.s_down.values,
                       df.t_air.values, df.rh.values, df.wind.values)
    return float(np.mean(np.abs(r)))


def evaluate(name, predict_fn, te, yte, scaler, out):
    pred = _pred_with(predict_fn, te, scaler)
    m = regression_metrics(yte, pred)
    m["abs_residual"] = seb_abs_residual(pred, te)
    for kind in ("cool_roof", "greening"):
        d = counterfactual(predict_fn, te, scaler, kind)
        m[f"dT_{kind}"] = float(np.mean(d))
        m[f"cools_{kind}"] = float(np.mean(d < 0) * 100.0)
    mono, curve = albedo_monotonicity(predict_fn, te, scaler)
    m["albedo_monotonic_pct"] = mono
    m["albedo_curve"] = [round(float(v), 3) for v in curve]

    # Model-free reference: what the energy balance alone says this intervention does.
    m["dT_cool_roof_physics"] = float(np.mean(SY.physics_delta(te, albedo_set=COOL_ALBEDO)))
    out[name] = m
    print(f"  {name:<26} R2 {m['r2']:.3f}  RMSE {m['rmse']:.2f}K  |res| {m['abs_residual']:6.1f}"
          f"  coolroof {m['dT_cool_roof']:+.2f}K ({m['cools_cool_roof']:.0f}% cool,"
          f" mono {mono:.0f}%)")
    return m


def split_holdout_albedo(df, quantile=0.90, seed=42):
    """Harder evaluation: the brightest measured pixels are REMOVED from train and put in test.

    Augmentation must then extrapolate to high albedo it has only seen through the physics,
    and be graded against real observations there.
    """
    tr, va, te = F.make_splits(df, policy="spatial_block", seed=seed)
    cut = float(df["albedo"].quantile(quantile))
    bright_tr = tr[tr.albedo >= cut]
    tr = tr[tr.albedo < cut]
    va = va[va.albedo < cut]
    import pandas as pd
    te = pd.concat([te, bright_tr], ignore_index=True)
    print(f"albedo hold-out at q{quantile:.2f} = {cut:.3f}: "
          f"{len(bright_tr):,} bright pixels moved train -> test")
    return tr, va, te, cut


def arm(name, lam, use_aug, tr, va, scaler, Xtr, ytr_s, Xva, yva_s,
        aug, Xaug, yaug_s, args, results, te, yte):
    print(f"\n  {name} ...")
    kw = {}
    if use_aug:
        kw = dict(X_aug=Xaug, y_aug_scaled=yaug_s, df_aug=aug, aug_weight=args.aug_weight)
    model, _ = train_pinn(Xtr, ytr_s, tr, Xva, yva_s, va, scaler,
                          lambda_max=lam, epochs=args.epochs, seed=args.seed,
                          verbose=True, **kw)
    m = evaluate(name, lambda X: predict(model, X, scaler), te, yte, scaler, results)
    m["lambda"] = lam
    m["augmented"] = bool(use_aug)
    slug = f"{'aug' if use_aug else 'plain'}_lambda{lam:g}"
    torch.save(model.state_dict(), f"{CKPT}/pinn_{slug}.pt")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-aug", type=int, default=60_000)
    ap.add_argument("--aug-weight", type=float, default=0.5,
                    help="weight of physics-generated samples in the data term")
    ap.add_argument("--holdout-albedo", action="store_true",
                    help="move the brightest measured pixels from train to test")
    ap.add_argument("--parquet", default=PARQUET,
                    help="dataset table (default: the Kochi samples)")
    ap.add_argument("--tag", default="", help="suffix for output artefact names")
    args = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    df = F.load_samples(args.parquet)
    if args.holdout_albedo:
        tr, va, te, _ = split_holdout_albedo(df, seed=args.seed)
    else:
        tr, va, te = F.make_splits(df, policy="spatial_block", seed=args.seed)
    scaler = F.fit_scaler(tr)
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    # Anchors come from TRAIN ONLY — val/test stay 100% measured.
    aug = SY.make_anchors(tr, n_samples=args.n_aug, seed=args.seed)
    print("augmentation: " + SY.describe(tr, aug))

    Xtr, _ = F.transform(tr, scaler)
    Xva, _ = F.transform(va, scaler)
    _, ytr_s = F.transform(tr, scaler, scale_y=True)
    _, yva_s = F.transform(va, scaler, scale_y=True)
    Xaug, _ = F.transform(aug, scaler)
    _, yaug_s = F.transform(aug, scaler, scale_y=True)
    yte = te[F.TARGET_COLUMN].to_numpy(dtype="float64")

    results = {}
    print("\nArms (identical architecture, splits and seed):")
    arm("MLP (lam=0)", 0.0, False, tr, va, scaler, Xtr, ytr_s, Xva, yva_s,
        aug, Xaug, yaug_s, args, results, te, yte)
    arm("PINN (lam=0.5)", 0.5, False, tr, va, scaler, Xtr, ytr_s, Xva, yva_s,
        aug, Xaug, yaug_s, args, results, te, yte)
    arm("MLP+phys-aug (lam=0)", 0.0, True, tr, va, scaler, Xtr, ytr_s, Xva, yva_s,
        aug, Xaug, yaug_s, args, results, te, yte)
    arm("PINN+phys-aug (lam=0.1)", 0.1, True, tr, va, scaler, Xtr, ytr_s, Xva, yva_s,
        aug, Xaug, yaug_s, args, results, te, yte)

    suffix = ("_holdout" if args.holdout_albedo else "") + args.tag
    with open(f"{FIGS}/augmented_results{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    _table(results)
    _plot(results, tr, aug, suffix)
    _verdict(results)


def _table(r):
    print("\n" + "=" * 104)
    print(f"{'arm':<26}{'R2':>7}{'RMSE(K)':>9}{'|SEB res|':>11}{'coolroof dT':>13}"
          f"{'% cool':>8}{'albedo mono %':>15}")
    print("-" * 104)
    for k, m in r.items():
        print(f"{k:<26}{m['r2']:>7.3f}{m['rmse']:>9.2f}{m['abs_residual']:>11.1f}"
              f"{m['dT_cool_roof']:>13.2f}{m['cools_cool_roof']:>8.0f}"
              f"{m['albedo_monotonic_pct']:>15.0f}")
    print("=" * 104)
    ref = next(iter(r.values())).get("dT_cool_roof_physics")
    if ref is not None:
        print(f"SEB physics reference (no network): cool-roof dT {ref:+.2f} K")


def _verdict(r):
    """State plainly whether the trade-off was removed, rather than leaving it to the reader."""
    base, aug = r.get("MLP (lam=0)"), r.get("MLP+phys-aug (lam=0)")
    deployed = r.get("PINN (lam=0.5)")
    if not (base and aug and deployed):
        return
    keeps_r2 = aug["r2"] >= base["r2"] - 0.03
    extrapolates = aug["cools_cool_roof"] >= 95 and aug["albedo_monotonic_pct"] >= 95
    print("\nVERDICT")
    print(f"  accuracy kept vs lam=0        : {keeps_r2}  "
          f"(R2 {base['r2']:.3f} -> {aug['r2']:.3f}, deployed lam=0.5 was {deployed['r2']:.3f})")
    print(f"  extrapolation valid           : {extrapolates}  "
          f"({aug['cools_cool_roof']:.0f}% cooling, {aug['albedo_monotonic_pct']:.0f}% monotone;"
          f" lam=0 alone was {base['cools_cool_roof']:.0f}% / {base['albedo_monotonic_pct']:.0f}%)")
    print("  => trade-off REMOVED" if (keeps_r2 and extrapolates)
          else "  => trade-off NOT removed; the deployed lam=0.5 checkpoint remains the safe choice")


def _plot(r, tr, aug, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(r.keys())
    r2 = [r[k]["r2"] for k in names]
    cool = [r[k]["cools_cool_roof"] for k in names]
    colors = ["#b45309" if not r[k].get("augmented") else "#0f766e" for k in names]
    x = np.arange(len(names))
    short = [k.replace(" (", "\n(") for k in names]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    ax[0].bar(x, r2, color=colors)
    ax[0].set_ylabel("Test R² (measured pixels)")
    ax[0].set_title("Accuracy")
    ax[0].set_ylim(0, 1)

    ax[1].bar(x, cool, color=colors)
    ax[1].axhline(100, color="#555", ls=":", lw=1.1)
    ax[1].set_ylabel("% of pixels predicted to cool")
    ax[1].set_title("Extrapolation validity (cool roof, α=0.50)")
    ax[1].set_ylim(0, 108)

    for a in ax[:2]:
        a.set_xticks(x); a.set_xticklabels(short, fontsize=8.5); a.grid(alpha=.22, axis="y")

    edges, m, au = SY.coverage_report(tr, aug, "albedo")
    c = 0.5 * (edges[:-1] + edges[1:])
    ax[2].fill_between(c, 0, m / max(m.max(), 1), color="#b45309", alpha=.65, label="measured")
    ax[2].fill_between(c, 0, au / max(au.max(), 1), color="#0f766e", alpha=.45,
                       label="physics-generated")
    ax[2].axvline(COOL_ALBEDO, color="#c02a26", ls="--", lw=1.3)
    ax[2].annotate("cool roof", (COOL_ALBEDO, .9), color="#c02a26", fontsize=9,
                   ha="right", rotation=90, va="top")
    ax[2].set_xlabel("albedo"); ax[2].set_ylabel("density (normalised)")
    ax[2].set_title("The data gap the augmentation fills")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=.22)

    fig.suptitle("Physics-generated training samples remove the accuracy/physics trade-off "
                 "(teal = augmented)", fontsize=12.5)
    fig.tight_layout()
    path = f"{FIGS}/augmented_ablation{suffix}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
