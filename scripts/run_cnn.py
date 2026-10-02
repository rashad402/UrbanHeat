"""CNN baseline vs the point-wise models (plan §7, §14).

Usage:
    python scripts/export_chips.py --patch 9          # preferred: dense chips from GEE
    python scripts/run_cnn.py --chips data/processed/kochi_chips_p9.npz

    python scripts/run_cnn.py                         # fallback: rebuild chips from the table

Closes the last outstanding baseline from the proposal. The CNN sees the NEIGHBOURHOOD of each
pixel (a patch x patch window of NDVI / NDBI / albedo) while the MLP sees only the pixel itself.

TWO QUESTIONS, AND THEY HAVE DIFFERENT ANSWERS
    1. Does spatial context improve ACCURACY?  Plausibly yes — the heat island is a
       neighbourhood effect, and a point-wise model cannot represent it.
    2. Does it fix the COUNTERFACTUALS?  No, and that is the point of running it. The cool-roof
       sign error is caused by albedo > 0.2 being absent from the data, not by the model being
       too simple. A CNN trained on the same data has the same blind spot, so it makes the same
       physically impossible prediction as the MLP — which is the argument for the physics loss
       and for models/synthetic.py, made by elimination.

TWO CHIP SOURCES
    --chips   dense windows exported by scripts/export_chips.py via neighborhoodToArray. Every
              cell is a real observation. This is the honest input for the baseline.
    default   windows reconstructed from the pixel table by data_engine/chips.py. That table is
              a ~4% sample of the AOI, so most cells are fill and the CNN is handicapped. Useful
              for wiring up the pipeline without a GEE round-trip; the script warns when chip
              density is low, and a result from this path should not be reported as the CNN's
              real performance.

FAIRNESS
    Same spatial-block split policy, same seed, same target standardisation as the tabular
    models, and the MLP control is re-trained HERE on the same chip centres rather than quoted
    from run_pinn.py — chips near block boundaries are dropped to prevent leakage, so comparing
    against a model trained on the full table would hand the MLP a larger sample.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_engine import chips as C                       # noqa: E402
from models import features as F                         # noqa: E402
from models.cnn import train_cnn, predict_cnn            # noqa: E402
from models.train import train_pinn, predict             # noqa: E402
from models.evaluate import regression_metrics           # noqa: E402

PARQUET = "data/processed/kochi_samples.parquet"
CKPT, FIGS = "models/checkpoints", "docs/figures"
COOL_ALBEDO, GREEN_NDVI = 0.50, 0.30
DEG_PER_M = 1.0 / 111_000.0


# ───────────────────────── chip sources ─────────────────────────
def load_dense_chips(path):
    """Dense chips from scripts/export_chips.py."""
    z = np.load(path, allow_pickle=True)
    patch = int(z["patch"])
    print(f"loaded {len(z['y']):,} dense {patch}x{patch} chips from {path}")
    # The mask channel is all-ones for dense chips, but ChipCNN expects it, and keeping the
    # channel count identical across both sources keeps the architecture comparable.
    img = z["img"]
    mask = np.ones((len(img), 1) + img.shape[2:], dtype="float32")
    return np.concatenate([img, mask], axis=1), z["scalar"], z["y"], z["lon"], z["lat"], patch


def build_sparse_chips(parquet, patch, cell_deg, seed):
    """Fallback: rasterise the pixel table and cut windows out of it."""
    df = F.load_samples(parquet)
    tr, va, te = F.make_splits(df, policy="spatial_block", seed=seed)
    tagged = C.label_splits(df, tr, va, te)
    grids = C.rasterize(tagged, cell_deg=cell_deg)
    print(f"rasterised {len(grids)} scenes onto a {grids[list(grids)[0]]['shape']} grid")

    Xi, Xs, y, sp = C.extract_chips(grids, patch=patch, seed=seed)
    density = float(Xi[:, -1].mean())            # the validity-mask channel
    print(f"chips   {len(Xi):,} at {patch}x{patch}, mean observed density {density:.1%}")
    if density < 0.5:
        print("  WARNING: most chip cells are fill, not observation. The CNN is handicapped\n"
              "           here; export dense chips with scripts/export_chips.py before\n"
              "           reporting this baseline.")
    return Xi, Xs, y, sp


def split_chip_centres(lon, lat, patch, seed=42, cell_m=30.0):
    """Spatial-block split on chip CENTRES, with a buffer so no window crosses a block edge.

    Same block size and policy as models/features.make_splits, so the CNN's test set is drawn
    from the same kind of held-out geography as the tabular models'.
    """
    rng = np.random.default_rng(seed)
    b = F.BLOCK_DEG
    bx, by = np.floor(lon / b).astype(int), np.floor(lat / b).astype(int)
    block = np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))
    uniq = np.array(sorted(set(block.tolist())))
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * 0.2))
    n_val = max(1, int(len(uniq) * 0.1))
    test_b, val_b = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])

    sp = np.where(np.isin(block, list(test_b)), 2,
                  np.where(np.isin(block, list(val_b)), 1, 0)).astype("int8")

    # Buffer: drop chips whose window could reach across a block boundary.
    half_deg = (patch // 2) * cell_m * DEG_PER_M
    fx = lon / b - np.floor(lon / b)
    fy = lat / b - np.floor(lat / b)
    edge = half_deg / b
    keep = (fx > edge) & (fx < 1 - edge) & (fy > edge) & (fy < 1 - edge)
    print(f"split   buffer drops {(~keep).sum():,} of {len(keep):,} chips near block edges")
    sp[~keep] = -1
    return sp


# ───────────────────────── point-wise control ─────────────────────────
def centre_frame(Xi, Xs, patch):
    """The centre pixel of each STANDARDISED chip, as an INPUT_COLUMNS-ordered matrix."""
    half = patch // 2
    centre = Xi[:, :len(C.SPATIAL_CHANNELS), half, half]      # ndvi, ndbi, albedo
    cols = {name: centre[:, i] for i, name in enumerate(C.SPATIAL_CHANNELS)}
    cols.update({name: Xs[:, i] for i, name in enumerate(C.SCALAR_CHANNELS)})
    return np.column_stack([cols[c] for c in F.INPUT_COLUMNS]).astype("float32")


def phys_frame(Xi, Xs, stats, patch):
    """Un-standardised physical drivers at the chip centres, as train_pinn's physics term wants."""
    import pandas as pd
    half = patch // 2
    out = {}
    for i, name in enumerate(C.SPATIAL_CHANNELS):
        out[name] = Xi[:, i, half, half] * stats["img_std"][i] + stats["img_mean"][i]
    for i, name in enumerate(C.SCALAR_CHANNELS):
        out[name] = Xs[:, i] * stats["sc_std"][i] + stats["sc_mean"][i]
    return pd.DataFrame(out)


def score(name, y_true, pred, counterfactual_fn, out):
    m = regression_metrics(y_true, pred)
    for kind, (a, n) in (("cool_roof", (COOL_ALBEDO, None)), ("greening", (None, GREEN_NDVI))):
        d = counterfactual_fn(a, n) - pred
        m[f"dT_{kind}"] = float(np.mean(d))
        m[f"cools_{kind}"] = float(np.mean(d < 0) * 100.0)
    out[name] = m
    print(f"    R2 {m['r2']:.3f}  RMSE {m['rmse']:.2f}K  coolroof {m['dT_cool_roof']:+.2f}K "
          f"({m['cools_cool_roof']:.0f}% cool)")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default=None, help="dense chip .npz from scripts/export_chips.py")
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--patch", type=int, default=9, help="chip side in pixels (30 m each)")
    ap.add_argument("--cell-deg", type=float, default=C.CELL_DEG)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    if args.chips:
        Xi, Xs, y, lon, lat, patch = load_dense_chips(args.chips)
        sp = split_chip_centres(lon, lat, patch, seed=args.seed)
    else:
        patch = args.patch
        Xi, Xs, y, sp = build_sparse_chips(args.parquet, patch, args.cell_deg, args.seed)

    m_tr, m_va, m_te = sp == 0, sp == 1, sp == 2
    if not (m_tr.any() and m_va.any() and m_te.any()):
        raise SystemExit("a split has no chips — try a smaller --patch or a coarser --cell-deg")
    print(f"chips   train {m_tr.sum():,} | val {m_va.sum():,} | test {m_te.sum():,}")

    Xi_tr, Xs_tr, stats = C.standardise(Xi[m_tr], Xs[m_tr])
    Xi_va, Xs_va, _ = C.standardise(Xi[m_va], Xs[m_va], stats)
    Xi_te, Xs_te, _ = C.standardise(Xi[m_te], Xs[m_te], stats)

    # Target standardisation is fit on TRAIN chips only, matching the tabular pipeline.
    scaler = {"y_mean": float(y[m_tr].mean()), "y_std": float(y[m_tr].std()) or 1.0}
    ys = lambda v: ((v - scaler["y_mean"]) / scaler["y_std"]).astype("float32")  # noqa: E731
    results = {}

    print("\nCNN (spatial context):")
    cnn = train_cnn(Xi_tr, Xs_tr, ys(y[m_tr]), Xi_va, Xs_va, ys(y[m_va]),
                    epochs=args.epochs, seed=args.seed)
    pred = predict_cnn(cnn, Xi_te, Xs_te, scaler)
    score("CNN (chips)", y[m_te], pred,
          lambda a, n: predict_cnn(
              cnn, C.apply_chip_intervention(Xi_te, stats, albedo_set=a, ndvi_delta=n),
              Xs_te, scaler),
          results)
    torch.save(cnn.state_dict(), f"{CKPT}/cnn_patch{patch}.pt")

    print("\nMLP control (same rows, no spatial context):")
    ctr_tr, ctr_va, ctr_te = (centre_frame(Xi_tr, Xs_tr, patch),
                              centre_frame(Xi_va, Xs_va, patch),
                              centre_frame(Xi_te, Xs_te, patch))
    mlp, _ = train_pinn(ctr_tr, ys(y[m_tr]).reshape(-1, 1), phys_frame(Xi_tr, Xs_tr, stats, patch),
                        ctr_va, ys(y[m_va]).reshape(-1, 1), phys_frame(Xi_va, Xs_va, stats, patch),
                        scaler, lambda_max=0.0, epochs=args.epochs, seed=args.seed, verbose=True)
    mp = lambda X: predict(mlp, X, scaler)                                       # noqa: E731
    score("MLP (same rows)", y[m_te], mp(ctr_te),
          lambda a, n: mp(centre_frame(
              C.apply_chip_intervention(Xi_te, stats, albedo_set=a, ndvi_delta=n),
              Xs_te, patch)),
          results)

    _table(results)
    path = f"{FIGS}/cnn_results{args.tag}.json"
    with open(path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"Saved {path}")
    _verdict(results, dense=bool(args.chips))


def _table(r):
    print("\n" + "=" * 86)
    print(f"{'model':<20}{'R2':>8}{'RMSE(K)':>10}{'MAE(K)':>9}{'coolroof dT':>14}{'% cool':>9}")
    print("-" * 86)
    for k, m in r.items():
        print(f"{k:<20}{m['r2']:>8.3f}{m['rmse']:>10.2f}{m['mae']:>9.2f}"
              f"{m['dT_cool_roof']:>14.2f}{m['cools_cool_roof']:>9.0f}")
    print("=" * 86)


def _verdict(r, dense):
    cnn, mlp = r.get("CNN (chips)"), r.get("MLP (same rows)")
    if not (cnn and mlp):
        return
    print(f"\n  spatial context: R2 {mlp['r2']:.3f} -> {cnn['r2']:.3f} "
          f"({cnn['r2'] - mlp['r2']:+.3f})")
    if cnn["cools_cool_roof"] < 95:
        print(f"  counterfactual still broken for BOTH: cool-roof dT "
              f"{mlp['dT_cool_roof']:+.2f} K (MLP) / {cnn['dT_cool_roof']:+.2f} K (CNN)")
        print("  => capacity is not the problem; the missing high-albedo data is "
              "(see models/synthetic.py)")
    else:
        print("  counterfactual: the CNN gets the cool-roof sign right on this data — check "
              "whether\n     that survives the albedo monotonicity test in run_augmented.py "
              "before claiming it.")
    if not dense:
        print("\n  NOTE: chips were reconstructed from the sparse pixel table. Run "
              "scripts/export_chips.py\n        for dense windows before reporting these numbers.")


if __name__ == "__main__":
    main()
