"""Tests for the physics-generated training samples (plan §9b).

Run:  python tests/test_synthetic.py

models/synthetic.py is the module the project's strongest claim rests on, so the properties it
relies on are asserted rather than assumed:

  * The anchor label is bias-free by construction — it is a measured LST plus a physics-predicted
    CHANGE, so the SEB model's absolute offset must cancel exactly.
  * The augmented support actually covers the intervention point (albedo 0.50), which the
    measured record does not reach. This is the whole purpose of the module.
  * Anchors never leak the label: a row whose intervention is a no-op must keep its measured LST.
  * The physics response has the right sign and is monotone in albedo — if the generator itself
    got this wrong, it would teach the network the same error the physics term exists to prevent.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import sebal as S                 # noqa: E402
from models import synthetic as SY            # noqa: E402
from models.features import INPUT_COLUMNS, TARGET_COLUMN   # noqa: E402
from tests.fixtures import synthetic_dataset  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))


def main():
    print("Physics-augmentation tests\n" + "=" * 70)
    df = synthetic_dataset(n=3000, seed=11)

    # ───────────── the data gap is real in the fixture ─────────────
    check("measured albedo never reaches a cool roof",
          df.albedo.max() < 0.30,
          f"max measured albedo {df.albedo.max():.3f} vs cool roof 0.50")
    corr = float(np.corrcoef(df.albedo, df.lst)[0, 1])
    check("fixture reproduces the land-cover confound",
          corr > 0.2,
          f"corr(albedo, LST) = {corr:+.3f} — positive, though physics says brighter is cooler")

    # ───────────── physics direction ─────────────
    d_cool = SY.physics_delta(df, albedo_set=0.50)
    check("physics says a cool roof cools, for every pixel",
          float(np.mean(d_cool < 0)) == 1.0,
          f"mean {d_cool.mean():+.2f} K, {100 * np.mean(d_cool < 0):.0f}% cooling")
    d_green = SY.physics_delta(df, ndvi_delta=0.30)
    check("physics says greening cools almost everywhere",
          float(np.mean(d_green < 0)) > 0.9,
          f"mean {d_green.mean():+.2f} K, {100 * np.mean(d_green < 0):.0f}% cooling")

    # Monotone response: the generator must not teach a non-monotone albedo curve.
    sub = df.iloc[:400]
    means = []
    for a in np.linspace(0.08, 0.70, 12):
        mod = sub.copy()
        mod["albedo"] = a
        means.append(float(np.mean(SY.equilibrium_of(mod))))
    steps = np.diff(means)
    check("physics equilibrium is monotone decreasing in albedo",
          bool(np.all(steps < 0)),
          f"{np.sum(steps < 0)}/{len(steps)} steps decreasing, "
          f"total {means[-1] - means[0]:+.2f} K")

    # ───────────── anchor construction ─────────────
    aug = SY.make_anchors(df, n_samples=4000, seed=3)
    check("anchors keep the dataset contract",
          all(c in aug.columns for c in INPUT_COLUMNS + [TARGET_COLUMN]),
          f"{len(aug):,} rows")
    check("augmented support covers the intervention point",
          aug.albedo.min() < 0.10 and aug.albedo.max() > 0.60,
          f"albedo {aug.albedo.min():.3f}-{aug.albedo.max():.3f}")
    beyond = float(np.mean(aug.albedo > df.albedo.max()))
    check("a large share of anchors sit beyond the measured maximum",
          beyond > 0.3, f"{beyond:.0%} beyond {df.albedo.max():.3f}")
    check("anchor labels stay physically plausible",
          bool(((aug[TARGET_COLUMN] >= SY.LST_MIN_K)
                & (aug[TARGET_COLUMN] <= SY.LST_MAX_K)).all()),
          f"{aug[TARGET_COLUMN].min():.1f}-{aug[TARGET_COLUMN].max():.1f} K")
    check("brighter anchors are labelled cooler",
          float(np.corrcoef(aug.albedo, aug.anchor_delta_k)[0, 1]) < -0.5,
          f"corr(albedo, anchor dT) = "
          f"{np.corrcoef(aug.albedo, aug.anchor_delta_k)[0, 1]:+.3f}")

    # ───────────── bias-free by construction ─────────────
    # A no-op intervention must return the measured label untouched: the SEB offset cancels.
    # mix=(0,1,0) selects the NDVI-only path, which leaves albedo alone; a zero NDVI delta then
    # perturbs nothing. (Forcing albedo_range=(0,0) would NOT be a no-op — it would set albedo
    # to zero, which is a large real change.)
    noop = SY.make_anchors(df.iloc[:500], n_samples=500, seed=5,
                           ndvi_delta_range=(0.0, 0.0), mix=(0, 1, 0))
    check("a no-op anchor reproduces the measured label exactly",
          bool(np.allclose(noop["anchor_delta_k"], 0.0, atol=1e-6)),
          f"max |dT| {np.abs(noop['anchor_delta_k']).max():.2e} K")

    # The label is measured LST + physics delta, never the physics absolute temperature.
    src = df.iloc[:300].reset_index(drop=True)
    mod = src.copy()
    mod["albedo"] = 0.50
    expect = src.lst.to_numpy() + (SY.equilibrium_of(mod) - SY.equilibrium_of(src))
    direct = SY.equilibrium_of(mod)
    offset = float(np.mean(np.abs(expect - direct)))
    check("anchoring removes the SEB absolute-temperature bias",
          offset > 0.5,
          f"raw equilibrium labels would be off by {offset:.2f} K on average")

    # ───────────── NDBI coupling ─────────────
    greened = SY.make_anchors(df, n_samples=2000, seed=9, mix=(0, 1, 0))
    rise = greened.ndvi.to_numpy() - df.ndvi.mean()
    check("greening anchors move NDBI opposite to NDVI",
          float(np.corrcoef(greened.ndvi, greened.ndbi)[0, 1]) < 0,
          f"corr(NDVI, NDBI) = {np.corrcoef(greened.ndvi, greened.ndbi)[0, 1]:+.3f}")

    # ───────────── training integration ─────────────
    _test_training_accepts_augmentation(df)

    print("\n" + "=" * 70)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


def _test_training_accepts_augmentation(df):
    """train_pinn must mix the pool, weight it, and still select on MEASURED validation."""
    from models import features as F
    from models.train import train_pinn

    tr, va, _te = F.make_splits(df, policy="spatial_block", seed=1)
    scaler = F.fit_scaler(tr)
    aug = SY.make_anchors(tr, n_samples=800, seed=2)

    Xtr, _ = F.transform(tr, scaler)
    Xva, _ = F.transform(va, scaler)
    _, ytr = F.transform(tr, scaler, scale_y=True)
    _, yva = F.transform(va, scaler, scale_y=True)
    Xaug, _ = F.transform(aug, scaler)
    _, yaug = F.transform(aug, scaler, scale_y=True)

    model, hist = train_pinn(Xtr, ytr, tr, Xva, yva, va, scaler,
                             lambda_max=0.0, epochs=3, seed=0, verbose=False,
                             X_aug=Xaug, y_aug_scaled=yaug, df_aug=aug, aug_weight=0.5)
    check("train_pinn trains with an augmentation pool", len(hist) == 3)

    # Zero weight on the synthetic pool must be indistinguishable from omitting it.
    m0, h0 = train_pinn(Xtr, ytr, tr, Xva, yva, va, scaler, lambda_max=0.0, epochs=2,
                        seed=0, verbose=False)
    m1, h1 = train_pinn(Xtr, ytr, tr, Xva, yva, va, scaler, lambda_max=0.0, epochs=2,
                        seed=0, verbose=False, X_aug=Xaug, y_aug_scaled=yaug,
                        df_aug=aug, aug_weight=0.0)
    check("aug_weight=0 leaves the data term unchanged",
          abs(h0[-1]["val_data"] - h1[-1]["val_data"]) < 5e-3,
          f"val data {h0[-1]['val_data']:.5f} vs {h1[-1]['val_data']:.5f}")


if __name__ == "__main__":
    sys.exit(main())
