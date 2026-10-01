"""Physics-generated training samples — removing the accuracy/physics trade-off (plan §9b).

THE PROBLEM THIS SOLVES
    The deployed checkpoint is lambda=0.5 (R2 0.60) rather than lambda~0.05 (R2 0.84), because
    only a strong physics weight makes the model extrapolate safely. The underlying cause is a
    DATA GAP, not a modelling failure: measured albedo over Kochi is 0.134 +/- 0.024 (p99 0.198),
    so albedo > 0.2 simply does not occur in training. A cool roof sets albedo to 0.50 — about
    15 standard deviations out. With no data there, the data term is silent and the network is
    free to continue the spurious in-sample trend (brighter = hotter, a land-cover confound),
    which is why lambda=0 predicts that cool roofs WARM the surface.

    Raising lambda fixes extrapolation by over-ruling the data term everywhere, including where
    the data was good. That is the trade-off. The fix is to remove the gap instead: give the
    DATA term correct supervision in the high-albedo region, so it no longer has to be over-ruled.

ANCHORED COUNTERFACTUALS (what this module generates)
    A naive approach — sample features uniformly and label them with the SEB equilibrium
    temperature — would train the network on the physics model's ABSOLUTE temperatures, which
    carry a parameterisation bias of several Kelvin against observed LST. That would poison the
    in-distribution accuracy we are trying to keep.

    Instead each synthetic sample is ANCHORED to a real training pixel. Take a real pixel with
    measured LST y and drivers x; perturb only the intervention variables (albedo, NDVI) to x';
    and label the synthetic sample

        y' = y + [ T_eq(x') - T_eq(x) ]

    where T_eq solves the surface energy balance (models.sebal.equilibrium_temperature). The
    label is the MEASURED temperature plus the PHYSICS-PREDICTED CHANGE. Two properties follow:

      1. Bias-free. Any systematic offset between T_eq and observed LST — roughness, moisture
         availability, the radiometric-vs-aerodynamic temperature gap — enters both terms and
         cancels in the difference. The physics supplies only the response, which is exactly the
         quantity the planner deploys (api/inference.whatif returns delta_t, never an absolute).
      2. Realistic covariates. lon/lat, S_down, T_air, RH, wind and the unperturbed index come
         from a real pixel, so the synthetic rows sit on the real data manifold in every
         dimension except the one being extrapolated.

    Perturbing albedo while HOLDING NDVI/NDBI FIXED is deliberate: it is the partial derivative
    dT/dalbedo at constant land cover — a roof that gets painted, not a roof that becomes a park.
    That partial is precisely what the observational data cannot identify, because in the record
    albedo only ever varies together with land cover.

LEAKAGE
    Anchors are drawn from the TRAIN split only (enforced by the caller, scripts/run_augmented.py).
    Validation and test remain 100% measured pixels, so reported R2/RMSE are still honest
    out-of-sample numbers on real observations.
"""

import numpy as np

from . import sebal as S
from .features import INPUT_COLUMNS, TARGET_COLUMN

# Physical plausibility bounds, matching data_engine/build_dataset.py.
LST_MIN_K, LST_MAX_K = 288.0, 333.0

# Albedo range to cover. The upper end exceeds a cool roof (0.50) so the intervention point is
# interior to the training support rather than sitting on its edge.
ALBEDO_LO, ALBEDO_HI = 0.05, 0.70

# NDVI perturbation range. Negative values matter too: they teach the loss of vegetation, which
# keeps the response monotonic rather than one-sided.
NDVI_DELTA_LO, NDVI_DELTA_HI = -0.20, 0.50
NDVI_MAX = 0.95

# Greening a built surface also reduces its built-up index. NDVI and NDBI are strongly
# anti-correlated in the measured record; this keeps synthetic rows on that manifold instead of
# inventing "dense vegetation that is also dense concrete" pixels the network could exploit.
NDBI_COUPLING = 0.5

# Reject anchors whose physics response is implausible — a guard against the bisection solver
# being pushed somewhere pathological by an outlier driver combination.
MAX_ABS_DELTA_K = 25.0

PHYS_DRIVERS = ["ndvi", "albedo", "s_down", "t_air", "rh", "wind"]


def equilibrium_of(df):
    """SEB equilibrium surface temperature [K] for every row of `df`."""
    return S.equilibrium_temperature(
        df["ndvi"].to_numpy(dtype="float64"), df["albedo"].to_numpy(dtype="float64"),
        df["s_down"].to_numpy(dtype="float64"), df["t_air"].to_numpy(dtype="float64"),
        df["rh"].to_numpy(dtype="float64"), df["wind"].to_numpy(dtype="float64"))


def physics_delta(df, albedo_set=None, ndvi_delta=None):
    """Physics-predicted dT [K] for an intervention, from the SEB equilibrium alone.

    This is the model-free reference the PINN's counterfactuals are measured against: it uses no
    network at all, only the energy balance. `albedo_set` raises albedo to at least that value
    (as a coating would); `ndvi_delta` adds vegetation.
    """
    mod = df.copy()
    if albedo_set is not None:
        mod["albedo"] = np.maximum(mod["albedo"].to_numpy(dtype="float64"), float(albedo_set))
    if ndvi_delta:
        mod["ndvi"] = np.clip(mod["ndvi"].to_numpy(dtype="float64") + float(ndvi_delta),
                              -1.0, NDVI_MAX)
    return equilibrium_of(mod) - equilibrium_of(df)


def make_anchors(train_df, n_samples=60_000, seed=42, mix=(0.5, 0.25, 0.25),
                 albedo_range=(ALBEDO_LO, ALBEDO_HI),
                 ndvi_delta_range=(NDVI_DELTA_LO, NDVI_DELTA_HI),
                 ndbi_coupling=NDBI_COUPLING):
    """Generate anchored counterfactual training rows.

    Args:
        train_df: measured TRAIN pixels (never val/test — see the leakage note above).
        n_samples: how many synthetic rows to draw, with replacement, before filtering.
        mix: probability of (albedo-only, ndvi-only, both) perturbations. Albedo is weighted
            highest because that is where the data gap is most extreme.

    Returns a DataFrame with the same columns the model consumes (INPUT_COLUMNS + lst), plus
    `anchor_delta_k` for diagnostics.
    """
    rng = np.random.default_rng(seed)
    src = train_df.iloc[rng.integers(0, len(train_df), size=int(n_samples))].reset_index(drop=True)

    kind = rng.choice(3, size=len(src), p=np.asarray(mix, dtype="float64") / np.sum(mix))
    out = src.copy()

    do_albedo = kind != 1
    do_ndvi = kind != 0

    new_albedo = rng.uniform(albedo_range[0], albedo_range[1], size=len(src))
    out.loc[do_albedo, "albedo"] = new_albedo[do_albedo]

    delta = rng.uniform(ndvi_delta_range[0], ndvi_delta_range[1], size=len(src))
    delta = np.where(do_ndvi, delta, 0.0)
    out["ndvi"] = np.clip(src["ndvi"].to_numpy(dtype="float64") + delta, -1.0, NDVI_MAX)
    # Actual applied change after clipping — NDBI must follow the same amount.
    applied = out["ndvi"].to_numpy(dtype="float64") - src["ndvi"].to_numpy(dtype="float64")
    if "ndbi" in out:
        out["ndbi"] = np.clip(src["ndbi"].to_numpy(dtype="float64") - ndbi_coupling * applied,
                              -1.0, 1.0)

    d = equilibrium_of(out) - equilibrium_of(src)
    out[TARGET_COLUMN] = src[TARGET_COLUMN].to_numpy(dtype="float64") + d
    out["anchor_delta_k"] = d

    ok = (np.isfinite(d) & (np.abs(d) <= MAX_ABS_DELTA_K)
          & (out[TARGET_COLUMN] >= LST_MIN_K) & (out[TARGET_COLUMN] <= LST_MAX_K))
    keep = [c for c in INPUT_COLUMNS if c in out] + [TARGET_COLUMN, "anchor_delta_k"]
    return out.loc[ok.to_numpy(), keep].reset_index(drop=True)


def coverage_report(train_df, aug_df, column="albedo", edges=None):
    """Histogram of measured vs augmented support in one feature — the data-gap figure.

    Returns (edges, measured_counts, augmented_counts).
    """
    if edges is None:
        lo = float(min(train_df[column].min(), aug_df[column].min()))
        hi = float(max(train_df[column].max(), aug_df[column].max()))
        edges = np.linspace(lo, hi, 41)
    m, _ = np.histogram(train_df[column].to_numpy(dtype="float64"), bins=edges)
    a, _ = np.histogram(aug_df[column].to_numpy(dtype="float64"), bins=edges)
    return edges, m, a


def describe(train_df, aug_df):
    """One-line summary of what the augmentation added, for the run log."""
    a_tr = train_df["albedo"].to_numpy(dtype="float64")
    a_au = aug_df["albedo"].to_numpy(dtype="float64")
    hi = float(np.mean(a_au > a_tr.max()) * 100.0)
    return (f"{len(aug_df):,} anchored samples | albedo measured "
            f"[{a_tr.min():.3f}, {a_tr.max():.3f}] -> augmented [{a_au.min():.3f}, {a_au.max():.3f}] "
            f"| {hi:.0f}% beyond the measured maximum | "
            f"mean anchor dT {aug_df['anchor_delta_k'].mean():+.2f} K")
