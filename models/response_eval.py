"""Response-centred evaluation: score the model on the CHANGE it predicts, not on absolute LST.

WHY A SECOND EVALUATION
    The headline metric so far is pixel R^2 on absolute land surface temperature. The planner never
    uses an absolute prediction: its baseline is the MEASURED Landsat LST, and the model supplies
    only the response, delta-T, to a change in the surface (api/inference.py). R^2 on absolute LST
    therefore scores a quantity the deployment does not depend on, and it is dominated by things
    that do not matter for a counterfactual (for example a constant bias, which cancels in delta-T).
    This module scores the quantity the deployment DOES depend on, against real observations.

THE NATURAL EXPERIMENT IN THE DATA
    Of the 75,762 Landsat pixels in the table, 51,611 were sampled on two or more dates, at exactly
    the same coordinates. Between two dates a pixel's inputs change: weather always, and sometimes
    the surface itself (vegetation cleared or grown, roofs and paving added). The observed change in
    LST is then a measured response to a measured change in inputs, which is exactly what a
    counterfactual claims to predict. A model that has learned the right physics should predict the
    observed change; a model that has learned only "what kind of place is hot" should not.

PROTOCOL (fixed before any result was seen; do not tune these to the outcome)
    Data      held-out TEST pixels of the standard spatial-block split (seed 42), the same pixels
              the absolute-R^2 table uses. No pair ever involves a training pixel.
    Pairs     each pixel's observations sorted by date; every CONSECUTIVE pair (t1, t2) is one
              sample. Consecutive pairs keep samples close to independent within a pixel.
    Tier A    total change:   d_obs = y(t2) - y(t1),  d_pred = f(x(t2)) - f(x(t1)).
              Includes city-wide weather, so it is partly easy.
    Tier B    local change:   the same, after subtracting each date's mean (computed separately
              for observations and for each model's predictions, over the test pixels). Weather
              that moves the whole city is removed; what is left is the pixel's change relative to
              its surroundings.
    Tier C    surface change: Tier B restricted to pairs whose SURFACE changed by a material
              amount, |dNDVI| >= 0.10 or |d albedo| >= 0.02 (about 0.7 and 0.8 standard deviations).
              These are real interventions that happened, and the primary test of a counterfactual.
    Metrics   r       Pearson correlation of predicted and observed change.
              R2      1 - SSE/SST on the observed change (can be negative).
              slope   OLS slope of observed on predicted change. 1 = right size; below 1 = the
                      model over-reacts; above 1 = it under-reacts.
              sign    share of pairs with |d_obs| >= 0.5 K where the predicted change has the same
                      sign as the observed one.
              rmse    in K.
    Spread    95% intervals by bootstrapping SPATIAL BLOCKS (the split's own 0.01 deg blocks), so
              neighbouring pixels are not treated as independent evidence.
"""

import numpy as np
import pandas as pd

from .features import BLOCK_DEG

NDVI_CHANGE = 0.10        # Tier C: material vegetation change ...
ALBEDO_CHANGE = 0.02      # ... or material brightness change
SIGN_MIN_K = 0.5          # sign agreement ignores observed changes smaller than this
COORD_DECIMALS = 6        # pixel identity: coordinates agree to ~0.1 m on the Landsat grid


def build_pairs(df):
    """Consecutive same-pixel pairs. Returns two positional index arrays (i1 earlier, i2 later)
    into `df` as given (position, not label)."""
    d = pd.DataFrame({
        "pix": df["lon"].round(COORD_DECIMALS).astype(str).values + "_"
               + df["lat"].round(COORD_DECIMALS).astype(str).values,
        "date": pd.to_datetime(df["date"]).values,
        "pos": np.arange(len(df)),
    }).sort_values(["pix", "date"], kind="mergesort")
    same = d["pix"].values[1:] == d["pix"].values[:-1]
    later = d["date"].values[1:] > d["date"].values[:-1]      # a pixel twice on one date is not a pair
    keep = same & later
    return d["pos"].values[:-1][keep], d["pos"].values[1:][keep]


def date_anomaly(values, dates):
    """Subtract each date's mean, so city-wide weather is removed."""
    v = pd.Series(np.asarray(values, dtype="float64"))
    return (v - v.groupby(np.asarray(dates)).transform("mean")).to_numpy()


def change_metrics(d_obs, d_pred):
    """Metrics of a predicted change against the observed change."""
    d_obs = np.asarray(d_obs, dtype="float64")
    d_pred = np.asarray(d_pred, dtype="float64")
    n = len(d_obs)
    if n < 3:
        return {"n": n}
    sst = float(np.sum((d_obs - d_obs.mean()) ** 2))
    sse = float(np.sum((d_obs - d_pred) ** 2))
    if np.std(d_pred) > 1e-12 and np.std(d_obs) > 1e-12:
        r = float(np.corrcoef(d_obs, d_pred)[0, 1])
        slope = float(np.polyfit(d_pred, d_obs, 1)[0])
    else:
        r, slope = float("nan"), float("nan")
    big = np.abs(d_obs) >= SIGN_MIN_K
    sign = float(np.mean(np.sign(d_pred[big]) == np.sign(d_obs[big]))) if big.any() else float("nan")
    return {"n": n, "r": r, "r2": 1.0 - sse / sst if sst > 0 else float("nan"), "slope": slope,
            "sign": sign, "rmse": float(np.sqrt(np.mean((d_obs - d_pred) ** 2)))}


def decompose_r2(y, y_hat, dates):
    """Split absolute skill into BETWEEN-date (does it track each day's weather?) and WITHIN-date
    (does it put the heat in the right places on a given day?)."""
    y = np.asarray(y, dtype="float64")
    y_hat = np.asarray(y_hat, dtype="float64")
    frame = pd.DataFrame({"y": y, "p": y_hat, "d": np.asarray(dates)})
    means = frame.groupby("d")[["y", "p"]].mean()

    def r2(a, b):
        sst = np.sum((a - a.mean()) ** 2)
        return float(1 - np.sum((a - b) ** 2) / sst) if sst > 0 else float("nan")

    return {"between_date_r2": r2(means["y"].values, means["p"].values),
            "within_date_r2": r2(date_anomaly(y, frame["d"]), date_anomaly(y_hat, frame["d"]))}


def tiers(test_df, y_pred):
    """The three change tiers for one model's predictions on the test rows (Kelvin).
    Returns {tier: (d_obs, d_pred, block_id)} so callers can score or bootstrap them."""
    y = test_df["lst"].to_numpy(dtype="float64")
    p = np.asarray(y_pred, dtype="float64")
    dates = test_df["date"].to_numpy()
    i1, i2 = build_pairs(test_df)
    block = (np.floor(test_df["lon"].to_numpy() / BLOCK_DEG).astype(np.int64) * 100000
             + np.floor(test_df["lat"].to_numpy() / BLOCK_DEG).astype(np.int64))[i1]

    a_obs, a_pred = y[i2] - y[i1], p[i2] - p[i1]
    ya, pa = date_anomaly(y, dates), date_anomaly(p, dates)
    b_obs, b_pred = ya[i2] - ya[i1], pa[i2] - pa[i1]
    nd = test_df["ndvi"].to_numpy()
    al = test_df["albedo"].to_numpy()
    surf = (np.abs(nd[i2] - nd[i1]) >= NDVI_CHANGE) | (np.abs(al[i2] - al[i1]) >= ALBEDO_CHANGE)
    return {"A_total": (a_obs, a_pred, block),
            "B_local": (b_obs, b_pred, block),
            "C_surface": (b_obs[surf], b_pred[surf], block[surf])}


def ward_counterfactual(predict, df, input_columns, cool_albedo=0.50, steps=10):
    """The cool-roof test the deployed checkpoint was selected on: every ward's mean feature vector,
    albedo raised to max(albedo, cool_albedo) in `steps` steps. `predict` maps a feature dict to K.

    Returns the mean change, the WORST ward (positive = predicted to heat), the share of wards that
    cool, and the share whose temperature falls at every step (a steady, monotone response)."""
    w = df.groupby("ward_id")[input_columns].mean()
    feats = {c: w[c].to_numpy() for c in input_columns}
    target = np.maximum(feats["albedo"], cool_albedo)
    curves = []
    for k in range(steps + 1):
        f = dict(feats)
        f["albedo"] = feats["albedo"] + (target - feats["albedo"]) * (k / steps)
        curves.append(np.asarray(predict(f), dtype="float64"))
    curves = np.array(curves)
    dt = curves[-1] - curves[0]
    return {"cool_roof_dt": float(dt.mean()), "worst_ward": float(dt.max()),
            "wards_cooling_pct": float(100 * np.mean(dt < 0)),
            "monotone_pct": float(100 * np.mean(np.all(np.diff(curves, axis=0) <= 0.01, axis=0)))}


def scorecard(predict, test_df, full_df, input_columns, bootstrap=True):
    """Every metric of the protocol for one model: absolute, observed change (tiers A-C), and the
    ward counterfactual. `predict` maps a feature dict to LST in K. Used by every experiment so they
    are all scored identically."""
    from .evaluate import regression_metrics
    y = test_df["lst"].to_numpy(dtype="float64")
    p = np.asarray(predict({c: test_df[c].to_numpy() for c in input_columns}), dtype="float64")
    absolute = regression_metrics(y, p)
    absolute["bias"] = float(np.mean(p - y))                 # positive = predicts too hot
    absolute.update(decompose_r2(y, p, test_df["date"].to_numpy()))
    t = tiers(test_df, p)
    change = {name: change_metrics(o, q) for name, (o, q, _) in t.items()}
    if bootstrap and change["C_surface"].get("n", 0) >= 30:
        o, q, b = t["C_surface"]
        change["C_surface"]["r_ci"] = block_bootstrap(o, q, b, "r", seed=1)
    return {"absolute": absolute, "change": change,
            "counterfactual": ward_counterfactual(predict, full_df, input_columns)}


def block_bootstrap(d_obs, d_pred, block, metric, n_boot=300, seed=0):
    """95% interval of one metric, resampling whole spatial blocks."""
    rng = np.random.default_rng(seed)
    ids = np.unique(block)
    members = {b: np.flatnonzero(block == b) for b in ids}
    vals = []
    for _ in range(n_boot):
        take = np.concatenate([members[b] for b in rng.choice(ids, size=len(ids), replace=True)])
        v = change_metrics(d_obs[take], d_pred[take]).get(metric)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
