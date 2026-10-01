"""Gridded image chips from the per-pixel table — the input the CNN baseline needs (plan §7).

models/baselines.py notes that the proposal's CNN baseline was deferred because "a CNN needs
gridded image patches, but our dataset is a per-pixel table". This module closes that gap
without a second Earth Engine export: the sampled pixels carry lon/lat/date, so they can be
rasterised back onto the regular 30 m grid they were drawn from and cut into chips.

WHY A CNN AT ALL
    The tabular models see each pixel in isolation. A CNN sees its NEIGHBOURHOOD, which is where
    the urban heat island actually lives — a courtyard surrounded by concrete is hotter than the
    same courtyard surrounded by canopy, and no point-wise model can represent that. If spatial
    context carries real signal, the CNN should beat the point-wise MLP on identical splits.

SPARSITY
    The table is a random SAMPLE of pixels, so the reconstructed grid is mostly empty. Chips
    therefore carry an explicit VALIDITY MASK channel and missing cells are filled with the
    per-scene mean, so the network can tell "no observation" from "observed average" instead of
    silently learning the fill value as signal.

LEAKAGE (the part that is easy to get wrong)
    Chips overlap. A chip centred on a test pixel can contain train pixels in its window, which
    would leak the very correlation the spatial-block split exists to break. `extract_chips`
    therefore DROPS any chip whose window touches a cell belonging to a different split, rather
    than assigning chips by centre alone. That costs chips near block boundaries and is the
    reason the CNN trains on fewer samples than the tabular models — a fair trade for a
    comparison that means something.
"""

import numpy as np
import pandas as pd

# Spatial channels: these vary pixel-to-pixel and are what the convolutions actually see.
SPATIAL_CHANNELS = ["ndvi", "ndbi", "albedo"]
# Scalar inputs: per-scene climate plus position. Constant across a chip, so feeding them as
# image planes would waste the convolutions; they enter through a side head instead.
SCALAR_CHANNELS = ["lon", "lat", "s_down", "t_air", "rh", "wind"]

CELL_DEG = 0.00027        # ~30 m at Kochi's latitude — the native Landsat grid


def _grid_index(df, cell_deg=CELL_DEG):
    """Map lon/lat to integer grid cells, origin at the dataset's south-west corner."""
    lon0, lat0 = float(df.lon.min()), float(df.lat.min())
    gx = np.round((df.lon.to_numpy(dtype="float64") - lon0) / cell_deg).astype(int)
    gy = np.round((df.lat.to_numpy(dtype="float64") - lat0) / cell_deg).astype(int)
    return gx, gy, lon0, lat0


def rasterize(df, cell_deg=CELL_DEG):
    """Rasterise the sample table back onto a regular grid, one stack per date.

    Returns {date: dict(channels=…, lst=…, valid=…, split=…)} where each entry is a 2D array.
    `split` is present only when the frame carries a `split` column.
    """
    gx, gy, lon0, lat0 = _grid_index(df, cell_deg)
    h, w = int(gy.max()) + 1, int(gx.max()) + 1
    out = {}
    has_split = "split" in df.columns
    codes = {"train": 0, "val": 1, "test": 2}

    for date, sub in df.assign(_gx=gx, _gy=gy).groupby("date"):
        valid = np.zeros((h, w), dtype=bool)
        chans = {c: np.zeros((h, w), dtype="float32") for c in SPATIAL_CHANNELS}
        lst = np.zeros((h, w), dtype="float32")
        split = np.full((h, w), -1, dtype="int8")
        yy, xx = sub._gy.to_numpy(), sub._gx.to_numpy()
        valid[yy, xx] = True
        for c in SPATIAL_CHANNELS:
            chans[c][yy, xx] = sub[c].to_numpy(dtype="float32")
        lst[yy, xx] = sub["lst"].to_numpy(dtype="float32")
        if has_split:
            split[yy, xx] = [codes[s] for s in sub["split"]]

        # Fill unobserved cells with the per-scene mean of observed ones. The mask channel keeps
        # this distinguishable from a real observation at that value.
        for c in SPATIAL_CHANNELS:
            m = float(chans[c][valid].mean()) if valid.any() else 0.0
            chans[c][~valid] = m

        scalars = {c: float(sub[c].mean()) for c in SCALAR_CHANNELS if c in sub}
        out[date] = {"channels": chans, "lst": lst, "valid": valid, "split": split,
                     "scalars": scalars, "shape": (h, w),
                     "rows": sub[["_gy", "_gx"]].to_numpy()}
    return out


def extract_chips(grids, patch=9, require_pure_split=True, max_per_date=None, seed=42):
    """Cut `patch`x`patch` chips centred on every observed pixel.

    Returns (X_img, X_scalar, y, split) with X_img shaped [N, C, patch, patch]; C is
    len(SPATIAL_CHANNELS) + 1 for the validity mask.

    With `require_pure_split`, a chip is kept only if every observed cell in its window belongs
    to the same split as its centre — see the leakage note in the module docstring.
    """
    rng = np.random.default_rng(seed)
    half = patch // 2
    imgs, scal, ys, sps = [], [], [], []

    for _date, g in grids.items():
        h, w = g["shape"]
        valid, split, lst = g["valid"], g["split"], g["lst"]
        stack = np.stack([g["channels"][c] for c in SPATIAL_CHANNELS]
                         + [valid.astype("float32")], axis=0)
        sv = np.array([g["scalars"].get(c, 0.0) for c in SCALAR_CHANNELS], dtype="float32")

        centres = g["rows"]
        if max_per_date and len(centres) > max_per_date:
            centres = centres[rng.choice(len(centres), max_per_date, replace=False)]

        for cy, cx in centres:
            if cy < half or cx < half or cy + half >= h or cx + half >= w:
                continue                                   # window would run off the grid
            win_split = split[cy - half:cy + half + 1, cx - half:cx + half + 1]
            own = split[cy, cx]
            if require_pure_split and own >= 0:
                others = win_split[(win_split >= 0) & (win_split != own)]
                if others.size:
                    continue                               # window straddles a split boundary
            imgs.append(stack[:, cy - half:cy + half + 1, cx - half:cx + half + 1])
            scal.append(sv)
            ys.append(lst[cy, cx])
            sps.append(own)

    if not imgs:
        raise ValueError("no chips extracted — grid too sparse for this patch size")
    return (np.stack(imgs).astype("float32"), np.stack(scal).astype("float32"),
            np.asarray(ys, dtype="float32"), np.asarray(sps, dtype="int8"))


def label_splits(df, train, val, test):
    """Tag each row of `df` with the split it landed in, for leakage-aware chip extraction."""
    split = pd.Series("none", index=df.index, dtype=object)
    split.loc[train.index] = "train"
    split.loc[val.index] = "val"
    split.loc[test.index] = "test"
    out = df.copy()
    out["split"] = split
    return out[out.split != "none"]


def standardise(X_img, X_scalar, stats=None):
    """Per-channel standardisation. Fit on train (`stats=None`), then reuse those stats."""
    if stats is None:
        # The mask channel is already 0/1 — leave it alone so it stays interpretable.
        n_sp = len(SPATIAL_CHANNELS)
        mean = X_img[:, :n_sp].mean(axis=(0, 2, 3))
        std = X_img[:, :n_sp].std(axis=(0, 2, 3))
        std[std == 0] = 1.0
        smean, sstd = X_scalar.mean(axis=0), X_scalar.std(axis=0)
        sstd[sstd == 0] = 1.0
        stats = {"img_mean": mean, "img_std": std, "sc_mean": smean, "sc_std": sstd}

    n_sp = len(SPATIAL_CHANNELS)
    Xi = X_img.copy()
    Xi[:, :n_sp] = ((Xi[:, :n_sp] - stats["img_mean"][None, :, None, None])
                    / stats["img_std"][None, :, None, None])
    Xs = (X_scalar - stats["sc_mean"][None, :]) / stats["sc_std"][None, :]
    return Xi.astype("float32"), Xs.astype("float32"), stats


def apply_chip_intervention(X_img, stats, albedo_set=None, ndvi_delta=None):
    """Apply a counterfactual to STANDARDISED chips, so the CNN can be compared like the others.

    The intervention is defined in physical units, so it is un-standardised, applied, and
    re-standardised per channel.
    """
    Xi = X_img.copy()
    idx = {c: i for i, c in enumerate(SPATIAL_CHANNELS)}

    def edit(ch, fn):
        i = idx[ch]
        raw = Xi[:, i] * stats["img_std"][i] + stats["img_mean"][i]
        Xi[:, i] = (fn(raw) - stats["img_mean"][i]) / stats["img_std"][i]

    if albedo_set is not None:
        edit("albedo", lambda v: np.maximum(v, float(albedo_set)))
    if ndvi_delta:
        edit("ndvi", lambda v: np.clip(v + float(ndvi_delta), -1.0, 0.95))
    return Xi
