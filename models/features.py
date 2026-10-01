"""Feature matrix assembly, normalization, and train/val/test splits (plan §6).

Reads the dataset artifact from data_engine/build_dataset.py and produces the X/y arrays
consumed by the baselines and the PINN.

SPLIT POLICY (important): neighbouring 30 m pixels are highly correlated, so a random pixel
split leaks information between train and test and badly inflates R². We split by SPATIAL
BLOCKS instead — the AOI is diced into ~1 km cells and whole cells go to train/val/test, so
test pixels are in locations the model never saw. `random` is available only to demonstrate
that inflation in the report.
"""

import json

import numpy as np
import pandas as pd

# Spatial coords use lon/lat directly (normalized during training, so units don't matter).
INPUT_COLUMNS = ["lon", "lat", "ndvi", "ndbi", "albedo", "s_down", "t_air", "rh", "wind"]
TARGET_COLUMN = "lst"          # Kelvin

BLOCK_DEG = 0.01               # ~1.1 km spatial blocks


def load_samples(parquet_path):
    """Load the pixel samples table."""
    return pd.read_parquet(parquet_path)


def make_splits(df, policy="spatial_block", test_fraction=0.2, val_fraction=0.1, seed=42):
    """Return (train_df, val_df, test_df).

    policy='spatial_block' assigns whole ~1 km cells to a split (leakage-safe, the default).
    policy='random' shuffles individual pixels (leaky — for the comparison experiment only).
    """
    rng = np.random.default_rng(seed)

    if policy == "random":
        idx = rng.permutation(len(df))
        n_test = int(len(df) * test_fraction)
        n_val = int(len(df) * val_fraction)
        test_i, val_i, train_i = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
        return df.iloc[train_i], df.iloc[val_i], df.iloc[test_i]

    if policy != "spatial_block":
        raise ValueError(f"unknown split policy: {policy}")

    block = (np.floor(df.lon / BLOCK_DEG).astype(int).astype(str) + "_" +
             np.floor(df.lat / BLOCK_DEG).astype(int).astype(str))
    blocks = np.array(sorted(block.unique()))
    rng.shuffle(blocks)
    n_test = max(1, int(len(blocks) * test_fraction))
    n_val = max(1, int(len(blocks) * val_fraction))
    test_b = set(blocks[:n_test])
    val_b = set(blocks[n_test:n_test + n_val])

    is_test = block.isin(test_b)
    is_val = block.isin(val_b)
    return df[~(is_test | is_val)], df[is_val], df[is_test]


def fit_scaler(train_df):
    """Fit standardization stats on TRAIN ONLY (features and target)."""
    s = {"x_mean": {}, "x_std": {}}
    for c in INPUT_COLUMNS:
        s["x_mean"][c] = float(train_df[c].mean())
        s["x_std"][c] = float(train_df[c].std()) or 1.0
    s["y_mean"] = float(train_df[TARGET_COLUMN].mean())
    s["y_std"] = float(train_df[TARGET_COLUMN].std()) or 1.0
    return s


def save_scaler(scaler, path):
    with open(path, "w") as f:
        json.dump(scaler, f, indent=2)


def load_scaler(path):
    with open(path) as f:
        return json.load(f)


def transform(df, scaler, scale_y=False):
    """Return (X, y) float32 arrays in INPUT_COLUMNS order.

    X is always standardized. y is raw Kelvin unless scale_y=True (used for NN training;
    invert with `inverse_y`).
    """
    X = np.column_stack([
        (df[c].to_numpy(dtype="float64") - scaler["x_mean"][c]) / scaler["x_std"][c]
        for c in INPUT_COLUMNS
    ]).astype("float32")
    y = df[TARGET_COLUMN].to_numpy(dtype="float64")
    if scale_y:
        y = (y - scaler["y_mean"]) / scaler["y_std"]
    return X, y.astype("float32")


def inverse_y(y_scaled, scaler):
    """Map standardized predictions back to Kelvin."""
    return np.asarray(y_scaled) * scaler["y_std"] + scaler["y_mean"]
