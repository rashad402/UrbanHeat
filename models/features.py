"""Feature matrix assembly, normalization, and train/val/test splits (plan §6).

Reads the dataset artifact produced by data_engine/build_dataset.py and produces the
X/y arrays consumed by baselines and the PINN. Splits use a spatial-block policy to
avoid neighbour leakage (a random pixel split inflates R^2 — do not use it).
"""

import json

# Spatial coords use lon/lat directly (normalized during training, so units don't matter).
INPUT_COLUMNS = ["lon", "lat", "ndvi", "ndbi", "albedo", "s_down", "t_air", "rh", "wind"]
TARGET_COLUMN = "lst"


def load_samples(parquet_path):
    """Load kochi_samples.parquet into a DataFrame."""
    raise NotImplementedError


def make_splits(df, policy="spatial_block", test_fraction=0.2, val_fraction=0.1, seed=42):
    """Return (train_df, val_df, test_df) using a leakage-safe split policy."""
    raise NotImplementedError


def fit_scaler(train_df):
    """Fit standardization stats on TRAIN ONLY. Returns dict of per-feature mean/std."""
    raise NotImplementedError


def save_scaler(scaler, path):
    with open(path, "w") as f:
        json.dump(scaler, f, indent=2)


def load_scaler(path):
    with open(path) as f:
        return json.load(f)


def transform(df, scaler):
    """Apply the scaler and return (X, y) arrays in INPUT_COLUMNS order."""
    raise NotImplementedError
