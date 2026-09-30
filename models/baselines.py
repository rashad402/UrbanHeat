"""Baseline LST regressors: Random Forest, XGBoost, CNN (plan §7).

Trained on the SAME feature matrix as the PINN for a controlled comparison, and used as a
data sanity gate: if baselines can't reach reasonable LST R^2 (~0.7+), fix the data/features
before building the PINN. RF/XGB use the point table; the CNN uses image patches (gridded).
"""


def train_random_forest(X_train, y_train, **kwargs):
    raise NotImplementedError


def train_xgboost(X_train, y_train, **kwargs):
    raise NotImplementedError


def train_cnn(patches_train, y_train, **kwargs):
    """CNN operates on HxW feature patches rather than point vectors (needs gridded input)."""
    raise NotImplementedError
