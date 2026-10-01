"""Baseline LST regressors (plan §7).

Trained on the SAME feature matrix as the PINN for a controlled comparison, and used as a
DATA SANITY GATE: if these can't reach reasonable LST R^2 (~0.7), the data/features are wrong
and must be fixed before building the PINN.

Models:
  * Random Forest   - non-linear, no feature scaling needed
  * XGBoost         - gradient boosting, usually the strongest tabular baseline
  * MLP             - SAME architecture as the planned PINN (6x128 SiLU) but with DATA LOSS
                      ONLY. This is the critical ablation: PINN minus the physics term, so the
                      PINN-vs-MLP gap isolates exactly what the SEB physics loss buys.

NOTE ON THE CNN: the proposal lists a CNN baseline. A CNN needs gridded image patches, but our
dataset is a per-pixel table, so it requires a separate GEE chip export. It is deferred — the
MLP above is the more informative ablation for the physics-loss claim, since it matches the
PINN architecture exactly. Add the CNN later for completeness against the proposal.
"""

import numpy as np
import torch
import torch.nn as nn


def train_random_forest(X_train, y_train, seed=42, **kw):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=kw.get("n_estimators", 300),
                              max_depth=kw.get("max_depth", None),
                              min_samples_leaf=kw.get("min_samples_leaf", 2),
                              n_jobs=-1, random_state=seed)
    m.fit(X_train, y_train)
    return m


def train_xgboost(X_train, y_train, X_val=None, y_val=None, seed=42, **kw):
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=kw.get("n_estimators", 600),
                     max_depth=kw.get("max_depth", 8),
                     learning_rate=kw.get("learning_rate", 0.05),
                     subsample=0.8, colsample_bytree=0.8,
                     early_stopping_rounds=40 if X_val is not None else None,
                     n_jobs=-1, random_state=seed, tree_method="hist")
    if X_val is not None:
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        m.fit(X_train, y_train)
    return m


class MLP(nn.Module):
    """Same shape as the planned PINN (see models/pinn.py) — data loss only."""

    def __init__(self, in_dim=9, hidden_layers=6, hidden_units=128, activation="silu"):
        super().__init__()
        act = {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        layers = [nn.Linear(in_dim, hidden_units), act()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), act()]
        layers += [nn.Linear(hidden_units, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, seed=42, epochs=200, batch_size=4096,
              lr=1e-3, patience=20, device=None, verbose=True):
    """Train the PINN-shaped MLP on standardized features + standardized target."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xt = torch.tensor(X_train, device=device)
    yt = torch.tensor(y_train, device=device).view(-1, 1)
    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val, device=device).view(-1, 1)

    n = len(Xt)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = torch.mean((model(Xt[idx]) - yt[idx]) ** 2)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vloss = float(torch.mean((model(Xv) - yv) ** 2))
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop at epoch {ep} (best val MSE {best:.5f})")
                break
        if verbose and ep % 25 == 0:
            print(f"    epoch {ep:3d}  val MSE {vloss:.5f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def mlp_predict(model, X, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, device=device)).cpu().numpy().ravel()
