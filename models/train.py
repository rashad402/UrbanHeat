"""PINN training loop (plan §9).

The physics weight is ramped from 0 to `lambda_max` so the network first learns the data and
only then has the SEB constraint tightened — starting with full physics weight tends to trap
the optimiser in a physically consistent but inaccurate solution.

`lambda_max = 0` reproduces the plain MLP baseline exactly (same architecture, same seed),
which is what makes the comparison clean.
"""

import numpy as np
import torch

from .pinn import PINN, composite_loss


def lambda_schedule(epoch, maximum, ramp_epochs, start=0.0):
    """Linearly ramp the physics weight from `start` to `maximum` over `ramp_epochs`."""
    if ramp_epochs <= 0 or epoch >= ramp_epochs:
        return maximum
    return start + (maximum - start) * (epoch / ramp_epochs)


def _phys_tensors(df, device):
    """Raw (unstandardised) physical drivers needed by the SEB residual."""
    return {c: torch.tensor(df[c].to_numpy(dtype="float64"), dtype=torch.float32, device=device)
            for c in ["ndvi", "albedo", "s_down", "t_air", "rh", "wind"]}


def train_pinn(X_train, y_train_scaled, df_train,
               X_val, y_val_scaled, df_val,
               scaler, lambda_max=0.1, ramp_epochs=40, epochs=200, batch_size=4096,
               lr=1e-3, patience=20, seed=42, device=None, verbose=True,
               X_aug=None, y_aug_scaled=None, df_aug=None, aug_weight=1.0):
    """Train the PINN. Selection is on validation DATA loss, so accuracy is never traded away
    silently — the physics term shapes the solution but does not choose the checkpoint.

    PHYSICS AUGMENTATION (optional). Passing X_aug/y_aug_scaled/df_aug appends the anchored
    counterfactual samples of models/synthetic.py to the training pool, at `aug_weight` in the
    data term. They give the DATA term supervision in the high-albedo region the satellite
    record never visits, which is what lets a small lambda extrapolate correctly — see
    models/synthetic.py and scripts/run_augmented.py.

    VALIDATION STAYS MEASURED. df_val is untouched, so early stopping and every reported metric
    are still judged on real observations only.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = PINN(in_dim=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xt = torch.tensor(X_train, device=device)
    yt = torch.tensor(y_train_scaled, device=device).view(-1, 1)
    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val_scaled, device=device).view(-1, 1)
    pt = _phys_tensors(df_train, device)
    pv = _phys_tensors(df_val, device)
    wt = torch.ones(len(Xt), device=device)

    if X_aug is not None and len(X_aug):
        Xt = torch.cat([Xt, torch.tensor(X_aug, device=device)])
        yt = torch.cat([yt, torch.tensor(y_aug_scaled, device=device).view(-1, 1)])
        pa = _phys_tensors(df_aug, device)
        pt = {k: torch.cat([v, pa[k]]) for k, v in pt.items()}
        wt = torch.cat([wt, torch.full((len(X_aug),), float(aug_weight), device=device)])
        if verbose:
            print(f"    + {len(X_aug):,} physics-generated samples at weight {aug_weight:g}")

    n = len(Xt)
    best, best_state, bad = float("inf"), None, 0
    history = []

    for ep in range(epochs):
        lam = lambda_schedule(ep, lambda_max, ramp_epochs)
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_phys = {k: v[idx] for k, v in pt.items()}
            opt.zero_grad()
            total, _, _, _ = composite_loss(model(Xt[idx]), yt[idx], batch_phys, scaler, lam,
                                            weights=wt[idx])
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            _, vdata, vphys, vres = composite_loss(model(Xv), yv, pv, scaler, lam)
            vdata, vphys, vres = float(vdata), float(vphys), float(vres)
        history.append({"epoch": ep, "lambda": lam, "val_data": vdata,
                        "val_physics": vphys, "val_abs_residual": vres})

        if vdata < best - 1e-5:
            best, bad = vdata, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop at epoch {ep} (best val data MSE {best:.5f})")
                break
        if verbose and ep % 25 == 0:
            print(f"    epoch {ep:3d}  lambda {lam:.3f}  val data {vdata:.5f} "
                  f" |residual| {vres:7.1f} W/m2")

    if best_state:
        model.load_state_dict(best_state)
    return model, history


def predict(model, X, scaler, device=None):
    """Return predicted LST in KELVIN."""
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, device=device)).cpu().numpy().ravel()
    return out * scaler["y_std"] + scaler["y_mean"]
