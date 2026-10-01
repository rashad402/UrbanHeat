"""CNN baseline over gridded chips (plan §7) — the one baseline the proposal still lacked.

ARCHITECTURE
    Two inputs, because the dataset has two kinds of predictor:

      image head   NDVI / NDBI / albedo / validity mask over a patch x patch window, through
                   three Conv-SiLU blocks and global average pooling. This is the part the
                   tabular models cannot express: the NEIGHBOURHOOD of a pixel.
      scalar head  lon, lat and the per-scene ERA5 drivers (S_down, T_air, RH, wind), which are
                   constant across a chip. Feeding them as image planes would spend convolutions
                   on constants, so they bypass the stack and join at the fusion layer.

    Capacity is matched to models/pinn.PINN as closely as two different architectures allow
    (~128 wide, SiLU, same optimiser and schedule), so a win is attributable to spatial context
    rather than to parameter count.

WHAT THIS BASELINE IS FOR
    It tests whether neighbourhood context adds predictive signal over point-wise features. It
    is NOT a candidate for deployment: the planner needs counterfactuals, and a CNN trained on
    data alone inherits exactly the land-cover confound that makes the plain MLP predict cool
    roofs warm the city. Its counterfactuals are reported for that reason — to show that more
    capacity does not fix a problem that is about missing data, not missing flexibility.
"""

import numpy as np
import torch
import torch.nn as nn


class ChipCNN(nn.Module):
    def __init__(self, in_channels=4, n_scalars=6, width=64, fusion=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.scalar = nn.Sequential(nn.Linear(n_scalars, fusion), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(width * 2 + fusion, fusion), nn.SiLU(),
            nn.Linear(fusion, fusion), nn.SiLU(),
            nn.Linear(fusion, 1),
        )

    def forward(self, img, scalars):
        z = self.conv(img).flatten(1)
        return self.head(torch.cat([z, self.scalar(scalars)], dim=1))


def train_cnn(Xi_tr, Xs_tr, y_tr_scaled, Xi_va, Xs_va, y_va_scaled,
              epochs=60, batch_size=256, lr=1e-3, patience=10, seed=42,
              device=None, verbose=True):
    """Train the chip CNN on standardised inputs and a standardised target.

    Selection is on validation MSE, matching models/train.train_pinn so the comparison is clean.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = ChipCNN(in_channels=Xi_tr.shape[1], n_scalars=Xs_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ti = torch.tensor(Xi_tr, device=device)
    ts = torch.tensor(Xs_tr, device=device)
    ty = torch.tensor(y_tr_scaled, device=device).view(-1, 1)
    vi = torch.tensor(Xi_va, device=device)
    vs = torch.tensor(Xs_va, device=device)
    vy = torch.tensor(y_va_scaled, device=device).view(-1, 1)

    n = len(ti)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = torch.mean((model(ti[idx], ts[idx]) - ty[idx]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vloss = float(torch.mean((model(vi, vs) - vy) ** 2))
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop at epoch {ep} (best val MSE {best:.5f})")
                break
        if verbose and ep % 10 == 0:
            print(f"    epoch {ep:3d}  val MSE {vloss:.5f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_cnn(model, Xi, Xs, scaler, batch_size=512, device=None):
    """Return predicted LST in KELVIN."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xi), batch_size):
            img = torch.tensor(Xi[i:i + batch_size], device=device)
            sc = torch.tensor(Xs[i:i + batch_size], device=device)
            out.append(model(img, sc).cpu().numpy().ravel())
    return np.concatenate(out) * scaler["y_std"] + scaler["y_mean"]
