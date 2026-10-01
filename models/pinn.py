"""Physics-Informed Neural Network core (plan §9).

A point-wise MLP: inputs (lon, lat, NDVI, NDBI, albedo, S_down, T_air, RH, wind) -> LST.
Identical architecture to models/baselines.MLP, so the only difference between them is the
physics term — the PINN-vs-MLP gap isolates exactly what the SEB loss buys.

    L = mean((T_pred - T_obs)^2)_standardised  +  lambda * mean((residual / FLUX_SCALE)^2)

The residual comes from models/sebal.seb_residual and is in W m^-2, which is O(100) while the
standardised data term is O(0.1). Dividing by FLUX_SCALE puts both terms near O(1) so lambda is
an interpretable weight rather than a tiny fudge factor.

Pure PyTorch rather than DeepXDE: the SEB residual is algebraic (no spatial/temporal
derivatives), so DeepXDE's PDE machinery would add dependency without doing any work here.
"""

import torch
import torch.nn as nn

from .sebal import seb_residual

FLUX_SCALE = 100.0          # W m^-2 — normalises the physics term to O(1)


class PINN(nn.Module):
    def __init__(self, in_dim=9, hidden_layers=6, hidden_units=128, activation="silu"):
        super().__init__()
        act = {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        layers = [nn.Linear(in_dim, hidden_units), act()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), act()]
        layers += [nn.Linear(hidden_units, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: [N, in_dim] standardised features -> standardised LST [N, 1]."""
        return self.net(x)


def physics_residual(t_pred_kelvin, phys):
    """SEB residual [W m^-2] for predicted temperatures given raw (unstandardised) drivers.

    `phys` is a dict of tensors: ndvi, albedo, s_down, t_air, rh, wind — in physical units.
    """
    return seb_residual(t_pred_kelvin,
                        ndvi=phys["ndvi"], albedo=phys["albedo"], s_down=phys["s_down"],
                        t_air=phys["t_air"], rh=phys["rh"], wind=phys["wind"])


def composite_loss(pred_scaled, target_scaled, phys, scaler, lambda_physics, weights=None):
    """Data MSE (standardised) + lambda * normalised SEB-residual MSE.

    `weights` is an optional per-sample weight on the DATA term only, used to mix measured
    pixels with the physics-generated samples of models/synthetic.py at a lower weight. The
    physics term is unweighted: the SEB residual is meaningful for every sample regardless of
    where its label came from.

    Returns (total, data_loss, physics_loss, mean_abs_residual_W_m2).
    """
    sq = (pred_scaled - target_scaled) ** 2
    if weights is None:
        data_loss = torch.mean(sq)
    else:
        w = weights.view(-1, 1)
        data_loss = torch.sum(w * sq) / torch.clamp(torch.sum(w), min=1e-8)

    # back to Kelvin so the physics sees real temperatures
    t_pred_k = pred_scaled.squeeze(-1) * scaler["y_std"] + scaler["y_mean"]
    res = physics_residual(t_pred_k, phys)
    physics_loss = torch.mean((res / FLUX_SCALE) ** 2)

    total = data_loss + lambda_physics * physics_loss
    return total, data_loss, physics_loss, res.abs().mean().detach()
