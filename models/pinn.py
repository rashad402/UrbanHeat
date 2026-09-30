"""Physics-Informed Neural Network core (plan §9).

A point-wise MLP surrogate: inputs (x, y, NDVI, NDBI, albedo, S_down, T_air, RH, wind) -> LST.
Trained with a composite loss:

    L = data_mse  +  lambda * physics_mse
      = mean((T_pred - T_landsat)^2)  +  lambda * mean(seb_residual(T_pred)^2)

Recommendation: pure PyTorch (the SEB residual is algebraic, no derivatives needed).
Use DeepXDE only if a diffusion/PDE term is added later. Config: model_config.yaml.
"""

import torch
import torch.nn as nn

from .sebal import seb_residual


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
        """x: [N, in_dim] normalized features -> T_pred [N, 1] (Kelvin, denormalized upstream)."""
        return self.net(x)


def composite_loss(t_pred, t_true, features, lambda_physics):
    """Data MSE + lambda * SEB-residual MSE. Returns (total, data_loss, physics_loss)."""
    data_loss = torch.mean((t_pred - t_true) ** 2)
    residual = seb_residual(t_pred, features)
    physics_loss = torch.mean(residual ** 2)
    total = data_loss + lambda_physics * physics_loss
    return total, data_loss, physics_loss
