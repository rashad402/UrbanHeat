"""PINN training loop (plan §9).

Usage:
    python models/train.py --config configs/model_config.yaml

Handles: feature loading, scaler fit/persist, lambda ramp for the physics term,
Adam + cosine LR, early stopping on validation data-loss, best-checkpoint saving,
and per-component loss logging (data vs physics — log them separately).
"""

import argparse


def lambda_schedule(epoch, start, maximum, ramp_epochs):
    """Linearly ramp the physics weight from `start` to `maximum` over `ramp_epochs`."""
    if epoch >= ramp_epochs:
        return maximum
    return start + (maximum - start) * (epoch / max(1, ramp_epochs))


def train(config_path):
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_config.yaml")
    args = ap.parse_args()
    train(args.config)
