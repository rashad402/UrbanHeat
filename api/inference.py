"""Model loading and ward-level batch inference for the API (plan §11)."""


def load_model(checkpoint_path, scaler_path):
    """Load the trained PINN + feature scaler. Returns an object with .predict(features)."""
    raise NotImplementedError


def ward_features(ward_ids):
    """Fetch the base feature rows for the given wards (from the processed dataset)."""
    raise NotImplementedError


def infer_delta_t(model, ward_ids, intervention, scaler):
    """Run base + counterfactual inference; return per-ward {t_base, t_new, delta_t}."""
    raise NotImplementedError
