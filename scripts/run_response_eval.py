"""Score every model on the response-centred protocol of models/response_eval.py.

Run:  python scripts/run_response_eval.py        (~2 min; retrains XGBoost, loads the rest)

One scorecard, three families of evidence, for every model on the SAME held-out pixels:
  1. absolute LST      R2, RMSE, and R2 split into between-date and within-date skill
  2. observed change   Tiers A/B/C: does the model predict how a pixel's LST actually changed
                       when its weather, its neighbourhood or its own surface changed?
  3. counterfactual    the ward-level cool-roof test: share of the 74 wards that cool, the worst
                       ward, and whether cooling rises steadily as albedo is raised.

The SEB physics on its own (no network) is included as a reference: it shows how much of the
response the physics alone gets right, which is what the PINN's loss is pulling towards.

Writes docs/figures/response_eval.json and docs/figures/response_eval.png. Nothing is saved over
the deployed checkpoint or its scaler (scripts/run_baselines.py would do that; this does not).
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.inference import PinnModel                                  # noqa: E402
from models import features as F                                     # noqa: E402
from models import response_eval as R                                # noqa: E402
from models.baselines import train_xgboost                           # noqa: E402
from models.evaluate import regression_metrics                       # noqa: E402
from models.sebal import equilibrium_temperature                     # noqa: E402

PARQUET = "data/processed/kochi_samples.parquet"
CKPT = "models/checkpoints"
FIGS = "docs/figures"
SEED = 42

MODELS = [   # label, checkpoint (None = built here)
    ("XGBoost", None),
    ("MLP (no physics)", "mlp_lambda0.pt"),
    ("PINN lam=0.1", "pinn_lambda0.1.pt"),
    ("PINN lam=0.2", "pinn_plain_lambda0.2.pt"),
    ("PINN lam=0.5 (deployed)", "pinn_lambda0.5.pt"),
    ("PINN lam=2", "pinn_lambda2.pt"),
    ("MLP + aug", "pinn_aug_lambda0.pt"),
    ("PINN + aug lam=0.1", "pinn_aug_lambda0.1.pt"),
    ("PINN + aug lam=0.2", "pinn_aug_lambda0.2.pt"),
    ("PINN + aug lam=0.5", "pinn_aug_lambda0.5.pt"),
    ("SEB physics only", "physics"),
]
PHYS = ["ndvi", "albedo", "s_down", "t_air", "rh", "wind"]


def ward_counterfactual(predict, df):
    """The cool-roof ward test; the single definition lives in models/response_eval.py."""
    return R.ward_counterfactual(predict, df, F.INPUT_COLUMNS)


def main():
    t0 = time.time()
    df = F.load_samples(PARQUET)
    tr, va, te = F.make_splits(df, policy="spatial_block", seed=SEED)
    te = te.reset_index(drop=True)
    print(f"test pixels {len(te):,}; pairs: ", end="")
    i1, _ = R.build_pairs(te)
    print(f"{len(i1):,} consecutive same-pixel pairs")

    # The checkpoints were trained with scaler.json; confirm it is the train-split scaler, so the
    # XGBoost baseline built here sees identically standardised inputs.
    saved = F.load_scaler(f"{CKPT}/scaler.json")
    fresh = F.fit_scaler(tr)
    drift = max(abs(saved["x_mean"][c] - fresh["x_mean"][c]) / saved["x_std"][c] for c in F.INPUT_COLUMNS)
    print(f"scaler check: max mean drift {drift:.2e} sd ({'same split' if drift < 1e-6 else 'DIFFERENT'})")

    def std_X(feats):
        return np.column_stack([(np.asarray(feats[c], dtype="float64") - saved["x_mean"][c]) / saved["x_std"][c]
                                for c in F.INPUT_COLUMNS]).astype("float32")

    predictors = {}
    for label, ck in MODELS:
        if ck is None:
            Xtr, ytr = F.transform(tr, saved)
            Xva, yva = F.transform(va, saved)
            print("training XGBoost on the train split ...", flush=True)
            xgb = train_xgboost(Xtr, ytr, Xva, yva, seed=SEED)
            predictors[label] = lambda f, m=xgb: m.predict(std_X(f)).astype("float64")
        elif ck == "physics":
            predictors[label] = lambda f: equilibrium_temperature(*[np.asarray(f[c], dtype="float64") for c in PHYS])
        else:
            m = PinnModel(f"{CKPT}/{ck}", f"{CKPT}/scaler.json")
            predictors[label] = lambda f, m=m: m.predict_k(f)

    test_feats = {c: te[c].to_numpy() for c in F.INPUT_COLUMNS}
    y = te["lst"].to_numpy(dtype="float64")
    results = {}
    for label, predict in predictors.items():
        p = predict(test_feats)
        absm = regression_metrics(y, p)
        absm.update(R.decompose_r2(y, p, te["date"].to_numpy()))
        tiers = R.tiers(te, p)
        change = {name: R.change_metrics(o, q) for name, (o, q, _) in tiers.items()}
        o, q, b = tiers["C_surface"]
        change["C_surface"]["r_ci"] = R.block_bootstrap(o, q, b, "r", seed=1)
        change["C_surface"]["slope_ci"] = R.block_bootstrap(o, q, b, "slope", seed=2)
        cf = ward_counterfactual(predict, df)
        results[label] = {"absolute": absm, "change": change, "counterfactual": cf}
        c = change["C_surface"]
        print(f"  {label:<24} absR2 {absm['r2']:6.3f} | within {absm['within_date_r2']:6.3f} | "
              f"A r {change['A_total']['r']:.3f} | B r {change['B_local']['r']:.3f} | "
              f"C r {c['r']:.3f} [{c['r_ci'][0]:.2f},{c['r_ci'][1]:.2f}] slope {c['slope']:.2f} sign {c['sign']:.0%} | "
              f"wards cool {cf['wards_cooling_pct']:.0f}% worst {cf['worst_ward']:+.2f} mono {cf['monotone_pct']:.0f}%",
              flush=True)

    meta = {"n_test_pixels": len(te), "n_pairs": int(len(i1)),
            "n_surface_pairs": int(len(R.tiers(te, y)["C_surface"][0])),
            "protocol": {"ndvi_change": R.NDVI_CHANGE, "albedo_change": R.ALBEDO_CHANGE,
                         "sign_min_k": R.SIGN_MIN_K, "split": "spatial_block", "seed": SEED}}
    os.makedirs(FIGS, exist_ok=True)
    with open(f"{FIGS}/response_eval.json", "w") as fh:
        json.dump({"meta": meta, "results": results}, fh, indent=2)
    _plot(results, meta)
    print(f"\n{meta['n_pairs']:,} pairs, {meta['n_surface_pairs']:,} with a material surface change. "
          f"Wrote {FIGS}/response_eval.json and .png in {time.time() - t0:.0f} s")


def _plot(results, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results)
    short = labels                         # full names, rotated: nine of them do not fit flat
    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))

    vals = [[results[l]["absolute"]["r2"] for l in labels],
            [results[l]["absolute"]["within_date_r2"] for l in labels],
            [results[l]["change"]["C_surface"]["r"] for l in labels]]
    for k, (v, name) in enumerate(zip(vals, ["absolute LST R²", "within-date R²", "surface-change r (Tier C)"])):
        ax[0].bar(x + (k - 1) * 0.27, v, 0.27, label=name)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xticks(x, short, fontsize=8, rotation=35, ha="right")
    ax[0].set_title("Old score vs new: what each model gets right")
    ax[0].legend(fontsize=8)

    r = [results[l]["change"]["C_surface"]["r"] for l in labels]
    ci = np.array([results[l]["change"]["C_surface"]["r_ci"] for l in labels])
    ax[1].errorbar(r, x, xerr=[np.array(r) - ci[:, 0], ci[:, 1] - np.array(r)], fmt="o", capsize=3)
    ax[1].set_yticks(x, [l.replace("\n", " ") for l in labels], fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("correlation, predicted vs observed change")
    ax[1].set_title(f"Tier C: {meta['n_surface_pairs']:,} real surface changes (95% block bootstrap)")

    cool = [results[l]["counterfactual"]["wards_cooling_pct"] for l in labels]
    mono = [results[l]["counterfactual"]["monotone_pct"] for l in labels]
    ax[2].bar(x - 0.2, cool, 0.4, label="wards cooling under cool roof")
    ax[2].bar(x + 0.2, mono, 0.4, label="albedo response monotone")
    ax[2].set_xticks(x, short, fontsize=8, rotation=35, ha="right")
    ax[2].set_ylim(0, 105)
    ax[2].set_title("Counterfactual safety (74 wards)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/response_eval.png", dpi=130)


if __name__ == "__main__":
    main()
