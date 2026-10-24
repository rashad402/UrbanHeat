"""Robustness of the model comparison: random seeds, extrapolation to unseen brightness, the future.

  python scripts/run_robustness.py seeds [--seeds 1 2 3 4]   (~100 min on 8 CPU cores)
  python scripts/run_robustness.py extrapolation              (~25 min)
  python scripts/run_robustness.py temporal                   (~25 min)
  python scripts/run_robustness.py report                     (seconds; tables + figure)

Each experiment is resumable: results are appended to docs/figures/robustness/<experiment>.json
after every model, and a model already there is skipped. Checkpoints go to
models/checkpoints/robustness/<experiment>/ and nothing existing is overwritten.

PROTOCOL (fixed before running; the metrics are those of models/response_eval.py)
  seeds          The standard spatial-block split stays FIXED; only training randomness changes:
                 weight initialisation, batch order, and which physics-generated samples are drawn.
                 Seed 42 is the existing checkpoints, seeds 1-4 are new. Reported as mean and SD, so
                 a difference between two models can be compared with how much one model varies.
  extrapolation  Every pixel with albedo at or above the 90th percentile is removed from training
                 and validation. The test set is those bright pixels inside held-out blocks, unseen
                 in place AND in brightness, which is the situation a cool-roof counterfactual is in.
                 Primary: RMSE and bias on bright pixels, against the same model's in-range RMSE.
  temporal       Train and validate on 2015-2023 scenes; test on every 2024-2026 scene.
                 Primary: R2, RMSE, bias and the observed-change tiers on the future scenes.
  Every network uses the recipe of scripts/run_augmented.py: 150 epochs, lambda ramped over 40
  epochs, and for augmented arms 60,000 physics-generated samples at weight 0.5.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import features as F                                     # noqa: E402
from models import response_eval as R                                # noqa: E402
from models import synthetic as SY                                   # noqa: E402
from models.baselines import train_xgboost                           # noqa: E402
from models.pinn import PINN                                         # noqa: E402
from models.train import train_pinn, predict as nn_predict           # noqa: E402

PARQUET = "data/processed/kochi_samples.parquet"
CKPT = "models/checkpoints"
OUT = "docs/figures/robustness"
ROBUST_CKPT = f"{CKPT}/robustness"
SPLIT_SEED = 42
EPOCHS, N_AUG, AUG_WEIGHT = 150, 60_000, 0.5

ARMS = [   # label, lambda, augmented, seed-42 checkpoint trained with this same recipe
    ("MLP (no physics)", 0.0, False, "pinn_plain_lambda0.pt"),
    ("PINN lam=0.5", 0.5, False, "pinn_plain_lambda0.5.pt"),
    ("MLP + aug", 0.0, True, "pinn_aug_lambda0.pt"),
    ("PINN + aug lam=0.1", 0.1, True, "pinn_aug_lambda0.1.pt"),
    ("PINN + aug lam=0.2", 0.2, True, "pinn_aug_lambda0.2.pt"),
    ("PINN + aug lam=0.5", 0.5, True, "pinn_aug_lambda0.5.pt"),
]


def std_X(feats, scaler):
    return np.column_stack([(np.asarray(feats[c], dtype="float64") - scaler["x_mean"][c]) / scaler["x_std"][c]
                            for c in F.INPUT_COLUMNS]).astype("float32")


def nn_predictor(model, scaler):
    return lambda feats: nn_predict(model, std_X(feats, scaler), scaler)


def load_results(exp):
    path = f"{OUT}/{exp}.json"
    return json.load(open(path)) if os.path.exists(path) else {}


def save_results(exp, results):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{exp}.json", "w") as fh:
        json.dump(results, fh, indent=2)


class Trainer:
    """Trains every arm on one split, with one training seed, and scores it on the given test sets."""

    def __init__(self, exp, tr, va, tests, df, seed):
        self.exp, self.tr, self.va, self.tests, self.df, self.seed = exp, tr, va, tests, df, seed
        self.scaler = F.fit_scaler(tr)
        self.Xtr, _ = F.transform(tr, self.scaler)
        self.Xva, _ = F.transform(va, self.scaler)
        _, self.ytr_s = F.transform(tr, self.scaler, scale_y=True)
        _, self.yva_s = F.transform(va, self.scaler, scale_y=True)
        self._aug = None
        os.makedirs(f"{ROBUST_CKPT}/{exp}", exist_ok=True)

    def aug(self):
        if self._aug is None:
            a = SY.make_anchors(self.tr, n_samples=N_AUG, seed=self.seed)
            Xa, _ = F.transform(a, self.scaler)
            _, ya = F.transform(a, self.scaler, scale_y=True)
            self._aug = (a, Xa, ya)
        return self._aug

    def score(self, predict):
        out = {}
        for name, (test_df, full) in self.tests.items():
            if full:
                out[name] = R.scorecard(predict, test_df, self.df, F.INPUT_COLUMNS)
            else:   # accuracy only, for the in-range comparison set
                y = test_df["lst"].to_numpy(dtype="float64")
                p = predict({c: test_df[c].to_numpy() for c in F.INPUT_COLUMNS})
                out[name] = {"absolute": {"rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                                          "bias": float(np.mean(p - y))}}
        return out

    def xgboost(self):
        Xtr, ytr = F.transform(self.tr, self.scaler)
        Xva, yva = F.transform(self.va, self.scaler)
        m = train_xgboost(Xtr, ytr, Xva, yva, seed=self.seed)
        return lambda feats: m.predict(std_X(feats, self.scaler)).astype("float64")

    def network(self, lam, use_aug, reuse=None):
        if reuse:   # an existing seed-42 checkpoint trained with this exact recipe and split
            model = PINN(in_dim=len(F.INPUT_COLUMNS))
            model.load_state_dict(torch.load(f"{CKPT}/{reuse}", map_location="cpu"))
            model.eval()
            return nn_predictor(model, self.scaler)
        kw = {}
        if use_aug:
            a, Xa, ya = self.aug()
            kw = dict(X_aug=Xa, y_aug_scaled=ya, df_aug=a, aug_weight=AUG_WEIGHT)
        model, _ = train_pinn(self.Xtr, self.ytr_s, self.tr, self.Xva, self.yva_s, self.va, self.scaler,
                              lambda_max=lam, epochs=EPOCHS, seed=self.seed, verbose=False, **kw)
        slug = f"{'aug' if use_aug else 'plain'}_lambda{lam:g}_s{self.seed}"
        torch.save(model.state_dict(), f"{ROBUST_CKPT}/{self.exp}/pinn_{slug}.pt")
        return nn_predictor(model, self.scaler)


def run(exp, trainer, results, key_suffix, reuse_seed42=False):
    jobs = [("XGBoost", None, None, None)] + [(l, lam, aug, ck) for l, lam, aug, ck in ARMS]
    for label, lam, use_aug, ck in jobs:
        key = f"{label}{key_suffix}"
        if key in results:
            print(f"  {key:<34} done earlier, skipped", flush=True)
            continue
        t0 = time.time()
        predict = (trainer.xgboost() if label == "XGBoost"
                   else trainer.network(lam, use_aug, reuse=ck if reuse_seed42 else None))
        results[key] = trainer.score(predict)
        save_results(exp, results)
        first = next(iter(results[key].values()))
        a, cf = first["absolute"], first.get("counterfactual", {})
        c = first.get("change", {}).get("C_surface", {})
        print(f"  {key:<34} R2 {a.get('r2', float('nan')):6.3f} RMSE {a['rmse']:.2f} bias {a['bias']:+.2f} "
              f"| C r {c.get('r', float('nan')):.3f} | worst ward {cf.get('worst_ward', float('nan')):+.2f} "
              f"| {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["seeds", "extrapolation", "temporal", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--smoke", metavar="DIR", help="2-epoch dry run writing only into DIR")
    args = ap.parse_args()
    if args.smoke:
        global EPOCHS, N_AUG, OUT, ROBUST_CKPT
        EPOCHS, N_AUG, OUT, ROBUST_CKPT = 2, 3000, f"{args.smoke}/out", f"{args.smoke}/ckpt"
    if args.experiment == "report":
        return report()

    torch.set_num_threads(4)          # two experiments share the 8 cores
    df = F.load_samples(PARQUET)
    exp = args.experiment
    results = load_results(exp)
    print(f"== {exp} ==", flush=True)

    if exp == "seeds":
        tr, va, te = F.make_splits(df, policy="spatial_block", seed=SPLIT_SEED)
        te = te.reset_index(drop=True)
        for seed in [42] + [s for s in args.seeds if s != 42]:
            print(f"-- training seed {seed} --", flush=True)
            run(exp, Trainer(exp, tr, va, {"test": (te, True)}, df, seed), results,
                f" | seed {seed}", reuse_seed42=(seed == 42))
    elif exp == "extrapolation":
        tr, va, te_x, te_in, cut = F.split_feature_extrapolation(df, "albedo", 0.90, seed=SPLIT_SEED)
        print(f"albedo >= {cut:.3f} held out: train {len(tr):,} | bright test {len(te_x):,} | "
              f"in-range test {len(te_in):,}", flush=True)
        tests = {"bright (extrapolated)": (te_x.reset_index(drop=True), True),
                 "in range": (te_in.reset_index(drop=True), False)}
        run(exp, Trainer(exp, tr, va, tests, df, SPLIT_SEED), results, "")
    elif exp == "temporal":
        tr, va, te = F.split_temporal(df, test_from_year=2024, seed=SPLIT_SEED)
        print(f"train 2015-2023 {len(tr):,} | val {len(va):,} | test 2024-2026 {len(te):,}", flush=True)
        run(exp, Trainer(exp, tr, va, {"future": (te.reset_index(drop=True), True)}, df, SPLIT_SEED),
            results, "")


# ─────────────────────────────── report ───────────────────────────────
METRICS = [("absolute R2", ("absolute", "r2")), ("RMSE K", ("absolute", "rmse")),
           ("real-change r", ("change", "C_surface", "r")), ("worst ward K", ("counterfactual", "worst_ward")),
           ("wards cool %", ("counterfactual", "wards_cooling_pct")), ("steady %", ("counterfactual", "monotone_pct"))]


def _get(d, path):
    for k in path:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if isinstance(d, (int, float)) else float("nan")


def report():
    labels = ["XGBoost"] + [a[0] for a in ARMS]
    summary = {}

    seeds = load_results("seeds")
    if seeds:
        print("\nSEEDS: mean ± SD over training seeds (fixed spatial-block split)")
        print(f"{'model':<22}{'n':>3}" + "".join(f"{m:>18}" for m, _ in METRICS))
        summary["seeds"] = {}
        for lab in labels:
            runs = [v["test"] for k, v in seeds.items() if k.split(" | ")[0] == lab]
            if not runs:
                continue
            row = {}
            for name, path in METRICS:
                vals = np.array([_get(r, path) for r in runs])
                row[name] = (float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0)
            summary["seeds"][lab] = {"n": len(runs), **row}
            print(f"{lab:<22}{len(runs):>3}" + "".join(f"{row[m][0]:>11.3f} ±{row[m][1]:<5.3f}" for m, _ in METRICS))

    for exp, test_name in (("extrapolation", "bright (extrapolated)"), ("temporal", "future")):
        res = load_results(exp)
        if not res:
            continue
        print(f"\n{exp.upper()}: scored on '{test_name}'")
        extra = " in-range RMSE  bias(bright)" if exp == "extrapolation" else "        bias"
        print(f"{'model':<22}" + "".join(f"{m:>15}" for m, _ in METRICS) + extra)
        summary[exp] = {}
        for lab in labels:
            if lab not in res:
                continue
            r = res[lab][test_name]
            row = {m: _get(r, p) for m, p in METRICS}
            row["bias"] = _get(r, ("absolute", "bias"))
            if exp == "extrapolation":
                row["in-range RMSE"] = _get(res[lab]["in range"], ("absolute", "rmse"))
            summary[exp][lab] = row
            tail = (f"{row['in-range RMSE']:>15.3f}{row['bias']:>+13.3f}" if exp == "extrapolation"
                    else f"{row['bias']:>+12.3f}")
            print(f"{lab:<22}" + "".join(f"{row[m]:>15.3f}" for m, _ in METRICS) + tail)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    _plot(summary, labels)
    print(f"\nwrote {OUT}/summary.json and {OUT}/robustness.png")


def _plot(summary, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    x = np.arange(len(labels))
    s = summary.get("seeds", {})
    if s:
        for k, (m, off) in enumerate((("absolute R2", -0.2), ("real-change r", 0.2))):
            mu = [s.get(l, {}).get(m, (np.nan, 0))[0] for l in labels]
            sd = [s.get(l, {}).get(m, (np.nan, 0))[1] for l in labels]
            ax[0].bar(x + off, mu, 0.4, yerr=sd, capsize=3, label=f"{m} (mean ± SD)")
        ax[0].set_title(f"Seed variability: {max(v['n'] for v in s.values())} training seeds per model")
        ax[0].legend(fontsize=8)
    e = summary.get("extrapolation", {})
    if e:
        ax[1].bar(x - 0.2, [e.get(l, {}).get("in-range RMSE", np.nan) for l in labels], 0.4, label="in-range pixels")
        ax[1].bar(x + 0.2, [e.get(l, {}).get("RMSE K", np.nan) for l in labels], 0.4, label="bright pixels never seen")
        ax[1].set_ylabel("RMSE (K)")
        ax[1].set_title("Extrapolation to unseen brightness (albedo ≥ p90)")
        ax[1].legend(fontsize=8)
    t = summary.get("temporal", {})
    if t:
        ax[2].bar(x - 0.2, [s.get(l, {}).get("absolute R2", (np.nan,))[0] for l in labels], 0.4, label="same years (seed mean)")
        ax[2].bar(x + 0.2, [t.get(l, {}).get("absolute R2", np.nan) for l in labels], 0.4, label="future scenes 2024-26")
        ax[2].set_ylabel("R²")
        ax[2].set_title("Temporal holdout: trained on 2015-23")
        ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xticks(x, labels, fontsize=8, rotation=35, ha="right")
        a.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/robustness.png", dpi=130)


if __name__ == "__main__":
    main()
