# PIML-UrbanHeat

**A Physics-Informed Neural Framework for Counterfactual Urban Heat Mitigation in Humid-Tropical Coastal Cities**

A physics-constrained decision-support platform that combines satellite remote sensing with
Physics-Informed Neural Networks (PINNs) to predict Land Surface Temperature (LST) and simulate
urban heat mitigation strategies (green roofs, cool roofs, urban greening) for Kerala's coastal cities.

Dept. of Computer Science and Engineering, Government College of Engineering Kannur.
Team: Jesmina E · Judha C K · Muhammad Rashad Hashim · Najih V — Supervisor: Dr. Bindu P V

---

## What it does

1. **Data Engine** — extracts LST (Landsat 8/9), spectral indices (NDVI/NDBI/albedo from Sentinel-2),
   and climate variables (ERA5-Land) via Google Earth Engine; cloud-masks and gap-fills with MODIS.
2. **PINN Core** — an MLP that predicts LST, trained with a composite loss that penalizes both
   data error and Surface Energy Balance (SEB) residual, so predictions stay thermodynamically consistent.
3. **What-If Simulator** — a Next.js + Mapbox web app where planners modify ward-level surface parameters
   and see an instant ΔT cooling heatmap.

See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the full engineering plan and 28-week schedule.

## Repository layout

```
data_engine/   satellite + climate extraction, cloud masking, gap-filling, dataset assembly,
               chip gridding for the CNN
models/        features, baselines (RF/XGB/MLP), CNN, SEBAL physics, PINN, physics-generated
               training samples, training, evaluation
api/           FastAPI services — planner.py (the console), main.py, live_demo.py, inference.py
web/planner/   the planning console (Mapbox GL JS); tiles/ holds the baked thermal overlay
configs/       data & model configuration, AOI + ward boundaries, measured roof shares
tests/         SEB physics, physics augmentation, overlay tiling, and API-level tests (no GEE required)
notebooks/     exploratory only (never a source of reported results)
docs/          final report (LaTeX) and figures
```

## The planning console

`web/planner/` + `api/planner.py` — the decision-support application for municipal planners.
Select wards (or draw a custom area), apply cool roofs / greening, and get the predicted land
surface temperature from the trained PINN.

```bash
uvicorn api.planner:app --port 8080
```

Then open **http://localhost:8080** (first start takes ~15 s: model, dataset and Earth Engine sign-in).

- Baseline LST is **measured** Landsat; the PINN supplies the **response** (ΔT) only.
- Map is **Mapbox GL JS** (satellite basemap) with the Landsat thermal layer as **baked static
  tiles** (`web/planner/tiles/`, see below), so it pans and zooms at any scale and appears
  instantly. If no bake exists the app falls back to live Earth Engine tiles, whose signed URLs
  expire *silently*, so those are re-minted on a timer and via `/api/refresh_tiles`. A static
  server-rendered scene remains as the no-WebGL fallback (`/api/scene`).
- Every selection is analysed **per pixel**, then averaged. The model is non-linear, so averaging
  features first and predicting once is a different (and wrong) number — see `tests/test_api.py`,
  which pins the difference numerically.
- Coverage scaling uses the **measured** per-ward roof share where
  `configs/roof_share_kochi.json` exists (`scripts/build_roof_share.py`), falling back to a
  documented assumption. The slider is an override, not the live value.
- Deployed checkpoint is **λ=0.5**, not the highest-R² one — see `api/inference.py` for why
  (counterfactuals are an extrapolation problem). `scripts/run_augmented.py` tests whether that
  trade-off can be removed entirely.

What a planner can do with it: select wards from the map **or** the keyboard-navigable ward list,
draw a box or an arbitrary **zone**, shade wards by baseline temperature or predicted ΔT, rank all
wards by severity / cooling / cooling-per-km²-treated, save and name scenarios (persisted in the
browser), compare two side by side, and export CSV or print a one-page sheet.

### Baking the thermal overlay

The overlay is a per-pixel **median** of land surface temperature over every usable dry-season
Landsat 8/9 scene (94 of them). Served live, Earth Engine recomputes that median tile by tile on
every first view (measured: 12.6 s per tile on average, up to 24 s), so the layer filled in slowly and
needed an Earth Engine login just to draw. The scenes are history, so it is computed **once**:

```bash
python scripts/bake_overlay.py            # ~1 min; writes web/planner/tiles/ (~5 MB, committed)
python scripts/bake_overlay.py --reuse    # re-tile the last export without Earth Engine
```

Re-run it when new scenes should enter the median (monthly is plenty) or when the colour range,
palette, extent or zoom range change, since colours are baked into the tiles. The tile maths is in
`data_engine/tiling.py` and the single definition of the layer, shared with the live fallback, is
`data_engine/overlay.py`. `URBANHEAT_OVERLAY=earthengine` forces the live layer.

Box and Zone still query Earth Engine live: a stored layer cannot supply NDVI, albedo and
climate for an arbitrary shape.

## Quick start (Python side)

```bash
conda env create -f environment.yml
conda activate urbanheat
earthengine authenticate          # requires a (free) Google Earth Engine account
python -m data_engine.build_dataset --config configs/data_config.yaml   # a module: it uses relative imports
```

## Experiments

```bash
python scripts/run_baselines.py          # RF / XGBoost / MLP, leakage-safe splits
python scripts/run_pinn.py               # physics-weight sweep: the accuracy/physics trade-off
python scripts/run_augmented.py          # can physics-generated samples REMOVE that trade-off?
python scripts/run_augmented.py --holdout-albedo   # the harder test: unseen real high albedo
python scripts/export_chips.py --patch 9 # dense CNN chips from Earth Engine
python scripts/run_cnn.py --chips data/processed/kochi_chips_p9.npz
python scripts/build_roof_share.py       # measured roof share per ward, from building footprints
```

`scripts/run_augmented.py` is the one to read first. The deployed model trades R² 0.85 → 0.52 for
safe extrapolation, and the argument in `models/synthetic.py` is that this trade-off is an
artefact of a **data gap** (albedo > 0.2 never occurs in the satellite record) rather than a real
tension. It fills that gap with counterfactuals anchored to real pixels and labelled
`measured LST + physics-predicted change`, so the SEB closure's absolute bias cancels.

## Tests

```bash
python tests/run_all.py        # all suites; no Earth Engine, no dataset, no checkpoints needed
```

Earth Engine is faked (`tests/fake_ee.py`) and the dataset is synthesised (`tests/fixtures.py`)
with the two properties that matter: a narrow albedo range and the land-cover confound that makes
a data-only model predict cool roofs *warm* the city.

## Status

Interface contracts (dataset / model / API) are defined in the implementation plan §3.
Work is organised by module owner — see plan §16.

## License

TBD (academic project).
