# PIML-UrbanHeat — Implementation Plan

**Project:** A Physics-Informed Neural Framework for Counterfactual Urban Heat Mitigation in Humid-Tropical Coastal Cities
**Team:** Jesmina E · Judha C K · Muhammad Rashad Hashim · Najih V — Supervisor: Dr. Bindu P V
**Institution:** Dept. of CSE, Government College of Engineering Kannur
**Plan version:** 1.0 · 2026-09-30

> This document turns the approved proposal + interim presentation into an engineering execution plan:
> what to build, in what order, with which interfaces, and how to de-risk the hard parts early.
> It maps onto the 28-week schedule from the interim presentation (see [§13](#13-mapping-to-the-28-week-schedule)).

---

## 1. Guiding principles

1. **Data first, model second.** The single biggest risk is *not the PINN* — it is assembling a clean, gap-filled, co-registered satellite + climate dataset for a Kerala coastal city. Prove the data pipeline end-to-end on one city before touching the neural network.
2. **MVP-first, vertically.** Get a thin slice working through *every* layer (data → simple model → API → map showing one ΔT) as early as possible, then deepen. Avoid building any one module to perfection in isolation.
3. **Baselines are not a formality.** RF / XGBoost / CNN are trained *before* the PINN, on the exact same feature matrix. They double as (a) the paper's comparison and (b) a sanity check that the features actually predict LST.
4. **Physics is a loss term, not a separate simulator.** All the "physics" lives in the training loss and the inference-time recomputation of surface energy balance (SEB). Keep it in one well-tested module.
5. **Reproducibility.** Every dataset artifact is produced by a versioned script with a fixed config (dates, AOI, cloud threshold, random seed). No manual clicking in the GEE Code Editor for anything that ends up in results.

---

## 2. Scope decisions (recommended defaults)

These are decisions the proposal left open. Recommended defaults are marked ✅ — confirm with the supervisor, but don't block on them.

| Decision | Recommended default | Rationale |
|---|---|---|
| **Primary study area** | ✅ **Kochi (Kochi Municipal Corporation + suburbs)** | Largest coastal UHI signal in Kerala; has published UHI studies (Ramesh & Nair 2025) for ΔT plausibility cross-checks; clear ward boundaries. |
| Secondary/validation city | Thiruvananthapuram (stretch goal only) | Second city proves geographic transfer; defer until single-city works. |
| **Temporal window** | ✅ 2015–2026, dry-season scenes (Dec–Apr) for training; MODIS gap-fill for monsoon | Landsat 8 launched 2013; Sentinel-2 from 2015 (limits the common window). Dry season = most cloud-free Landsat. |
| **Spatial resolution** | ✅ 30 m (Landsat/Sentinel native); MODIS 1 km downscaled to 30 m | Matches proposal; 30 m is the ward-level planning resolution. |
| **PINN input domain** | ✅ Per-pixel feature vector (not full PDE field) | The proposal's PINN is a *point-wise* surrogate: inputs (x,y,NDVI,NDBI,α,S↓,Tₐ,RH,u) → Tₛ. No spatial derivatives are in the SEB residual, so this is a physics-*constrained* regressor, not a PDE solver. This simplifies DeepXDE usage massively — confirm this interpretation early (see [§9](#9-phase-4--pinn-core)). |
| Compute | ✅ Google Colab (free/Pro T4) + GEE cloud | 8–16 GB local RAM per hardware slide is fine for a point-wise MLP; GPU only speeds training. No HPC needed. |
| Frontend hosting | ✅ Vercel (Next.js) + FastAPI inference on Render/Railway or Colab tunnel for demo | Zero-cost tiers cover a demo. |

---

## 3. Target repository structure

Single monorepo, initialised with `git` (the working dir is **not** a git repo yet — do this in Phase 0).

```
UrbanHeat/
├── README.md
├── IMPLEMENTATION_PLAN.md          # this file
├── .gitignore                      # exclude data/, *.tif, node_modules, .venv, model checkpoints
├── environment.yml                 # conda env for Python side
├── configs/
│   ├── aoi_kochi.geojson           # area of interest polygon
│   ├── data_config.yaml            # dates, cloud %, bands, seed
│   └── model_config.yaml           # PINN arch, loss weights, LR schedule
├── data_engine/                    # Module 1–4: satellite + climate data
│   ├── gee_export.py               # Earth Engine extraction (Landsat, Sentinel-2, MODIS, ERA5)
│   ├── indices.py                  # NDVI, NDBI, albedo, LST retrieval
│   ├── cloud_mask.py               # QA_PIXEL masking
│   ├── gap_fill.py                 # MODIS→Landsat downscaling / temporal fill
│   ├── build_dataset.py            # co-register, stack, sample → parquet/npz
│   └── era5.py                     # ERA5-Land climate variable ingestion
├── models/
│   ├── features.py                 # feature matrix assembly, normalization, splits
│   ├── baselines.py                # RF, XGBoost, CNN regressors
│   ├── sebal.py                    # SEB flux parameterizations (Rn, G, H, LE)  ← physics core
│   ├── pinn.py                     # network + composite loss (DeepXDE or pure PyTorch)
│   ├── train.py                    # training loop, checkpointing, logging
│   ├── evaluate.py                 # R², RMSE, MAE, SEB-residual, ΔT plausibility
│   └── counterfactual.py           # apply interventions, recompute T under SEB
├── api/
│   ├── main.py                     # FastAPI: /predict, /whatif endpoints
│   └── inference.py                # load checkpoint, ward-level batch inference
├── web/                            # Module 8: Next.js + Mapbox GL JS
│   ├── package.json
│   ├── app/                        # Next.js app router
│   ├── components/Map, Sliders, LegendΔT
│   └── lib/api.ts                  # calls FastAPI
├── notebooks/                      # exploratory only — never a source of results
└── docs/
    ├── report/                     # final project report (LaTeX)
    └── figures/
```

**Interface contracts (define these on day one so members can work in parallel):**

- **Dataset artifact:** `data/processed/kochi_samples.parquet` with columns
  `x, y, lat, lon, date, ndvi, ndbi, albedo, s_down, t_air, rh, wind, lst_landsat, lst_source ∈ {landsat, modis_fill}, ward_id`.
- **Model artifact:** `models/checkpoints/pinn_best.pt` + `scaler.json` (feature means/stds).
- **API contract:** `POST /whatif` → `{ward_ids:[...], intervention:{ndvi_delta, albedo_set, ...}}` returns `{ward_id: {t_base, t_new, delta_t}}`.

Everything downstream depends only on these three contracts, not on internal implementation.

---

## 4. Phase 0 — Foundations (Weeks 1–2)

- [ ] `git init` the project; push to the team GitHub repo; add `.gitignore` and branch protection on `main`.
- [ ] Create `environment.yml` (Python 3.11): `earthengine-api`, `geemap`, `rasterio`, `geopandas`, `xarray`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `deepxde`, `matplotlib`, `pyyaml`, `fastapi`, `uvicorn`.
- [ ] Each member authenticates Google Earth Engine (`earthengine authenticate`) — **requires a (free) GEE account tied to a Google Cloud project; register early, approval can take a day.**
- [ ] Draw `aoi_kochi.geojson` (bounding polygon) and obtain **ward boundary GeoJSON** for Kochi Municipal Corporation (source: municipal open data / OpenStreetMap admin relations / Kerala GIS). This is a known long-pole — start immediately.
- [ ] Write `configs/data_config.yaml` (date ranges, cloud threshold = 20%, bands, AOI path, seed=42).
- [ ] Agree the three interface contracts in [§3](#3-target-repository-structure).
- [ ] Set up experiment logging (Weights & Biases free tier, or a CSV/`mlflow` local). Decide once.

**Exit criterion:** repo builds the conda env; every member can run a trivial GEE query for the AOI.

---

## 5. Phase 1 — Spatio-Temporal Data Engine (Weeks 3–5)

This is the highest-risk phase. Budget the most time here.

**5.1 Satellite extraction (`gee_export.py`)**
- [ ] Landsat 8/9 Collection 2 L2 — thermal Band 10 for LST, plus optical bands. Use the L2 `ST_B10` surface temperature product (already atmospherically corrected) *or* implement Sobrino single-channel retrieval from TOA — **prefer the L2 ST product** to save weeks; note the choice.
- [ ] Sentinel-2 SR (Harmonized) — bands for NDVI (B8, B4), NDBI (B11, B8), albedo (Liang 2001 broadband coefficients from B2,B4,B8,B11,B12).
- [ ] MODIS `MOD11A2` (8-day LST) for gap-filling.
- [ ] ERA5-Land hourly → resample to satellite overpass time: `t_air` (2 m temp), `rh` (from 2 m temp + dewpoint), `wind` (√(u10²+v10²)), `s_down` (surface solar radiation downwards).

**5.2 Cloud masking (`cloud_mask.py`)**
- [ ] Landsat: bit-mask `QA_PIXEL` (cloud, cloud shadow, cirrus). Sentinel-2: `SCL` / `QA60`.

**5.3 Indices & LST (`indices.py`)**
- [ ] NDVI = (NIR−Red)/(NIR+Red); NDBI = (SWIR1−NIR)/(SWIR1+NIR); broadband albedo; LST in °C (or K — **fix units in a constants module and use them everywhere**).

**5.4 Gap-filling (`gap_fill.py`)**
- [ ] Regress cloud-free Landsat LST on MODIS LST + indices during clear periods; apply to downscale MODIS 1 km → 30 m during monsoon gaps. Keep it simple (linear/RF regression per the proposal's "regression relationship"). Tag every filled pixel with `lst_source = modis_fill` so evaluation can exclude them if needed.

**5.5 Dataset assembly (`build_dataset.py`)**
- [ ] Co-register all layers to a common 30 m grid + CRS (UTM 43N for Kerala). Stack, spatially join ward IDs, sample to the point table, write parquet.

**Exit criterion:** `kochi_samples.parquet` exists with ≥ ~10⁵ clean rows across ≥ 8–10 dates, and a decadal UHI trend map (2015–2026) renders. **Demo this before Week 6.**

---

## 6. Phase 2 — Feature engineering & splits (Week 6)

- [ ] `features.py`: assemble X = [x,y,ndvi,ndbi,albedo,s_down,t_air,rh,wind], y = lst. Normalize (standardize) using **train-set statistics only**; persist `scaler.json`.
- [ ] Splits: **spatial-block or by-date holdout**, not random pixel split (random split leaks neighbours and inflates R²). Document the split policy — reviewers will ask.

---

## 7. Phase 3 — Baseline models (Weeks 7–8, was Weeks 8–10)

- [ ] `baselines.py`: Random Forest, XGBoost, and a small CNN (operate on image patches rather than point vectors — this is the one baseline needing gridded input, so decide patch size, e.g. 32×32).
- [ ] Report R², RMSE, MAE on the holdout for each. Persist predictions.
- [ ] **Gate:** if baselines can't reach reasonable LST R² (~0.7+), the *features/data* are wrong — fix before building the PINN. This gate is the whole point of doing baselines first.

---

## 8. Phase 4 — Surface Energy Balance physics module (Weeks 9–10)

Build and unit-test this *before* wiring it into the PINN loss. It's pure functions — easy to test.

`sebal.py` implements (all in SI units, Tₛ in Kelvin):

```
Rn = (1−α)·S↓ + ε·L↓ − (1−ε)·L↓ − ε·σ·Tₛ⁴          # net radiation
G  = Rn · Γ(NDVI, α, Tₛ)                            # ground heat flux (SEBAL fraction, Bastiaanssen 1998)
H  = ρ·cp·(Tₛ − Tₐ) / r_ah(u)                       # sensible heat (aerodynamic resistance)
LE = Rn − G − H                                     # latent heat as SEB residual
SEB_residual = Rn − G − H − LE   (≡ 0 by construction if LE is the residual)
```

- [ ] Implement L↓ (incoming longwave) from ERA5 or Stefan-Boltzmann with air-temp + Brutsaert emissivity.
- [ ] Emissivity ε from NDVI (NDVI-threshold method).
- [ ] `r_ah(u)` aerodynamic resistance as a function of wind speed; document the roughness-length assumptions (the humid-coastal calibration novelty lives here — parameterize with Kerala ERA5 wind/humidity).
- [ ] **Design note on the loss:** if LE is *defined* as the residual, `Rn−G−H−LE ≡ 0` trivially and the physics loss is zero regardless of Tₛ — which defeats the purpose. The physics constraint must instead penalize the network Tₛ against the **Tₛ that closes the budget given an *independent* LE estimate** (e.g. LE from a Penman-Monteith / Priestley-Taylor estimate using RH), OR enforce that predicted Tₛ satisfies `Rn(Tₛ) = G + H(Tₛ) + LE_independent`. **Resolve this formulation with the supervisor in Week 9** — it is the single most important physics decision in the project and the proposal's equation as written needs this clarification. Unit-test the residual is non-trivial before proceeding.

**Exit criterion:** given a row of features + a candidate Tₛ, `sebal.py` returns each flux and a *non-degenerate* SEB residual, with tests.

---

## 9. Phase 4 — PINN core (Weeks 11–13)

- [ ] `pinn.py`: fully-connected MLP (start 5–8 hidden layers × 64–128 units, `tanh` or `SiLU`). Input dim 9, output Tₛ.
- [ ] **DeepXDE vs pure PyTorch:** because the SEB residual is algebraic (no spatial/temporal derivatives in the point-wise formulation), a **plain PyTorch custom loss is simpler and more transparent than DeepXDE**. Use DeepXDE only if you later add a diffusion PDE term. Recommend starting in pure PyTorch; mention DeepXDE in the report as the library option. Confirm with supervisor (proposal names DeepXDE).
- [ ] Composite loss:
  `L = (1/N)·Σ(T_pred − T_landsat)²  +  λ·(1/M)·Σ|SEB_residual(T_pred)|²`
- [ ] λ scheduling: start λ small, ramp up (loss-balancing / gradual physics weighting) to avoid the physics term dominating early. Log both loss components separately.
- [ ] `train.py`: Adam + cosine/step LR, early stopping on val data-loss, checkpoint best.

**Exit criterion:** PINN matches or beats XGBoost on LST R²/RMSE **and** has lower SEB-residual than the baselines under held-out data.

---

## 10. Phase 5 — Counterfactual engine (Weeks 14–17)

- [ ] `counterfactual.py`: given ward pixels + an intervention, modify features and re-infer:
  - **Green Roof / Urban greening:** NDVI += 0.2…0.4 (clamp ≤ ~0.85); correlated ε and (optionally) albedo adjustment.
  - **Cool Roof:** albedo 0.15 → 0.50 (set, not add).
  - **Tree Canopy:** NDVI increase + effective wind/roughness change if modelled.
- [ ] Recompute SEB at the new Tₛ to confirm the predicted ΔT is energy-consistent (increasing vegetation ↑LE must be compensated by ↓H/↓G).
- [ ] Output per-ward ΔT = T_new − T_base.
- [ ] **Validation (no ground truth exists):** cross-check predicted ΔT magnitudes against published tropical field ranges — cool roofs 1.5–4.0 °C, greening 0.5–3.0 °C. Flag any out-of-range output. This is the proposal's stated validation method.

**Exit criterion:** `/whatif` returns plausible, SEB-consistent ΔT for a real Kochi ward.

---

## 11. Phase 6 — Inference API (Weeks 15–17, parallel with §10)

- [ ] `api/main.py` (FastAPI): `/predict` (base LST for AOI/ward) and `/whatif` (intervention → ΔT). Load checkpoint + scaler once at startup. Target < 1 s per ward batch (the proposal's real-time claim — easy for an MLP).
- [ ] Return GeoJSON or a lightweight per-ward JSON the frontend can join to boundaries.
- [ ] CORS for the Next.js origin.

---

## 12. Phase 7 — Interactive web simulator (Weeks 18–21)

- [ ] `web/` Next.js app + Mapbox GL JS: Kochi ward boundaries over a satellite basemap.
- [ ] Ward selection (click/brush) → intervention controls (NDVI slider, albedo slider, canopy toggle) → call `/whatif` → render ΔT choropleth/heatmap with a diverging cooling legend.
- [ ] Base-vs-intervention toggle; strategy comparison view; export map as PNG.
- [ ] Keep secrets (Mapbox token) in env vars, not committed.

**Exit criterion:** a planner can select a ward, apply cool-roof + greening, and see an instant ΔT map — the headline demo.

---

## 13. Mapping to the 28-week schedule

The presentation's schedule is preserved; this plan reorders slightly to front-load data risk and pull baselines earlier.

| Presentation weeks | Presentation plan | This plan's phase |
|---|---|---|
| 1–2 | Lit review, scope | Phase 0 Foundations |
| 3–4 | Dataset identification & collection | Phase 1 Data Engine (start) |
| 5–7 | Preprocessing, cloud masking, feature extraction | Phase 1 finish + Phase 2 |
| 8–10 | Baselines RF/XGB/CNN | Phase 3 (pull to wk 7–8) + Phase 4 SEBAL module (wk 9–10) |
| 11–13 | PINN development | Phase 4 PINN core |
| 14–16 | SEB physics loss integration | folded into Phase 4 + start Phase 5 |
| 17–20 | Counterfactual simulation | Phase 5 + Phase 6 API |
| 21–24 | Interactive viz, testing, evaluation | Phase 7 Web + Phase 8 Evaluation |
| 25–28 | Integration, docs, demo, report | Phase 8 |

---

## 14. Evaluation plan (implements the interim Evaluation table)

| Component | Metric | Where |
|---|---|---|
| LST prediction accuracy | R², RMSE, MAE (holdout) | `evaluate.py` |
| Physics consistency | mean/max SEB residual under interventions | `evaluate.py` + `sebal.py` |
| Counterfactual plausibility | predicted ΔT vs published field ranges | `counterfactual.py` |
| Baseline benefit | PINN vs RF/XGB/CNN on all above | comparison table |
| System performance | `/whatif` latency, throughput | API load test |

Deliver one results table + figures (predicted vs observed scatter, ΔT maps, loss curves) into `docs/figures/`.

---

## 15. Risks & engineering mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **SEB physics loss is degenerate** (LE-as-residual makes residual ≡ 0) | Physics term does nothing — kills the novelty | Resolve formulation in Phase 4 Week 9 (independent LE estimate). Highest priority. |
| Ward boundary data unavailable/poor for Kochi | Blocks simulator | Start sourcing Week 1; fall back to grid cells or OSM admin polygons. |
| Monsoon cloud gaps too large; gap-fill noisy | Biased training data | Restrict primary training to dry season; treat MODIS-fill rows as auxiliary, tag & optionally down-weight. |
| Random split inflates metrics | Reviewers reject results | Spatial-block/by-date splits from the start. |
| GEE quota / export size limits | Slow iteration | Export at AOI scale, tile if needed, cache locally. |
| Scope creep (2nd city, PDE terms, mobile app) | Miss deadline | Single city + point-wise PINN is the committed MVP; everything else is a stretch goal. |
| Colab session limits during training | Lost runs | Checkpoint every epoch to Drive; keep MLP small (fast to retrain). |

---

## 16. Suggested work split (4 members)

Assign by module ownership but pair on integration boundaries.

- **Member A — Data Engine lead** (`data_engine/`): GEE, cloud masking, gap-fill, dataset assembly.
- **Member B — Modeling/Physics lead** (`models/sebal.py`, `pinn.py`, `train.py`): SEB formulation, PINN, loss.
- **Member C — Baselines & Evaluation lead** (`models/baselines.py`, `evaluate.py`, `counterfactual.py`): comparison experiments, validation, figures.
- **Member D — Product lead** (`api/`, `web/`): FastAPI inference, Next.js + Mapbox simulator, deployment/demo.

Shared: config contracts, the final report (`docs/report/`), and the interface definitions in [§3](#3-target-repository-structure).

---

## 17. Immediate next actions (this week)

1. `git init` + push repo skeleton with the structure in [§3](#3-target-repository-structure).
2. All four register/authenticate Google Earth Engine.
3. Member A: source Kochi AOI + ward boundary GeoJSON.
4. Member B: draft and confirm the **SEB residual formulation** with Dr. Bindu P V (settles the [§8](#8-phase-4--surface-energy-balance-physics-module-weeks-910) open question).
5. Write `configs/data_config.yaml` and the three interface contracts; commit.
6. Run one end-to-end GEE query for the AOI to confirm access — the thin first slice.
```
