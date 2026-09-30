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
data_engine/   satellite + climate extraction, cloud masking, gap-filling, dataset assembly
models/        features, baselines (RF/XGB/CNN), SEBAL physics, PINN, training, evaluation
api/           FastAPI inference service (/predict, /whatif)
web/           Next.js + Mapbox GL JS interactive simulator
configs/        data & model configuration, AOI + ward boundaries
notebooks/      exploratory only (never a source of reported results)
docs/           final report (LaTeX) and figures
```

## Quick start (Python side)

```bash
conda env create -f environment.yml
conda activate urbanheat
earthengine authenticate          # requires a (free) Google Earth Engine account
python data_engine/gee_export.py --config configs/data_config.yaml
```

## Status

Early scaffold. Interface contracts (dataset / model / API) are defined in the implementation plan §3.
Work is organised by module owner — see plan §16.

## License

TBD (academic project).
