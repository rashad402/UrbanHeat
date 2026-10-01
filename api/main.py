"""FastAPI inference service running the trained PINN (plan §11).

Run:
    uvicorn api.main:app --port 8001
    # http://localhost:8001/docs  for the interactive API browser

Endpoints (the API INTERFACE CONTRACT — plan §3):
    GET  /wards                                              -> ward list with baseline LST
    POST /predict  {ward_ids:[...]}                          -> {ward_id: {t_base}}
    POST /whatif   {ward_ids:[...], intervention:{...}}      -> {ward_id: {t_base, t_new, delta_t}}

Ward feature vectors are the per-ward means of the pixel dataset, so the model sees exactly the
features it was trained on. Temperatures are returned in DEGREES CELSIUS for the UI; the model
works in Kelvin internally.
"""

import os
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.inference import PinnModel, whatif        # noqa: E402
from models.features import INPUT_COLUMNS          # noqa: E402

PARQUET = "data/processed/kochi_samples.parquet"
app = FastAPI(title="PIML-UrbanHeat PINN API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

S = {}


@app.on_event("startup")
def _startup():
    S["model"] = PinnModel()
    df = pd.read_parquet(PARQUET)
    df = df[df.ward_id.notna()]
    # per-ward mean feature vector + observed baseline, for the model to run on
    agg = df.groupby("ward_id")[INPUT_COLUMNS + ["lst"]].mean()
    agg["npix"] = df.groupby("ward_id").size()
    S["wards"] = agg
    print(f"Loaded PINN '{S['model'].name}' and {len(agg)} ward feature vectors")


def _ward_feats(ward_ids):
    missing = [w for w in ward_ids if w not in S["wards"].index]
    if missing:
        raise HTTPException(404, f"unknown ward_id(s): {missing[:5]}")
    sub = S["wards"].loc[ward_ids]
    return {c: sub[c].to_numpy() for c in INPUT_COLUMNS}, sub


class Intervention(BaseModel):
    type: str = "cool_roof"            # cool_roof | green_roof | urban_greening | tree_canopy
    ndvi_delta: float | None = None
    albedo_set: float | None = None


class PredictRequest(BaseModel):
    ward_ids: list[str]


class WhatIfRequest(BaseModel):
    ward_ids: list[str]
    intervention: Intervention


DEFAULTS = {
    "cool_roof": dict(albedo_set=0.50),
    "green_roof": dict(ndvi_delta=0.30),
    "urban_greening": dict(ndvi_delta=0.30),
    "tree_canopy": dict(ndvi_delta=0.25),
}


@app.get("/health")
def health():
    return {"ok": True, "model": S["model"].name, "n_wards": len(S["wards"])}


@app.get("/wards")
def wards():
    w = S["wards"]
    feats = {c: w[c].to_numpy() for c in INPUT_COLUMNS}
    t = S["model"].predict_k(feats) - 273.15
    return {wid: {"t_base": round(float(t[i]), 2),
                  "observed": round(float(w.lst.iloc[i] - 273.15), 2),
                  "ndvi": round(float(w.ndvi.iloc[i]), 3),
                  "albedo": round(float(w.albedo.iloc[i]), 3),
                  "npix": int(w.npix.iloc[i])}
            for i, wid in enumerate(w.index)}


@app.post("/predict")
def predict(req: PredictRequest):
    feats, sub = _ward_feats(req.ward_ids)
    t = S["model"].predict_k(feats) - 273.15
    return {wid: {"t_base": round(float(t[i]), 2),
                  "observed": round(float(sub.lst.iloc[i] - 273.15), 2)}
            for i, wid in enumerate(req.ward_ids)}


@app.post("/whatif")
def whatif_endpoint(req: WhatIfRequest):
    feats, _ = _ward_feats(req.ward_ids)
    iv = req.intervention
    d = dict(DEFAULTS.get(iv.type, {}))
    if iv.albedo_set is not None:
        d["albedo_set"] = iv.albedo_set
    if iv.ndvi_delta is not None:
        d["ndvi_delta"] = iv.ndvi_delta
    if not d:
        raise HTTPException(400, f"unknown intervention type '{iv.type}' and no parameters given")

    r = whatif(S["model"], feats, albedo_set=d.get("albedo_set"), ndvi_delta=d.get("ndvi_delta"))
    # Baseline is the MEASURED ward LST; the model contributes only the response (delta_t).
    observed = S["wards"].loc[req.ward_ids, "lst"].to_numpy() - 273.15
    return {wid: {"t_base": round(float(observed[i]), 2),
                  "t_new": round(float(observed[i] + r["delta_t_applied"][i]), 2),
                  "delta_t": round(float(r["delta_t_applied"][i]), 2),
                  "delta_t_full_surface": round(float(r["delta_t"][i]), 2),
                  "t_model_base": round(float(r["t_base"][i] - 273.15), 2),
                  "built_frac": round(float(r["built_frac"][i]), 2),
                  "seb_residual": round(float(r["seb_residual"][i]), 1)}
            for i, wid in enumerate(req.ward_ids)}
