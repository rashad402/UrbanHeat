"""FastAPI inference service (plan §11).

Run:
    uvicorn api.main:app --reload

Endpoints (the API INTERFACE CONTRACT — plan §3):
    POST /predict  {ward_ids:[...]}                         -> {ward_id: {t_base}}
    POST /whatif   {ward_ids:[...], intervention:{...}}      -> {ward_id: {t_base, t_new, delta_t}}

Loads the trained checkpoint + scaler once at startup. Target < 1 s per ward batch.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="PIML-UrbanHeat API")

# Restrict to the web app origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Intervention(BaseModel):
    type: str                      # green_roof | cool_roof | urban_greening | tree_canopy
    ndvi_delta: float | None = None
    albedo_set: float | None = None


class WhatIfRequest(BaseModel):
    ward_ids: list[str]
    intervention: Intervention


class PredictRequest(BaseModel):
    ward_ids: list[str]


@app.on_event("startup")
def _load_model():
    """Load checkpoint + scaler into module state. See api/inference.py."""
    # from .inference import load_model
    # app.state.model = load_model(...)
    raise NotImplementedError


@app.post("/predict")
def predict(req: PredictRequest):
    raise NotImplementedError


@app.post("/whatif")
def whatif(req: WhatIfRequest):
    raise NotImplementedError
