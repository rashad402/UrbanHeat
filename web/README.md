# Web — What-If Urban Heat Mitigation Simulator

Next.js + Mapbox GL JS frontend (plan §12).

Not yet scaffolded. When starting this module (Phase 6, ~Weeks 18–21):

```bash
cd web
npx create-next-app@latest . --ts --app --eslint
npm install mapbox-gl
```

Then build:
- Map of Kochi ward boundaries over a satellite basemap.
- Ward selection (click/brush) + intervention controls (NDVI slider, albedo slider, canopy toggle).
- Call the FastAPI `/whatif` endpoint and render a ΔT choropleth with a diverging cooling legend.
- Base-vs-intervention toggle; strategy comparison; PNG export.

Config:
- `NEXT_PUBLIC_MAPBOX_TOKEN` and `NEXT_PUBLIC_API_URL` in `web/.env.local` (never commit).

API contract (see plan §3):
`POST /whatif` → `{ ward_ids:[...], intervention:{ type, ndvi_delta?, albedo_set? } }`
returns `{ ward_id: { t_base, t_new, delta_t } }`.
