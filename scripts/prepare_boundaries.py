"""Fetch and prepare Kochi AOI + ward boundaries (plan §4, Phase 0).

Source: DataMeet community "Municipal_Spatial_Data" (Kochi Municipal Corporation, 77 wards),
derived from Oorvani Foundation / OpenCity. Files there are newline-delimited GeoJSON
(one Feature per line); this script assembles them into standard FeatureCollections,
normalizes ward properties, and writes:

    configs/aoi_kochi.geojson     (corporation boundary -> AOI)
    configs/wards_kochi.geojson   (77 wards, properties: ward_id, ward_name, ward_no)

Run:
    python scripts/prepare_boundaries.py

CRS: WGS84 (EPSG:4326), lon/lat — matches Earth Engine's expected input.
Attribution: DataMeet Municipal Spatial Data (ODbL). Keep this credit in the report.
"""

import json
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Kochi/"
WARDS_SRC = BASE + "KCH_wards.geojson"
BOUNDARY_SRC = BASE + "KCH_Corporation_Boundary.geojson"

OUT_AOI = "configs/aoi_kochi.geojson"
OUT_WARDS = "configs/wards_kochi.geojson"


def load_ndjson(url):
    """Read newline-delimited GeoJSON (one Feature per line) into a list of features."""
    req = urllib.request.Request(url, headers={"User-Agent": "UrbanHeat-academic/1.0"})
    text = urllib.request.urlopen(req, timeout=90).read().decode("utf-8")
    feats = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            feats.append(json.loads(line))
    return feats


def feature_collection(features):
    return {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features}


def main():
    os.makedirs("configs", exist_ok=True)

    # --- Wards ---
    # NOTE: the source has 77 polygons: 71 with Ward_No 1..71, and 6 real wards whose
    # attributes are missing (empty properties). We keep all 77 and give the unlabeled ones
    # a stable synthetic id ("x<index>") so every ward is clickable in the simulator.
    raw_wards = load_ndjson(WARDS_SRC)
    wards = []
    n_unlabeled = 0
    for i, f in enumerate(raw_wards):
        p = f.get("properties", {}) or {}
        ward_no = p.get("Ward_No")
        if ward_no:
            ward_id = str(ward_no)
            ward_no_int = int(ward_no)
            ward_name = p.get("Ward_Name")
        else:
            n_unlabeled += 1
            ward_id = f"x{i}"
            ward_no_int = None
            ward_name = None
        wards.append({
            "type": "Feature",
            "geometry": f["geometry"],
            "properties": {
                "ward_id": ward_id,
                "ward_no": ward_no_int,
                "ward_name": ward_name,
                "area_sqkm": float(p["Area"]) if p.get("Area") else None,
            },
        })
    with open(OUT_WARDS, "w", encoding="utf-8") as fh:
        json.dump(feature_collection(wards), fh, ensure_ascii=False)
    print(f"Wrote {OUT_WARDS} ({len(wards)} wards; {n_unlabeled} unlabeled -> synthetic ids)")

    # --- AOI (corporation boundary) ---
    raw_boundary = load_ndjson(BOUNDARY_SRC)
    boundary = raw_boundary[0]
    boundary["properties"] = {"name": "Kochi Municipal Corporation"}
    with open(OUT_AOI, "w", encoding="utf-8") as fh:
        json.dump(feature_collection([boundary]), fh, ensure_ascii=False)
    print(f"Wrote {OUT_AOI} (geometry: {boundary['geometry']['type']})")


if __name__ == "__main__":
    main()
