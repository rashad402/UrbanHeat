"""API-level tests for the planner backend (plan §11, §12).

Run:  python tests/test_api.py           (or: pytest tests/test_api.py)

The repository had no tests outside tests/test_sebal.py, and the bugs this app actually shipped
were integration bugs — a layer hidden behind an overlay, a stale cached asset, a negative CSS
scale, a degenerate physics loss, features averaged before a non-linear model. Unit tests on the
physics would not have caught any of them. These exercise the real FastAPI app end to end, with
Earth Engine faked (tests/fake_ee.py) and a synthetic dataset (tests/fixtures.py).

WHAT IS WORTH ASSERTING HERE
    * Per-pixel aggregation, not mean-feature. The model is non-linear, so running it on averaged
      features is not the same as averaging per-pixel predictions. This is checked NUMERICALLY,
      by computing both and requiring they differ — a regression test that fails loudly if anyone
      reintroduces the shortcut.
    * Coverage scaling is actually applied and actually monotone in roof share.
    * The budget metric divides by treated area rather than multiplying by total area (the bug
      that made `cooling_per_km2` report K.km2).
    * Tile URLs can be re-minted, since an expired signed URL fails silently.
    * Wards with no land pixels are reported as such rather than as an error.
    * The error path returns a real HTTP status, so the UI can show a toast rather than printing
      an exception into a KPI tile.
"""

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import tests.fake_ee as fake_ee          # noqa: E402
fake_ee.install()                        # must precede any import of api.planner

from tests.fixtures import write_fixture, synthetic_dataset   # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))


def _make_checkpoint(tmp, df):
    """A randomly initialised PINN plus a scaler fitted on the fixture.

    The weights are arbitrary — these tests are about the API's arithmetic, not the model's
    accuracy (that lives in scripts/run_pinn.py and scripts/run_augmented.py). A random network
    is still strongly NON-LINEAR, which is exactly what the aggregation test needs.

    WEIGHT_GAIN scales the parameters so responses land in the Kelvin range rather than the
    millikelvin range. At default initialisation the network is nearly affine over this data, the
    two aggregations agree to four decimal places, and the Jensen test below would pass
    vacuously — unable to tell a correct implementation from the bug it is meant to guard.
    """
    import torch
    from models import features as F
    from models.pinn import PINN

    WEIGHT_GAIN = 2.0
    torch.manual_seed(0)
    net = PINN(in_dim=len(F.INPUT_COLUMNS))
    with torch.no_grad():
        for prm in net.parameters():
            prm.mul_(WEIGHT_GAIN)
    ckpt = os.path.join(tmp, "pinn.pt")
    scaler_path = os.path.join(tmp, "scaler.json")
    torch.save(net.state_dict(), ckpt)
    F.save_scaler(F.fit_scaler(df), scaler_path)
    return ckpt, scaler_path


def build_client(tmp, roof_share_file=None):
    """Boot the real FastAPI app against the fixture."""
    from fastapi.testclient import TestClient

    pq, gj = write_fixture(tmp, n=5000)
    df = synthetic_dataset(n=5000, seed=7)
    ckpt, scaler_path = _make_checkpoint(tmp, df)
    os.environ["PINN_CKPT"], os.environ["PINN_SCALER"] = ckpt, scaler_path

    import api.planner as P
    P.PARQUET, P.WARDS_PATH = pq, gj
    P.ROOF_SHARE_PATH = roof_share_file or os.path.join(tmp, "absent.json")
    P.CFG_PATH = os.path.join(ROOT, "configs", "data_config.yaml")
    P.SCENE_CACHE = os.path.join(tmp, "scene_cache.json")
    return TestClient(P.app), P


def post(client, path, albedo=None, ndvi=None, wards=None, bbox=None,
         season=None, roof_share=None):
    sel = ({"kind": "bbox", "bounds": bbox} if bbox
           else {"kind": "wards", "ward_ids": wards or []})
    return client.post(path, json={
        "selection": sel,
        "interventions": {"albedo_set": albedo, "ndvi_delta": ndvi, "roof_share": roof_share},
        "season": season,
    })


def main():
    print("Planner API tests\n" + "=" * 70)
    tmp = tempfile.mkdtemp(prefix="urbanheat-api-")
    client, P = build_client(tmp)

    with client:
        # ───────────── bootstrap ─────────────
        meta = client.get("/api/meta").json()
        check("/api/meta returns model, bounds and tiles",
              all(k in meta for k in ("model", "bounds", "lst_tiles", "n_wards")),
              f"{meta.get('n_wards')} wards")
        check("/api/meta reports data vintage",
              meta["vintage"].get("n_scenes") == fake_ee.N_SCENES
              and meta["vintage"].get("n_pixels", 0) > 0,
              f"{meta['vintage'].get('n_scenes')} scenes, "
              f"{meta['vintage'].get('n_pixels')} px")
        check("/api/meta reports the no-data ward count",
              meta["n_wards_no_data"] >= 1,
              f"{meta['n_wards_no_data']} of {meta['n_wards_total']} inert")
        check("roof share reported as an assumption when unmeasured",
              meta["roof_share_measured"] is False)

        wards = client.get("/api/wards").json()
        ids = [f["properties"]["ward_id"] for f in wards["features"]
               if f["properties"]["has_data"]]
        inert = [f["properties"] for f in wards["features"]
                 if not f["properties"]["has_data"]]
        check("/api/wards marks data-free wards has_data=false",
              len(inert) >= 1 and "lst_c" not in inert[0],
              f"{len(inert)} inert")

        # ───────────── the aggregation bug ─────────────
        r = post(client, "/api/analyze", albedo=0.5, wards=ids[:4]).json()
        check("/api/analyze returns per-ward items and a summary",
              len(r["items"]) == 4 and "mean_delta_t" in r["summary"])

        per_pixel, mean_feature = _aggregation_comparison(P, ids[:4])
        gap = abs(per_pixel - mean_feature)
        # The gap must exceed the reporting resolution, or the next two checks prove nothing.
        check("per-pixel aggregation differs from mean-feature (Jensen)",
              gap > 0.05,
              f"per-pixel {per_pixel:+.4f} K vs mean-feature {mean_feature:+.4f} K "
              f"(gap {gap:.4f} K)")
        check("/api/analyze reports the per-pixel value",
              _matches(r, per_pixel),
              f"api {_api_mean(r):+.4f} K, expected {per_pixel:+.4f} K")
        check("/api/analyze does NOT report the mean-feature value",
              not _matches(r, mean_feature),
              f"mean-feature would be {mean_feature:+.4f} K")

        # ───────────── coverage scaling ─────────────
        lo = post(client, "/api/analyze", albedo=0.5, wards=ids[:4], roof_share=0.2).json()
        hi = post(client, "/api/analyze", albedo=0.5, wards=ids[:4], roof_share=1.0).json()
        check("roof share scales the reported cooling",
              abs(hi["summary"]["mean_delta_t"]) > abs(lo["summary"]["mean_delta_t"]),
              f"20% -> {lo['summary']['mean_delta_t']:+.2f} K, "
              f"100% -> {hi['summary']['mean_delta_t']:+.2f} K")
        check("roof share is echoed back for auditability",
              abs(lo["summary"]["roof_share"] - 0.2) < 1e-9
              and lo["summary"]["roof_share_source"] == "user")

        # ───────────── budget metric ─────────────
        sm = hi["summary"]
        treated = sm["treated_area_sqkm"]
        check("treated area is a strict subset of selected area",
              0 < treated < sm["area_sqkm"],
              f"{treated} of {sm['area_sqkm']} km²")
        expect = sum(i["delta_t"] * (i["area_sqkm"] or 0) for i in hi["items"])
        check("cooling_k_km2 is the area-weighted total",
              abs(sm["cooling_k_km2"] - expect) < 1e-2,
              f"{sm['cooling_k_km2']} K·km²")
        check("cooling_per_treated_km2 divides by treated area",
              abs(sm["cooling_per_treated_km2"] - expect / treated) < 1e-2,
              f"{sm['cooling_per_treated_km2']} K·km²/km²")

        # ───────────── uncertainty ─────────────
        check("summary carries a p10-p90 dT spread",
              sm["delta_t_p10"] <= sm["mean_delta_t"] + 1e-9
              and sm["delta_t_p90"] >= sm["delta_t_p10"],
              f"p10 {sm['delta_t_p10']}, mean {sm['mean_delta_t']}, p90 {sm['delta_t_p90']}")
        check("items carry their own spread",
              all("delta_t_p10" in i and "delta_t_p90" in i for i in hi["items"]))

        # ───────────── seasons ─────────────
        dry = post(client, "/api/analyze", albedo=0.5, wards=ids[:4], season="dry").json()
        pre = post(client, "/api/analyze", albedo=0.5, wards=ids[:4], season="premonsoon").json()
        check("season filter changes the pixel count",
              dry["summary"]["n_pixels"] != pre["summary"]["n_pixels"]
              and dry["summary"]["n_pixels"] < r["summary"]["n_pixels"],
              f"all {r['summary']['n_pixels']}, dry {dry['summary']['n_pixels']}, "
              f"pre-monsoon {pre['summary']['n_pixels']}")
        check("season is echoed in the summary", dry["summary"]["season"] == "dry")

        # ───────────── no-intervention case ─────────────
        base = post(client, "/api/analyze", wards=ids[:4]).json()
        check("no intervention reports no_change and zero cooling",
              base["summary"]["no_change"] is True
              and abs(base["summary"]["mean_delta_t"]) < 1e-9)

        # ───────────── ranking ─────────────
        rk = post(client, "/api/rank", albedo=0.5).json()
        check("/api/rank scores every ward with data",
              len(rk["items"]) == meta["n_wards"],
              f"{len(rk['items'])} ranked")
        check("/api/rank supplies all three orderings",
              all(k in rk["items"][0] for k in
                  ("t_base", "delta_t", "cooling_per_treated_km2")))

        # ───────────── export ─────────────
        ex = client.post("/api/export", json={
            "selection": {"kind": "wards", "ward_ids": ids[:3]},
            "interventions": {"albedo_set": 0.5, "ndvi_delta": None, "roof_share": None},
            "season": "dry"})
        body = ex.text
        check("/api/export returns CSV with a provenance header",
              ex.status_code == 200 and body.startswith("# UrbanHeat Planner scenario")
              and "baseline_scenes" in body and "roof_share_source" in body)
        check("/api/export rows match the analysis",
              len([l for l in body.splitlines() if l and not l.startswith("#")]) == 4,
              "3 wards + header")

        # ───────────── tile refresh ─────────────
        before = meta["lst_tiles"]
        after = client.post("/api/refresh_tiles").json()["lst_tiles"]
        check("/api/refresh_tiles re-mints a different signed URL",
              after != before, f"{before[-16:]} -> {after[-16:]}")

        # ───────────── drawn area ─────────────
        fake_ee.SAMPLE_ROWS = _bbox_rows()
        bb = post(client, "/api/analyze", albedo=0.5,
                  bbox=[76.25, 9.95, 76.27, 9.97]).json()
        check("bbox selection analyses sampled pixels per pixel",
              bb["summary"]["n_pixels"] == len(fake_ee.SAMPLE_ROWS)
              and bb["items"][0]["npix"] == len(fake_ee.SAMPLE_ROWS),
              f"{bb['summary']['n_pixels']} px")

        check("drawn area is measured from the geometry, not the sample size",
              abs(bb["items"][0]["area_sqkm"] - fake_ee.REGION_AREA_M2 / 1e6) < 1e-3,
              f"{bb['items'][0]['area_sqkm']} km² from geometry; "
              f"counting {len(fake_ee.SAMPLE_ROWS)} sampled pixels would give "
              f"{len(fake_ee.SAMPLE_ROWS) * 900 / 1e6:.3f} km²")

        poly = client.post("/api/analyze", json={
            "selection": {"kind": "polygon",
                          "coordinates": [[76.25, 9.95], [76.27, 9.95],
                                          [76.27, 9.97], [76.26, 9.98]]},
            "interventions": {"albedo_set": 0.5}, "season": None})
        check("polygon selection is analysed like a bbox",
              poly.status_code == 200 and poly.json()["items"][0]["name"] == "Drawn zone",
              f"HTTP {poly.status_code}")
        badpoly = client.post("/api/analyze", json={
            "selection": {"kind": "polygon", "coordinates": [[76.25, 9.95], [76.27, 9.95]]},
            "interventions": {"albedo_set": 0.5}})
        check("degenerate polygon returns 400", badpoly.status_code == 400)

        # a bow-tie: corners given in column order, so the diagonals cross
        bow = client.post("/api/analyze", json={
            "selection": {"kind": "polygon",
                          "coordinates": [[76.25, 9.97], [76.25, 9.95], [76.27, 9.97], [76.27, 9.95]]},
            "interventions": {"albedo_set": 0.5}})
        check("self-crossing zone is refused with a readable 400",
              bow.status_code == 400 and "crosses itself" in bow.json().get("detail", ""),
              f"HTTP {bow.status_code}: {bow.json().get('detail', '')[:60]}")
        # the same four corners in order are fine, and so is a concave L drawn on purpose
        square = client.post("/api/analyze", json={
            "selection": {"kind": "polygon",
                          "coordinates": [[76.25, 9.97], [76.25, 9.95], [76.27, 9.95], [76.27, 9.97]]},
            "interventions": {"albedo_set": 0.5}})
        ell = client.post("/api/analyze", json={
            "selection": {"kind": "polygon",
                          "coordinates": [[76.25, 9.95], [76.28, 9.95], [76.28, 9.96],
                                          [76.26, 9.96], [76.26, 9.98], [76.25, 9.98]]},
            "interventions": {"albedo_set": 0.5}})
        check("the same corners in order are accepted", square.status_code == 200,
              f"HTTP {square.status_code}")
        check("a concave (L-shaped) zone is accepted, not mistaken for a crossing",
              ell.status_code == 200, f"HTTP {ell.status_code}")

        fake_ee.SAMPLE_ROWS = []
        empty = post(client, "/api/analyze", albedo=0.5, bbox=[76.0, 9.0, 76.01, 9.01])
        check("empty drawn area returns 422, not a 500",
              empty.status_code == 422, f"HTTP {empty.status_code}")

        # ───────────── error paths ─────────────
        bad = post(client, "/api/analyze", albedo=0.5, wards=["NOT_A_WARD"])
        check("unknown ward returns 400 with a readable detail",
              bad.status_code == 400 and "detail" in bad.json(),
              bad.json().get("detail", ""))
        badkind = client.post("/api/analyze", json={
            "selection": {"kind": "nonsense"},
            "interventions": {"albedo_set": 0.5}})
        check("unknown selection kind returns 400", badkind.status_code == 400)

    # ───────────── measured roof share ─────────────
    _test_measured_roof_share(tmp, ids)

    print("\n" + "=" * 70)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


def _api_mean(res):
    return res["summary"]["mean_delta_t"]


def _matches(res, value):
    return abs(_api_mean(res) - round(value, 2)) < 0.011


def _aggregation_comparison(P, ward_ids):
    """Compute area-weighted mean dT both ways, to prove they are not the same number.

    per-pixel  : predict for every pixel, then average      (what the API must do)
    mean-feature: average the features, then predict once    (the bug)
    """
    from models.features import INPUT_COLUMNS
    from api.inference import whatif, built_fraction

    S, model = P.S, P.S["model"]
    frame = P.S["px"][P.S["px"].ward_id.isin(ward_ids)]
    geo = {f["properties"]["ward_id"]: f["properties"]
           for f in S["wards_geo"]["features"]}

    def weighted(per_ward):
        num = den = 0.0
        for wid, dt in per_ward.items():
            a = geo.get(wid, {}).get("area_sqkm") or 0.0
            num += dt * a
            den += a
        return num / den if den else float("nan")

    share = P.ROOF_SHARE_DEFAULT
    pp, mf = {}, {}
    for wid, g in frame.groupby("ward_id"):
        feats = {c: g[c].to_numpy(dtype="float64") for c in INPUT_COLUMNS}
        r = whatif(model, feats, albedo_set=0.5)
        pp[wid] = float(np.mean(r["delta_t"] * built_fraction(feats["ndvi"]) * share))

        m = {c: np.array([g[c].mean()], dtype="float64") for c in INPUT_COLUMNS}
        rm = whatif(model, m, albedo_set=0.5)
        mf[wid] = float(rm["delta_t"][0] * built_fraction(m["ndvi"])[0] * share)
    return weighted(pp), weighted(mf)


def _bbox_rows(n=60, seed=3):
    """Pixel rows as Earth Engine would return them for a drawn rectangle."""
    df = synthetic_dataset(n=n, seed=seed)
    cols = ["lon", "lat", "ndvi", "ndbi", "albedo", "s_down", "t_air", "rh", "wind", "lst"]
    return df[cols].to_dict("records")


def _test_measured_roof_share(tmp, ids):
    """A roof_share_kochi.json on disk must override the hard-coded assumption, per ward."""
    from fastapi.testclient import TestClient
    import importlib
    import api.planner as P

    path = os.path.join(tmp, "roof_share.json")
    wards = {w: {"roof_share": 0.1 + 0.02 * i} for i, w in enumerate(ids)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"source": "test", "city_default": 0.33, "wards": wards}, fh)

    importlib.reload(P)
    P.PARQUET = os.path.join(tmp, "samples.parquet")
    P.WARDS_PATH = os.path.join(tmp, "wards.geojson")
    P.ROOF_SHARE_PATH = path
    P.CFG_PATH = os.path.join(ROOT, "configs", "data_config.yaml")
    P.SCENE_CACHE = os.path.join(tmp, "scene_cache2.json")

    with TestClient(P.app) as c:
        meta = c.get("/api/meta").json()
        check("measured roof share is detected and flagged",
              meta["roof_share_measured"] is True
              and abs(meta["roof_share_default"] - 0.33) < 1e-9)
        res = post(c, "/api/analyze", albedo=0.5, wards=ids[:3]).json()
        check("measured per-ward roof share is used, not the 0.5 constant",
              res["summary"]["roof_share_source"] == "measured"
              and abs(res["summary"]["roof_share"] - 0.5) > 1e-6,
              f"mean share {res['summary']['roof_share']}")
        got = {i["ward_id"]: i["roof_share"] for i in res["items"]}
        check("each ward gets its own measured share",
              all(abs(got[w] - wards[w]["roof_share"]) < 1e-3 for w in got),
              str(got))
        wg = c.get("/api/wards").json()
        check("/api/wards exposes per-ward roof share",
              any("roof_share" in f["properties"] for f in wg["features"]))


if __name__ == "__main__":
    sys.exit(main())
