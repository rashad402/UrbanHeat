"""A stand-in for the `earthengine-api` module, so the API can be tested without GEE.

api/planner.py talks to Earth Engine at import time and during startup: it builds a collection,
clips a median composite, signs a tile URL and samples pixels for drawn areas. None of that is
reachable from CI, and none of it is what the API tests are checking — they are checking the
aggregation, scaling, budget and uncertainty logic that sits on top.

This module provides a chainable fake. Every attribute access returns something callable that
returns another link in the chain, so arbitrary Earth Engine expressions evaluate without error.
Only the handful of calls that actually RETURN DATA are given real behaviour:

    <geometry>.bounds().coordinates().getInfo()   the AOI bounding ring
    <collection>.size().getInfo()                 the scene count behind the baseline
    <image>.getMapId(...)                         a signed tile URL (counted, so tests can prove
                                                  /api/refresh_tiles actually re-mints it)
    <image>.sample(...).getInfo()                 pixels for a drawn bbox

Install it with `install()` BEFORE importing api.planner.
"""

import sys
import types

# Pixels handed back by a bbox sample; tests overwrite this to drive the drawn-area path.
SAMPLE_ROWS = []
# Bounding ring returned by aoi.bounds().coordinates().getInfo().
BOUNDS = [[[76.20, 9.90], [76.36, 9.90], [76.36, 10.06], [76.20, 10.06], [76.20, 9.90]]]
N_SCENES = 37

mapid_calls = {"n": 0}


class _TileFetcher:
    def __init__(self, n):
        self.url_format = f"https://earthengine.example/tiles/{n}/{{z}}/{{x}}/{{y}}"


class _Chain:
    """Any Earth Engine expression. `kind` remembers what the chain last asked for."""

    def __init__(self, kind=None):
        self._kind = kind

    def __call__(self, *a, **kw):
        return _Chain(self._kind)

    def __getattr__(self, name):
        if name == "getMapId":
            def get_map_id(*a, **kw):
                mapid_calls["n"] += 1
                return {"tile_fetcher": _TileFetcher(mapid_calls["n"])}
            return get_map_id
        if name == "getInfo":
            def get_info(*a, **kw):
                if self._kind == "coordinates":
                    return BOUNDS
                if self._kind == "size":
                    return N_SCENES
                if self._kind == "sample":
                    return {"features": [{"properties": dict(r)} for r in SAMPLE_ROWS]}
                return None
            return get_info
        # Remember the calls whose results the planner actually reads.
        if name in ("coordinates", "size", "sample"):
            return _Chain(name)
        return _Chain(self._kind)

    # Earth Engine objects are sometimes used where a value is expected.
    def __iter__(self):
        return iter(())


def _module():
    m = types.ModuleType("ee")

    def initialize(*a, **kw):
        return None

    m.Initialize = initialize
    m.Authenticate = lambda *a, **kw: None
    for name in ("Image", "ImageCollection", "FeatureCollection", "Feature", "Geometry",
                 "Filter", "Reducer", "Kernel", "Number", "String", "List", "Dictionary",
                 "Date", "Array", "Algorithms"):
        setattr(m, name, _Chain(name.lower()))
    return m


def install():
    """Put the fake in sys.modules. Call before importing anything that imports `ee`."""
    sys.modules["ee"] = _module()
    return sys.modules["ee"]


def reset():
    mapid_calls["n"] = 0
