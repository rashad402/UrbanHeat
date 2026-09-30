"""MODIS -> Landsat temporal gap-filling / downscaling (plan §5.4).

Kerala's monsoon (Jun-Sep) leaves multi-month gaps in cloud-free Landsat thermal coverage.
Fit a regression of cloud-free Landsat LST on MODIS LST (+ spectral indices) during clear
periods, then apply it to downscale MODIS 1 km -> 30 m during gaps.

Tag every filled pixel with lst_source = 'modis_fill' so evaluation can exclude/down-weight it.
"""


def fit_downscaling_model(landsat_lst, modis_lst, indices):
    """Fit LST_landsat ~ f(LST_modis, ndvi, ndbi, albedo) on co-located clear-sky pixels."""
    raise NotImplementedError


def apply_downscaling(model, modis_lst, indices):
    """Predict 30 m LST for gap periods. Returns filled LST + a source mask."""
    raise NotImplementedError
