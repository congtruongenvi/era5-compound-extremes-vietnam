"""
ERA5 retrieval and subsetting.

Kept deliberately small: the point of this module is that the analysis in
`indices.py` can be run against either a real CDS download or a synthetic
field, so that the science logic is testable without a licence or a 40 GB
download.

Requires a configured CDS API key at ~/.cdsapirc for the real path:
    url: https://cds.climate.copernicus.eu/api
    key: <your-key>

Accept the ERA5 licence once at https://cds.climate.copernicus.eu before
the first request, or every request will fail with a licence error rather
than a data error.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Study domain: southern Vietnam -- Mekong Delta, Southeast region,
# and the southern Central Highlands fringe.
SOUTHERN_VIETNAM = {
    "north": 12.5,
    "west": 104.0,
    "south": 8.3,
    "east": 107.8,
}

# Variables needed for the compound analysis.
ERA5_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
]

# ERA5 monthly means are used for SPEI at monthly resolution. For daily
# ETCCDI indices the hourly or daily product is required instead -- see
# `fetch_daily()`.
ERA5_MONTHLY_DATASET = "reanalysis-era5-single-levels-monthly-means"


def fetch_monthly(
    start: str,
    end: str,
    bbox: dict | None = None,
    variables: list[str] | None = None,
    cache_dir: str = "data/raw",
) -> xr.Dataset:
    """Download and open ERA5 monthly means for the study domain.

    Returns a dataset with `t2m` (K) and `tp` (m/month), renamed to explicit
    units so downstream code cannot silently mix accumulation windows.
    """
    import cdsapi
    import os

    bbox = bbox or SOUTHERN_VIETNAM
    variables = variables or ERA5_VARIABLES

    area = [bbox["north"], bbox["west"], bbox["south"], bbox["east"]]

    start_year, start_month = (int(x) for x in start.split("-")[:2])
    end_year, end_month = (int(x) for x in end.split("-")[:2])

    years = [str(y) for y in range(start_year, end_year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    target = os.path.join(cache_dir, f"era5_monthly_{start_year}_{end_year}.nc")
    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(target):
        client = cdsapi.Client()
        client.retrieve(
            ERA5_MONTHLY_DATASET,
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": variables,
                "year": years,
                "month": months,
                "time": "00:00",
                "area": area,
                "data_format": "netcdf",
            },
            target,
        )

    ds = xr.open_dataset(target)

    # Rename to short, unambiguous names.
    rename = {}
    if "t2m" in ds:
        rename["t2m"] = "t2m"
    if "tp" in ds:
        rename["tp"] = "tp"
    ds = ds.rename(rename)

    # Normalise coordinate names to lat/lon and ensure ascending order --
    # CDS returns descending latitude by default, which silently reverses
    # every latitude-weighted calculation.
    ds = _normalise_coords(ds)

    ds["tp"].attrs["units"] = "m/month"
    ds["t2m"].attrs["units"] = "K"

    return ds


def _normalise_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for candidate, canonical in (("latitude", "lat"), ("longitude", "lon")):
        if candidate in ds.coords:
            rename[candidate] = canonical
    if rename:
        ds = ds.rename(rename)

    for dim in ("lat", "lon"):
        if dim in ds.coords and ds[dim].values[0] > ds[dim].values[-1]:
            ds = ds.isel({dim: slice(None, None, -1)})
    return ds


def fetch_daily(
    start: str,
    end: str,
    bbox: dict | None = None,
    cache_dir: str = "data/raw",
) -> xr.Dataset:
    """ERA5 hourly fields aggregated to daily Tmax / Tmin / precipitation.

    Needed for ETCCDI-style daily indices (TX90p). Separated from
    `fetch_monthly()` because the hourly product is roughly two orders of
    magnitude larger for the same period and domain.
    """
    raise NotImplementedError(
        "Use dataset 'reanalysis-era5-single-levels' with "
        "'2m_temperature' at times 00:00-23:00, then "
        "ds.resample(time='1D').max() for Tmax and .sum() for precipitation. "
        "Kept unimplemented so the monthly workflow stays the default path."
    )


def synthetic_field(
    start: str = "1991-01-01",
    end: str = "2024-12-31",
    bbox: dict | None = None,
    resolution: float = 0.25,
    seed: int = 42,
    warming_per_decade: float = 0.25,
) -> xr.Dataset:
    """Physically-plausible synthetic ERA5 substitute with a known trend.

    Purpose: exercise the full index -> compound -> footprint -> trend chain
    without a CDS account, and verify that a trend the code *should* detect,
    it does detect.

    Construction
    ------------
    - Temperature: seasonal cycle + monotonic warming at a controlled rate
      (degrees per decade) + spatially-correlated noise.
    - Precipitation: monsoon-like seasonal cycle (May-November peak), skewed,
      with a fixed baseline so rank-based indices have something to rank.

    The imposed warming rate is returned in the attributes so a trend test can
    be checked against ground truth rather than eyeballed.
    """
    bbox = bbox or SOUTHERN_VIETNAM

    time = xr.date_range(start=start, end=end, freq="MS")
    lat = np.arange(bbox["south"], bbox["north"] + resolution / 2, resolution)
    lon = np.arange(bbox["west"], bbox["east"] + resolution / 2, resolution)

    rng = np.random.default_rng(seed)

    n_t, n_lat, n_lon = time.size, lat.size, lon.size
    t_years = np.array(
        [(t.year - time[0].year) + (t.month - 1) / 12 for t in time], dtype=float
    )
    month = np.array([t.month for t in time])

    # --- Temperature -----------------------------------------------------
    # Seasonal cycle: warm in April-May, modulated by latitude.
    seasonal = 27.0 + 3.0 * np.sin(2 * np.pi * (month - 3) / 12)
    lat_grad = -0.08 * (lat - lat.mean())

    # warming_per_decade is expressed per DECADE, so the per-year rate is a
    # tenth of it. Mixing the two is a silent 10x error that a trend test
    # against the imposed value catches immediately -- which is the point of
    # returning the imposed value in the attributes.
    warming = (warming_per_decade / 10.0) * t_years
    noise_t = rng.normal(0, 0.8, size=(n_t, n_lat, n_lon))

    t2m_c = (
        seasonal[:, None, None]
        + lat_grad[None, :, None]
        + warming[:, None, None]
        + noise_t
    )

    # --- Precipitation ---------------------------------------------------
    # Monsoon seasonality, May-November.
    monsoon = 1.0 + 1.6 * np.clip(
        np.sin(2 * np.pi * (month - 4) / 12), 0, None
    )
    base_p = 180.0  # mm/month baseline
    gamma_shape = 1.3
    raw = rng.gamma(gamma_shape, 1.0, size=(n_t, n_lat, n_lon))
    # Longitude gradient: wetter toward the coast.
    lon_grad = 0.02 * (lon - lon.mean())
    tp = base_p * monsoon[:, None, None] * raw / gamma_shape + lon_grad[None, None, :]
    tp = np.maximum(tp, 0.0)

    ds = xr.Dataset(
        {
            "t2m": (
                ("time", "lat", "lon"),
                t2m_c + 273.15,
                {"units": "K", "long_name": "2m temperature"},
            ),
            "tp": (
                ("time", "lat", "lon"),
                tp / 1000.0,
                {"units": "m/month", "long_name": "total precipitation"},
            ),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "source": "synthetic",
            "warming_per_decade_K": warming_per_decade,
            "note": (
                "Synthetic field for pipeline validation. Not ERA5, "
                "not observations. Imposed warming is the ground truth a "
                "trend test should recover."
            ),
        },
    )
    return ds
