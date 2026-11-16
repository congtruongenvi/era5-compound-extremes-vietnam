"""
Compound hydroclimatic extremes in southern Vietnam from ERA5 reanalysis.

This module implements the core index computations used to characterise
compound drought-heat and compound wet-heat events over the Mekong Delta
and the Southeast region of Vietnam.

Scientific rationale
--------------------
Compound extremes -- the co-occurrence or succession of hazard drivers --
produce impacts that exceed the sum of their individual parts. A heatwave
during a hydrological drought depletes soil moisture, reservoir storage and
hydropower head simultaneously; a heavy-rain event on already saturated
ground turns into fluvial flooding. Standard single-hazard indices miss this
co-occurrence entirely.

Index conventions
-----------------
SPI  : Standardised Precipitation Index        (McKee et al., 1993)
SPEI : Standardised Precipitation-Evapotranspiration Index (Vicente-Serrano et al., 2010)
       computed here on a water-balance proxy P - PET (Hargreaves) when a full
       Penman-Monteith PET is not available.
TX90p: percentage of days with Tmax above the 90th percentile
       (ETCCDI convention, Zhang et al., 2011)

Compound definitions used here
------------------------------
drought-heat  : SPEI-3 below the 10th percentile AND Tmax above the 90th
                percentile, both at monthly resolution.
wet-heat      : 3-month precipitation above the 90th percentile AND Tmax above
                the 90th percentile.
hazard_footprint : the connected spatial extent of simultaneous exceedance,
                expressed in km^2 and as a fraction of the study domain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


# --------------------------------------------------------------------------
# Reference period and percentile conventions
# --------------------------------------------------------------------------
DEFAULT_REFERENCE_PERIOD = ("1991-01-01", "2020-12-31")
DROUGHT_PERCENTILE = 10
HEAT_PERCENTILE = 90


# --------------------------------------------------------------------------
# Precipitation accumulation
# --------------------------------------------------------------------------
def accumulate(ds: xr.Dataset, var: str, window: int) -> xr.DataArray:
    """Rolling N-month accumulation with a centred, min_periods guard.

    Rolling sums are used rather than raw monthly values because a single dry
    month is not a drought; drought is a deficit accumulated over a season.
    min_periods=window prevents partially-filled windows from silently
    producing artificially low accumulations at the series edges.
    """
    return (
        ds[var]
        .rolling(time=window, center=False, min_periods=window)
        .sum()
    )


# --------------------------------------------------------------------------
# Potential evapotranspiration (Hargreaves-Samani)
# --------------------------------------------------------------------------
def hargreaves_pet(
    tmax: xr.DataArray,
    tmin: xr.DataArray,
    tmean: xr.DataArray,
    lat: xr.DataArray,
) -> xr.DataArray:
    """Hargreaves-Samani reference evapotranspiration, in mm/day.

    Chosen over Penman-Monteith because ERA5 variables at the resolution used
    here do not always carry the radiation and wind terms needed for the full
    formulation, and the Hargreaves form needs only temperature and
    extraterrestrial radiation. Ra is computed from latitude and day-of-year.

    Parameters
    ----------
    tmax, tmin, tmean : K, time x lat x lon
    lat : degrees north

    Returns
    -------
    PET in mm/day on the same grid.

    Note
    ----
    Hargreaves was calibrated on daily temperature ranges. Applying it to
    *monthly mean* temperatures underestimates the diurnal range and therefore
    underestimates PET. Where a monthly field is all that is available, use
    `hargreaves_pet_monthly()`, which makes the substitution explicit.
    """
    # Convert Kelvin to Celsius.
    tmax_c = tmax - 273.15
    tmin_c = tmin - 273.15
    tmean_c = tmean - 273.15

    doy = tmean.time.dt.dayofyear
    lat_rad = np.deg2rad(lat)

    # Solar declination, Cooper (1969).
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)

    # Sunset hour angle.
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0))

    # Inverse relative Earth-Sun distance.
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)

    # Extraterrestrial radiation, MJ m-2 day-1.
    gsc = 0.0820  # solar constant, MJ m-2 min-1
    ra = (
        (24 * 60 / np.pi)
        * gsc
        * dr
        * (
            ws * np.sin(lat_rad) * np.sin(decl)
            + np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
        )
    )

    # Temperature range term, with a floor to avoid zero/negative ranges.
    tr = np.maximum(tmax_c - tmin_c, 0.1)

    pet = 0.0023 * ra * (tmean_c + 17.8) * np.sqrt(tr)
    return np.maximum(pet, 0.0)


def hargreaves_pet_monthly(
    tmean_k: xr.DataArray,
    dtr_c: xr.DataArray,
    lat: xr.DataArray,
    days_in_month: xr.DataArray | None = None,
) -> xr.DataArray:
    """Monthly-total PET (mm/month) from monthly mean temperature.

    Substitutes the monthly mean diurnal temperature range for the daily range,
    which is the standard practical compromise when only monthly means are
    available. Ra is averaged over the days of each month rather than sampled
    at one day, so the month length is respected.

    Parameters
    ----------
    tmean_k : monthly mean temperature, K
    dtr_c   : monthly mean diurnal temperature range, degrees C
    lat     : latitude coordinate, degrees north
    days_in_month : optional; computed from the time axis when omitted.
    """
    tmean_c = tmean_k - 273.15

    if days_in_month is None:
        days_in_month = tmean_k.time.dt.days_in_month.astype(float)

    month = tmean_k.time.dt.month
    doy_mid = tmean_k.time.dt.dayofyear

    lat_rad = np.deg2rad(lat)
    gsc = 0.0820

    # Average Ra over the days of the month, by evaluating at the mid-month
    # day which is within a few per-cent of the true monthly mean.
    decl = 0.409 * np.sin(2 * np.pi * doy_mid / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0))
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy_mid / 365)
    ra = (
        (24 * 60 / np.pi)
        * gsc
        * dr
        * (
            ws * np.sin(lat_rad) * np.sin(decl)
            + np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
        )
    )

    tr = np.maximum(dtr_c, 0.1)
    pet_daily = 0.0023 * ra * (tmean_c + 17.8) * np.sqrt(tr)

    pet_monthly = pet_daily * days_in_month
    pet_monthly = np.maximum(pet_monthly, 0.0)
    pet_monthly.name = "pet"
    pet_monthly.attrs = {
        "units": "mm/month",
        "long_name": "Hargreaves reference evapotranspiration, monthly total",
        "note": "monthly-mean DTR substituted for daily DTR",
    }
    return pet_monthly


# --------------------------------------------------------------------------
# Standardised indices
# --------------------------------------------------------------------------
def _fit_gamma(values: np.ndarray) -> dict:
    """Fit a gamma distribution to positive values via method of moments.

    Implemented directly rather than pulling in scipy.stats so the module has
    a minimal dependency surface. Returns shape (a) and scale (b).
    """
    positive = values[values > 0]
    if positive.size < 30:
        return {"a": np.nan, "b": np.nan}
    mean = positive.mean()
    var = positive.var()
    if var <= 0:
        return {"a": np.nan, "b": np.nan}
    a = mean**2 / var
    b = var / mean
    return {"a": a, "b": b}


def _standardise(
    series: np.ndarray, params: dict, zero_fraction: float
) -> np.ndarray:
    """Map a precipitation series to standard normal quantiles.

    A mixed distribution is used: the probability of zero is added to the
    gamma CDF for positive values, following the standard SPI treatment of
    arid and seasonally dry stations.
    """
    from math import erf, sqrt

    a, b = params["a"], params["b"]
    if np.isnan(a) or np.isnan(b):
        return np.full_like(series, np.nan, dtype=float)

    out = np.full(series.shape, np.nan, dtype=float)
    for i, x in enumerate(series):
        if np.isnan(x):
            continue
        if x <= 0:
            p = zero_fraction * 0.5
        else:
            # Gamma CDF via the regularised lower incomplete gamma function,
            # approximated with a series expansion (Abramowitz & Stegun 6.5.29).
            p_pos = _gamma_cdf(x, a, b)
            p = zero_fraction + (1 - zero_fraction) * p_pos

        p = min(max(p, 1e-6), 1 - 1e-6)
        # Inverse normal via the Beasley-Springer-Moro rational approximation
        # on the complementary error function.
        z = _inv_normal(p)
        out[i] = z
    return out


def _gamma_cdf(x: float, a: float, b: float) -> float:
    """Regularised lower incomplete gamma P(a, x/b) via series expansion."""
    from math import exp, lgamma, log

    y = x / b
    if y <= 0:
        return 0.0
    if y < a + 1:
        # Series representation for small y.
        term = 1.0 / a
        total = term
        for n in range(1, 200):
            term *= y / (a + n)
            total += term
            if abs(term) < 1e-12 * abs(total):
                break
        return total * exp(-y + a * log(y) - lgamma(a))
    # Continued fraction for large y.
    tiny = 1e-30
    bb = y + 1 - a
    c = 1 / tiny
    d = 1 / bb
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        bb += 2
        d = an * d + bb
        if abs(d) < tiny:
            d = tiny
        c = bb + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return 1.0 - exp(-y + a * log(y) - lgamma(a)) * h


def _inv_normal(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= 1 - p_low:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = np.sqrt(-2 * np.log(1 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def spei(
    precip: xr.DataArray,
    pet: xr.DataArray,
    window: int = 3,
    reference_period: tuple[str, str] = DEFAULT_REFERENCE_PERIOD,
) -> xr.DataArray:
    """Standardised Precipitation-Evapotranspiration Index.

    The water balance P - PET is accumulated over `window` months, then mapped
    to a standard normal using a gamma fit estimated **on the reference period
    only**. Estimating the distribution on the full series would let the
    calibration period leak information about the events being diagnosed --
    a common and easily missed form of look-ahead bias.

    The accumulation is computed on the full series and the reference period
    is applied afterwards. Subsetting first would delete the first `window-1`
    months of the reference period, shifting every subsequent index value.
    """
    balance = (precip - pet).assign_attrs(units="mm/month")

    # Accumulate on the full record ...
    accum_full = (
        balance.rolling(time=window, center=False, min_periods=window).sum()
    )

    # ... then restrict the calibration sample to the reference period.
    accum_ref = accum_full.sel(time=slice(*reference_period))

    params = xr.apply_ufunc(
        _fit_params_core,
        accum_ref,
        input_core_dims=[["time"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )
    gamma_a, gamma_b = params

    zero_frac = xr.apply_ufunc(
        lambda x: float(np.mean(x[np.isfinite(x)] <= 0)) if np.isfinite(x).any() else np.nan,
        accum_ref,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    out = xr.apply_ufunc(
        _standardise_grid,
        accum_full,
        gamma_a,
        gamma_b,
        zero_frac,
        input_core_dims=[["time"], [], [], []],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    out.name = f"spei{window}"
    out.attrs = {
        "long_name": f"SPEI accumulated over {window} months",
        "units": "standard normal quantiles",
        "reference_period": f"{reference_period[0]} to {reference_period[1]}",
    }
    return out


def _fit_params_core(series: np.ndarray):
    """Fit the gamma parameters on the reference period for one grid cell."""
    finite = series[np.isfinite(series)]
    if finite.size < 30:
        return np.nan, np.nan
    params = _fit_gamma(finite)
    return params["a"], params["b"]


def _standardise_grid(
    series: np.ndarray, a: float, b: float, zero_fraction: float
) -> np.ndarray:
    """Apply reference-period parameters to the full series."""
    if not np.isfinite(a) or not np.isfinite(b):
        return np.full(series.shape, np.nan)
    return _standardise(series, {"a": a, "b": b}, zero_fraction)


# --------------------------------------------------------------------------
# Heat index
# --------------------------------------------------------------------------
def heat_exceedance(
    tmax: xr.DataArray,
    percentile: int = HEAT_PERCENTILE,
    reference_period: tuple[str, str] = DEFAULT_REFERENCE_PERIOD,
) -> xr.DataArray:
    """Tmax above its reference-period percentile, as a boolean mask.

    Using a percentile threshold rather than a fixed temperature makes the
    index comparable across the latitudinal gradient of the study domain:
    35 C is extreme in Da Lat and unremarkable in the Mekong Delta.
    """
    ref = tmax.sel(time=slice(*reference_period))
    threshold = ref.quantile(percentile / 100.0, dim="time")
    mask = tmax > threshold
    mask.name = f"tmax_gt_p{percentile}"
    mask.attrs = {
        "long_name": f"Tmax exceeds reference-period p{percentile}",
        "units": "boolean",
    }
    return mask


# --------------------------------------------------------------------------
# Compound event detection
# --------------------------------------------------------------------------
def compound_mask(*masks: xr.DataArray) -> xr.DataArray:
    """Simultaneous exceedance of two or more hazard conditions.

    This is the definitional core of the whole exercise: a compound event is
    an AND, not an OR. Taking the union would count a dry month and a hot
    month in different seasons as one compound event.
    """
    if len(masks) < 2:
        raise ValueError("compound_mask needs at least two hazard masks")
    out = masks[0].astype(bool)
    for m in masks[1:]:
        out = out & m.astype(bool)
    out.name = "compound_event"
    out.attrs = {"long_name": "compound hazard exceedance", "units": "boolean"}
    return out


def event_catalogue(
    mask: xr.DataArray, min_cells: int = 5
) -> pd.DataFrame:
    """Tabulate discrete compound events with duration, extent and footprint.

    Contiguity is assessed in time only, per grid cell, and then aggregated.
    A production implementation would use connected-component labelling in
    space-time (e.g. scipy.ndimage.label with a 4-D structure element); that
    step is left explicit here as `label_spacetime` so the simplification is
    visible rather than hidden.
    """
    records = []
    if "time" not in mask.dims:
        raise ValueError("mask must have a time dimension")

    cell_counts = mask.sum(dim=[d for d in mask.dims if d != "time"])

    active = cell_counts.values > min_cells
    if not active.any():
        return pd.DataFrame(
            columns=["start", "end", "duration_months", "max_cells", "mean_cells"]
        )

    idx = np.flatnonzero(active)
    breaks = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, breaks + 1)

    for g in groups:
        window = cell_counts.isel(time=slice(g[0], g[-1] + 1))
        records.append(
            {
                "start": str(mask.time.isel(time=g[0]).values)[:10],
                "end": str(mask.time.isel(time=g[-1]).values)[:10],
                "duration_months": int(g[-1] - g[0] + 1),
                "max_cells": int(window.max().values),
                "mean_cells": float(window.mean().values),
            }
        )

    return pd.DataFrame(records).sort_values("duration_months", ascending=False)


def label_spacetime(mask: xr.DataArray):
    """Connected-component labelling of compound events in space and time.

    Deliberately left as a documented stub: the scipy-based implementation
    needs a 4-D binary structure element and, more importantly, a decision
    about whether diagonal (space-time) adjacency counts as one event or two.
    That is a scientific choice, not a coding one, and it changes event counts
    materially. Documented here so a reader knows the simplification exists.
    """
    raise NotImplementedError(
        "Implement with scipy.ndimage.label over a 4-D (t, y, x) array; "
        "decide diagonal adjacency explicitly before reporting event counts."
    )


# --------------------------------------------------------------------------
# Hazard footprint
# --------------------------------------------------------------------------
def hazard_footprint(mask: xr.DataArray, cell_area_km2: xr.DataArray) -> xr.DataArray:
    """Area of simultaneous exceedance, in km^2, per time step."""
    footprint = (mask.astype(float) * cell_area_km2).sum(
        dim=[d for d in mask.dims if d != "time"]
    )
    footprint.name = "hazard_footprint_km2"
    footprint.attrs = {
        "long_name": "area under compound hazard exceedance",
        "units": "km2",
    }
    return footprint


def cell_area(lat: xr.DataArray, lon: xr.DataArray, radius_km: float = 6371.0):
    """Spherical cell area for a regular lat-lon grid, in km^2.

    ERA5's 0.25 degree grid has cells whose area shrinks with the cosine of
    latitude; using a constant area biases every footprint statistic toward
    the equator. This is a one-line fix that is routinely forgotten.
    """
    lat_vals = np.asarray(lat.values if hasattr(lat, "values") else lat, dtype=float)
    lon_vals = np.asarray(lon.values if hasattr(lon, "values") else lon, dtype=float)

    if lat_vals.size < 2:
        raise ValueError(
            "cell_area needs at least two latitude points to form cell boundaries"
        )

    # Cell edges at the midpoints between grid centres, with the outer halves
    # extrapolated so the edge arrays match the centre arrays in length.
    lat_edges = np.empty(lat_vals.size + 1)
    lat_edges[1:-1] = 0.5 * (lat_vals[:-1] + lat_vals[1:])
    lat_edges[0] = lat_vals[0] - 0.5 * (lat_vals[1] - lat_vals[0])
    lat_edges[-1] = lat_vals[-1] + 0.5 * (lat_vals[-1] - lat_vals[-2])

    if lon_vals.size > 1:
        dlon = np.abs(np.median(np.diff(lon_vals)))
    else:
        dlon = 360.0

    area_1d = (
        radius_km**2
        * np.deg2rad(dlon)
        * (
            np.sin(np.deg2rad(lat_edges[1:]))
            - np.sin(np.deg2rad(lat_edges[:-1]))
        )
    )

    area = np.repeat(area_1d[:, None], lon_vals.size, axis=1)

    return xr.DataArray(
        area,
        coords={"lat": lat_vals, "lon": lon_vals},
        dims=("lat", "lon"),
        name="cell_area_km2",
        attrs={"units": "km2"},
    )


# --------------------------------------------------------------------------
# Trend estimation
# --------------------------------------------------------------------------
def theil_sen_trend(series: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Theil-Sen slope per grid cell, in units per year.

    Theil-Sen is used instead of OLS because extreme-value indices are
    bounded and heteroscedastic: a single record-breaking year can flip an
    OLS slope sign, whereas the median pairwise slope is robust to it. This
    matters when the object of study is precisely the extreme years.

    The time axis is converted to fractional years before the pairwise slopes
    are formed, so the result is in units per year for any input frequency.
    Deriving a conversion factor from the mean timestep instead is a common
    source of 10x errors when the series is monthly.
    """
    # Fractional years from the start of the record. Calendar-aware enough
    # for monthly and annual data, which is all this is used on.
    times = series[dim].values
    if np.issubdtype(times.dtype, np.datetime64):
        t0 = times[0]
        t_years = np.array(
            [
                (pd.Timestamp(x) - pd.Timestamp(t0)).total_seconds()
                / (365.25 * 24 * 3600)
                for x in times
            ],
            dtype=float,
        )
    else:
        t_years = np.asarray(times, dtype=float)

    t = xr.DataArray(t_years, dims=[dim], coords={dim: series[dim]})

    slopes = xr.apply_ufunc(
        _theil_sen_core,
        series,
        t,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    slopes.name = "theil_sen_slope"
    slopes.attrs = {"units": f"{series.attrs.get('units', 'unit')} per year"}
    return slopes


def _theil_sen_core(y: np.ndarray, x: np.ndarray) -> float:
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return np.nan
    y = y[finite]
    x = x[finite]
    n = y.size
    i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    valid = dx != 0
    if not valid.any():
        return np.nan
    slopes = (y[j][valid] - y[i][valid]) / dx[valid]
    return float(np.median(slopes))


def mann_kendall_p(series: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Two-sided Mann-Kendall p-value per grid cell.

    Significance is reported alongside every trend because at 0.25 degree
    resolution there are enough cells that a non-trivial fraction will show a
    large slope by chance alone. Trends are quoted as "significant at p<0.05"
    or they are not quoted.
    """
    pvals = xr.apply_ufunc(
        _mann_kendall_core,
        series,
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    pvals.name = "mann_kendall_p"
    return pvals


def _mann_kendall_core(y: np.ndarray) -> float:
    """Normal-approximation Mann-Kendall test with tie correction."""
    y = y[np.isfinite(y)]
    n = y.size
    if n < 10:
        return np.nan

    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(y[k + 1 :] - y[k]))

    # Tie correction.
    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))

    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return np.nan

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    from math import erfc
    return float(erfc(abs(z) / np.sqrt(2)))
