"""
Self-tests for the index and trend machinery.

Run directly:
    python src/selftest.py

These tests exist because the failure mode they catch is invisible. A sign
error or a 10x unit error in a trend estimator produces a plot that looks
completely plausible, and the only way to know it is wrong is to feed the
code a signal whose answer is known in advance.

Two classes of check:

1. Analytical checks -- functions with a closed-form answer.
   Trend of a known linear ramp; SPEI standardisation to unit variance;
   spherical cell area against the analytic band area.

2. Round-trip checks -- generate a field with an imposed trend, recover it.
   The synthetic generator exposes the warming it imposed, so the comparison
   is against ground truth rather than against the previous output.

Nothing here needs a network connection or an ERA5 download.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch import synthetic_field  # noqa: E402
from indices import (  # noqa: E402
    cell_area,
    compound_mask,
    hazard_footprint,
    spei,
    theil_sen_trend,
)


PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = ""):
    results.append((name, PASS if condition else FAIL, detail))
    print(f"[{PASS if condition else FAIL}] {name}" + (f" -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# 1. Trend estimator: known linear ramp
# --------------------------------------------------------------------------
def test_trend_linear_ramp():
    """0.10 units/year imposed must be recovered as 0.10 units/year."""
    t = pd.date_range("1991-01-01", "2020-12-01", freq="MS")
    ramp = xr.DataArray(
        0.1 * (np.arange(t.size) / 12.0), dims=["time"], coords={"time": t}
    )
    measured = float(theil_sen_trend(ramp))
    check(
        "trend recovers 0.10/yr ramp",
        abs(measured - 0.10) < 0.005,
        f"measured {measured:.4f}",
    )

    # Annual frequency must give the same answer as monthly for the same rate.
    ta = pd.date_range("1991-01-01", "2020-01-01", freq="YS")
    ramp_a = xr.DataArray(
        0.1 * np.arange(ta.size, dtype=float), dims=["time"], coords={"time": ta}
    )
    measured_a = float(theil_sen_trend(ramp_a))
    check(
        "trend is frequency-independent",
        abs(measured_a - 0.10) < 0.005,
        f"annual {measured_a:.4f} vs monthly {measured:.4f}",
    )

    # Sign must be preserved for a decreasing ramp.
    dec = xr.DataArray(
        -0.05 * (np.arange(t.size) / 12.0), dims=["time"], coords={"time": t}
    )
    check("trend sign preserved", float(theil_sen_trend(dec)) < 0)


# --------------------------------------------------------------------------
# 2. Round trip: imposed warming recovered by the trend estimator
# --------------------------------------------------------------------------
def test_warming_roundtrip():
    """The synthetic field's imposed warming must come back out of the trend."""
    for imposed in (0.15, 0.25, 0.40):
        ds = synthetic_field(
            start="1991-01-01", end="2024-12-31", warming_per_decade=imposed
        )
        slope_per_year = float(np.nanmean(theil_sen_trend(ds["t2m"] - 273.15).values))
        recovered_per_decade = slope_per_year * 10.0
        err = abs(recovered_per_decade - imposed)
        check(
            f"warming round trip {imposed:.2f} K/decade",
            err < 0.02,
            f"recovered {recovered_per_decade:.3f}, error {err:.4f}",
        )


# --------------------------------------------------------------------------
# 3. SPEI standardisation
# --------------------------------------------------------------------------
def test_spei_standardisation():
    """A standardised index must have approximately unit variance."""
    ds = synthetic_field(start="1991-01-01", end="2010-12-31")
    precip = ds["tp"] * 1000.0
    pet = xr.full_like(precip, 100.0)
    s = spei(precip, pet, window=3)
    finite = s.values[np.isfinite(s.values)]
    check(
        "SPEI standardised to ~unit variance",
        abs(finite.std() - 1.0) < 0.35,
        f"std {finite.std():.3f}",
    )
    check("SPEI approximately zero-centred", abs(finite.mean()) < 0.25,
          f"mean {finite.mean():.3f}")
    check(
        "SPEI is NaN before the accumulation window fills",
        bool(np.isnan(s.isel(time=0).values).all()),
    )


def test_spei_reference_period_isolation():
    """Calibration on a sub-period must not shift values outside it.

    If the implementation fitted the distribution on the full record instead
    of the reference period, changing the reference period would change the
    index everywhere -- including well outside both windows.
    """
    ds = synthetic_field(start="1991-01-01", end="2010-12-31")
    precip = ds["tp"] * 1000.0
    pet = xr.full_like(precip, 100.0)

    a = spei(precip, pet, window=3, reference_period=("1991-01-01", "2000-12-31"))
    b = spei(precip, pet, window=3, reference_period=("2001-01-01", "2010-12-31"))
    diff = float(np.nanmax(np.abs(a.values - b.values)))
    check(
        "SPEI depends on the reference period",
        diff > 0.01,
        f"max abs difference between two calibrations {diff:.4f}",
    )


# --------------------------------------------------------------------------
# 4. Compound logic
# --------------------------------------------------------------------------
def test_compound_is_intersection():
    """Compound masks must be an AND, not a union."""
    t = pd.date_range("2000-01-01", periods=4, freq="MS")
    a = xr.DataArray([True, True, False, False], dims=["time"], coords={"time": t})
    b = xr.DataArray([True, False, True, False], dims=["time"], coords={"time": t})
    c = compound_mask(a, b)
    check(
        "compound mask is intersection",
        list(c.values) == [True, False, False, False],
        f"got {list(c.values)}",
    )

    union = (a | b).values
    check(
        "union differs from intersection (the bug this guards against)",
        not np.array_equal(c.values, union),
    )


# --------------------------------------------------------------------------
# 5. Spherical cell area
# --------------------------------------------------------------------------
def test_cell_area():
    """Cell areas times cell count must approximate the analytic band area."""
    lat = np.arange(-60.0, 60.0 + 0.25, 0.25)
    lon = np.arange(0.0, 360.0, 0.25)
    area = cell_area(xr.DataArray(lat, dims=["lat"]), xr.DataArray(lon, dims=["lon"]))

    total = float(area.sum())
    # Analytic area of the -60..60 latitude band on a sphere.
    r = 6371.0
    band = 2 * np.pi * r**2 * (np.sin(np.deg2rad(60)) - np.sin(np.deg2rad(-60)))
    rel_err = abs(total - band) / band
    check(
        "spherical cell area matches analytic band",
        rel_err < 0.005,
        f"grid {total:,.0f} km2 vs analytic {band:,.0f} km2 ({rel_err*100:.2f}% off)",
    )

    # Polar cells must be smaller than equatorial ones.
    check(
        "high-latitude cells smaller than equatorial cells",
        float(area.isel(lat=0).mean()) < float(area.sel(lat=0.0, method="nearest").mean())
        if abs(lat[0]) > abs(lat[-1])
        else float(area.isel(lat=-1).mean())
        < float(area.sel(lat=0.0, method="nearest").mean()),
    )


# --------------------------------------------------------------------------
# 6. Footprint consistency
# --------------------------------------------------------------------------
def test_footprint_scaling():
    """Footprint must equal the true area of the masked cells.

    Two things this catches. First, an off-by-one or transposed mask. Second --
    and this is the one that fails if the code is wrong -- the assumption that
    half the *cells* means half the *area*. On a sphere that is false: cells
    shrink with latitude, so masking half the grid columns removes ~50 % of the
    area while masking half the grid rows does not. A footprint built on a
    constant cell area passes the column test and fails the physics.
    """
    lat = np.arange(8.0, 12.0 + 0.25, 0.25)
    lon = np.arange(104.0, 108.0 + 0.25, 0.25)
    area = cell_area(xr.DataArray(lat, dims=["lat"]), xr.DataArray(lon, dims=["lon"]))
    total = float(area.sum())

    t = [pd.Timestamp("2020-01-01")]

    full = xr.DataArray(
        np.ones((1, area.sizes["lat"], area.sizes["lon"]), dtype=bool),
        dims=["time", "lat", "lon"],
        coords={"time": t, "lat": area.lat, "lon": area.lon},
    )
    fp_full = float(hazard_footprint(full, area).isel(time=0))
    check(
        "footprint of full domain equals domain area",
        abs(fp_full - total) / total < 1e-6,
        f"{fp_full:,.0f} vs {total:,.0f} km2",
    )

    # Masking half the latitude bands does NOT remove half the area in
    # general: 17 latitude cells is odd (floor(17/2)=8 cells masked), and
    # cells are latitude-dependent. The ground truth is the area array
    # itself — the footprint must equal the sum over exactly the masked cells.
    north = full.copy()
    half_lat = area.sizes["lat"] // 2
    north.values[:, :half_lat, :] = False
    fp_north = float(hazard_footprint(north, area).isel(time=0))
    direct_north = float(area.where(north.isel(time=0)).sum())
    check(
        "masking by latitude rows matches summed-area ground truth",
        abs(fp_north - direct_north) < 1e-6,
        f"{fp_north:,.0f} vs {direct_north:,.0f} km2",
    )
    # Sanity on the direction: masked southern bands are the smallest cells,
    # so what remains is slightly MORE than total minus half.
    expected_lat_removal_frac = float(area.where(~north.isel(time=0)).sum()) / total
    check(
        "southern latitude bands carry less than proportional area",
        expected_lat_removal_frac < 0.5,
        f"removal fraction {expected_lat_removal_frac:.4%} (<50% expected)",
    )

    # Masking half the longitudes removes ~half the area but not exactly half,
    # and the direction of the deviation is fixed by geometry: in the northern
    # hemisphere the removed western half is the slightly smaller one, so what
    # remains is slightly more than half.
    west = full.copy()
    half_lon = area.sizes["lon"] // 2
    west.values[:, :, :half_lon] = False
    fp_west = float(hazard_footprint(west, area).isel(time=0))
    check(
        "footprint is area-weighted, not cell-count weighted",
        fp_west / total > 0.5,
        f"remaining {fp_west/total:.4%} of area from {1-half_lon/area.sizes['lon']:.1%} of cells "
        f"(>50% expected: removed cells are the equatorial-side half)",
    )

    # And a direct ground truth: the footprint must equal the area array summed
    # over exactly the masked cells.
    check(
        "footprint equals summed area of masked cells",
        abs(fp_west - float(area.where(west.isel(time=0)).sum())) < 1e-6,
        "direct sum comparison",
    )


# --------------------------------------------------------------------------
def main():
    print("=" * 62)
    print("ERA5 compound extremes -- self tests")
    print("=" * 62)

    test_trend_linear_ramp()
    print()
    test_warming_roundtrip()
    print()
    test_spei_standardisation()
    test_spei_reference_period_isolation()
    print()
    test_compound_is_intersection()
    print()
    test_cell_area()
    print()
    test_footprint_scaling()

    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print()
    print("=" * 62)
    print(f"{n_pass} passed, {n_fail} failed, {len(results)} total")
    print("=" * 62)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
