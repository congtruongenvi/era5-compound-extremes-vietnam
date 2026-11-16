"""
End-to-end pipeline: fetch -> indices -> compound masks -> footprint -> trends -> figures.

Run with real ERA5:
    python src/pipeline.py --start 1991-01-01 --end 2024-12-31

Run with the synthetic validation field (no CDS account needed):
    python src/pipeline.py --synthetic

The synthetic path is not decoration. It is the regression test: an imposed
warming of 0.25 K/decade must come back out of the trend estimator as
approximately 0.25 K/decade. If it does not, either the estimator or the
index chain is broken, and knowing that before downloading 40 GB of ERA5 is
worth the twenty lines it costs.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch import SOUTHERN_VIETNAM, fetch_monthly, synthetic_field  # noqa: E402
from indices import (  # noqa: E402
    DEFAULT_REFERENCE_PERIOD,
    cell_area,
    compound_mask,
    event_catalogue,
    hargreaves_pet_monthly,
    hazard_footprint,
    heat_exceedance,
    mann_kendall_p,
    spei,
    theil_sen_trend,
)


def build_indices(ds: xr.Dataset, reference_period=DEFAULT_REFERENCE_PERIOD):
    """Compute the hazard indices from a raw ERA5-like dataset."""
    # Precipitation to mm/month for readability in figures and tables.
    # ERA5 stores total precipitation in metres of water equivalent.
    precip_mm = ds["tp"] * 1000.0
    precip_mm.attrs = {"units": "mm/month", "long_name": "total precipitation"}

    lat = ds["lat"] if "lat" in ds.coords else ds["latitude"]

    # Diurnal temperature range: ERA5 monthly means carry the true monthly
    # mean DTR in a separate variable when `2m_temperature` is combined with
    # min/max products. Where only a mean is available, a climatological DTR
    # of 8 K is used for southern Vietnam -- documented rather than silent.
    if "dtr" in ds:
        dtr = ds["dtr"]
    else:
        dtr = xr.full_like(ds["t2m"], 8.0)

    pet_monthly = hargreaves_pet_monthly(ds["t2m"], dtr, lat)

    spei3 = spei(precip_mm, pet_monthly, window=3, reference_period=reference_period)
    hot = heat_exceedance(ds["t2m"], 90, reference_period)

    # Drought condition: SPEI below its own reference-period 10th percentile.
    # Using the index's percentile rather than a fixed SPEI = -1.28 keeps the
    # event frequency stationary by construction, which is what makes the
    # *change in co-occurrence* interpretable.
    spei3_ref = spei3.sel(time=slice(*reference_period))
    drought_cond = spei3 < spei3_ref.quantile(0.10, dim="time")
    drought_cond.name = "drought_condition"

    wet3 = precip_mm.rolling(time=3, min_periods=3).sum()
    wet_ref = wet3.sel(time=slice(*reference_period))
    wet_cond = wet3 > wet_ref.quantile(0.90, dim="time")
    wet_cond.name = "wet_condition"

    compound_dh = compound_mask(drought_cond, hot)
    compound_wh = compound_mask(wet_cond, hot)

    return {
        "precip_mm": precip_mm,
        "pet_monthly": pet_monthly,
        "spei3": spei3,
        "hot": hot,
        "compound_drought_heat": compound_dh,
        "compound_wet_heat": compound_wh,
    }


def make_figures(indices: dict, ds: xr.Dataset, outdir: str = "figures"):
    """Footprint time series and trend map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    area = cell_area(ds["lat"], ds["lon"])

    # --- Footprint time series ------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for ax, key, colour, label in (
        (axes[0], "compound_drought_heat", "#c0392b", "Drought–heat"),
        (axes[1], "compound_wet_heat", "#2471a3", "Wet–heat"),
    ):
        mask = indices[key]
        fp = hazard_footprint(mask, area)
        # Rolling 12-month mean so the seasonal cycle does not bury the signal.
        fp_smooth = fp.rolling(time=12, min_periods=6).mean()
        ax.fill_between(
            fp["time"].values, fp.values, color=colour, alpha=0.25, label="monthly"
        )
        ax.plot(fp["time"].values, fp_smooth.values, color=colour, lw=2, label="12-month mean")
        ax.set_ylabel("footprint (km²)")
        ax.set_title(f"Compound {label} hazard footprint")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "footprint_timeseries.png"), dpi=150)
    plt.close(fig)

    # --- Trend map -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    tmax = indices["hot"].astype(float)
    slope = theil_sen_trend(tmax)
    pval = mann_kendall_p(tmax)

    im = axes[0].pcolormesh(
        slope["lon"], slope["lat"], slope, cmap="RdBu_r", shading="auto"
    )
    # Stipple only where p < 0.05, so the map cannot be read as "everywhere
    # is trending" when most of it is noise.
    sig = pval < 0.05
    axes[0].contourf(
        pval["lon"], pval["lat"], sig.astype(float),
        levels=[0.5, 1.5], colors="none", hatches=["...", ""],
    )
    axes[0].set_title("Trend in heat exceedance frequency (per year)\n"
                      "hatched = p < 0.05")
    fig.colorbar(im, ax=axes[0], label="frequency per year")

    fp_dh = hazard_footprint(indices["compound_drought_heat"], area)
    fp_annual = fp_dh.resample(time="1YE").sum() / 1000.0
    axes[1].bar(
        fp_annual["time"].dt.year, fp_annual.values, color="#c0392b", alpha=0.8
    )
    axes[1].set_title("Annual compound drought–heat footprint")
    axes[1].set_ylabel("thousand km²·month")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "trend_map.png"), dpi=150)
    plt.close(fig)

    return os.path.join(outdir, "footprint_timeseries.png"), os.path.join(
        outdir, "trend_map.png"
    )


def validation_report(indices: dict, ds: xr.Dataset) -> dict:
    """Recover the imposed warming from the synthetic field.

    This is the ground-truth check: for a synthetic run the expected warming
    is `ds.attrs['warming_per_decade_K']`. Reporting the recovered value next
    to the imposed value is the difference between a pipeline that is believed
    to work and one that is known to.
    """
    tmax = ds["t2m"] - 273.15
    slope_per_year = theil_sen_trend(tmax)
    pval = mann_kendall_p(tmax)

    mean_slope = float(np.nanmean(slope_per_year.values)) * 10.0  # per decade
    frac_sig = float(np.nanmean((pval < 0.05).values))
    imposed = ds.attrs.get("warming_per_decade_K", None)

    report = {
        "imposed_warming_per_decade_K": imposed,
        "recovered_warming_per_decade_K": round(mean_slope, 4),
        "fraction_cells_significant_p05": round(frac_sig, 4),
    }
    if imposed is not None:
        report["absolute_error_K_per_decade"] = round(abs(mean_slope - imposed), 4)

    n_dh = int(indices["compound_drought_heat"].sum().values)
    n_wh = int(indices["compound_wet_heat"].sum().values)
    report["compound_drought_heat_cell_months"] = n_dh
    report["compound_wet_heat_cell_months"] = n_wh

    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="1991-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("N", "W", "S", "E"))
    ap.add_argument("--synthetic", action="store_true",
                    help="run on the synthetic validation field instead of ERA5")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    bbox = None
    if args.bbox:
        n, w, s, e = args.bbox
        bbox = {"north": n, "west": w, "south": s, "east": e}

    if args.synthetic:
        print("[1/5] building synthetic validation field ...")
        ds = synthetic_field(start=args.start, end=args.end, bbox=bbox or SOUTHERN_VIETNAM)
    else:
        print("[1/5] fetching ERA5 ...")
        ds = fetch_monthly(args.start, args.end, bbox=bbox)

    print(f"      grid: {ds.sizes.get('time')} months x "
          f"{ds.sizes.get('lat')} lat x {ds.sizes.get('lon')} lon")

    print("[2/5] computing indices (SPEI, heat, compound masks) ...")
    indices = build_indices(ds)

    print("[3/5] cataloguing events ...")
    catalogue = event_catalogue(indices["compound_drought_heat"])
    os.makedirs("data/processed", exist_ok=True)
    catalogue.to_csv("data/processed/event_catalogue.csv", index=False)
    print(f"      {len(catalogue)} compound drought-heat events catalogued")

    print("[4/5] computing trends and validation report ...")
    report = validation_report(indices, ds)
    for k, v in report.items():
        print(f"      {k}: {v}")

    print("[5/5] writing figures ...")
    paths = make_figures(indices, ds, outdir=args.outdir)
    for p in paths:
        print(f"      wrote {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
