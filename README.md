# ERA5 Compound Hydroclimatic Extremes — Southern Vietnam

Quantifying **compound drought–heat and wet–heat extremes** over the Mekong Delta and
Southeast Vietnam from ERA5 reanalysis, and translating them into **hazard footprints**
(area under simultaneous exceedance, km²) that can be linked to observed disaster losses
and adaptation policy.

> **Why compound, not single-hazard?**
> A heatwave during a hydrological drought depletes soil moisture, reservoir storage, and
> hydropower head at the same time. A heavy-rain event on saturated ground becomes fluvial
> flooding. Drought indices and heat indices computed separately both miss this
> co-occurrence — and it is the co-occurrence that drives the losses.

---

## Research questions

1. How often do drought and heat conditions occur **simultaneously** in southern Vietnam,
   and how has that frequency changed since 1991?
2. What is the **spatial footprint** of these compound events — how many km² are under
   concurrent exceedance, and does the footprint grow with event severity?
3. Are trends in compound frequency **statistically significant**, or within the range
   expected from internal variability?
4. Which sub-regions (Mekong Delta, Southeast, Central Highlands fringe) carry the
   highest compound exposure?

---

## Method

| Step | Index / method | Convention |
|---|---|---|
| Precipitation deficit | **SPEI-3, SPEI-6** | Vicente-Serrano et al. (2010) |
| Heat exceedance | **Tmax > reference-period p90** | ETCCDI TX90p (Zhang et al. 2011) |
| Compound drought–heat | SPEI-3 < p10 **AND** Tmax > p90 | co-occurrence, not union |
| Compound wet–heat | 3-month P > p90 **AND** Tmax > p90 | co-occurrence, not union |
| Hazard footprint | Σ cell area under concurrent exceedance | spherical cell area, km² |
| Trend | **Theil–Sen slope + Mann–Kendall p** | robust to record-breaking years |

### Two methodological choices worth defending

**1. Distribution fitted on the reference period only.**
SPEI is calibrated on 1991–2020. Fitting the gamma distribution on the full series would
let the calibration period absorb information about the very events being diagnosed —
a form of look-ahead bias that inflates apparent trends. The code enforces the split in
`spei()`.

**2. Theil–Sen rather than OLS for trends.**
Extreme-value indices are bounded and heteroscedastic. One record-breaking year can flip
an OLS slope's sign; the median pairwise slope is robust to it. When the object of study
*is* the extreme years, that robustness is not a luxury.

**3. Spherical cell area.**
ERA5's 0.25° grid has cells whose area shrinks with the cosine of latitude. Using a
constant area biases every footprint statistic toward the equator. `cell_area()` computes
the true spherical area per cell.

---

## Repository layout

```
.
├── src/
│   ├── fetch.py       # ERA5 retrieval (CDS API), subsetting, caching
│   ├── indices.py     # SPEI, heat exceedance, compound masks, footprint, trends
│   └── pipeline.py    # end-to-end run: fetch -> indices -> catalogue -> figures
├── notebooks/
│   └── 01_explore_compound_extremes.ipynb
├── figures/           # generated output
├── data/              # raw and processed (gitignored; see data/README.md)
├── requirements.txt
└── Makefile
```

---

## Status

**Working analysis pipeline, pre-results stage.** The index chain, trend machinery,
footprint accounting and unit conventions are implemented and validated (18 self-tests).
ERA5 download is reproducible via CDS; the included figures are generated from a
physically-plausible synthetic field because the ERA5 run is intentionally deferred until
the method is locked. Real-data figures will replace synthetic ones in the working branch.

This repository demonstrates **method implementation and verification**, not a finished
climate assessment. Do not cite `figures/` outputs as observed trends.

---

## Quick start

```bash
# 1. Environment
uv venv --python 3.12 .venv
uv pip install -r requirements.txt

# 2. ERA5 access (one-time)
#    Register at https://cds.climate.copernicus.eu and accept the ERA5 licence,
#    then place your key at ~/.cdsapirc

# 3. Run the full pipeline
python src/pipeline.py --start 1991-01-01 --end 2024-12-31 --bbox 8.5 23.5 102.0 110.5

# Or, to validate the analysis logic without any download:
python src/pipeline.py --synthetic
```

`--synthetic` generates a physically-plausible synthetic field with a known imposed
trend and runs the entire index → footprint → trend chain on it. Use it to check that
code changes have not broken the pipeline, and to see expected output formats without
a CDS account.

---

## Outputs

| File | Contents |
|---|---|
| `data/processed/spei3.nc` | SPEI-3 field, monthly |
| `data/processed/compound_drought_heat.nc` | Boolean compound mask |
| `figures/footprint_timeseries.png` | Hazard footprint (km²) through time |
| `figures/trend_map.png` | Theil–Sen trend with significance stippling |
| `data/processed/event_catalogue.csv` | Discrete events: start, end, duration, extent |

---

## Data sources

- **ERA5** hourly/monthly reanalysis — Hersbach et al. (2020), Copernicus Climate Change
  Service. Variables: `total_precipitation`, `2m_temperature`, `2m_dewpoint_temperature`,
  `surface_net_solar_radiation`, `10m_u_component_of_wind`.
- **ERA5-Land** for soil moisture and land-surface fluxes where higher resolution is
  needed.
- Administrative boundaries: GADM v4.1.

---

## Limitations, stated plainly

- ERA5 is a **model reanalysis, not an observation network.** Its precipitation over the
  Mekong Delta diverges from gauge records most strongly in the wet season; results here
  characterise *large-scale hydroclimatic conditions*, not station-level rainfall.
- The compound definition uses **monthly** resolution, which under-counts short,
  intense events (a 5-day flash drought will not register). Daily resolution is the
  natural extension.
- `label_spacetime()` is a documented stub. Connected-component labelling in space-time
  would give more defensible event counts than the per-cell time-contiguity rule used in
  `event_catalogue()`; the simplification is visible rather than buried.
- Hargreaves PET is a **proxy** for Penman–Monteith. It needs only temperature and
  extraterrestrial radiation, which matters for data availability, but it is less
  accurate in humid, advection-dominated coastal settings.
- Socio-economic linkage (exposure, vulnerability, observed losses) is **not** in this
  repository — this is the hazard side of the chain.

---

## References

- McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought frequency
  and duration to time scales. *8th Conf. on Applied Climatology*, AMS.
- Vicente-Serrano, S.M., Beguería, S., López-Moreno, J.I. (2010). A multiscalar drought
  index sensitive to global warming: the Standardized Precipitation
  Evapotranspiration Index. *Journal of Climate*, 23, 1696–1718.
- Zhang, X. et al. (2011). Indices for monitoring changes in extremes based on daily
  temperature and precipitation data. *WIREs Climate Change*, 2, 851–870.
- Zscheischler, J. et al. (2018). Future climate risk from compound events.
  *Nature Climate Change*, 8, 469–477.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146, 1999–2049.

---

## Author

**Truong Cong Truong** — MSc Environmental Science (VNU-HCM), 15+ years in numerical
hydrology and hydrodynamic modelling across the Mekong Delta and Dong Nai River basin.

*Python · xarray · ERA5/CMIP6 · spatial statistics*
