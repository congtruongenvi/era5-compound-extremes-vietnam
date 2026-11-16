# data/

Nothing in this directory is committed to git.

## raw/

ERA5 subsets written by `src/fetch.py`. Naming convention:

```
era5_monthly_<start_year>_<end_year>.nc
```

Typical size for southern Vietnam at 0.25° over 1991–2024: a few hundred MB.
Regenerate with:

```bash
python src/pipeline.py --start 1991-01-01 --end 2024-12-31
```

## processed/

Intermediate and final analysis products:

| File | Contents |
|---|---|
| `spei3.nc` | SPEI-3 field, monthly, standard normal quantiles |
| `compound_drought_heat.nc` | Boolean mask, compound drought–heat |
| `compound_wet_heat.nc` | Boolean mask, compound wet–heat |
| `event_catalogue.csv` | Discrete events: start, end, duration, extent |

## Access

ERA5 is distributed by the Copernicus Climate Change Service. To retrieve it:

1. Register at https://cds.climate.copernicus.eu
2. Accept the ERA5 licence on the dataset page (requests fail with a licence
   error, not a data error, if this is skipped)
3. Save your key to `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-key>
```

Nothing in this repository redistributes ERA5 data.
