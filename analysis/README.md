# analysis

Dataset-construction-adjacent analyses and manuscript figure generation. This folder is in the middle of being split up — see the cleanup status table below.

## Current status

| File | Verdict | Notes |
| --- | --- | --- |

| `create_analysis_parquets.ipynb`, `aggregate_analysis_parquets.ipynb` | Move to `global_snowmelt_runoff_onset_analysis` | Tile-level parquet aggregation infrastructure feeding the broader-science notebooks below; not manuscript-figure-specific. |
| `global_analysis.ipynb`, `river_basin_analysis.ipynb`, `sierra_nevada.ipynb` | Move to `global_snowmelt_runoff_onset_analysis` | Broader hydrology/geography science (continent-scale lapse rates, basin-scale population/precipitation impacts, regional case study) — not on the manuscript figure list. |
| `mountain_range_geo_and_topo_analysis.ipynb`, `mountain_range_era5_analysis.ipynb`, `compare_runoff_onset_anomaly_and_climate.ipynb`, `get_era5_data.ipynb` | Move to `global_snowmelt_runoff_onset_analysis` | Per-mountain-range climate correlation science; not manuscript-figure-specific. |
| `mountain_range_combined_analysis.ipynb` | **CHECK** | Appears to be the source notebook later split into `mountain_range_geo_and_topo_analysis.ipynb` + `mountain_range_era5_analysis.ipynb` — verify the split covers everything. |
| `quick_era5_view.ipynb` | **CHECK** | Ad hoc exploratory preview, superseded by `get_era5_data.ipynb` / `mountain_range_era5_analysis.ipynb`. |
| `bareerah_lidar.ipynb`, `ross_mower_stuff.ipynb`, `snowmelt_mahboubeh.ipynb` | **CHECK** | One-off collaborator requests with no manuscript tie-in. |
| `scratch/` | **Delete** | Raw backscatter/DEM data used only by `ross_mower_stuff.ipynb`, plus preview PNGs referenced only in a commented-out line in `quick_era5_view.ipynb`. |

## Data directories

`aggregated_results/`, `era5_data/`, `csvs/`, `colorbars/`, `geometries/`, and most of `figures/` are inputs/outputs of the notebooks above and should move alongside their parent notebook. **None of these directories are currently gitignored** — see the root README's cleanup notes before running any broad `git add`.

## Blocker

Moving notebooks to `global_snowmelt_runoff_onset_analysis` requires that repository to exist first — it hasn't been created/cloned yet.
