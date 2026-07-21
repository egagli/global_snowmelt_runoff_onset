# analysis

Dataset-construction-adjacent analyses and manuscript figure generation. This folder is in the middle of being split up — see the cleanup status table below.

## Current status

| File | Verdict | Notes |
| --- | --- | --- |
| [`methods_fig.ipynb`](methods_fig.ipynb) | **Keep here** | Produces Fig. 1 workflow-diagram panels (GMBA mountains map, valid-tiles map, Rainier Sentinel-1/Sentinel-2 example imagery) and draft global Robinson-projection maps. |
| `rainier_figure.ipynb` | Verify against `methods_fig.ipynb` | Overlaps substantially with `methods_fig.ipynb`'s Rainier panels (same bbox/imagery) — check for redundancy before deciding keep vs. move. |
| `create_analysis_parquets.ipynb`, `aggregate_analysis_parquets.ipynb` | Move to `global_snowmelt_runoff_onset_analysis` | Tile-level parquet aggregation infrastructure feeding the broader-science notebooks below; not manuscript-figure-specific. |
| `global_analysis.ipynb`, `river_basin_analysis.ipynb`, `sierra_nevada.ipynb` | Move to `global_snowmelt_runoff_onset_analysis` | Broader hydrology/geography science (continent-scale lapse rates, basin-scale population/precipitation impacts, regional case study) — not on the manuscript figure list. |
| `mountain_range_geo_and_topo_analysis.ipynb`, `mountain_range_era5_analysis.ipynb`, `compare_runoff_onset_anomaly_and_climate.ipynb`, `get_era5_data.ipynb` | Move to `global_snowmelt_runoff_onset_analysis` | Per-mountain-range climate correlation science; not manuscript-figure-specific. |
| `mountain_range_combined_analysis.ipynb` | **Delete (superseded)** | Appears to be the source notebook later split into `mountain_range_geo_and_topo_analysis.ipynb` + `mountain_range_era5_analysis.ipynb` — verify the split covers everything, then delete. |
| `quick_era5_view.ipynb` | **Delete (superseded)** | Ad hoc exploratory preview, superseded by `get_era5_data.ipynb` / `mountain_range_era5_analysis.ipynb`. |
| `bareerah_lidar.ipynb`, `ross_mower_stuff.ipynb`, `snowmelt_mahboubeh.ipynb` | **Delete (scratch)** | One-off collaborator requests with no manuscript tie-in. |
| `scratch/` | **Delete** | Raw backscatter/DEM data used only by `ross_mower_stuff.ipynb`, plus preview PNGs referenced only in a commented-out line in `quick_era5_view.ipynb`. |

## Data directories

`aggregated_results/`, `era5_data/`, `csvs/`, `colorbars/`, `geometries/`, and most of `figures/` are inputs/outputs of the notebooks above and should move or be deleted alongside their parent notebook. `figures/methods/` should stay (output of `methods_fig.ipynb`). **None of these directories are currently gitignored** — see the root README's cleanup notes before running any broad `git add`.

## Blocker

Moving notebooks to `global_snowmelt_runoff_onset_analysis` requires that repository to exist first — it hasn't been created/cloned yet.
