# global_snowmelt_runoff_onset

Core Python package implementing the dataset creation algorithm from manuscript Sect. 2.2. Installed as an editable dependency via pixi (`global-snowmelt-runoff-onset = { path = ".", editable = true }` in `pixi.toml`); imported by the notebooks/scripts in `processing/`, `dataset_utils/`, `dataset_evaluation/`, and `analysis/`.

## Modules

| Module | Purpose |
| --- | --- |
| [`config.py`](config.py) | `Config` — loads a versioned `config/global_config_vN.txt` file, builds the global geobox and tile grid (`odc.geo`), defines the three-stage chunking strategy, connects to Azure Blob Storage, and tracks per-tile processing status (`get_list_of_tiles()`, backed by `tile_data/tile_results_*.csv`). `Tile` — represents a single processing unit (bbox, row/col, processing status/timing fields). |
| [`processing.py`](processing.py) | The core algorithm: `get_sentinel1_rtc()` (Sentinel-1 RTC acquisition from Planetary Computer), `get_spatiotemporal_snow_cover_mask()` (MODIS-derived snow phenology masking — reads the `MODIS_snow_phenology` icechunk store for configs ≥ v10, or the legacy `MODIS_seasonal_snow_mask` Zarr v2 store for configs ≤ v9), `apply_all_masks()`, `filter_insufficient_pixels_per_orbit()`, `calculate_runoff_onset()`, `dataarrays_to_dataset()`, and supporting helpers (temporal resolution, DOWY conversion, gap analysis). |
| [`plot_utils.py`](plot_utils.py) | Shared plotting helpers used across `visualize/` and `analysis/`: `create_month_colorbar()`, `create_diverging_colorbar()`, `plot_geoms()`, and water-year calendar constants (DOWY↔month mapping for both hemispheres). |
| [`analysis.py`](analysis.py) | Optional (imported best-effort in `__init__.py`, requires `xdem` and other extra dependencies) — helpers for enriching per-tile datasets with topography, CHILI, snow classification, ESA WorldCover, forest cover, and mountain range/basin/continent labels, and for exporting tile-level datasets to analysis parquet files (used by `analysis/create_analysis_parquets.ipynb`). |
| `global_snowmelt_runoff_onset.py` | Empty (0 bytes) — stale packaging-scaffold stub, safe to delete. |

## Related

- [`processing/README.md`](../processing/README.md) — how `processing.py`/`config.py` are driven at scale (Coiled notebooks and GitHub Actions).
- [`config/`](../config) — the versioned config files `Config` reads.
