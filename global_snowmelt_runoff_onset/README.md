# global_snowmelt_runoff_onset

Core Python package implementing the dataset creation algorithm from manuscript Sect. 2.2. Installed as an editable dependency via pixi (`global-snowmelt-runoff-onset = { path = ".", editable = true }` in `pixi.toml`); imported by the notebooks/scripts in `processing/`, `dataset_utils/`, `dataset_evaluation/`, and `visualize/`, and by the standalone [global_snowmelt_runoff_onset_analysis](https://github.com/egagli/global_snowmelt_runoff_onset_analysis) repo (which editable-installs this package from a sibling clone).

## Modules

| Module | Purpose |
| --- | --- |
| [`config.py`](config.py) | `Config` — loads a versioned `config/global_config_vN.txt` file, builds the global geobox and tile grid (`odc.geo`), defines the chunking/sharding strategy, and connects to Azure Blob Storage. For configs ≥ v10 it opens the Icechunk output repository (`open_output_repo()`, with per-water-year manifest splitting and storage-retry settings); for configs ≤ v9 it keeps the legacy pre-allocated Zarr v2 store plus CSV status tracking (`get_list_of_tiles()`). `Tile` — a single processing unit (bbox, row/col, geobox). |
| [`processing.py`](processing.py) | The core algorithm: `search_sentinel1_items()` (retried Planetary Computer STAC search — distinguishes transient MPC failures from genuinely-empty regions) + `load_sentinel1_rtc()` (lazy odc-stac load; the two compose into `get_sentinel1_rtc()`), `get_spatiotemporal_snow_cover_mask()` (MODIS-derived snow phenology masking — reads the `MODIS_snow_phenology` icechunk store for configs ≥ v10, or the legacy `MODIS_seasonal_snow_mask` Zarr v2 store for configs ≤ v9), `apply_all_masks()`, `filter_insufficient_pixels_per_orbit()`, `calculate_runoff_onset()`, and supporting helpers (temporal resolution, DOWY conversion, gap analysis). Also home of the odc-stac antimeridian self-check/workaround (`ensure_antimeridian_footprint_fix()`). |
| [`store.py`](store.py) | v10+ output store schema and initialization: the sharded Zarr v3 template (one tile × water year per shard, small inner chunks), metadata-only `initialize_store()`, and `tile_region_slices()` (exact integer region slices per tile). |
| [`status.py`](status.py) | Processing status derived from Icechunk commit history: structured commit metadata schema (`build_commit_metadata()`), `get_tile_status_gdf()` (per-tile × water-year status incl. composite staleness), `get_remaining_work()` (dispatch lists for GitHub Actions / `run_tiles.py`). Replaces the ≤ v9 CSV tracking. |
| [`provenance.py`](provenance.py) | `collect_provenance()` — compute-platform metadata embedded in every commit (GitHub Actions run/runner, JupyterHub/CryoCloud session, or local hostname; package versions; code SHA). |
| [`plot_utils.py`](plot_utils.py) | Shared plotting helpers used across `visualize/` and `analysis/`: `create_month_colorbar()`, `create_diverging_colorbar()`, `plot_geoms()`, and water-year calendar constants (DOWY↔month mapping for both hemispheres). |
| `global_snowmelt_runoff_onset.py` | Empty (0 bytes) — stale packaging-scaffold stub, safe to delete. |

`analysis.py` (tile-dataset enrichment + parquet export helpers) moved to the `gsro_analysis` package in the [global_snowmelt_runoff_onset_analysis](https://github.com/egagli/global_snowmelt_runoff_onset_analysis) repo in July 2026.

## Related

- [`processing/README.md`](../processing/README.md) — how these modules are driven at scale (GitHub Actions, `run_tiles.py` locally/CryoCloud).
- [`config/`](../config) — the versioned config files `Config` reads.
