# processing

Implements the dataset creation methodology from manuscript Sect. 2.2 — tiling, the pre-allocated Zarr output store, and per-tile runoff onset computation across the 23,520-tile global grid.

## Two processing paths (currently coexisting)

There are two independent implementations of the same per-tile algorithm today:

1. **Coiled/Dask notebooks** (`process_tiles.ipynb`, `process_tiles_serverless.ipynb`) — the original bulk-processing path. Spin up a Coiled cluster or serverless function and submit tiles (all 10 water years per tile) via `dask.distributed`.
2. **GitHub Actions script** (`scripts/process_single_tile.py`, called from `.github/workflows/`) — processes one tile per job using local threaded Dask, no Coiled dependency. This is the actively-used path going forward, and the intended target for the planned Icechunk migration (see root README migration notes).

These should eventually converge on a single entrypoint; until then, treat the GH Actions script as the source of truth.

## Notebooks

| Notebook | Description |
| --- | --- |
| [`select_tiles_to_process.ipynb`](select_tiles_to_process.ipynb) | One-time builder of the 23,520-tile global grid; filters to the ~4,700 tiles with meaningful seasonal snow (Sturm & Liston classification) and writes `tile_data/global_tiles_with_seasonal_snow.geojson`, the static tile registry. |
| [`create_zarr_store.ipynb`](create_zarr_store.ipynb) | Pre-allocates the empty global Zarr v2 output store (all variables, all-nodata, `to_zarr(..., compute=False, mode='w')`) that individual tile jobs then fill in via region writes. |
| [`process_tiles.ipynb`](process_tiles.ipynb) | Coiled `Cluster`-based bulk processing (persistent cluster, `n_workers=40-60`), batches of 10 tiles submitted via `client.submit`. |
| [`process_tiles_serverless.ipynb`](process_tiles_serverless.ipynb) | Same algorithm as above, using a Coiled serverless `@coiled.function` instead of a persistent cluster. |
| [`quality_check_tiles.ipynb`](quality_check_tiles.ipynb) | Compares two dataset versions (e.g. `global_config_v8.txt` vs `v9.txt`) tile-by-tile for consistency after a config or algorithm change. |
| [`test_chunking.ipynb`](test_chunking.ipynb) | Scratch notebook for tuning dask chunk sizes during processing. |
| [`precompute_spatiotemporal_snow_mask_reprojection.ipynb`](precompute_spatiotemporal_snow_mask_reprojection.ipynb) | Builds a pre-reprojected (80 m, EPSG:4326) copy of the MODIS snow phenology store, used by the `'precomputed'` branch of `get_spatiotemporal_snow_cover_mask()`'s `reproject_method` option. Currently targets the *old* `MODIS_seasonal_snow_mask`-derived store; needs regenerating once the pipeline migrates to `MODIS_snow_phenology`. |
| `download_and_compress_zarr.ipynb` | Empty (0 bytes) — stale placeholder, safe to delete. |

## Scripts (`scripts/`)

| Script | Description |
| --- | --- |
| [`process_single_tile.py`](scripts/process_single_tile.py) | GitHub Actions entrypoint. Runs the full algorithm (`global_snowmelt_runoff_onset.processing`) for one tile using local threaded Dask, writes to the pre-allocated global Zarr store via a region write, and appends a per-tile result row to `tile_data/tile_results_*.csv`. |
| [`get_tiles_for_batch.py`](scripts/get_tiles_for_batch.py) | Builds a JSON tile matrix (filtered by `which_tiles_to_process`: all/processed/failed/unprocessed/etc.) for the GitHub Actions batch workflows. |
| [`consolidate_artifacts.py`](scripts/consolidate_artifacts.py) | Downloads per-tile result CSVs from GitHub Actions run artifacts and merges them into `tile_data/tile_results_*.csv`. |
| `consolidate_artifacts_clean.py` | Empty (0 bytes) — stale placeholder, safe to delete. |

## `tile_data/`

- `global_tiles_with_seasonal_snow.geojson` — the static tile registry (row/col, bbox, snow-presence flag) produced by `select_tiles_to_process.ipynb`.
- `tile_results_v2.csv` … `tile_results_v9.csv` — historical per-tile processing status, one row per tile (columns include `success`, `error_messages`, `start_time`, `total_time`, and per-water-year `tr_YYYY`/`pix_ct_YYYY`). `v9` is current; older versions are kept for provenance but not actively read.
- `valid_tiles.geojson`, `valid_tiles_v2.geojson` — superseded by `global_tiles_with_seasonal_snow.geojson`.
- `quality_check/`, `reprojected_snow_mask/` — outputs of `quality_check_tiles.ipynb` and `precompute_spatiotemporal_snow_mask_reprojection.ipynb` respectively.

## Related

- [`global_snowmelt_runoff_onset/README.md`](../global_snowmelt_runoff_onset/README.md) — the core algorithm these notebooks/scripts call.
- [`.github/workflows/README.md`](../.github/workflows/README.md) — how `scripts/process_single_tile.py` is invoked at scale.
- [`config/`](../config) — versioned processing configuration files (`global_config_v2.txt` … `global_config_v9.txt`) consumed via `global_snowmelt_runoff_onset.config.Config`.
