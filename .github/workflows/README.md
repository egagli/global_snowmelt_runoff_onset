# .github/workflows

GitHub Actions implementation of the tile-processing pipeline described in manuscript Sect. 2.2.3. All four workflows are `workflow_dispatch` (manually triggered), use [pixi](https://pixi.sh)'s `ci` environment exclusively (`prefix-dev/setup-pixi`, no conda), run on plain `ubuntu-latest` runners, and authenticate to Azure Blob Storage via the `AZURE_STORAGE_SAS_TOKEN`/`AZURE_STORAGE_ACCOUNT` repository secrets.

## Workflows

| Workflow | Trigger | What it does |
| --- | --- | --- |
| [`process_single_tile.yml`](process_single_tile.yml) | manual (`tile_row`, `tile_col`, `config_file`) | Runs `processing/scripts/process_single_tile.py` for one tile (120 min timeout). On failure, writes a synthetic failure-status CSV so the tile shows up as failed rather than silently missing. Uploads the result CSV as a workflow artifact (30-day retention). |
| [`process_batch_small.yml`](process_batch_small.yml) | manual, or `workflow_call` (reusable) | For ≤256 tiles. One job builds a tile matrix via `processing/scripts/get_tiles_for_batch.py` (filtered by `which_tiles_to_process`: all/processed/failed/unprocessed/etc.), a second job runs one `process_single_tile.py` invocation per matrix entry — true tile-level parallelism via the GitHub Actions matrix, not Dask. |
| [`process_batch_large.yml`](process_batch_large.yml) | manual | For >256 tiles (beyond the GitHub Actions matrix limit). Splits the requested tile count into batches and matrixes over batch index, calling `process_batch_small.yml` as a reusable workflow for each batch. |
| [`consolidate_tile_results.yml`](consolidate_tile_results.yml) | manual (`days_back`, `config_version`) | Downloads recent tile-result artifacts via `processing/scripts/consolidate_artifacts.py`, merges them into `processing/tile_data/tile_results_*.csv`, and commits/pushes the update back to `main`. This is how per-tile processing status is currently tracked (see caveat below). |

## Processing status caveat

Status is tracked as one CSV row per **tile** (not per tile × water year) — a single `success` flag covers all 10 water years for that tile. This is different from the `MODIS_snow_phenology` sibling repo, which derives status directly from Icechunk commit history instead of a CSV. See the root README's migration notes for the planned move to an Icechunk-backed store with finer-grained (tile × water year) status tracking.

## Related

See [`processing/README.md`](../../processing/README.md) for the scripts these workflows call and how they relate to the (soon to be retired) Coiled/Dask notebooks.
