# .github/workflows

GitHub Actions implementation of the tile-processing pipeline described in manuscript Sect. 2.2.3. All workflows are `workflow_dispatch` (manually triggered), use [pixi](https://pixi.sh)'s `ci` environment exclusively (`prefix-dev/setup-pixi`, no conda), run on plain `ubuntu-latest` runners, and authenticate to Azure Blob Storage via the `AZURE_STORAGE_SAS_TOKEN`/`AZURE_STORAGE_ACCOUNT` repository secrets.

As of config v10, the output store is an **Icechunk repository** and processing status is derived entirely from its commit history (structured commit metadata; see `global_snowmelt_runoff_onset/status.py`) — there are no status CSVs, no artifacts, and no consolidation step. A failed tile×water-year simply never commits and is re-dispatched on the next run; a verified-empty one gets an explicit empty-marker commit and is never retried (unless requested).

## Workflows

| Workflow | Trigger | What it does |
| --- | --- | --- |
| [`process_single_tile.yml`](process_single_tile.yml) | manual (`tile_row`, `tile_col`, `water_years`, `config_file`) | Runs `processing/scripts/process_single_tile.py` for one tile (120 min timeout). Each water year is processed and committed separately; `water_years` can be `all`, a subset like `2019,2020`, or `none` to only refresh the cross-year composites. |
| [`process_batch.yml`](process_batch.yml) | manual, or `workflow_call` (reusable) | For ≤256 tiles. One job derives the remaining work from icechunk commit history via `processing/scripts/get_tiles_for_batch.py` (each matrix entry carries the tile's missing water years), a second job runs one `process_single_tile.py` invocation per matrix entry — true tile-level parallelism via the GitHub Actions matrix. |
| [`process_all_tiles.yml`](process_all_tiles.yml) | manual | Splits the full remaining work into ≤256-tile batches and matrixes over batch index, calling `process_batch.yml` as a reusable workflow for each. The work list is pinned to a single icechunk snapshot so every batch computes identical batch boundaries even though they start at different times. |

## Status tracking

Per **tile × water year** (finer than the pre-v10 per-tile CSVs): each commit's metadata records the tile, water year, `data`/`empty` status, per-year stats, config version, and compute-platform provenance (GitHub Actions run/runner IDs here; hostname/JupyterHub info elsewhere). `which_tiles` options:

- `incomplete` (default) — tiles with missing water years (including previously failed ones) or missing/stale composites
- `unprocessed` — tiles with no commits at all
- `all` — full reprocess

The same status derivation powers `processing/scripts/run_tiles.py`, so a run started on GitHub Actions can be finished locally or on CryoCloud (or vice versa) with no coordination beyond the icechunk repository itself.

## Related

See [`processing/README.md`](../../processing/README.md) for the scripts these workflows call.
