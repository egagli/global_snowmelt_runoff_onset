# .github/workflows

GitHub Actions implementation of the tile-processing pipeline described in manuscript Sect. 2.2.3. The three tile-processing workflows are `workflow_dispatch` (manually triggered), use [pixi](https://pixi.sh)'s `ci` environment (`prefix-dev/setup-pixi`, no conda), run on plain `ubuntu-latest` runners, and authenticate to Azure Blob Storage via the `AZURE_STORAGE_SAS_TOKEN`/`AZURE_STORAGE_ACCOUNT` repository secrets. `water_year_watch.yml` is the exception to all three: it is cron-scheduled, uses plain `python3` (no pixi), and needs only `issues: write` (no Azure access).

As of config v10, the output store is an **Icechunk repository** and processing status is derived entirely from its commit history (structured commit metadata; see `global_snowmelt_runoff_onset/status.py`) — there are no status CSVs, no artifacts, and no consolidation step. A failed tile×water-year simply never commits and is re-dispatched on the next run; a verified-empty one gets an explicit empty-marker commit and is never retried (unless requested).

## Workflows

| Workflow | Trigger | What it does |
| --- | --- | --- |
| [`process_single_tile.yml`](process_single_tile.yml) | manual (`tile_row`, `tile_col`, `water_years`, `config_file`) | Runs `processing/scripts/process_single_tile.py` for one tile (`timeout-minutes: 330`, same for the per-tile job in `process_batch.yml`). Each water year is processed and committed separately; `water_years` can be `all`, `missing` (resume mode — only years with no commit yet), a subset like `2019,2020`, or `none` to only refresh the cross-year composites. |
| [`process_batch.yml`](process_batch.yml) | manual, or `workflow_call` (reusable; inputs incl. `as_of_snapshot`, `batch_index`, `tiles_file`, `how_many`) | For ≤256 tiles. One job derives the remaining work from icechunk commit history via `processing/scripts/get_tiles_for_batch.py` (each matrix entry carries the tile's missing water years), a second job runs one `process_single_tile.py` invocation per matrix entry — true tile-level parallelism via the GitHub Actions matrix. |
| [`process_all_tiles.yml`](process_all_tiles.yml) | manual | Splits the full remaining work into ≤256-tile batches and matrixes over batch index, calling `process_batch.yml` as a reusable workflow for each. The work list is pinned to a single icechunk snapshot (`as_of_snapshot`) so every batch computes identical batch boundaries even though they start at different times. |
| [`test_tiles.yml`](test_tiles.yml) | manual (`tiles`, `water_years`, `config_file`) | **Throwaway v10 shakedown battery** (self-labelled; delete once real batches are running). Matrixes over a hand-picked tile list — default `10,0` (antimeridian), `9,138` (scene-densest, memory stress), `25,39` (Rainier v9 cross-check), `28,65` (empty-year mix), `25,190` (catalog backfill), `80,240` (`no_seasonal_snow`), `56,69`/`57,69` (equator-straddling Cayambe pair) — on the extended v10 grid, and exercises icechunk's `ConflictDetector` under real concurrent commits. Introduced the `water_years: missing` resume mode. |
| [`water_year_watch.yml`](water_year_watch.yml) | monthly cron (`0 12 3 * *`) + manual | Runs `processing/scripts/water_year_watch.py`: opens one deduplicated GitHub issue per (water year, hemisphere) once that season has fully elapsed plus the config's `trailing_buffer_days`, signalling the new water year can be appended and dispatched. No pixi, no Azure. |

Any of the batch workflows can be restricted to an explicit tile list via the `tiles_file` input — a repo-relative path with one `row,col` per line, e.g. `tiles_file: processing/tile_data/station_tiles_v10.txt` (built by `processing/scripts/make_station_tile_list.py`) for the prioritized weather-station tiles.

## Status tracking

Per **tile × water year** (finer than the pre-v10 per-tile CSVs): each commit's metadata records the tile, water year, `data`/`empty` status, per-year stats, config version, and compute-platform provenance (GitHub Actions run/runner IDs here; hostname/JupyterHub info elsewhere). `which_tiles` options:

- `incomplete` (default) — tiles with missing water years (including previously failed ones) or missing/stale composites
- `unprocessed` — tiles with no commits at all
- `all` — full reprocess

The same status derivation powers `processing/scripts/run_tiles.py`, so a run started on GitHub Actions can be finished locally or on CryoCloud (or vice versa) with no coordination beyond the icechunk repository itself.

## Related

See [`processing/README.md`](../../processing/README.md) for the scripts these workflows call.
