# Maintenance and processing workflow

The operational runbook for this repository: how to (re)build the dataset, run and
babysit the GitHub Actions fleet, add new water years, and maintain the store and
registry. For the *design* — why status lives in icechunk commits, why failures never
commit, how the whole pattern generalizes — see
[`icechunk-github-actions-pattern.md`](icechunk-github-actions-pattern.md). For
mapping outputs to the manuscript, see
[`results_and_figures.md`](results_and_figures.md). Deeper reference for every knob
lives in [`processing/README.md`](../processing/README.md).

Numbered notebooks in `processing/` are the lifecycle, in order:
`0_select_tiles_to_process` → `1_create_icechunk_store` → (fleet runs) →
`2_check_tile_status` → `3_quality_check_tiles` → `4_finalize_icechunk_store`
(release) → `5_add_water_year` (extension).

## 0. Credentials

Everything needs two environment variables (locally) or repo secrets (GitHub Actions):

```bash
export AZURE_STORAGE_ACCOUNT=...        # storage account hosting the icechunk repo
export AZURE_STORAGE_SAS_TOKEN=...      # SAS token with read/write/list/delete on the container
```

The SAS token **expires** — `Config` prints time-to-expiry at load. A fleet dispatched
with a nearly-dead token fails en masse with auth errors; refresh the repo secret
first. Planetary Computer needs no credentials (anonymous + auto-fetched tokens).

## 1. Tile registry (one-time per config era; refresh on demand)

`processing/0_select_tiles_to_process.ipynb` builds
`processing/tile_data/global_tiles_with_seasonal_snow_v10.geojson`:

1. Global grid from the config geobox (asserted against `expected_grid_shape` /
   `expected_tile_grid` tripwires — a sub-pixel bbox edit silently renumbers every
   tile; run `processing/scripts/verify_grid_alignment.py` after any bbox change).
2. Seasonal-snow filter (> 0.001 % Sturm seasonal-snow pixels, via `exactextract`).
3. **S1 VV availability from the MPC GeoParquet catalog index** (per-tile VV item
   counts) → `to_process` flag + `tile_notes`. No manual geographic exclusions; the
   catalog is the authority. Registry keeps excluded rows for documentation;
   `status.get_tile_status_gdf` filters on `to_process`.
4. Manual overrides: `processing/scripts/apply_manual_tiles.py` +
   `tile_data/manual_tiles_v10.txt` force-select tiles the catalog rule missed.
5. Priority subsets: `processing/scripts/make_station_tile_list.py` →
   `tile_data/station_tiles_v10.txt` (tiles containing snow-pillow stations, for
   evaluation-first runs).

The registry is versioned with the config (`valid_tiles_geojson_path` in
`config/global_config_v10.txt`); a v11 would write `..._v11.geojson` and point its
config at it.

## 2. Store initialization (one-time per store version)

`processing/1_create_icechunk_store.ipynb`: metadata-only Zarr v3 template on Azure —
full-extent arrays, shards `(1 water_year, 2048, 2048)` (= exactly one tile × water
year), inner chunks `(1, 256, 256)`, `fill_value = -9999`, water years from the
config (`WY_start`–`WY_end`). Re-initializing **destroys all processed data** — it is
only for new store versions or pre-fleet schema changes. Sanity checks afterward:
the dry-run cell of `processing/5_add_water_year.ipynb`
(`store.extend_water_years(config, repo, dry_run=True)` — should report nothing to
append) and `verify_grid_alignment.py`.

## 3. Running the fleet

```text
GitHub → Actions → "Process All Tiles" → Run workflow
  which_tiles: incomplete          # the only mode you normally need
  tiles_file:  (optional)          # e.g. processing/tile_data/station_tiles_v10.txt
  how_many:    0                   # 0 = no limit
```

- The dispatcher folds icechunk commit history (snapshot-pinned for the whole run),
  emits batches of ≤ 256 tiles, and each tile job processes **only its missing water
  years** then refreshes composites. `incomplete` is idempotent: **re-dispatch until
  the remaining count is zero** — completed work is never redone, failures never
  commit and are simply picked up again.
- Monitor with `processing/2_check_tile_status.ipynb` (authoritative store-derived
  status, per-WY completion, fleet compute accounting, remaining-work list — writes
  the `processing/results/v10/*.csv` snapshots) or `gh run list` for job-level view.
- Single tile: "Process Single Tile" workflow, or locally
  `pixi run python processing/scripts/process_single_tile.py --tile-row R --tile-col C
  --water-years missing`. `--water-years` semantics: `missing` (resume; default in the
  workflows), `all` (recompute + supersede, warned loudly), `none` (composites only),
  or an explicit list. `--local-store PATH` runs the full pipeline against a local
  icechunk repo for testing (single worker only — local FS storage is unsafe for
  concurrent commits).
- Local batch runs (CryoCloud etc.): `processing/scripts/run_tiles.py` — same status
  derivation, no coordination needed beyond the store.

### Failure triage

Only *patterns* deserve investigation; scattered one-off failures are runner
roulette and converge on redispatch.

| Symptom in the job log | Meaning | Action |
| --- | --- | --- |
| `Network is unreachable` on the first external call | Runner boot-time egress failure (hits a few % of runners) | None — jittered retries absorb most; redispatch catches the rest |
| Load dies at exactly the SAS `se=` timestamp (403s, `WarpOperationError`) | Dense year outran the ~45-min MPC token window on a slow runner | Redispatch (fresh runner). Persistent: the year is structurally too big — raise its worker tier in `scale_workers_by_density` or split |
| `over the drop cap ... failing instead of committing a thinned year` | More confirmed-dead scenes (blob 404 but STAC-listed) than the caps allow | Investigate — this is either real archive loss or an outage; caps are `≤3 scenes` and `≤max(1, ceil(5%))` |
| `ALREADY have commits and will be recomputed and superseded` | An explicit-year dispatch is redoing committed work | Fine if intentional (algorithm change); otherwise use `--water-years missing` |
| Same tile failing fast on every redispatch | Real bug | Pull the log; this is how the drop-cap and Severnaya Zemlya issues were found |

Runner facts to remember: ~16 GB RAM (workflows set `MALLOC_ARENA_MAX=2`; dask
workers auto-tier by scene density), 330-min timeout (timeouts are cheap — committed
years survive), throughput varies ~3× between runners.

## 4. Quality checking

- `processing/3_quality_check_tiles.ipynb` — per-tile stats and v9-vs-v10 visual
  comparison for the test-tile battery (antimeridian, scene-densest, Rainier,
  empty-year mix, catalog backfill, Cayambe equator pair). Read its expectations
  cell before alarm: `pct_identical` tracks S1 acquisition redundancy, not
  correctness — the corrected QC filter legitimately changes sparse years.
- `processing/scripts/list_dropped_scenes.py` — global inventory of every scene
  dropped as missing-from-storage, from commit metadata.
- Spot-verify store bytes against commit `stats` (valid-pixel counts must match
  exactly); `2_check_tile_status.ipynb` does this on samples.

## 5. Adding a new water year

The **Water Year Watch** workflow (monthly cron) opens a GitHub issue when a water
year becomes processable for a hemisphere — northern WY N closes Sep 30 N, southern
Mar 31 N+1, plus `trailing_buffer_days = 120` (phenology's own 90-day buffer + fleet
grace). Eligibility is enforced in both the dispatcher and the processor, so nothing
can be committed for a season that hasn't fully elapsed — an ineligible year just
stays `missing` until its date passes. The two hemispheres of the same water year
become eligible ~6 months apart; no manual coordination is needed.

Order of operations (also in the issue template):

1. **Upstream first**: the [`MODIS_snow_phenology`](https://github.com/egagli/MODIS_snow_phenology)
   store must have the new year committed for the eligible hemisphere(s) (it has the
   same watch workflow + its own extend script).
2. Run `processing/5_add_water_year.ipynb`: checks hemisphere eligibility, spot-checks
   that the phenology store actually holds the new year (known snowy windows per
   hemisphere), dry-runs the plan, then does the shard-aligned metadata-only append
   via `store.extend_water_years` (discovers `water_year`-dimensioned arrays by their
   zarr `dimension_names`; verifies the new slab reads as fill).
3. Bump `WY_end` in `config/global_config_v10.txt`, commit, push.
4. Dispatch **Process All Tiles** (`incomplete`) — the new (tile, year) pairs appear
   in the work list automatically; composites go stale and refresh per tile.

## 6. Store & repo maintenance

- **After a full fleet run**: `processing/4_finalize_icechunk_store.ipynb` — freezes
  the ledger (lossless export of every commit's metadata), tags the release (`v10.x`;
  tags are GC roots), garbage-collects (dry-run first, then real), sweeps repo-info
  backups, and verifies. Regenerate `processing/results/v10/bulk_processing_stats.csv`
  (final section of `2_check_tile_status.ipynb`) so the manuscript's
  compute-accounting numbers are final (done 2026-08-11 for the initial fleet).
- **Reprocessing after an algorithm change**: newest-wins — just dispatch with
  explicit years (`all` for full recompute); superseded commits remain in history
  until expired. Store-schema changes (new variable, grid change) require a new
  store version instead.
- **Registry refresh**: rerun the catalog probe in `0_select_tiles_to_process.ipynb`
  when S1 coverage evolves (e.g. Greenland VV backfill); newly-`to_process` tiles
  flow into the next `incomplete` dispatch as wholly-missing tiles.
- **Downstream after data changes**: rerun the evaluation chains
  (`dataset_evaluation/compare_to_all_public_snow_pillows/` notebooks 1→5) and the
  `visualize/` figure notebooks; every manuscript number must land in a `results/`
  CSV via `results.save_result_table` (see
  [`results_and_figures.md`](results_and_figures.md)).

## Known caveats (documented, deliberate)

- **Equator**: tile row 56 overhangs lat 0 by ~2.8 px; the sliver is masked
  (~223 m no-data strip) and the phenology reproject blends hemisphere conventions
  within ~500 m of the equator — materially affects only Volcán Cayambe, which is in
  the test battery. See the "future grid epoch" issue for the alignment fix.
- **No VV polarization** over Greenland/Canadian Arctic: excluded by the registry's
  catalog rule, not by grid extent — the grid already reserves the space for a
  future HH-capable pipeline.
- **Read chunking is deliberately not bit-reproducible against v9** (2048-px reads);
  `--read-chunk-dim 512 --read-chunk-time 10` reproduces v9 exactly when needed.
