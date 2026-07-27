# processing

Implements the dataset creation methodology from manuscript Sect. 2.2 — tiling, the global Icechunk/Zarr v3 output store, and per-tile-per-water-year runoff onset computation across the 23,520-tile global grid (~4,800 tiles with seasonal snow).

## Pipeline (config v10+)

One platform-agnostic entrypoint, [`scripts/process_single_tile.py`](scripts/process_single_tile.py), runs everywhere (GitHub Actions, CryoCloud, local) and processes one tile **one water year at a time**, committing each year to the Icechunk store individually:

- **Memory-bounded**: only one water year's Sentinel-1 stack is computed at a time. Each year is searched, signed, and lazily loaded separately — Planetary Computer asset tokens expire ~45 min after signing, so a single up-front search would leave later years of a 10-year job reading from expired URLs. A year whose load/compute still fails (transient blob error or token expiry on a slow link) is retried once from a fresh, freshly-signed search.
- **Read tuning**: S1 is read in whole-tile, single-scene dask tasks by default (`--read-chunk-dim 2048 --read-chunk-time 1`): ~27% fewer bytes and ~2× faster loading than the v9-equivalent 512-px chunks. This is deliberately **not bit-reproducible** against v9 (decision July 2026): measured end-to-end, 99.56% of pixels are identical, 0.04% of coverage shifts at scene-footprint edges, and ~0.4% of pixels flip between near-tied backscatter minima (tile statistics unchanged). Pass `--read-chunk-dim 512 --read-chunk-time 10` for exact v9-equivalent output (e.g. version cross-checks). On high-latency home connections, raise `--dask-workers` (e.g. 12) — S1 loading is latency-bound and throughput scales nearly linearly with workers (benchmarked 10 → 31 MB/s), bringing even the scene-densest tile-year inside the Planetary Computer token lifetime. Note odc-stac reads COG *overviews* automatically: the level follows the min-axis scale ratio between source (10 m UTM) and the 4326 grid — ×4 (40 m) at mid-latitudes, only ×2 (20 m) near 70°N where a 0.00072° longitude pixel is ~27 m (a longstanding v9 property, unchanged in v10).
- **Failure isolation**: a timeout/crash mid-tile loses at most one year; committed years are never redone. A failed tile×water-year never commits — status derivation treats absence as "missing" and re-dispatches it.
- **Verified-empty markers**: a year with genuinely nothing to process gets an empty commit with a reason (`no_seasonal_snow` from the phenology mask, checked before any S1 work; `no_s1_data` after a *successful* STAC search returns nothing; `no_valid_pixels` when nothing survives quality filtering). Transient Planetary Computer failures are retried and, if persistent, fail the job — they are never recorded as empty.
- **Composites**: after the annual layers, the cross-year composites (`runoff_onset_median`/`mad`, `temporal_resolution_median`) are recomputed from in-memory results plus store readback of years from previous runs, and committed last. Reprocessing a single year automatically marks the tile's composites stale for refresh.
- **Provenance**: every commit's metadata embeds where it ran (GitHub Actions run/runner, JupyterHub/CryoCloud session, or hostname), package versions, and the code SHA.

Status lives in the commit history alone (`global_snowmelt_runoff_onset/status.py`); the published v9 store and its CSV-based tracking remain frozen for provenance.

## Memory profile and the per-orbit escape hatch

The scene-densest tile is **(7, 138)** (~70°N, Scandinavia): up to **594 scenes/year** across **10 relative orbits** (~60 scenes each; one full-tile single-orbit stack ≈ 1 GB). The current path (`process_one_year`: mask → denoise → per-orbit quality filter → temporal resolution + runoff onset in **one fused `dask.compute`**) peaks **3.8–5.15 GB at 4 workers** on the densest tiles — comfortably inside a GitHub Actions runner's 16 GB (>3× headroom). **No memory change is required for GitHub Actions**, and the pipeline is validated as-is.

A **per-orbit Python loop** (materialize one orbit at a time, `idxmin` it, accumulate the 2-D constituent minima, then combine) was benchmarked against the current path on tile (7, 138) WY2019, full 2048² tile, identical bytes, single-threaded:

| scenes | current (groupby + dask) | per-orbit loop |
| --- | --- | --- |
| 150 | 3.82 GB | **2.65 GB** (−31 %) |
| 300 | >4.6 GB (OOM on a 4.6 GB-headroom host) | **3.38 GB** (completes) |

The two paths are **bit-for-bit identical** (onset and temporal resolution; max diff 0.0, identical valid-pixel masks) — consistent with the earlier 512² check. The per-orbit loop's per-orbit phase is bounded at ~1 GB, but its peak is set by the **shared constituent → median combine step** (all orbits' 2-D minima held while the cross-orbit median is taken), which it does *not* reduce; the current path's peak instead grows with total scene count. Net effect at the densest tile: roughly **−1.5 GB (~30 %)**.

**Decision (July 2026): not integrated.** The current path already fits GitHub Actions with large margin, and the per-orbit win is modest (the dominant consumer is the shared combine, unaffected). The per-orbit loop stays the documented fallback if a future denser tile or a smaller runner ever creates memory pressure. Two gotchas for that fallback, both learned the hard way while benchmarking:

- **Detach each orbit's reduction result to a plain NumPy array before accumulating.** Appending the xarray `idxmin` result directly pins its source orbit stack (~1 GB each), so RSS grows ~1 GB/orbit instead of staying bounded.
- **Cap glibc arenas (`MALLOC_ARENA_MAX=1`) or run the per-orbit reductions single-threaded.** Under the dask *threaded* scheduler the per-thread arenas fragment across the loop, so RSS climbs and can OOM on a low-RAM host even though the true working set is bounded.

For **memory-tight local hosts** (not GHA): the per-year loop leaves a ~3 GB RSS baseline (allocator retention across repeated computes), and the composite step's 10-year readback spikes on top of it. If a tile OOMs near the end, split it into two invocations so composites start from a fresh process: `process_single_tile.py ... --skip-composites`, then `process_single_tile.py ... --water-years none`. Also drop `--dask-workers` (3 on a ~5 GB-free host) and optionally set `MALLOC_ARENA_MAX=2`. None of this is needed on 16 GB GitHub Actions runners, where the single-invocation path is the intended production mode.

## Possible future variable: `runoff_onset_spread`

A natural sixth store variable would be a **per-pixel, per-water-year spread of the constituent onset estimates** — the MAD (or std) across the per-orbit/polarization backscatter-minimum dates that the median currently collapses into `runoff_onset`. This is a direct, empirical per-pixel/per-year standard error that automatically encodes speckle (latitude-dependent effective looks), sampling density, backscatter-trough flatness, and orbit count. Analyses could weight or filter by it, and interannual-variability studies could subtract it in quadrature to separate true year-to-year variability from estimation noise.

Example implementation (small, localized):

- **`store.py`**: add `"runoff_onset_spread"` to `VARIABLE_DESCRIPTIONS` (`unit=days`) and to `SCALED_VARIABLES` (0.1-day precision in int16), and add an `empty_3d("runoff_onset_spread")` to `build_template`'s `combine_by_coords`. Encoding is then auto-derived by the existing loop (3-D → shards `(1, 2048, 2048)`, inner chunks `(1, 256, 256)`).
- **`process_single_tile.process_one_year`**: the constituents already exist one call away — `processing.calculate_runoff_onset(..., return_constituent_runoff_onsets=True)` returns dims `(sat:relative_orbit, polarization, latitude, longitude)`. Reduce across those constituent dims (MAD or std, in day units) to a 2-D `float32` field and return it beside `onset_2d`/`tr_2d`.
- **the per-year writer**: add `runoff_onset_spread` to `ds_write` so it is written and committed in the same region/commit as `runoff_onset`.
- optional: a cross-year `runoff_onset_spread_median` composite if a single-layer summary is wanted.

**Cost:** one additional 3-D variable with the same shape/footprint as `runoff_onset` — on the order of **+one-third of the store** on disk (likely less, since small spread values compress well). **This is a schema decision with a closing window:** trivial to add before `create_icechunk_store.ipynb` initializes the production store; a data migration afterward.

## Notebooks

| Notebook | Description |
| --- | --- |
| [`select_tiles_to_process.ipynb`](select_tiles_to_process.ipynb) | One-time builder of the 23,520-tile global grid; filters to the tiles with meaningful seasonal snow (Sturm & Liston classification) and writes `tile_data/global_tiles_with_seasonal_snow.geojson`, the static tile registry. |
| [`create_icechunk_store.ipynb`](create_icechunk_store.ipynb) | One-time init of the v10+ Icechunk output repository: metadata-only Zarr v3 template with shards of (1 water_year, 2048, 2048) — exactly one tile×water-year per shard — and (1, 256, 256) inner chunks, plus per-water-year manifest splitting persisted in the repo config. |
| [`create_zarr_store.ipynb`](create_zarr_store.ipynb) | Legacy (≤ v9): pre-allocated the plain Zarr v2 store backing the published dataset. Kept for provenance. |
| [`quality_check_tiles.ipynb`](quality_check_tiles.ipynb) | Compares two dataset versions tile-by-tile for consistency after a config or algorithm change. |
| [`process_tiles.ipynb`](process_tiles.ipynb), [`process_tiles_serverless.ipynb`](process_tiles_serverless.ipynb) | Legacy Coiled/Dask bulk processing (≤ v9). Superseded by the GitHub Actions + `run_tiles.py` paths; kept for provenance until v10 is fully validated. |

## Scripts (`scripts/`)

| Script | Description |
| --- | --- |
| [`process_single_tile.py`](scripts/process_single_tile.py) | The single entrypoint described above. `--water-years all\|none\|2019,2020`, `--local-store` for testing against a local icechunk repo. Exits nonzero on failure (no commit) so GitHub Actions/`run_tiles.py` surface it. |
| [`get_tiles_for_batch.py`](scripts/get_tiles_for_batch.py) | Emits GitHub Actions matrix JSON of remaining work derived from icechunk commit history; each entry carries the tile's missing water years. `--list-batches` mode also pins the fleet run to one snapshot for consistent batching. |
| [`run_tiles.py`](scripts/run_tiles.py) | Local/CryoCloud batch driver: same status derivation, runs `process_single_tile.py` as subprocesses (`--max-workers N`, `--dry-run`). Note: keep `--max-workers 1` when writing to a `--local-store` filesystem repo (local icechunk storage is not safe for concurrent commits; Azure is). |
| [`consolidate_artifacts.py`](scripts/consolidate_artifacts.py) | Legacy (≤ v9) CSV artifact consolidation. Superseded — status now comes from commit history. |

## `tile_data/`

- `global_tiles_with_seasonal_snow.geojson` — the static tile registry (row/col, bbox, snow fraction) produced by `select_tiles_to_process.ipynb`. This is the only tile bookkeeping file; it never changes during processing.
- `tile_results_v2.csv` … `tile_results_v9.csv` — historical per-tile processing status for the legacy pipeline; kept for provenance, not read by v10+.

## Related

- [`global_snowmelt_runoff_onset/README.md`](../global_snowmelt_runoff_onset/README.md) — the core algorithm package (`processing.py`), plus the v10 pipeline modules: `store.py` (schema/init), `status.py` (commit-history status), `provenance.py` (compute-platform metadata).
- [`.github/workflows/README.md`](../.github/workflows/README.md) — how `process_single_tile.py` is invoked at scale.
- [`config/`](../config) — versioned processing configuration files consumed via `global_snowmelt_runoff_onset.config.Config`; v10 is the first icechunk config.
