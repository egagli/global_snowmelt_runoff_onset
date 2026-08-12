# processing

Implements the dataset creation methodology from manuscript Sect. 2.2 — tiling, the global Icechunk/Zarr v3 output store, and per-tile-per-water-year runoff onset computation across the 24,500-tile global grid (~4,400 selected tiles: seasonal snow *and* usable VV data per the catalog rule, plus manual additions via `tile_data/manual_tiles_v10.txt`).

## Grid extension

The global grid is a single EPSG:4326 geobox at 0.00072000072000072° (~80 m), cut into 2048 × 2048-pixel tiles that are also the Zarr shards. Config `v10` (2026-07-30) extends it north and south of the ≤ v9 grid:

| | ≤ v9 | v10 |
| --- | --- | --- |
| `bbox_top` → realized | 81.099 → 81.0994411 | **84.048 → 84.0485640** |
| `bbox_bottom` → realized | −59.999 → −59.9990999991 | **−63.4074 → −63.4075834** |
| pixels (lat × lon) | 195,970 × 499,998 | **204,800 × 499,998** |
| tiles (rows × cols) | 96 × 245, last row partial (1,410 px) | **100 × 245, all rows full** |

**Why these numbers.** The bbox edges snap *outward* to the lattice of `resolution` multiples, so the realized grid is not exactly what you type and a sub-pixel edit changes the row count: `84.0486` yields +4,097 rows (not +4,096) and `−63.4076` leaves a 1-pixel partial tile row. Both are asserted at config load via `expected_grid_shape` / `expected_tile_grid`; `processing/scripts/verify_grid_alignment.py` re-derives the whole mapping from the config files and is the check to run after any bbox change.

- **North (+2 tile rows, to 84.05°N) was the one-shot decision.** The geobox origin is top-left, so latitude rows can only be added *before* index 0 — which renumbers every tile and cannot be done in place. Two rows cover **all land on Earth** (northernmost is Kaffeklubben Island, 83.67°N; Ellesmere 83.1°N), so nothing terrestrial will ever require moving this edge again. Going further (e.g. to the pole) costs almost nothing in storage but leaves permanently dead rows in every global figure and pyramid level.
- **South (to a tile boundary, −63.41°S) is reversible.** Rows added at the *end* of a dimension are an in-place Zarr `resize`, so the south edge can grow later without renumbering anything — provided it lands on a tile boundary, which it now does. It reaches South Orkney, the South Shetlands and the Antarctic Peninsula tip; further south is a clean append whenever it's wanted.
- **Longitude was left alone.** ±180° is not on the lattice (it falls at ±249,999.75 px), so the current edges are already the nearest lattice points; the residual seam at the antimeridian is under two pixels wide, and closing it would shift every column index and introduce a sub-pixel overlap.

**Mapping to earlier versions.** The lattice is unchanged, so no pixel center moves:

> **v9 tile (r, c) == v10 tile (r + 2, c)** and **v9 pixel (i, j) == v10 pixel (i + 4096, j)**

verified exhaustively over all 23,520 v9 tiles (byte-identical geobox transforms). Use `store.grid_pixel_offset()` and `store.tile_region_slices_on_grid()` rather than hardcoding the offset, and compare the two stores **by integer index, not by coordinate value**: the materialized `latitude` arrays are computed from different origins and differ by ~3e-14° (one float64 ULP), so `xr.align` / exact coordinate equality fails while the grids are in fact identical.

> ⚠️ **Tile indices recorded before 2026-07-30 — including every benchmark table below, GitHub Actions job names, and log artifacts from runs #1–#4 — are on the ≤ v9 grid.** Add 2 to the row to get the current index: the densest tile (7,138) is now **(9,138)**, Rainier (23,39) is now **(25,39)**, the antimeridian tile (8,0) is now **(10,0)**.

Extending the grid does **not** by itself add coverage: northern Greenland and the Canadian Arctic sit inside the old grid already and are excluded because Planetary Computer's IW-only `sentinel-1-rtc` collection carries **HH/HV** there, not the VV this pipeline uses. At the 2026-08-03 full registry re-probe, 848 snowy tiles are excluded by the S1 catalog: **584** carry only non-VV items (≈364 Greenland/NE-Canada, ≈187 Canadian Arctic, ≈33 elsewhere), **200** have ≤ 40 VV items (stray acquisitions below the `to_process` threshold), and **64** have no S1 RTC items at all. (Northern Greenland has ~80,000 HH items against a handful of stray VV ones, first appearing 2025-02-26.) The extension reserves the space so that adding HH support later is a pipeline change, not another grid rebuild. Tiles the catalog rule excludes can be force-selected via `tile_data/manual_tiles_v10.txt` (see *tile_data* below) — used 2026-08-11 to restore the 165 v9-data tiles whose only snow is Sturm class-4 ephemeral.

### The equator

Tile row 56 overhangs latitude 0 by 2.78 px, so `processing.remove_equator_crossing` masks the sliver on the wrong side of the equator (a ~223 m no-data strip) to keep one hemisphere's water-year convention per tile; separately, the bilinear phenology reproject blends the MODIS phenology's northern/southern DOWY conventions within ~500 m of the equator. Volcán Cayambe — tiles (56,69)/(57,69) — is the only place on Earth where seasonal snow touches the equator, and that tile pair is in the v10 test-tile battery (`test_tiles.yml`, `3_quality_check_tiles.ipynb`) for exactly this reason.

## Pipeline (config v10+)

One platform-agnostic entrypoint, [`scripts/process_single_tile.py`](scripts/process_single_tile.py), runs everywhere (GitHub Actions, CryoCloud, local) and processes one tile **one water year at a time**, committing each year to the Icechunk store individually:

- **Memory-bounded**: only one water year's Sentinel-1 stack is computed at a time. Each year is searched, signed, and lazily loaded separately — Planetary Computer asset tokens expire ~45 min after signing, so a single up-front search would leave later years of a 10-year job reading from expired URLs. A year whose load/compute still fails (transient blob error or token expiry on a slow link) is retried once from a fresh, freshly-signed search.
- **Read tuning**: S1 is read in whole-tile, single-scene dask tasks by default (`--read-chunk-dim 2048 --read-chunk-time 1`): ~27% fewer bytes and ~2× faster loading than the v9-equivalent 512-px chunks. This is deliberately **not bit-reproducible** against v9 (decision July 2026): measured end-to-end, 99.56% of pixels are identical, 0.04% of coverage shifts at scene-footprint edges, and ~0.4% of pixels flip between near-tied backscatter minima (tile statistics unchanged). Pass `--read-chunk-dim 512 --read-chunk-time 10` for exact v9-equivalent output (e.g. version cross-checks). On high-latency home connections, raise `--dask-workers` — S1 loading is latency-bound and throughput scales nearly linearly with workers (quantified in *Chunking and dask* below: 4/8/12/16 workers → 5.4/9.6/10.9/16.1 MB/s), bringing even the scene-densest tile-year inside the Planetary Computer token lifetime. Note odc-stac reads COG *overviews* automatically: the level follows the min-axis scale ratio between source (10 m UTM) and the 4326 grid — ×4 (40 m) at mid-latitudes, only ×2 (20 m) near 70°N where a 0.00072° longitude pixel is ~27 m (a longstanding v9 property, unchanged in v10).
- **Failure isolation**: a timeout/crash mid-tile loses at most one year; committed years are never redone. A failed tile×water-year never commits — status derivation treats absence as "missing" and re-dispatches it.
- **Verified-empty markers**: a year with genuinely nothing to process gets an empty commit with a reason (`no_seasonal_snow` from the phenology mask, checked before any S1 work; `no_s1_data` after a *successful* STAC search returns nothing; `no_valid_pixels` when nothing survives quality filtering). Transient Planetary Computer failures are retried and, if persistent, fail the job — they are never recorded as empty.
- **Composites**: after the annual layers, the cross-year composites (`runoff_onset_median`/`mad`, `temporal_resolution_median`) are recomputed from in-memory results plus store readback of years from previous runs, and committed last. Reprocessing a single year automatically marks the tile's composites stale for refresh.
- **Provenance**: every commit's metadata embeds where it ran (GitHub Actions run/runner, JupyterHub/CryoCloud session, or hostname), package versions, the code SHA, and the machine size (`cpu_count`, `memory_gb`) so CPU-core-hours are derivable per commit (`duration_s / 3600 × cpu_count`).
- **Bulk-processing stats in every data commit** (added 2026-08-04; commits before then lack them): `stats.dest_gb` — the in-memory size of the year's S1 stack on the destination (tile) grid, i.e. the data volume read at the 80 m overview level — plus `stats.mb_s_effective` (throughput over the fused download+compute), `stats.dask_workers`, and `stats.peak_rss_gb` (process max-so-far; Linux `ru_maxrss`). Summed over the commit history (e.g. in `2_check_tile_status.ipynb`, where `status.get_commit_records` flattens them to `stats.*` columns) these replace the manuscript Sect. 2.2.3 bulk numbers with exact fleet-wide values — no log scraping. Verified-empty `no_valid_pixels` years keep their stats too (they still downloaded the full stack).

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

**Measured on GitHub Actions runners (2026-07-27, TEST tiles runs #2/#3):** ~15.5 GB visible RAM. Without the arena cap (run #2, 4 workers), Rainier's 10-year loop crept 2.38 → 5.62 GB and (7,138) killed its runner outright at 102 min (log never uploaded). With `MALLOC_ARENA_MAX=2` + 8 dask workers (run #3): Rainier 34 → 20 min (1.7×), and **(7,138) survives in memory — RSS plateaus flat at 14.2–14.7 GB (95%)** — but hits the wall-time limit instead: dense ~70°N years are 10–20× mid-latitude volume (×2 overview instead of ×4, ~600 scenes) at 40–50 min each, so a full 10-year dense tile exceeds any single job. Consequences, all applied: workflows set the arena cap, run `--dask-workers auto` (per-water-year thread count scaled down as scene density grows — see *Chunking and dask* below; a fixed 8 was the proven setting through 2026-07), and use `timeout-minutes: 330`; the handful of densest tiles simply finish on a second status-driven dispatch (each committed year is durable, so a timeout costs nothing).

For **memory-tight local hosts** (not GHA): the per-year loop leaves a ~3 GB RSS baseline (allocator retention across repeated computes), and the composite step's 10-year readback spikes on top of it. If a tile OOMs near the end, split it into two invocations so composites start from a fresh process: `process_single_tile.py ... --skip-composites`, then `process_single_tile.py ... --water-years none`. Also drop `--dask-workers` (3 on a ~5 GB-free host) and optionally set `MALLOC_ARENA_MAX=2`. None of this is needed on 16 GB GitHub Actions runners, where the single-invocation path is the intended production mode.

## Chunking and dask: measured answers (2026-07-27)

Benchmarked on tile (23,190) WY2020 (55 scenes, full 2048² tile, real MPC downloads, each variant a fresh process; all variants produced bit-identical outputs):

| variant | wall | peak RSS | verdict |
| --- | --- | --- | --- |
| production (read 2048/t=1, mask 512, fused compute), ×2 runs | 225–231 s | 3.3–3.5 GB | **keep** |
| mask chunked 2048 (aligned, no chunk-split warning) | 234 s | 3.9 GB | no win, +0.5 GB peak |
| `persist()` masked stack, then reduce | 234 s (219 dl + 15 compute) | 3.2 GB | no win |
| synchronous scheduler (no dask) | 606 s | 3.0 GB | 2.6× slower |

What this settles:

- **The `PerformanceWarning: Increasing number of chunks by factor of 16` is benign and the design behind it is correct.** S1 reads land as one (1, 2048, 2048) chunk per scene; masking against the (512, 512)-chunked snow mask splits each into 16. The pipeline therefore *downloads at 2048* (fewest HTTP round trips; ~27 % fewer bytes and ~2× faster loading than 512 reads) and *computes at 512* (smaller tasks, lower peak). Forcing compute to 2048 removes the warning but is measurably worse (+0.5 GB peak, no speedup). Ignore the warning.
- **`persist()`/staged `compute()` buys nothing**: the fused single `dask.compute` per year is equal within noise, and persist holds the whole masked stack in RAM (a liability on dense years). The persist run also cleanly measures the wall-time split: **~95 % download, ~5 % compute** — this pipeline is a download job with a small compute tail.
- **Dask is helping, and specifically as a download parallelizer**: the threaded scheduler at 3 workers is 2.6× faster than synchronous; compute itself is too small to matter. Consequently `--dask-workers` is the main throughput lever (S1 loading is latency-bound; the drift-cancelled mirrored sweep below is the reference measurement), bounded by memory (peak RSS scales with concurrent scene chunks).
- **Scene density only strengthens these conclusions**: denser years shift the balance even further toward download, and memory (the one scene-sensitive axis) is characterized in the section above.
- **GDAL HTTP tuning: refuted (interleaved A/B, same tile-year, ×2 each).** The popular cloud bundle (`CPL_VSIL_CURL_CHUNK_SIZE=5MB` + big curl cache + HTTP/2 + threaded decode) was **38% slower** than GDAL defaults (117 s → 187 s): overview-level COG reads fetch scattered 16–512 KB blocks, so 5 MB range granularity over-reads massively. The surgical subset (`GDAL_INGESTED_BYTES_AT_OPEN=65536` + `GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES`) was a statistical tie. Keep `configure_rio(cloud_defaults=True)` and otherwise leave GDAL alone. Worker count remains the only proven download lever (GHA measured: 4→8 workers = 1.7× on Rainier; peak RSS 5.6→6.7 GB of ~14.4 GB).
- **On-runner measurement beats extrapolation**: local and GHA performance differ (bandwidth, latency to West Europe, cores). Every job now logs a startup perf line (dask workers, CPUs, perf-relevant env) and a per-year `GB dest … MB/s effective` line, so worker-count or env experiments on runners are read directly from job logs.
- **Thread-pool oversubscription keeps paying through 16 threads** (mirrored sweep, same tile-year, drift-cancelled): 4/8/12/16 workers → 5.4/9.6/10.9/16.1 MB/s. dask's threaded scheduler *is* a thread pool (a custom `ThreadPoolExecutor` via `dask.config.set(pool=…)` is equivalent to `num_workers`); GDAL issues range requests serially within a task, so in-flight requests ≈ thread count, and the pool is effectively the I/O-concurrency knob. `dask.distributed`/process pools would only add IPC and duplicate the mask (the GIL is already released during I/O). The binding constraint is memory: concurrency must shrink as scene density grows. Hence **`--dask-workers auto`**: per-water-year thread count from scene count (16/12/8/6 for ≤150/≤300/≤450/>450 scenes), applied per `dask.compute(..., num_workers=N)`. **Validated on GHA and adopted** — the fleet workflows (`process_single_tile.yml`, `process_batch.yml`) now pass `--dask-workers auto` (was a fixed 8 through 2026-07).

## QC filter corrected in v10: edge-anchored max-gap, count filter removed

Through v9, `calc_max_gap_pixelwise` documented "max gap between consecutive **valid** acquisitions
within the search window" but implemented the inverse: an `xr.where` polarity flip measured the DOWY
spacing between consecutive **invalid** acquisitions (valid ones broke the diff chain), and the
window-edge sentinels produced negative diffs that never survived the `max` — in practice the filter
approximated an orbit-revisit check with no working edge rule (synthetic check: returned 50 where
the documented answer is 150).

**v10 fixes the semantics** (2026-07-28): per pixel/orbit/polarization, the max gap between
consecutive valid acquisitions must be ≤ `max_allowed_days_gap_per_orbit`, **with the window edges
anchoring the first and last gap** (window start → first valid acquisition; last valid acquisition →
window end; the window is the MODIS-derived `[SAD + max_consec_snow_days/2, SDD + 16 d]`). The
implementation is vectorized (bottleneck-backed `ffill`, dask-friendly) and verified against a
brute-force reference (300/300 randomized cases; dask/numpy parity).

**`min_monthly_acquisitions` was removed** along with the fix: gap ≤ G over a window of length W
implies ≥ W/G − 1 valid acquisitions, so the 1/month count floor was implied up to one acquisition;
denser requirements are expressed by lowering the gap threshold (2/month evenly ≡ gap ≤ 15). Old
config files may still contain the key; it is ignored.

Pipeline-equivalence verification (Rainier 512² WY2020, 120 scenes): onset and temporal_resolution
are **bit-identical wherever no per-orbit QC verdict changed**; all differing pixels (173 orbit
verdicts, all old-pass-only) are exactly the intended QC change propagating through the cross-orbit
median.

**Measured coverage delta, old vs corrected QC (2026-07-28; one dense + one sparse year per test
tile; `@1024` = center quarter-tile where the full tile exceeded local memory):**

| tile | WY | scenes | coverage old → new | net | orbit-verdict flips (old-only / new-only) |
| --- | --- | --- | --- | --- | --- |
| (8,0) | 2018 | 30 | 2,749,264 → 2,749,264 | 0 | 0 / 0 |
| (8,0) | 2015 | 3 | 1,171 → 974 | **−17 %** | 572 / 375 |
| (23,39) | 2015 | 49 | 1,394,116 → 2,198,012 | **+58 %** | 1.35 M / 3.99 M |
| (23,39)@1024 | 2020 | 179 | 961,757 → 961,757 | 0 | 2,242 / 18 |
| (26,65) | 2018 | 80 | 1,244,701 → 1,244,730 | +29 px | 96 / 2,030 |
| (26,65) | 2017 | 78 | 7,109 → 7,109 | 0 | 0 / 3 |
| (23,190) | 2020 | 55 | 2,672,387 → 2,672,387 | 0 | 437 k / 164 k |
| (23,190) | 2016 | 14 | 597 → 29,867 | **50×** | 135 / 32 k |
| (78,240) | 2023, 2024 | 43 | 161 / 39 → identical | 0 | 0 / 0 |
| (7,138)@1024 | 2019 | 478 | 1,031,919 → 1,031,919 | 0 | 2,517 / 3 |
| (7,138)@1024 | 2015 | 114 | 1,031,920 → 1,031,920 | 0 | 17 k / 38 k |

Pattern: **dense years are unchanged** (union coverage identical everywhere; verdict churn ≤0.25 %,
absorbed by orbit redundancy). **Sparse years with few orbits change substantially and mostly GAIN
coverage** — dropping the count filter releases pixels at its knife edge (gap-pass guarantees
W/30 − 1 acquisitions; the count rule demanded W/30), e.g. Rainier WY2015 +58 %, (23,190) WY2016
50×. The gained pixels carry coarse `temporal_resolution` values, so downstream tr-based filtering
still applies. Ultra-sparse single-orbit years can shrink slightly (edge anchoring; (8,0) WY2015
−17 %). The published v9 dataset used the old semantics — v9-vs-v10 comparisons will show these
sparse-year coverage differences.

## Adding new water years (WY2025+): requirements and the hemisphere problem

**Water-year clocks differ by hemisphere** (easysnowdata convention, used by both repos): northern
WY N = Oct 1 (N−1) → Sep 30 (N); southern WY N = **Apr 1 (N) → Mar 31 (N+1)** — the south finishes
the "same" water year ~6 months later. The config already anticipates the southern tail
(`end_date = {WY_end+1}-03-31`).

**Phenology is extendable today — the "MOD10A2 is dead" worry is wrong** (verified against CMR
2026-07-28: MOD10A2 granules exist through 2026-07-12 — Terra is still observing — with full
coverage of both hemispheres' WY2025 windows, 14k+ granules each). What ended was the *Planetary
Computer mirror* of MOD10A2 (mid-2025); the MODIS_snow_phenology repo already fetches from NSIDC
via `earthaccess`, which is unaffected. So extending phenology to WY2025 (and to WY2026 as each
hemisphere's season closes) is a run of the existing MODIS pipeline plus a water_year append to its
store. Watch-item, not blocker: Terra's orbit is drifting, which may eventually degrade product
consistency and force a VIIRS transition — check granule continuity (CMR query above takes seconds)
before each new-year extension. The S1 side is unaffected: the pipeline is platform-agnostic and
S1C/S1D share the 175-track relative-orbit grid (recent catalog months are exclusively S1C/S1D).

**Status of the above (2026-07-30): WY2015–2025 are the initial store range** (phenology completed
both hemispheres of WY2025, `WY_end = 2025`, and the store is created with 11 years from the
start), so the first extension this machinery faces is WY2026.

**Mechanics in this repo once phenology exists for the new year:**
1. Bump `WY_end` in the config (water_years and the search dates derive from it).
2. **Extend the store's water_year dimension**: run
   [`5_add_water_year.ipynb`](5_add_water_year.ipynb), which checks the preconditions
   (hemisphere eligibility, and that the phenology store actually contains the new year —
   the hemisphere trap below), then calls `store.extend_water_years` (zarr v3 resize +
   coordinate append in one icechunk commit; shard-aligned and metadata-only — new year
   slabs read as fill until written, composites are 2-D and get overwritten in place;
   dry-run first, verified append). No store rebuild.
3. Nothing else changes: `get_remaining_work` derives missing years from config × commit history, so
   every eligible tile automatically shows the new year as missing; the composite-staleness rule
   marks composites stale once new-year commits land; the fleet processes exactly the new year +
   composite refreshes.

**The hemisphere trap (now guarded in code, twice):** if the phenology store gains a new water-year
*dimension value* while only one hemisphere's phenology is actually computed, the pipeline's
missing-phenology guard passes (the year exists in the store) and
`binary_seasonal_snow_cover_presence.any()` is False for the not-yet-computed hemisphere — tiles
there would be **durably and wrongly committed as `no_seasonal_snow`** (an all-fill phenology slab
is indistinguishable from verified no-snow). The guard is the hemisphere-aware eligibility rule
(`status.season_end` / `status.wy_eligible`): a water year is dispatchable/processable only once
its season has fully elapsed for the tile's hemisphere (centroid latitude; matches the UTM-EPSG
rule the algorithm uses) plus `trailing_buffer_days` (config key, default 120 = phenology's 90-day
trailing buffer + grace for its fleet to run). It is applied in **both** layers:
- **dispatch** — `get_remaining_work` marks ineligible years `'ineligible'` (never `missing`) in
  every mode incl. `all`, and `tile_status == 'complete'` counts only *eligible* years, so a
  config bump months before a season closes doesn't flip the fleet to 'partial';
- **processor** — `process_single_tile.process_tile` vetoes ineligible years before the phenology
  check, committing **nothing** for them (they stay `missing` and are dispatched automatically once
  eligible; if every requested year is vetoed, the job exits without touching composites).

The **Water Year Watch** workflow (monthly cron) opens a GitHub issue per (water year, hemisphere)
when it becomes eligible, with this checklist inline — ~1 month after the MODIS_snow_phenology
repo's equivalent reminder, since phenology must extend first.

## Possible future variable: `runoff_onset_spread`

A natural sixth store variable would be a **per-pixel, per-water-year spread of the constituent onset estimates** — the MAD (or std) across the per-orbit/polarization backscatter-minimum dates that the median currently collapses into `runoff_onset`. This is a direct, empirical per-pixel/per-year standard error that automatically encodes speckle (latitude-dependent effective looks), sampling density, backscatter-trough flatness, and orbit count. Analyses could weight or filter by it, and interannual-variability studies could subtract it in quadrature to separate true year-to-year variability from estimation noise.

Example implementation (small, localized):

- **`store.py`**: add `"runoff_onset_spread"` to `VARIABLE_DESCRIPTIONS` (`unit=days`) and to `SCALED_VARIABLES` (0.1-day precision in int16), and add an `empty_3d("runoff_onset_spread")` to `build_template`'s `combine_by_coords`. Encoding is then auto-derived by the existing loop (3-D → shards `(1, 2048, 2048)`, inner chunks `(1, 256, 256)`).
- **`process_single_tile.process_one_year`**: the constituents already exist one call away — `processing.calculate_runoff_onset(..., return_constituent_runoff_onsets=True)` returns dims `(sat:relative_orbit, polarization, latitude, longitude)`. Reduce across those constituent dims (MAD or std, in day units) to a 2-D `float32` field and return it beside `onset_2d`/`tr_2d`.
- **the per-year writer**: add `runoff_onset_spread` to `ds_write` so it is written and committed in the same region/commit as `runoff_onset`.
- optional: a cross-year `runoff_onset_spread_median` composite if a single-layer summary is wanted.

**Cost:** one additional 3-D variable with the same shape/footprint as `runoff_onset` — on the order of **+one-third of the store** on disk (likely less, since small spread values compress well). **The schema window has closed:** the v10 production store was initialized 2026-08-04 with the five-variable schema and the fleet is committing into it, so adding `runoff_onset_spread` now means a store-level migration (add the array + backfill committed years) or deferring to v11.

## Notebooks

| Notebook | Description |
| --- | --- |
| [`0_select_tiles_to_process.ipynb`](0_select_tiles_to_process.ipynb) | Builder of the 24,500-tile global registry; **rerun whenever the grid or the S1 probe needs refreshing** (last run 2026-08-03, which re-probed the whole catalog and evaluated the four new v10 grid rows). Filters to the tiles with meaningful seasonal snow (Sturm & Liston classification), probes the S1 catalog per polarization, sets `to_process = has_seasonal_snow & (n_vv_items > 40)`, and writes `tile_data/global_tiles_with_seasonal_snow_v10.geojson`, the static tile registry (path set by `valid_tiles_geojson_path` in the config). A final section estimates the **Sentinel-1 input volume** (manuscript Sect. 2.2.3, replacing ">100 TB"): unique VV items intersecting `to_process` tiles × stratified `Content-Length` HEAD-sampled VV asset sizes → `results/<version>/s1_input_volume.csv`. Figures (per-polarization item-count maps, selected-tiles map) → `figures/<version>/`. |
| [`1_create_icechunk_store.ipynb`](1_create_icechunk_store.ipynb) | One-time init of the v10+ Icechunk output repository: metadata-only Zarr v3 template with shards of (1 water_year, 2048, 2048) — exactly one tile×water-year per shard — and (1, 256, 256) inner chunks, plus per-water-year manifest splitting persisted in the repo config. |
| [`create_zarr_store.ipynb`](create_zarr_store.ipynb) | Legacy (≤ v9): pre-allocated the plain Zarr v2 store backing the published dataset. Kept for provenance. |
| [`2_check_tile_status.ipynb`](2_check_tile_status.ipynb) | Fleet-progress dashboard (read-only against the store), safe to run while a fleet run is writing: per-tile / per-water-year status tables, all commit metadata (stats + provenance flattened to columns), throughput plots with a naive ETA, interactive `explore()` maps, the dropped-scene (`missing_assets`) list, and the remaining-work list. Everything derives from the icechunk commit history via `status.py`. Also persists the **bulk-processing stats** (manuscript Sect. 2.2.3: CPU-core-hours, wall time, TB read at 80 m, throughput, peak RSS, platform mix — per water year + totals, all commits incl. superseded reprocesses) → `results/<version>/bulk_processing_stats.csv`, plus a final section of **fleet-diagnostic snapshots** → `results/<version>/{tile_status, wy_completion_breakdown, commit_outcome_summary, dropped_scenes, remaining_work}.csv` (per-tile status **and processing stats** — wall/core-hours, GB read, valid px, scenes, throughput, peak RSS, dropped scenes, last processed, retry counts — plus per-WY completion, per-outcome commit medians, the thinned-year dropped-scene list, and the dispatcher's remaining-work view — each stamped `_written_at` since the fleet keeps moving). |
| [`3_quality_check_tiles.ipynb`](3_quality_check_tiles.ipynb) | Two-part version comparison. The **v9-vs-v10 section** is the current one: the eight v10 test tiles on the extended grid — (10,0) antimeridian, (9,138) scene-densest, (25,39) Rainier, (28,65) empty-year mix, (25,190) catalog backfill, (80,240) `no_seasonal_snow`, and the (56,69)/(57,69) equator-straddling Cayambe pair — read via `store.tile_region_slices` / `tile_region_slices_on_grid` so each store is sliced on its own grid, plus a coverage matrix and an expected-differences rubric (QC fix, read-chunk change). The **older `validate_dataset_comparison` cells at the top are ≤ v9 only** and raise `ValueError` against any icechunk config. |
| [`4_finalize_icechunk_store.ipynb`](4_finalize_icechunk_store.ipynb) | One-time post-fleet cleanup, run only once `2_check_tile_status.ipynb` shows every tile complete: freezes the full commit-metadata ledger → `results/<version>/commit_ledger.parquet` (git-tracked, lossless — every derived table can be rebuilt from it), tags the release (`v10.0`, what the pyramid/map/evaluation notebooks should open), garbage-collects the objects abandoned by failed rebase attempts under fleet commit contention (~699k snapshot objects vs ~53k surviving commits at v10 completion), and sweeps the `overwritten/` repo-info backups that icechunk never deletes itself (2.34 TB of the repo's 2.49 TB at v10 completion). Deliberately **never calls `expire_snapshots`**: the commit history is the ledger `status.py` reads, and expiration would let GC delete it — GC alone keeps the ancestry intact and reclaims ~97% of the dead weight. |
| [`5_add_water_year.ipynb`](5_add_water_year.ipynb) | Runbook for extending the dataset when a new water year becomes processable (the Water Year Watch issue is the trigger; bump `WY_end` in the config first). Checks hemisphere eligibility **and spot-checks the phenology store itself for the new year** (the hemisphere trap: an all-fill phenology slab would durably commit `no_seasonal_snow`, so the notebook asserts a known-snowy window per eligible hemisphere has ≥ 56-day pixels *before* touching the store), then dry-runs and commits the shard-aligned metadata-only append via `store.extend_water_years`, verifies the new (tile, year) items appear in `get_remaining_work` for the eligible hemisphere only, and prints the fleet dispatch command. See *Adding new water years* above; after the fleet, re-run `4_finalize_icechunk_store.ipynb` with a bumped tag. |
| [`explore_s1_rtc_IW_polarization_spatial_distribution.ipynb`](../visualize/s1_rtc_coverage/explore_s1_rtc_IW_polarization_spatial_distribution.ipynb) (moved to `visualize/s1_rtc_coverage/` 2026-08-12) | Full-catalog scan of MPC `sentinel-1-rtc` GeoParquet: per-water-year polarization mix (VV+VH / HH+HV / single-pol), S1A/B/C/D platform turnover, and a Greenland zoom. The evidence base behind the polarization exclusions in *Grid extension* above and the HH-support question. Figures → `../visualize/s1_rtc_coverage/figures/`. |
| [`process_tiles.ipynb`](process_tiles.ipynb), [`process_tiles_serverless.ipynb`](process_tiles_serverless.ipynb) | Legacy Coiled/Dask bulk processing (≤ v9). Superseded by the GitHub Actions + `run_tiles.py` paths; kept for provenance until v10 is fully validated. |

## Scripts (`scripts/`)

| Script | Description |
| --- | --- |
| [`process_single_tile.py`](scripts/process_single_tile.py) | The single entrypoint described above. `--water-years all\|missing\|none\|2019,2020` (`missing` = resume mode, only years with no commit yet), `--local-store` for testing against a local icechunk repo. Exits nonzero on failure (no commit) so GitHub Actions/`run_tiles.py` surface it. |
| [`get_tiles_for_batch.py`](scripts/get_tiles_for_batch.py) | Emits GitHub Actions matrix JSON of remaining work derived from icechunk commit history; each entry carries the tile's missing water years. `--list-batches` mode also pins the fleet run to one snapshot for consistent batching; `--tiles-file` restricts the work universe to an explicit `row,col`-per-line list (the mechanism behind the prioritized station-tile runs via the workflows' `tiles_file` input). |
| [`run_tiles.py`](scripts/run_tiles.py) | Local/CryoCloud batch driver: same status derivation, runs `process_single_tile.py` as subprocesses. Selection/behavior flags: `--which-tiles`, `--how-many`, `--include-empty-years`, `--dask-workers`, `--branch`, `--max-workers N`, `--dry-run`. Note: keep `--max-workers 1` when writing to a `--local-store` filesystem repo (local icechunk storage is not safe for concurrent commits; Azure is). |
| [`verify_grid_alignment.py`](scripts/verify_grid_alignment.py) | Re-derives the v9↔v10 lattice mapping from the config files alone (no credentials); the check to run after any bbox/resolution/tile-size change. |
| [`make_station_tile_list.py`](scripts/make_station_tile_list.py) | Writes `tile_data/station_tiles_v10.txt` (tiles containing evaluable snow-pillow stations) for prioritized runs via `--tiles-file` / the `tiles_file` workflow input. |
| [`apply_manual_tiles.py`](scripts/apply_manual_tiles.py) | Flips `to_process = True` in the registry for every tile in `tile_data/manual_tiles_v10.txt` and appends a `tile_notes` marker — the registry-level way to *add* tiles the catalog rule excludes (`--tiles-file` can only restrict). Idempotent; touches only the geojson, never the store or progress metadata; use instead of rerunning the registry notebook (which re-probes the catalog). `--dry-run` supported. |
| [`list_dropped_scenes.py`](scripts/list_dropped_scenes.py) | Walks commit history for `missing_assets` — scenes present in the STAC catalog whose blobs 404'd during processing — one row per (tile, water year, scene id). |
| [`water_year_watch.py`](scripts/water_year_watch.py) | Behind the monthly `water_year_watch.yml` cron: opens one deduplicated GitHub issue per (water year, hemisphere) at `trailing_buffer_days` past season end. |
| [`migrate_registry_to_extended_grid.py`](scripts/migrate_registry_to_extended_grid.py) | One-shot v9→v10 registry row shift (+2) with placeholders for the new rows; superseded by the 2026-08-03 full rerun of `0_select_tiles_to_process.ipynb`, kept for provenance. |
| [`consolidate_artifacts.py`](scripts/consolidate_artifacts.py) | Legacy (≤ v9) CSV artifact consolidation. Superseded — status now comes from commit history. (`consolidate_artifacts_clean.py` is an untracked 0-byte stub — delete.) |

## `tile_data/`

- `global_tiles_with_seasonal_snow_v10.geojson` — the static tile registry (row/col, bbox, snow fraction, per-polarization S1 item counts — `n_vv_items` / `n_vh_items` / `n_hh_items` / `n_hv_items` plus `n_items_total` — and `to_process`) read by v10+ via `valid_tiles_geojson_path`. The polarization counts are items whose footprint intersects the tile and whose `sar:polarizations` includes that channel, so dual-pol acquisitions appear in two columns and the four never sum to `n_items_total`. Produced by `0_select_tiles_to_process.ipynb`; registries are versioned alongside the config (v11 would write `..._v11.geojson` and point its config at it). Rows with `to_process == False` are kept for documentation and are excluded from the work universe in `status.get_tile_status_gdf`; the rule is **`to_process = has_seasonal_snow & (n_vv_items > 40)`** (the `> 40` floor drops tiles with only stray VV acquisitions — 200 such tiles at the 2026-08-03 probe; `tile_notes` records the exclusion reason per row). This file never changes during processing.
- `global_tiles_with_seasonal_snow.geojson` — the unversioned predecessor of the above, still referenced by the legacy (≤ v9) configs. (The v10 file began as a copy of it, but is now a full regeneration: all 24,500 rows re-probed 2026-08-03 with the per-polarization counts and probe-date columns.)
- `station_tiles_v10.txt` — git-tracked `row,col` list of tiles containing evaluable snow-pillow stations (from `scripts/make_station_tile_list.py`), consumed by the workflows' `tiles_file` input for prioritized runs.
- `manual_tiles_v10.txt` — git-tracked `row,col` list of **manual `to_process` additions**, applied on top of the catalog rule by `0_select_tiles_to_process.ipynb` at registry-build time or standalone by `scripts/apply_manual_tiles.py`. Currently the 165 tiles v9 processed with data (1,585 tile-WYs — Scotland, Baltic coast, west Japan, S Chile, Tasmania, NZ, …) whose only snow is Sturm class-4 *ephemeral*, which the v10 registry's `valid_snow_classes` excludes even though the pipeline's per-pixel MODIS ≥56-day gate finds seasonal snow there in some years (added 2026-08-11: 4,227 → 4,392 `to_process`). Comments in the file carry the per-region rationale.
- `tile_results_v2.csv` … `tile_results_v9.csv` — historical per-tile processing status for the legacy pipeline; kept for provenance, not read by v10+. (`tile_results.csv` and `valid_tiles*.geojson` are older unversioned leftovers of the same lineage.)
- `quality_check/` — git-tracked v7↔v8 and v8↔v9 validation JSON from the legacy comparison workflow.

`0_select_tiles_to_process.ipynb` counts snow classes straight from the Sturm & Liston 300 m GeoTIFF on blob storage (tiled/ZSTD as of 2026-07-29, so windowed reads over HTTPS are cheap — ~2.5 min, ~1.4 GB for all 24,500 tiles). It passes `exactextract` the file the `easysnowdata` handle was opened from rather than the dask-backed DataArray, which measured ~10x faster for the same numbers. A local copy of the GeoTIFF (`SnowClass_GL_300m_10.0arcsec_2021_v01.0_tiled.tif`, ~67 MB) may appear in this directory from a local run; it is gitignored and safe to delete.

## Related

- [`global_snowmelt_runoff_onset/README.md`](../global_snowmelt_runoff_onset/README.md) — the core algorithm package (`processing.py`), plus the v10 pipeline modules: `store.py` (schema/init), `status.py` (commit-history status), `provenance.py` (compute-platform metadata).
- [`.github/workflows/README.md`](../.github/workflows/README.md) — how `process_single_tile.py` is invoked at scale.
- [`config/`](../config) — versioned processing configuration files consumed via `global_snowmelt_runoff_onset.config.Config`; v10 is the first icechunk config.
