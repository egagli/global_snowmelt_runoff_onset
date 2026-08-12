# pyramid

Generation code for the **multiscale visualization store**: a standalone plain Zarr v3 pyramid
derived from the icechunk dataset at a release tag, consumed by *both* the static global
figures ([`../global/`](../global/) — replacing the v9 `create_coarsened_global_maps.ipynb` →
`coarsened/` store convention) and the interactive web map
([`../../interactive_map/`](../../interactive_map/README.md), which holds the map app itself
and the options survey). QGIS (GeoZarr plugin) and GDAL ≥ 3.13 read the same store directly.

**Status (2026-08-13):** builder implemented and shaken down against the finalized `v10.0` tag
(snapshot `5GXWVWYG3BG5XMFXNAH0`). Production build runs via
[`build_pyramid.yml`](../../.github/workflows/build_pyramid.yml).

## How it's built

[`build_pyramid.py`](build_pyramid.py) drives **topozarr end-to-end** (pinned `==0.1.4` in
both pixi envs). The original plan here was a hand-rolled template-then-region-write fan-out
(25 per-(variable, water_year) slab jobs — see git history); it was retired when inspection of
topozarr 0.1.4 showed its `Pyramid.write` already implements the same architecture internally:
level 0 is streamed region by region from the lazy source, each coarser level is block-reduced
from the previously written level (shard-sized regions on a thread pool, bounded memory), and
all-fill regions are skipped so ocean/no-data chunks never exist.

Semantics verified empirically before adoption (synthetic fill/3-D/append tests, 2026-08-13):

- **The mean kernel is fill-aware on raw int16**: windows average valid values only
  (`[10, 20, -9999, 30] → 20`), all-fill windows stay `_FillValue` and aren't written.
  So the source is read **raw** (`mask_and_scale=False`): level 0 is a value-exact copy and
  the cascade is mean-of-valid in encoded-integer space — identical to decode→mean→re-encode
  up to truncation toward zero (≤ 0.1 day on scaled variables, ≤ 1 day on `runoff_onset`).
- Integer dtype, `_FillValue`, `scale_factor`, and `long_name` propagate to every level, and
  the zarr-level `fill_value` is set from `_FillValue` (the v10-store init bug class doesn't
  reproduce here — still asserted in verification).
- 3-D `(water_year, latitude, longitude)` variables coarsen per-year with chunk 1 on the year
  axis; the `water_year` coordinate is written at every level.
- Separate `create_pyramid(...).write(store, mode='a')` calls merge into one store — the basis
  for the job split below.
- Emitted attrs are the registered convention trio (`zarr_conventions` + multiscales `layout`
  + `proj:code`/`proj:wkt2` + `spatial:transform`/`bbox`/`shape`/`dimensions`/`registration`)
  plus per-variable `zarr-layer` `layer_hints`.

**Three jobs, not 25**: topozarr writes whole arrays, so the finest safe parallel unit is a
variable group — `composites` (3 year-less vars), `runoff_onset`, `temporal_resolution`. The
composites job runs first, alone (it creates the root group, level groups, coords, and
convention attrs); the two yearly jobs then run in parallel — they own disjoint zarr arrays,
so they never write the same object. Jobs are idempotent: rerun a failed one whole (the source
is tag-pinned, so rewrites are value-stable). One caveat this inherits: all 11 water years of
a variable live in one job, so per-year parallelism isn't available — fine at this size.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Store | Separate plain Zarr v3 on `uwcryo/snowmelt`, versioned prefix `snowmelt_runoff_onset/global_runoff_onset_<tag>_multiscale_<n>` | No manifest indirection for the browser; CDN-cacheable; disposable/regenerable — the icechunk repo stays the only source of truth. Bump `<n>` to bust caches on regeneration. |
| Level 0 | Included (value-exact raw copy) | Exact point readouts in the map popup; self-contained for QGIS/GDAL. ~66 GB. |
| Source | Tag (`--source-tag v10.0`), never a branch | Map and figures provably show the released dataset. `provenance` root attr records store, tag, snapshot id, method, topozarr version. |
| Levels | 2× spacing, `/0` (499,998×204,800) … `/9` (976×400) | 2× is what browser-direct clients want; ~10 levels covers zoom 0 → native. Trailing partial windows trim (iterated floor-halving; 499,998 → 249,999 → 124,999 → …). |
| Aggregation | Fill-aware integer mean-of-valid, cascaded level-from-level | Verified above. Cascade reads ~88 GB total vs ~660 GB recomputing every level from level 0; unweighted mean-of-means bias at partially-valid parents is bounded and matches the v9 figure precedent. Weighted (exact) cascade only if QC ever shows edge artifacts. |
| Encoding | Same as source per variable: int16, `_FillValue=-9999`, scale 0.1 on the three scaled vars | Halves size vs float32; zarr-layer decodes fill/scale; proven by the MODIS pyramid. |
| Chunks / shards | topozarr defaults: ~512 KB target chunks → (1, 512, 512) int16, 4×4 chunks/shard → (1, 2048, 2048) shards, empty chunks skipped | Good HTTP fetch unit; matches the source tile granularity; ocean simply doesn't exist. |
| Writer | obstore (`zarr.storage.ObjectStore` + `AzureStore`, SAS from the config) | Rust object_store handles the Azure byte-range patterns for Zarr v3 shards that adlfs gets wrong. |
| `layer_hints` | DOWY vars: viridis, clim [110, 270] (plot_utils month-colorbar convention); MAD/TR: magma, [0, 30] / [0, 24] days | Defaults for generic zarr-layer viewers; the map app sets its own. |

## Files

- [`build_pyramid.py`](build_pyramid.py) — the driver. `--job composites|runoff_onset|temporal_resolution|all`,
  `--variables` override (shakedowns/partial rebuilds), `--source-tag`, `--dest-prefix`,
  `--levels`, `--max-workers`, `--plan-only`.
- [`2_verify_pyramid.ipynb`](2_verify_pyramid.ipynb) — acceptance gates + the
  `Cache-Control` pass (see below).
- [`../../.github/workflows/build_pyramid.yml`](../../.github/workflows/build_pyramid.yml) —
  production build: composites job, then the two yearly jobs as a parallel matrix.
  Azure-proximate runners move ~150 GB in ~1–2 h total; the same build through a home
  connection is an overnight run.

## Runbook

1. **Shakedown** (already done for v10.0, repeatable for any new tag): one small variable to a
   scratch prefix, then the verification gates:

   ```bash
   pixi run python visualize/pyramid/build_pyramid.py \
       --variables runoff_onset_median \
       --dest-prefix snowmelt/snowmelt_runoff_onset/scratch_pyramid_shakedown
   ```

2. **Production build**: `gh workflow run build_pyramid.yml`. The workflow has exactly two
   inputs — `source_tag` (default `v10.0`) and a `mode` dropdown (`fresh` / `resume`) — and
   always builds all five variables; the config file and destination prefix derive from the
   tag, and the cache-busting generation counter is the committed `MULTISCALE_GENERATION`
   constant in `build_pyramid.py` (bump it in code to regenerate a tag's pyramid at a new
   prefix). The driver writes a progress marker (`_build/<job>.json` under the pyramid prefix)
   after every completed level, so recovery from a timeout or failure is just **redispatch
   with `mode=resume`**: each job continues from its first incomplete level (a partially
   written level is rewritten deterministically — the source is tag-pinned) and
   already-complete jobs no-op. Resuming against a marker from a different snapshot fails
   loudly. Observed reality (first v10.0 build, 2026-08-12): composites 1h46m; each yearly
   job's level-0 pass alone ~5 h at topozarr's derived 4 workers, hitting the then-300-min
   timeout inside level 2 (~92% done) — hence the marker system, 360-min timeouts, and
   `--max-workers 16` (I/O here is latency-bound; the fleet's thread-pool sweep showed
   scaling through 16).
3. **Verify + headers**: run [`2_verify_pyramid.ipynb`](2_verify_pyramid.ipynb) against the
   production prefix — structure/attrs lint, level-0-vs-source exact comparison on the QC
   tiles, cross-level visuals, then the `Cache-Control: public, max-age=31536000, immutable`
   pass (cache-busting is by prefix version, so immutable is safe).
4. **Consumers** (each unblocked once 3 passes): the map app in `interactive_map/`
   (zarr-layer 0.8.0 reads the store self-describing); repoint
   `../global/global_annual_runoff_onset_and_temporal_res.ipynb` and
   `../global/global_composites.ipynb` at levels 4/5 (≈1.3/2.6 km) and retire
   `../global/create_coarsened_global_maps.ipynb`; QGIS/GDAL for free.

## Gotchas (learned or inherited)

- **Coordinate `_FillValue`**: the raw (`mask_and_scale=False`) open leaves `_FillValue: nan`
  in coordinate attrs, which xarray's CF encoder refuses at write time — the driver strips it
  from coords (data-var `_FillValue` stays; topozarr keys on it).
- **`pyramid.as_datatree()` eagerly coarsens the source** — fine on toy data, 191 GiB here.
  The driver's `--plan-only` computes level shapes itself; don't "just peek" via datatree.
- **ci env zarr pin**: topozarr needs `zarr>=3.2.1`; without the explicit pin in
  `[feature.ci.dependencies]` the conda solver settles on 3.1.6 and the pypi solve fails.
- **zarr v3 `fill_value` vs `_FillValue`** (the v10 store-init bug): topozarr sets both
  correctly; the verify notebook still asserts `fill_value == -9999` per array.
- **CORS**: the container allows `egagli.github.io`; a localhost zarr-layer dev page may need
  a CORS rule addition (service-level Azure setting) — check before debugging "failed fetch".
- Convention v0.1 → v1 (expected late 2026) likely means one regeneration: prefix bump + one
  workflow dispatch.

## Cost

~88 GB storage (66 level-0 + ~22 overviews, the 4/3 geometric factor); ~1–2 h wall on GHA.
Zero new services, zero new Azure configuration. Regeneration costs the same — which is what
makes the pyramid safely disposable.

## Explicitly deferred

- **Phase 2 (in-icechunk overviews + icechunk-js)** — benchmark after the map ships; topozarr
  can write into icechunk directly now, making the experiment cheap.
- Weighted (exact) mean cascade — only if QC finds edge artifacts.
- Zenodo/archival of the pyramid — regenerable viz artifact, not part of the dataset record.
