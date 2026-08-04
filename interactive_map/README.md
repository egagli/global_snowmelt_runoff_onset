# Interactive map viewer — options and recommendations

Planning document for an interactive web map of the global snowmelt runoff onset dataset, in the spirit of the [MODIS_snow_phenology map](https://egagli.github.io/MODIS_snow_phenology/) ([repo](https://github.com/egagli/MODIS_snow_phenology)). Written July 2026, while the v10 icechunk pipeline was being stood up; revisit maturity claims (marked below) before implementation.

> **Status: planning only.** This directory contains no code. Phase 1 hasn't started — the v10 production store was initialized 2026-08-04 and fleet runs are in progress (live progress: `processing/check_tile_status.ipynb`), so pyramid generation waits on filled data. None of the Phase-1 deliverables named below (`create_zarr_multiscales.ipynb` copy, a `map/` app, a Pages deploy workflow) exist yet anywhere in the repo.

## What we're starting from

- **The dataset** (v10): Icechunk repository on Azure (`uwcryo/snowmelt`), Zarr v3, EPSG:4326, ~80 m global (499,998 × 204,800 — 84.05°N to 63.41°S since the 2026-07-30 grid extension), five int16 variables with `_FillValue=-9999` (`runoff_onset` and `temporal_resolution` with a `water_year` dimension 2015–2025; three cross-year composites without it), shards of (1, 2048, 2048) with (1, 256, 256) inner chunks. Sparse: ~4,200 of 24,500 tiles contain data, and the extreme north/south rows are empty (no VV coverage) — the viewer should fit its default view to the *data* extent (~81°N–60°S), not the array extent.
- **Infrastructure we have**: GitHub Pages, the Azure `snowmelt` container (already anonymous-blob-readable and CORS-open to `egagli.github.io` — the MODIS map reads it with no SAS today), and the GitHub-Actions tile-fan-out + Icechunk commit machinery built for processing (lineage: [icechunk_github_actions_demo](https://github.com/egagli/icechunk_github_actions_demo)).
- **Infrastructure we don't have**: any always-on server.

**Requirements**: global pan/zoom at native detail, per-water-year layer with a year slider, composite layers (median/MAD), point-click value readout, colormap/range controls, shareable static site, and no (or almost no) new infrastructure.

## The 2026 stack in one paragraph

Every serverless browser option below is the same four layers with different choices at each: a **store** (plain Zarr v3 over HTTPS, or an Icechunk repo via [icechunk-js](https://github.com/EarthyScience/icechunk-js)), the **[multiscales convention](https://github.com/zarr-conventions/multiscales)** (+ GeoZarr `proj:`/`spatial:` attrs) describing overview levels, a **pyramid generator** ([topozarr](https://github.com/carbonplan/topozarr) for this convention; [ndpyramid](https://github.com/carbonplan/ndpyramid) for the legacy Web-Mercator profile), and a **browser client** ([zarrita.js](https://github.com/manzt/zarrita.js) underneath [@carbonplan/zarr-layer](https://github.com/carbonplan/zarr-layer) for 2D maps, or Cesium/other clients). The alternative to all of it is a **dynamic tile server** that reads the store and emits ordinary XYZ tiles.

---

## Option A — Standalone plain-Zarr pyramid + zarr-layer (the MODIS_snow_phenology pattern)

Generate a multiscale pyramid as a **separate plain Zarr v3 store** next to the icechunk repo; render it with `@carbonplan/zarr-layer` from a static Next.js site on GitHub Pages.

This is exactly what [MODIS_snow_phenology](https://github.com/egagli/MODIS_snow_phenology) ships in production (`map/create_zarr_multiscales.ipynb` + `map/`), verified in its code:

- **topozarr** `create_pyramid(method='mean', target_chunk_bytes=1MB, chunks_per_shard=4)` → sharded Zarr v3 levels `/0` (finest) … `/N`, `multiscales` layout attrs, per-variable `layer_hints` (colormap, clim) read directly by zarr-layer.
- Re-encoded to **int16** with explicit `_FillValue`, `write_empty_chunks=False` (sparse ocean/no-data chunks simply don't exist).
- Written with **obstore** (`zarr.storage.ObjectStore(obstore.store.AzureStore(...))`) — chosen there because obstore's Rust `object_store` handles the Azure *suffix byte-range requests* needed to read Zarr v3 shard indices, which adlfs gets wrong. `consolidated=False`.
- A post-write pass sets `Cache-Control: public, max-age=31536000` on every blob; cache-busting by versioning the prefix (`..._multiscale_v2`).
- Map: static Next.js + MapLibre + zarr-layer; `selector: { water_year: { selected: i, type: 'index' } }` drives the year slider; point queries hit pyramid level 0 directly with zarrita.

**Our dataset is strictly easier than the MODIS case**: it's already EPSG:4326 (zarr-layer supports it natively — MODIS needed a sinusoidal `proj4` string + GPU reprojection), `water_year` is already a dimension, and int16/−9999 maps directly to `fillValue`.

| Pros | Cons |
|---|---|
| Proven end-to-end by us, on the same Azure container, same GitHub Pages origin (public access + CORS already in place) | Duplicates data: with level 0 included, ~2.3× the annual-store footprint (v9-derived estimate: v9's store was ~60 GB compressed → pyramid ≈ +80 GB; **re-measure against the filled v10 icechunk store** before committing to level-0 inclusion); starting at level 1 (160 m) halves that but point readouts become 160 m values |
| Zero servers; CDN/browser-cacheable; fastest possible chunk reads (no manifest indirection) | A second artifact that goes stale whenever tiles are reprocessed; must be regenerated (cheap, but a step to remember) |
| Pyramid store can be regenerated/deleted freely — the icechunk repo stays the single source of truth | Requires the pyramid prefix to be public-read (already true for this container) |
| topozarr emits the modern multiscales convention → same pyramid also readable by Zarr-Cesium etc. | topozarr is young (v0.0.x, 2026) — pin the version |

**One improvement over the MODIS implementation**: its notebook writes the whole pyramid in one `DataTree.to_zarr` and carries an OOM warning (~278 GB uncompressed working set). The [earthmover multiscales post](https://www.earthmover.io/blog/multiscales-in-al/) generates overviews **incrementally and resumably — initialize the structure first, then populate only regions containing data, in parallel**. That is *exactly* the template-then-region-write pattern our processing pipeline (and [icechunk_github_actions_demo](https://github.com/egagli/icechunk_github_actions_demo) before it) already uses, so pyramid generation can reuse the same tile fan-out (locally via `run_tiles.py`-style batching or on GitHub Actions) instead of one monolithic write.

## Option B — Overviews inside the icechunk repo + icechunk-js (the earthmover pattern, serverless variant)

Store overview levels as **groups in the same icechunk repository** as the native data, following [earthmover's multiscales-in-Arraylake post](https://www.earthmover.io/blog/multiscales-in-al/) and the [icechunk-multiscales-demo](https://github.com/earth-mover/icechunk-multiscales-demo): e.g. `4x/`, `16x/`, `64x/` groups carrying GeoZarr `proj:`/`spatial:` metadata, a `multiscales` attr on the root, `(time, y, x)` dimension order (which we already have). The browser reads the repo **directly** via [icechunk-js](https://github.com/EarthyScience/icechunk-js) plugged into zarr-layer as a custom zarrita store — a documented integration (our local zarr-layer v0.5.0 clone ships the `IcechunkStore` example in its README/types), demonstrated publicly by CarbonPlan's `icechunk_prec` demo.

Key facts (vetted):

- **Storage economics flip**: the native data *is* level 0, so nothing is duplicated. Earthmover's numbers: a 2× pyramid adds ~33%, a **4× pyramid ~6.7%** — for us, roughly +4 GB instead of +80 GB.
- **Transactional consistency**: overviews and native data update in one commit; the map can pin a tag (`v10.0`) so viewers always see a released, self-consistent version. With Option A the pyramid is always slightly stale until regenerated.
- **icechunk-js** (as of v0.4.x, mid-2026): read-only (fine for a viewer), TypeScript, browsers + Node, auto-detects Icechunk format v1/v2, implements zarrita's `AsyncReadable` incl. range reads for **sharded** arrays and optional range coalescing, custom `fetchClient` for SAS/auth. Caveat: over plain HTTP, branch/tag *listing* is only reliable on format-v2 repos (ours is created by icechunk 2.x, so fine — and pinning a snapshot/tag avoids listing anyway).
- **Aggregation-method constraint** (from the earthmover post): only *composable* methods (mean) can cascade level-from-level; median/mode/nearest must resample from native data at every level. Mean-of-valid is fine for visualizing DOWY fields.
- **Level-spacing tradeoff the blog doesn't spell out**: 4×/16× spacing is efficient behind a tile server that cuts tiles from whichever level, but a browser client picking "the coarsest sufficient level" can fetch up to 16× more pixels than the screen needs at 4× spacing. Browser-direct rendering wants 2× spacing — which erodes the storage advantage to ~+33% (still ≪ Option A).

| Pros | Cons |
|---|---|
| Minimal extra storage (+7% at 4×, +33% at 2×) | Every chunk read pays icechunk manifest indirection → more HTTP round-trips than plain Zarr (the MODIS notebook cites exactly this as why it flattened its pyramid) |
| Atomic consistency + tag-pinned map versions ("the map shows v10.0") | icechunk-js is young (v0.4.x); the zarr-layer+icechunk path is demonstrated but has far fewer production miles than Option A |
| One artifact; no regeneration bookkeeping | Whole repo (incl. history) must be public-read, or every viewer needs a fetchClient/SAS scheme |
| Overview writes reuse our existing commit machinery | Overview commits must not race tile processing (root-attr `multiscales` updates conflict) — write them **after** processing, as a tagged post-processing commit |

## Option C — Dynamic tile server (xpublish-tiles / titiler.xarray; managed: Arraylake Flux)

Run a small server that reads the icechunk store and emits standard XYZ/WMS raster tiles: **xpublish-tiles** (the open-source backend behind earthmover's Flux product; it selects the coarsest sufficient GeoZarr multiscale level per request) or [titiler + titiler.xarray](https://github.com/developmentseed/titiler) (mature, Development Seed). Earthmover's **Arraylake/Flux** is the managed, paid version of the same idea — it's how the icechunk-multiscales-demo is actually served in their world.

| Pros | Cons |
|---|---|
| Tiny storage (pairs with 4× in-icechunk overviews); tiles work in any client incl. QGIS/ArcGIS/Leaflet | **Requires an always-on server** — the one thing we don't have; cost, maintenance, uptime |
| Data can stay private; auth handled server-side | Per-tile latency (data reads + encode) unless fronted by a CDN |
| Server-side band math/masking possible | Overkill for a manuscript companion site |

## Option D — Legacy @carbonplan/maps + ndpyramid (not recommended)

The pre-2026 CarbonPlan stack: [ndpyramid](https://github.com/carbonplan/ndpyramid) builds strict Web-Mercator slippy-tile pyramids consumed by [@carbonplan/maps](https://github.com/carbonplan/maps) (React + old open-source Mapbox GL v1). It works — but it forces a reprojected visualization copy in a rigid layout, is React/Mapbox-v1-bound, and CarbonPlan itself has moved new work to zarr-layer/topozarr. Only relevant if we hit a blocking bug in zarr-layer.

## Option E — Other clients worth knowing about

- **[Zarr-Cesium](https://github.com/NOC-OI/zarr-cesium)** (National Oceanography Centre): 3D globe, volumetric slices, animated velocity fields, reads multiscale Zarr (4326/3857) via zarrita. Overkill for 2D DOWY rasters, but a compelling outreach/3D-globe option that could read the *same* pyramid Option A produces.
- **[Browzarr](https://github.com/EarthyScience/Browzarr)**: browser app for ad-hoc exploration (cube/sphere/map views, pixel time series) of any Zarr URL — useful as an "explore the raw data" link for power users, not a curated map. Same group maintains icechunk-js.
- **[deck.gl-raster](https://github.com/developmentseed/deck.gl-raster)** (Development Seed): lower-level GPU raster building blocks (zarr-layer borrows its mesh reprojection). Only relevant if we outgrow zarr-layer and build custom rendering.

*(These descriptions come from the April 2026 ecosystem survey and repo docs; only zarr-layer/topozarr/icechunk-js were code-verified or primary-source-checked by us.)*

---

## Cross-cutting design notes (apply to A and B)

- **Coarsening**: mean-of-valid (composable, cascades cheaply, NaN-aware through the −9999→NaN decode). Keep median/MAD rigor in the analytical store; the pyramid is a visualization artifact.
- **Layers**: `runoff_onset` (year slider) + `temporal_resolution` (year slider) + the three composites as year-less layers. `clim=[1, 366]` DOWY with a month-aware colormap for onset (reuse `plot_utils` month conventions), days for TR/MAD.
- **Point queries**: read level 0 with zarrita (Option A) or icechunk-js (Option B) for exact station-comparison-grade values in the popup.
- **Sparse-aware, resumable generation**: initialize pyramid structure first, populate only data-bearing regions (the tile registry says which), parallelize with the existing tile fan-out. Avoids the MODIS notebook's monolithic-write OOM and matches the earthmover incremental recipe.
- **Grid quirk**: our dims (499,998 × 204,800) aren't powers of two — coarsening gets ragged edges; topozarr handles trim/pad. Cosmetic only. (Latitude is now an exact multiple of the 2048 tile size; longitude still isn't.)
- **Azure**: `snowmelt` container is already anonymous-read + CORS-open for the MODIS map — no new Azure admin work for Option A. Set long-lived `Cache-Control` on pyramid blobs and version the prefix to bust caches.
- **Scaffolding**: the MODIS `map/` Next.js app (basePath config, deploy-map.yml GitHub Pages workflow, tile-status GeoJSON layer) is directly copyable; our status layer can be driven by `status.py`'s GeoDataFrame (structured commit metadata) instead of commit-message regexes.

## Recommendation

**Phase 1 (with/right after the v10 production run): Option A.** It's proven twice over on our own infrastructure, requires zero new services or Azure changes, and decouples the map from the still-evolving store. Concretely: copy `MODIS_snow_phenology/map/` + `create_zarr_multiscales.ipynb`, swap in the runoff variables/colormaps, build the topozarr pyramid at 2× spacing **including level 0** (exact point readouts; ~+80 GB is acceptable on uwcryo), write with obstore, generate incrementally per region, deploy on GitHub Pages.

**Phase 2 (opportunistic, after v10 is complete and tagged): prototype Option B.** Write 4× (or 2×) overviews into the icechunk repo as a post-processing tagged commit (the icechunk-multiscales-demo notebook is the template; our commit-with-retry machinery already fits), point a branch of the map at it via icechunk-js, and benchmark against the plain pyramid. If interaction latency is acceptable, retire the standalone pyramid and reclaim ~75 GB — and the map becomes version-pinned to dataset tags for free. This is also the configuration that ages best: it's where the ecosystem (GeoZarr conventions, icechunk-js, xpublish-tiles) is converging.

**Option C only if** the map outgrows static hosting (heavy traffic, private data, or a lab server materializes) — at which point the Phase-2 in-icechunk overviews are already the right storage layout for xpublish-tiles.

## References

- [MODIS_snow_phenology](https://github.com/egagli/MODIS_snow_phenology) — our production reference implementation ([live map](https://egagli.github.io/MODIS_snow_phenology/))
- [icechunk_github_actions_demo](https://github.com/egagli/icechunk_github_actions_demo) — origin of the tile-parallel icechunk write/status patterns this repo's pipeline (and the pyramid generation plan) reuses
- [Earthmover: Multiscales in Arraylake](https://www.earthmover.io/blog/multiscales-in-al/) + [icechunk-multiscales-demo](https://github.com/earth-mover/icechunk-multiscales-demo) — in-icechunk overview conventions, storage math, incremental generation
- [CarbonPlan: Flexible Zarr visualization for web maps](https://carbonplan.org/blog/zarr-layer-maps) — zarr-layer announcement; [zarr-layer](https://github.com/carbonplan/zarr-layer) · [topozarr](https://github.com/carbonplan/topozarr)
- [CarbonPlan OCR: producing](https://carbonplan.org/blog/producing-ocr-data) / [mapping](https://carbonplan.org/blog/mapping-ocr-data) — the end-to-end serverless template (icechunk region writes → zarr-layer + PMTiles + DuckDB-WASM, no servers)
- [icechunk-js](https://github.com/EarthyScience/icechunk-js) · [zarrita.js](https://github.com/manzt/zarrita.js) · [multiscales convention](https://github.com/zarr-conventions/multiscales)
- [titiler](https://github.com/developmentseed/titiler) · [Zarr-Cesium](https://github.com/NOC-OI/zarr-cesium) · [Browzarr](https://github.com/EarthyScience/Browzarr) · [deck.gl-raster](https://github.com/developmentseed/deck.gl-raster)
- [Earthmover: serverless datacube pipeline](https://www.earthmover.io/blog/serverless-datacube-pipeline/) · [Icechunk GLAD ingest guide](https://icechunk.io/en/latest/guides/ingestion/glad-ingest/) · [NASA IMPACT Zarr visualization report](https://nasa-impact.github.io/zarr-visualization-report/) (tiling-vs-dynamic-client tradeoff benchmarks; further reading)

### Provenance of claims

Code-verified locally (July 2026): everything attributed to MODIS_snow_phenology's `map/` + pyramid notebook, and zarr-layer v0.5.0's capabilities (Zarr v3, sharding codec, 4326/3857 + proj4 untiled mode, multiscales `layout` parsing, selectors, custom-store/Icechunk hook). Fetched from primary sources: the earthmover multiscales post + demo README, icechunk-js README. From the April 2026 ecosystem survey (plausibility-checked but not independently verified): Zarr-Cesium/Browzarr/deck.gl-raster feature details and project release dates.
