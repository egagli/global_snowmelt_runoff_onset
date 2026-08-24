# Interactive map viewer — options and recommendations

Planning document for an interactive web map of the global snowmelt runoff onset dataset, in the spirit of the [MODIS_snow_phenology map](https://egagli.github.io/MODIS_snow_phenology/) ([repo](https://github.com/egagli/MODIS_snow_phenology)). Written July 2026, while the v10 icechunk pipeline was being stood up; revisit maturity claims (marked below) before implementation.

> **Status: planning, Phase 1 unblocked.** The v10 fleet completed 2026-08-11 (all 4,392 tiles × 11 water years + composites), the store was finalized and tagged `v10.0` (`processing/4_finalize_icechunk_store.ipynb`), and the concrete Phase-1 build plan — standalone plain-Zarr pyramid with level 0, populated by 25 independent (variable, water_year) slab jobs — is in [`visualize/pyramid/README.md`](../pyramid/README.md) (2026-08-13; it supersedes this file's Option-A implementation sketch where they differ). Pyramid generation lives under `visualize/` because the static global figures consume the same store; this directory holds only the map app itself. **The map app now exists in [`map/`](map/)** (2026-08-12): the MODIS_snow_phenology app adapted to this dataset — published `@carbonplan/zarr-layer` **0.8.0** from npm (no local clone/webpack aliases; the store is read self-describing from its `proj:`/`spatial:` attrs), five layers (onset + temporal resolution with the water-year slider, three year-less composites), level-0 zarrita point queries with CF scale decoding, DOWY→date readouts, globe/mercator + basemap toggles. Deployed to GitHub Pages by [`deploy_map.yml`](../../.github/workflows/deploy_map.yml); build locally with `pixi run install && pixi run dev` in `map/`. Tooling facts below were re-verified 2026-08-11: zarr-layer is at 0.8.0 (self-describing `proj:`/`spatial:` stores), topozarr at 0.1.4, icechunk-js at 0.6.0 plus official-but-npm-lagging `@earthmover/icechunk` WASM bindings; ndpyramid/@carbonplan/maps confirmed dormant.

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

## Snow classification store (`build_snow_class_store.py`)

[`build_snow_class_store.py`](build_snow_class_store.py) publishes the [Liston & Sturm (2021) Global Seasonal-Snow Classification](https://nsidc.org/data/nsidc-0768) (300 m / 10 arcsec, doi:10.5067/99FTCYYYLAQ0) as a standalone multiscale Zarr v3 pyramid next to the runoff pyramid. The map uses it two ways ([issue #9](https://github.com/egagli/global_snowmelt_runoff_onset/issues/9)):

- the **"snow class" row** of the point-query card, read from level 0, and
- the **"snow class" basemap**, a categorical zarr-layer — which is why it needs to be a pyramid at all: a single-level 300 m array would make a world-view render read every chunk (8.4 GB decoded).

Built with topozarr `method="nearest"`: class codes are categorical, so coarse levels **decimate** rather than average (averaging would invent codes that mean nothing). Documented caveat of nearest — it keeps each window's top-left cell, so a class present only away from corners can vanish at coarse zoom; topozarr has no majority `mode` yet. 10 levels (level 0 native 64800 × 129600 uint8, fill 9, down to 126 × 253), plain (1024, 1024) chunks at level 0 and **no shards**, so a point query is a single HTTP fetch of one explicit chunk blob (all-fill chunks are skipped, so a 404 means fill). The root carries topozarr's `multiscales` + `proj:`/`spatial:` attrs, so zarr-layer self-describes CRS and extent with no client-side georeferencing.

```bash
# production build (full grid -> Azure; immutable-prefix convention, bump _1 to regenerate)
pixi run python visualize/interactive_map/build_snow_class_store.py

# separate post-step: Cache-Control 'public, max-age=31536000, immutable' on every blob
pixi run python visualize/interactive_map/build_snow_class_store.py --set-cache-headers
```

Public level-0 array URL (zarrita FetchStore / `zarr.open_array` target):
`https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/snow_classification_300m_multiscale_1/0/snow_class`
— chunks at `.../0/snow_class/c/{row//1024}/{col//1024}` with `row = floor((89.99999999994958 − lat)/0.0027777777777770003)`, `col = floor((lon + 180.0)/0.0027777777777770003)`; decoded chunks are always full 1024×1024 (edge chunks are fill-padded). Classes 8 (Ocean) / 9 (Fill), missing chunks, and off-grid points are "no class" — the query card shows an em-dash and the basemap shader discards them, so the dark basemap shows through. The `class_info` attr maps codes to names. For local testing the script takes `--local-dest` plus a `--bbox lon_min lat_min lon_max lat_max` subset window and `--levels` (testing only — production runs omit both); `--dry-run` prints the level plan. Verified against the source GeoTIFF: level 0 bit-identical (point checks — Mt Rainier summit → 7 Ice, WA Cascades → 3 Maritime, Puget lowland → 4 Ephemeral, offshore Pacific → 8 Ocean — plus a 512 × 512 block), coarse levels contain only valid class codes, and levels halve exactly.

**Retired**: the pre-pyramid single-level generation `snow_classification_300m_1` (array at the store root, no levels). It was replaced rather than extended because adding levels would have meant rewriting its root `zarr.json` in place, against the immutable-cache convention. Delete it once a deployed map is reading the `_multiscale_1` prefix — see the maintenance runbook.

## Seasonal-snow display filter (issue #9)

The sidebar's "limit to seasonal snow (Sturm & Liston 2021)" toggle is a
shader-side discard against the `seasonal_snow_pct` pyramid variable (percent
seasonal-snow area per cell, 0–100, class 4 Ephemeral excluded; built by the
`seasonal_snow` job in `visualize/pyramid/build_pyramid.py`). Every layer
samples it as an **auxiliary band** and flipping the toggle is a pure
`setUniforms()` call — no layer rebuilds, no refetches. Two deployment notes:

- **zarr-layer fork.** Multi-variable sampling isn't in upstream
  `@carbonplan/zarr-layer` 0.8.0 (bands come only from a selector along one
  dimension of a single variable), so `map/package.json` consumes the local
  fork via `file:../../../../zarr-layer` (`egagli/zarr-layer`, branch
  `aux-variables`, which adds the `auxVariables` option). `deploy_map.yml`
  checks that branch out adjacent to the repo and builds it before `npm ci` —
  the same pattern MODIS_snow_phenology's deploy uses. Fold the branch into
  the fork's `main` (or upstream it) and update the workflow `ref` when it
  settles.
- **Deploys independently of the mask job.** The map probes
  `{ZARR_URL}/0/seasonal_snow_pct/zarr.json` at startup; until the mask job
  has written the variable, layers are built without the aux band and the
  toggle renders disabled ("mask not yet available"). So map deploys and the
  pyramid mask job can land in either order.

Caveat worth keeping in the UI copy: excluding class 4 matches the tile
registry's accepted-class rule, but 165 registry tiles were manually added
*because* their only Sturm class is Ephemeral while the MODIS per-pixel gate
found real seasonal snow (UK/Ireland, S Scandinavia, lowland Japan, NZ, …) —
the toggle hides those regions' valid estimates entirely. It nevertheless
**defaults on** since issue #13 (2026-08-24; originally it defaulted off for
exactly this reason): the sidebar copy now says it's on by default and to turn
it off to see ephemeral-region estimates, and the point-query card shows the
snow class regardless of the toggle.

## Issue #13 additions (2026-08-24): version dropdown, GMBA overlay, zonal warnings

Three features added in one pass ([issue #13](https://github.com/egagli/global_snowmelt_runoff_onset/issues/13)),
plus the seasonal-filter default flip above:

- **Dataset-version dropdown** (sidebar). `VERSION_CONFIGS` in `map/lib/store.ts`
  maps each version to a pyramid URL and water-year list; the map probes every
  version's root `zarr.json` at startup, so a version whose pyramid doesn't
  exist renders disabled ("pyramid not yet published") rather than breaking.
  Per-version level-0 grids for point queries are parsed from each pyramid's
  own `spatial:transform`/`spatial:shape` attrs (`fetchVersionGrid` in
  `map/components/map.tsx`) — nothing per-version is hardcoded, so
  **publishing a v9 pyramid at
  `…/snowmelt_runoff_onset/global_runoff_onset_v9_multiscale_1` activates the
  v9 entry with zero map changes.** The build machinery for that landed
  2026-08-24: `build_pyramid.open_source` now branches on the config era
  (legacy ≤ v9 configs open the frozen Zarr-v2 store, pinned by its
  `.zmetadata` ETag in place of a tag), and `global_config_v9.txt` carries the
  pyramid prefix/generation keys — so all that remains is dispatching
  `build_pyramid.yml` with `config_file: global_config_v9.txt` (redispatch
  with `mode: resume` after any timeout), then running
  `2_verify_pyramid.ipynb` with the config flipped to v9 (its gates are
  era-aware) including its Cache-Control pass. Switching versions tears down
  and recreates the zarr layers, reopens the level-0 query arrays, re-probes
  the seasonal mask, and keeps the selected water *year* when the target
  version has it (v9 lacks 2025).
- **GMBA mountain-range overlay** (sidebar "overlays", default off). 290
  polygons from the GMBA Mountain Inventory v2.0 standard 300-selection
  (Snethlage et al. 2022, doi:10.1038/s41597-022-01256-y), simplified and
  published by [`prepare_gmba_overlay.py`](prepare_gmba_overlay.py) as a
  gzip-encoded (~2 MB transfer), immutable-cached GeoJSON blob:
  `https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/gmba_v2_standard_300_1.geojson`.
  The map lazy-loads it on first enable; hovering highlights the range
  (maplibre feature-state) and shows a card with name, feature type,
  countries, area, and elevation span. To regenerate, bump the `_1` suffix in
  the script's `--dest-blob` **and** `GMBA_URL` in `map/lib/store.ts` together.
- **Zonal warning toasts** (same pattern as the MODIS_snow_phenology map):
  `WARNING_ZONES` in `map/components/map.tsx` is checked against the viewport
  on every moveend/zoomend and surfaces a dismissible toast. Six zones:
  Greenland and the Canadian Arctic Archipelago (no data — MPC S1 RTC is
  HH/HV there, VV required), the equator hemisphere seam at Volcán Cayambe
  (issue #7), and the three MODIS-input artifact zones that propagate into
  our melt-search window (Atacama/Altiplano salt flats, Tibetan Plateau
  lakes, eastern tropical Andes cloud cover).

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
