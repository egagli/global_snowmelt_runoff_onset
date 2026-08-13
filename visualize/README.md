# visualize

This directory stores visualization notebooks and outputs — primarily the global composite figures used in the manuscript.

> **Version provenance:** figure outputs are scoped by dataset version (`figures/v9/` = the manuscript renders from the frozen Zarr v2 store `snowmelt/snowmelt_runoff_onset/global_v9.zarr`; `figures/v10/` = renders from the v10 icechunk store / pyramid). The `global/` figures were regenerated for v10 on 2026-08-12, `methods/` on 2026-08-12 (with the annual stacks and the "N-year composites" title parametrized by water-year count), and `regions/` on 2026-08-12 (year counts, colorbar labels and panel-grid slots all derived from the dataset's water-year count, which is 11 for v10).
>
> **One display source:** every notebook here that needs a below-native-resolution view of the dataset reads the **multiscale visualization pyramid** (`pyramid/`) via `build_pyramid.open_pyramid_level`. The old two-step path — coarsen the full-resolution store into `snowmelt/snowmelt_runoff_onset/coarsened/global_<version>_coarsened_20_ds.zarr`, then read that — is retired: it only ever existed for v9, the coarsened store is not regenerated, and `config.global_runoff_store` is `None` for icechunk configs. `global/create_coarsened_global_maps.ipynb`, the notebook that built it, was deleted 2026-08-13. The one remaining reference is `testing/test_antimeridian.ipynb`, a deliberately v9-pinned historical diagnostic (see below) — it is not expected to run against v10.

## Directory structure

```text
visualize/
├── colorbars/
│   ├── create_colorbars.ipynb          # demos colorbars from global_snowmelt_runoff_onset.plot_utils
│   └── figures/                        # two checked-in viridis_month_colorbar_110_270_*.svg files — see note below
├── data/
│   ├── download_and_preprocess_hillshade.ipynb
│   └── global_hillshade_robinson.tif   # global hillshade used as basemap in Robinson-projection figures
├── methods/
│   ├── create_methods_figure_components.ipynb   # renders each Fig. 1 panel (valid-tiles map, global maps from the pyramid, Rainier S1/S2/snow-phenology imagery) into figures/<version>/
│   ├── combine_methods_figure_components.ipynb  # assembles the panels into the final Fig. 1 (N_YEARS sets stack sheet counts + "N-year composites" title)
│   └── figures/                                 # v9/ + v10/ — per-version panel PNGs + combined methods_figure.png (Fig. 1)
├── global/
│   ├── global_composites.ipynb                       # Fig. 2, Fig. 3, Fig. A2, Fig. A4
│   ├── global_annual_runoff_onset_and_temporal_res.ipynb  # Fig. A3
│   └── figures/                                       # rendered PNGs for all global figures above (incl. global_all_composites_robinson_long.png, an alternate aspect ratio of Fig. 2)
├── interactive_map/
│   ├── README.md                       # options survey + phased plan for the web map
│   └── map/                            # the map app: Next.js + MapLibre + @carbonplan/zarr-layer 0.8.0 static export reading the pyramid store; deployed to GitHub Pages by .github/workflows/deploy_map.yml
├── pyramid/
│   ├── README.md                       # the v10 multiscale Zarr v3 store: decisions, verified topozarr semantics, runbook — one artifact serving the global figures here, the interactive_map/ app, and QGIS/GDAL
│   ├── build_pyramid.py                # topozarr end-to-end driver (3 jobs: composites first, then the two yearly vars in parallel); dispatched by .github/workflows/build_pyramid.yml
│   └── 2_verify_pyramid.ipynb          # acceptance gates (structure/attrs lint, level-0-vs-source exact, cross-level visuals) + Cache-Control pass
├── testing/
│   ├── test_antimeridian.ipynb    # closed diagnosis of the odc-stac antimeridian bug: root cause, upstream fix, and the retired in-repo workaround (see below)
│   └── inspect_tile.ipynb         # ad hoc single-tile inspection, not tied to a manuscript figure
└── regions/
    ├── alps/
    │   ├── alps.ipynb               # exploratory regional viewer, not a manuscript figure
    │   └── figures/                 # v9/ + v10/ — exploratory PNGs per dataset version
    ├── iceland/
    │   ├── iceland.ipynb            # exploratory regional viewer, not a manuscript figure
    │   └── figures/                 # v9/ + v10/ — exploratory PNGs per dataset version
    └── rainier/
        ├── rainier_figure.ipynb     # Rainier-region case study; still v9/Coiled-era and writes no figures (decision pending, see below)
        └── geometries/              # mt_baker, glacier_peak, mt_adams, mt_st_helens .geojson outlines
```

### `regions/` — regional viewers

`alps/alps.ipynb` and `iceland/iceland.ipynb` are the same notebook shape applied to two regions, and both were rebuilt for v10 (2026-08-12): they read `pyramid/` level 2 (~320 m, the closest analogue to the retired v9 "full-resolution store + `.coarsen(3, 3)`" display grid) for the region and level 5 (~2.6 km) for the context inset, and write to `figures/<config.version>/`.

Both are plotted in the region's UTM zone (`REGION_CRS`, one named constant per notebook) — projected metres, so shapes, areas and scalebars are honest, per figure convention 4. Two buffers set the two view scales: `MAP_BUFFER` (150 km alps / 80 km iceland) pads the main map out beyond the region outline so the range is shown in its surroundings rather than cropped hard, and `CONTEXT_BUFFER` (1000 km / 800 km) is the context inset's footprint. Every view in a notebook shares one `REGION_BOUNDS`, so all panels are directly comparable. Each produces:

| Figure | What it is |
| --- | --- |
| `orthographic_context_map.png` | globe-scale locator (region outlined on an orthographic land/ocean view) |
| `median_runoff_onset_map.png` | N-year median runoff onset over the regional hillshade, with the month colorbar |
| `median_runoff_onset_map_with_context.png` | the same map plus a **zoomed-out context inset** — small (~26% of the map width) and pinned in the upper-left corner with equal padding from both edges, so it costs as little of the map as possible; identical variable, hillshade, colormap and limits at `CONTEXT_LEVEL`, with the main map's extent outlined in red |
| `runoff_onset_anomaly_colorbar_horizontal.png` | the standalone diverging anomaly colorbar |
| `anomaly_maps_<R><C>grid_no_cbar.png`, `anomaly_maps_<R><C>grid_w_cbar.png` | per-water-year anomaly panels (anomaly = that year's onset minus the N-year median) on an R×C grid; the `_w_cbar` variants add a thin colorbar row. `R`/`C` in the filename are the actual grid, so the layouts adapt as water years are added — v9's 10-year `52grid`/`25grid` became 11-year `62grid`/`34grid`/`43grid` |

Scalebars follow convention 6: `add_scalebar()`, **frameless**, in the corner opposite whatever else occupies one — lower right on the median maps (the inset owns the upper left), and lower left on the **first panel only** of each anomaly grid, since all panels share `REGION_BOUNDS` and the water-year labels sit lower right. With no box behind it the bar's colour has to suit its corner, so it is passed explicitly: white over ocean (both median maps, the iceland panels), black over the alps' pale lowland panels.

The repetitive parts live in `global_snowmelt_runoff_onset.plot_utils` rather than being copy-pasted between the two notebooks: `variable_kw()`, `HILLSHADE_KW`, `load_hillshade()`, `style_map_axes()`, `add_context_inset()`, `add_scalebar()`, `orthographic_locator_map()` (see the package [README](../global_snowmelt_runoff_onset/README.md)). They are deliberately thin — one repetitive thing each, axes returned — so a notebook can always drop back to raw matplotlib.

**Figure conventions live in the `plot_utils` module docstring** — a numbered list (10 items) covering per-variable colour limits, month colorbars for day-of-water-year data, never hardcoding a water-year count, projected (UTM / equal-area) maps, equal aspect on those maps, scalebar placement and framelessness, hillshade `zorder`, versioned figure directories, save settings, and keeping insets small. Read it before adding a figure anywhere in this repo. Colour limits and the anomaly/median/MAD/temporal-resolution styling are not restated in the notebooks; they come from `variable_kw('<variable name>')`, so changing a convention is a one-line change in one place — `global/global_composites.ipynb` prints the whole `VARIABLE_KW` registry in its shared-styling cell so the colour decisions behind the manuscript figures are visible in the notebook rather than buried in an import.

`rainier/rainier_figure.ipynb` has **not** been ported: it still opens `config.global_runoff_store` (`None` for v10), imports the retired `coiled`, and saves nothing, so there is no `figures/` directory to version. Resolve the keep-or-delete question below before spending effort on it.

Colorbar code comes from `global_snowmelt_runoff_onset.plot_utils`. Colorbar demo in `colorbars/create_colorbars.ipynb`.

> **Note on `colorbars/figures/`:** the two checked-in `viridis_month_colorbar_110_270_*.svg` files were produced by a `create_month_colorbars(110, 270, ..., file_suffix='_110_270')` call that is currently **commented out** in the notebook, and the notebook's `base_path` is `colorbars/` (not `colorbars/figures/`) — so re-running the notebook as committed neither regenerates these files nor writes into `figures/`. Uncomment the call and fix the output path if they ever need regenerating.

Global hillshade in Robinson projection for visualization is in `data/global_hillshade_robinson.tif`. Source: [Natural Earth](https://www.naturalearthdata.com/downloads/10m-gray-earth/gray-earth-with-shaded-relief-ocean-bottom-and-drainages/).

Notebook to download hillshade is `data/download_and_preprocess_hillshade.ipynb`.

## Manuscript figures

Manuscript figure "Figure 1. Graphical representation of the workflow used to create the global snowmelt runoff onset dataset." is at `methods/figures/methods_figure.png`, assembled by `methods/combine_methods_figure_components.ipynb` from panels rendered in `methods/create_methods_figure_components.ipynb`.

Manuscript figure "Figure 2. Global snowmelt runoff onset composite products." can be found at `global/figures/v9/global_all_composites_robinson_wide.png` and figure creation code is in `global/global_composites.ipynb`

Manuscript figure "Figure 3. Polar stereographic projection of 10-year median snowmelt runoff onset date for the Northern Hemisphere." can be found at `global/figures/v9/global_composite_median_polar.png` and figure creation code is in `global/global_composites.ipynb`

Manuscript appendix figure "Figure A2. Global snowmelt runoff onset composite products with polar stereographic projection for the Northern Hemisphere." can be found at `global/figures/v9/global_all_composites_polar.png` and figure creation code is in `global/global_composites.ipynb`

Manuscript appendix figure "Figure A3. Snowmelt runoff onset (day of water year) and temporal resolution (days) for each water year." can be found at `global/figures/v9/global_annual_runoff_onset_and_temporal_res_with_hillshade_2015_2024.png` and figure creation code is in `global/global_annual_runoff_onset_and_temporal_res.ipynb`

Manuscript appendix figure "Figure A4. Per-pixel count of water years with valid annual snowmelt runoff onset estimates." can be found at `global/figures/v9/global_10yr_annual_runoff_onset_count.png` and figure creation code is in `global/global_composites.ipynb`

## Antimeridian diagnostic (`testing/test_antimeridian.ipynb`) — closed

Historical record of a solved problem, kept because it is the only place the diagnosis and the upstream research live. It root-causes a data gap in the westernmost tile column (crossing 180°/-180°) to a bug in `odc.stac.load()`: it computed an invalid footprint for UTM source zones abutting the antimeridian, so affected Sentinel-1 scenes were silently dropped during tile binning — 100% nodata output with `success=True` and no warning.

**Resolved upstream.** The fix (`wrapdateline=True`) was merged in [opendatacube/odc-stac#281](https://github.com/opendatacube/odc-stac/pull/281) on 2026-07-21 and released in **odc-stac 0.5.3**; `pixi.toml` pins `>=0.5.3` (2026-08-03). The interim self-check/monkeypatch in `processing.py` was **deleted 2026-08-13** — the pin makes it unreachable — and is preserved verbatim in the notebook's epilogue cells along with the timeline. The v10 store was built clean after the fix, so nothing needed reprocessing; tile (10,0) is the antimeridian entry in the standing validation battery (`processing/tile_data/test_tiles_v10.txt`) and holds real data in its westernmost columns.

Still open and only recorded here: [odc-geo#208](https://github.com/opendatacube/odc-geo/issues/208), and the notebook's finding that `pyproj`'s `force_over=True` fixes that *and* [odc-geo#176](https://github.com/opendatacube/odc-geo/issues/176) at the coordinate-transform level — a single fix for the shared root cause, never upstreamed. The notebook is deliberately v9-pinned and is not expected to run against v10.

## Non-manuscript notebooks

`testing/inspect_tile.ipynb` and `regions/alps/alps.ipynb` / `regions/iceland/iceland.ipynb` are ad hoc dataset viewers/regional explorations used for QA, not tied to a specific manuscript figure (the two regional ones are still kept runnable and versioned — see [`regions/`](#regions--regional-viewers) above). `regions/rainier/rainier_figure.ipynb` overlaps with the Rainier panels in `methods/` — related but not duplicates: `methods/create_methods_figure_components.ipynb` uses a single `rainier.geojson` and emits the tracked `rainier_*` panel PNGs, while `regions/rainier/rainier_figure.ipynb` uses four local volcano geometries (Baker, Glacier Peak, Adams, St. Helens) and emits nothing tracked. Keep-or-delete decision still open; it was left on the v9/Coiled path rather than ported to v10 pending that decision.
