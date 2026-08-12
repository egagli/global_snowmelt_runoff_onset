# visualize

This directory stores visualization notebooks and outputs — primarily the global composite figures used in the manuscript.

> **Version provenance:** figure outputs are scoped by dataset version (`figures/v9/` = the manuscript renders from the frozen Zarr v2 store `snowmelt/snowmelt_runoff_onset/global_v9.zarr`; `figures/v10/` = renders from the v10 icechunk store / pyramid). The `global/` figures were regenerated for v10 on 2026-08-12, and `methods/` on 2026-08-12 (with the annual stacks and the "N-year composites" title parametrized by water-year count).

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
│   ├── create_coarsened_global_maps.ipynb            # SUPERSEDED (v9/Coiled path, cannot run since Coiled was retired): the global figure notebooks above now read pyramid/ level 7 (~10 km) via build_pyramid.open_pyramid_level
│   └── figures/                                       # rendered PNGs for all global figures above (incl. global_all_composites_robinson_long.png, an alternate aspect ratio of Fig. 2)
├── interactive_map/
│   ├── README.md                       # options survey + phased plan for the web map
│   └── map/                            # the map app: Next.js + MapLibre + @carbonplan/zarr-layer 0.8.0 static export reading the pyramid store; deployed to GitHub Pages by .github/workflows/deploy_map.yml
├── pyramid/
│   ├── README.md                       # the v10 multiscale Zarr v3 store: decisions, verified topozarr semantics, runbook — one artifact serving the global figures here, the interactive_map/ app, and QGIS/GDAL
│   ├── build_pyramid.py                # topozarr end-to-end driver (3 jobs: composites first, then the two yearly vars in parallel); dispatched by .github/workflows/build_pyramid.yml
│   └── 2_verify_pyramid.ipynb          # acceptance gates (structure/attrs lint, level-0-vs-source exact, cross-level visuals) + Cache-Control pass
├── testing/
│   ├── test_antimeridian.ipynb    # diagnostic notebook for the odc-stac antimeridian bug (see below)
│   └── inspect_tile.ipynb         # ad hoc single-tile inspection, not tied to a manuscript figure
└── regions/
    ├── alps/
    │   ├── alps.ipynb               # exploratory regional viewer, not a manuscript figure
    │   └── figures/                 # 5 exploratory PNGs
    ├── iceland/
    │   ├── iceland.ipynb            # exploratory regional viewer, not a manuscript figure
    │   └── figures/                 # 6 exploratory PNGs
    └── rainier/
        ├── rainier_figure.ipynb     # Rainier-region case study; check against methods/ for overlap (decision pending)
        └── geometries/              # mt_baker, glacier_peak, mt_adams, mt_st_helens .geojson outlines
```

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

## Antimeridian diagnostic (`testing/test_antimeridian.ipynb`)

Root-causes a data gap in the westernmost tile column (crossing 180°/-180°) to a bug in `odc.stac.load()`: it computes an invalid footprint for UTM source zones abutting the antimeridian, which causes affected Sentinel-1 scenes to be silently dropped from tile binning. The fix (`wrapdateline=True`) was merged upstream in [opendatacube/odc-stac#281](https://github.com/opendatacube/odc-stac/pull/281) on 2026-07-21 and released as **odc-stac 0.5.3**; the pin in `pixi.toml` was bumped to `>=0.5.3` on 2026-08-03. `global_snowmelt_runoff_onset.processing.ensure_antimeridian_footprint_fix()` still runs a functional self-check at the top of `get_sentinel1_rtc()` and applies the same fix as a monkeypatch only if needed — with the fixed release installed it's a verified no-op, kept as a guard for older environments. The affected tiles (col 0–2) are covered by the clean-start v10 batch runs; tile (10,0) is in the `test_tiles.yml` battery to verify the fix end-to-end.

## Non-manuscript notebooks

`testing/inspect_tile.ipynb` and `regions/alps/alps.ipynb` / `regions/iceland/iceland.ipynb` are ad hoc dataset viewers/regional explorations used for QA, not tied to a specific manuscript figure. `regions/rainier/rainier_figure.ipynb` overlaps with the Rainier panels in `methods/` — related but not duplicates: `methods/create_methods_figure_components.ipynb` uses a single `rainier.geojson` and emits the tracked `rainier_*` panel PNGs, while `regions/rainier/rainier_figure.ipynb` uses four local volcano geometries (Baker, Glacier Peak, Adams, St. Helens) and emits nothing tracked. Keep-or-delete decision still open.
