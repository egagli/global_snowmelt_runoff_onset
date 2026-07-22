# visualize

This directory stores visualization notebooks and outputs — primarily the global composite figures used in the manuscript.

## Directory structure

```text
visualize/
├── colorbars/
│   ├── create_colorbars.ipynb          # demos colorbars from global_snowmelt_runoff_onset.plot_utils
│   └── figures/                        # rendered colorbar SVGs
├── data/
│   ├── download_and_preprocess_hillshade.ipynb
│   └── global_hillshade_robinson.tif   # global hillshade used as basemap in Robinson-projection figures
├── methods/
│   ├── create_methods_figure_components.ipynb   # renders each Fig. 1 panel (mountains map, valid-tiles map, Rainier S1/S2/snow-phenology imagery)
│   ├── combine_methods_figure_components.ipynb  # assembles the panels into the final Fig. 1
│   └── figures/                                 # panel PNGs + combined methods_figure.png (Fig. 1)
├── global/
│   ├── global_composites.ipynb                       # Fig. 2, Fig. 3, Fig. A2, Fig. A4
│   ├── global_annual_runoff_onset_and_temporal_res.ipynb  # Fig. A3
│   ├── create_coarsened_global_maps.ipynb            # builds coarsened rasters used by the global figures above
│   └── figures/                                       # rendered PNGs for all global figures above
├── testing/
│   ├── test_antimeridian.ipynb    # diagnostic notebook for the odc-stac antimeridian bug (see below)
│   └── inspect_tile.ipynb         # ad hoc single-tile inspection, not tied to a manuscript figure
└── regions/
    ├── alps/alps.ipynb              # exploratory regional viewer, not a manuscript figure
    ├── iceland/iceland.ipynb        # exploratory regional viewer, not a manuscript figure
    └── rainier/rainier_figure.ipynb # Rainier-region case study; check against methods/ for overlap (see root README remaining tasks)
```

Colorbar code comes from `global_snowmelt_runoff_onset.plot_utils`. Colorbar demo in `colorbars/create_colorbars.ipynb`.

Global hillshade in Robinson projection for visualization is in `data/global_hillshade_robinson.tif`. Source: [Natural Earth](https://www.naturalearthdata.com/downloads/10m-gray-earth/gray-earth-with-shaded-relief-ocean-bottom-and-drainages/).

Notebook to download hillshade is `data/download_and_preprocess_hillshade.ipynb`.

## Manuscript figures

Manuscript figure "Figure 1. Graphical representation of the workflow used to create the global snowmelt runoff onset dataset." is at `methods/figures/methods_figure.png`, assembled by `methods/combine_methods_figure_components.ipynb` from panels rendered in `methods/create_methods_figure_components.ipynb`.

Manuscript figure "Figure 2. Global snowmelt runoff onset composite products." can be found at `global/figures/global_all_composites_robinson_wide.png` and figure creation code is in `global/global_composites.ipynb`

Manuscript figure "Figure 3. Polar stereographic projection of 10-year median snowmelt runoff onset date for the Northern Hemisphere." can be found at `global/figures/global_composite_median_polar.png` and figure creation code is in `global/global_composites.ipynb`

Manuscript appendix figure "Figure A2. Global snowmelt runoff onset composite products with polar stereographic projection for the Northern Hemisphere." can be found at `global/figures/global_all_composites_polar.png` and figure creation code is in `global/global_composites.ipynb`

Manuscript appendix figure "Figure A3. Snowmelt runoff onset (day of water year) and temporal resolution (days) for each water year." can be found at `global/figures/global_annual_runoff_onset_and_temporal_res_with_hillshade_2015_2024.png` and figure creation code is in `global/global_annual_runoff_onset_and_temporal_res.ipynb`

Manuscript appendix figure "Figure A4. Per-pixel count of water years with valid annual snowmelt runoff onset estimates." can be found at `global/figures/global_10yr_annual_runoff_onset_count.png` and figure creation code is in `global/global_composites.ipynb`

## Antimeridian diagnostic (`testing/test_antimeridian.ipynb`)

Root-causes a data gap in the westernmost tile column (crossing 180°/-180°) to a bug in `odc.stac.load()`: it computes an invalid footprint for UTM source zones abutting the antimeridian, which causes affected Sentinel-1 scenes to be silently dropped from tile binning. The fix (`wrapdateline=True`) was merged upstream in [opendatacube/odc-stac#281](https://github.com/opendatacube/odc-stac/pull/281) on 2026-07-21 but isn't in a tagged release yet. In the meantime, `global_snowmelt_runoff_onset.processing.ensure_antimeridian_footprint_fix()` runs a functional self-check at the top of `get_sentinel1_rtc()` and applies the same fix as a monkeypatch only if needed — see the root README's remaining-tasks list for status and the still-outstanding tile reprocessing.

## Non-manuscript notebooks

`testing/inspect_tile.ipynb` and `regions/alps/alps.ipynb` / `regions/iceland/iceland.ipynb` are ad hoc dataset viewers/regional explorations used for QA, not tied to a specific manuscript figure. `regions/rainier/rainier_figure.ipynb` overlaps with the Rainier panels in `methods/` and needs a diff check (see root README).
