# visualize

This directory stores visualization notebooks and outputs — primarily the global composite figures used in the manuscript.

## Directory structure

```text
visualize/
├── colorbars/
│   └── create_colorbars.ipynb          # demos colorbars from global_snowmelt_runoff_onset.plot_utils
├── data/
│   ├── download_and_preprocess_hillshade.ipynb
│   └── global_hillshade_robinson.tif   # global hillshade used as basemap in Robinson-projection figures
├── global/
│   ├── global_composites.ipynb                       # Fig. 2, Fig. 3, Fig. A2, Fig. A4
│   ├── global_annual_runoff_onset_and_temporal_res.ipynb  # Fig. A3
│   ├── test_antimeridian.ipynb                        # diagnostic notebook for the odc-stac antimeridian bug (see below)
│   ├── view_maps.ipynb, view_global_maps.ipynb         # ad hoc dataset viewers, not tied to a manuscript figure
│   └── figures/                                       # rendered PNGs for all global figures below
└── regions/
    ├── alps/alps.ipynb                # exploratory regional viewer, not a manuscript figure
    └── iceland/iceland.ipynb          # exploratory regional viewer, not a manuscript figure
```

Colorbar code comes from `global_snowmelt_runoff_onset.plot_utils`. Colorbar demo in `colorbars/create_colorbars.ipynb`.

Global hillshade in Robinson projection for visualization is in `data/global_hillshade_robinson.tif`. Source: [Natural Earth](https://www.naturalearthdata.com/downloads/10m-gray-earth/gray-earth-with-shaded-relief-ocean-bottom-and-drainages/).

Notebook to download hillshade is `data/download_and_preprocess_hillshade.ipynb`.

## Manuscript figures

Manuscript figure "Figure 2. Global snowmelt runoff onset composite products." can be found at `global/figures/global_all_composites_robinson_wide.png` and figure creation code is in `global/global_composites.ipynb`

Manuscript figure "Figure 3. Polar stereographic projection of 10-year median snowmelt runoff onset date for the Northern Hemisphere." can be found at `global/figures/global_composite_median_polar.png` and figure creation code is in `global/global_composites.ipynb`

Manuscript appendix figure "Figure A2. Global snowmelt runoff onset composite products with polar stereographic projection for the Northern Hemisphere." can be found at `global/figures/global_all_composites_polar.png` and figure creation code is in `global/global_composites.ipynb`

Manuscript appendix figure "Figure A3. Snowmelt runoff onset (day of water year) and temporal resolution (days) for each water year." can be found at `global/figures/global_annual_runoff_onset_and_temporal_res_with_hillshade_2015_2024.png` and figure creation code is in `global/global_annual_runoff_onset_and_temporal_res.ipynb`

Manuscript appendix figure "Figure A4. Per-pixel count of water years with valid annual snowmelt runoff onset estimates." can be found at `global/figures/global_10yr_annual_runoff_onset_count.png` and figure creation code is in `global/global_composites.ipynb`

## Antimeridian diagnostic (`global/test_antimeridian.ipynb`)

Root-causes a data gap in the westernmost tile column (crossing 180°/-180°) to a bug in `odc.stac.load()`: it computes an invalid footprint for UTM source zones abutting the antimeridian, which causes affected Sentinel-1 scenes to be silently dropped from tile binning. A one-line `wrapdateline=True` fix is proposed upstream in [opendatacube/odc-stac#281](https://github.com/opendatacube/odc-stac/pull/281) (open, not yet merged/released). Until that lands, the fix needs to be applied as a workaround in `global_snowmelt_runoff_onset.processing.get_sentinel1_rtc()` and the affected tile column reprocessed.

## Non-manuscript notebooks

`global/view_maps.ipynb`, `global/view_global_maps.ipynb`, and `regions/alps/alps.ipynb` / `regions/iceland/iceland.ipynb` are ad hoc dataset viewers/regional explorations used for QA, not tied to a specific manuscript figure.
