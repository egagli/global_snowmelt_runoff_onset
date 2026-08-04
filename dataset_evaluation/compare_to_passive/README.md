# Comparison to Passive microwave

This directory evaluates the Global Snowmelt Runoff Onset dataset against the [ORNL DAAC Main Melt Onset Dates (ABoVE, 1841)](https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-main-melt-onset-dates-1841-1.0#documents-and-resources), an independent passive microwave–derived melt onset product covering 1988–2023 at 6.25km resolution over Alaska and northwestern Canada.

## Notebooks

| Notebook | Description |
|---|---|
| [alaska_range_comparison.ipynb](alaska_range_comparison.ipynb) | Spatial comparison over the Alaska Range for WY2020 — **manuscript Fig. 6** (median difference 7.8 days, MAD 22.2 days). Produces a multi-panel figure showing passive microwave melt onset, our SAR-based runoff onset, a difference map, and a histogram of pixel-wise differences with median bias and MAD. Writes `figures/<version>/alaska_range_passive_comparison.png`. |
| [kennicott_glacier_comparison.ipynb](kennicott_glacier_comparison.ipynb) | Zoomed case studies at **Kennicott Glacier and Denali**, with Sentinel-2 RGB context imagery (via `easysnowdata`). Writes `figures/<version>/passive_comparison/alaska_comparison_multipanel_v2.png`. |

## Data

`data/Main_Melt_Onset_Dates_1841/` — local copy of the ABoVE passive microwave melt onset GeoTIFFs, one file per year. Download from the ORNL DAAC link above; files are not tracked in git.

`geometries/` — GeoJSON boundary files used for clipping: `kennicott.geojson` and `denali.geojson`.

Other runtime inputs fetched over the network: the GMBA Mountain Inventory v2.0 standard polygons (earthenv.org), a US states GeoJSON (eric.clst.org), and the repo's `visualize/data/global_hillshade_robinson.tif` basemap.

## Outputs

Version-scoped under `figures/<version>/`:

- `figures/v9/alaska_range_passive_comparison.png` — Fig. 6 (tracked).
- `figures/v9/passive_comparison/` — **currently empty**: the Kennicott/Denali multipanel hasn't been regenerated since its output path was moved under `FIGURE_DIR`; rerun `kennicott_glacier_comparison.ipynb` to produce it.

> **Version note:** both notebooks still load `Config('config/global_config_v9.txt')` — Fig. 6 is a **v9** product, while the snow-pillow evaluation has moved to v10. Bump the config path and rerun for v10.

## Key notes

- The passive microwave product reports melt onset as **calendar day of year (DOY)**; our product uses **day of water year (DOWY)**. An offset of 92 days is applied when computing differences (DOY -> DOWY conversion for the October 1 water-year start).
- Spatial resolution differs substantially: passive microwave 6.25 km vs. our SAR product 80 m. For differencing, our product is coarsened 4× (`.coarsen(...).mean()`, → ~320 m) and the 6.25 km passive product is `reproject_match`-ed onto **that** grid.
- The passive microwave product covers only the ABoVE domain (Alaska and northwestern Canada), so spatial validation is limited to this region.
