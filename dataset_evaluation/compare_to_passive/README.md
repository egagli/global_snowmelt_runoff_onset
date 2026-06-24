# Comparison to Passive microwave

This directory evaluates the Global Snowmelt Runoff Onset dataset against the [ORNL DAAC Main Melt Onset Dates (ABoVE, 1841)](https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-main-melt-onset-dates-1841-1.0#documents-and-resources), an independent passive microwave–derived melt onset product covering 1988–2023 at 6.25km resolution over Alaska and northwestern Canada.

## Notebooks

| Notebook | Description |
|---|---|
| [alaska_range_comparison.ipynb](alaska_range_comparison.ipynb) | Spatial comparison over the Alaska Range for 2020. Produces a multi-panel figure showing passive microwave melt onset, our SAR-based runoff onset, a difference map, and a histogram of pixel-wise differences with median bias and MAD. |
| [kennicott_glacier_comparison.ipynb](kennicott_glacier_comparison.ipynb) | Focused comparison at Kennicott Glacier — a smaller-scale case study examining how the two products agree. |

## Data

`passive_data/Main_Melt_Onset_Dates_1841/` — local copy of the ABoVE passive microwave melt onset GeoTIFFs, one file per year. Download from the ORNL DAAC link above; files are not tracked in git.

`geometries/` — GeoJSON boundary files used for clipping (e.g., Kennicott Glacier outline).

## Key notes

- The passive microwave product reports melt onset as **calendar day of year (DOY)**; our product uses **day of water year (DOWY)**. An offset of 92 days is applied when computing differences (DOY -> DOWY conversion for the October 1 water-year start).
- Spatial resolution differs substantially: passive microwave 6.25 km vs. our SAR product 80 m. Comparisons use spatially aggregated versions of our product.
- The passive microwave product covers only the ABoVE domain (Alaska and northwestern Canada), so spatial validation is limited to this region.
