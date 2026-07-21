# Spatial coverage and temporal resolution

Computes the dataset's spatial coverage, global seasonal snow extent, and average temporal resolution per water year and for the 10-year composites — manuscript **Table 1** and the "how much seasonal snow do we miss" text stat in Sect. 3.3.

## Notebooks

| Notebook | Description |
| --- | --- |
| [`calculate_spatial_coverage_and_temporal_resolution.ipynb`](calculate_spatial_coverage_and_temporal_resolution.ipynb) | Loads the MODIS-derived snow phenology dataset and the runoff onset product; computes spatial coverage (km²), global seasonal snow extent, percent coverage, and average temporal resolution for each water year 2015–2024 and the 10-year composites. This is Table 1. |
| [`how_much_seasonal_snow_do_we_miss.ipynb`](how_much_seasonal_snow_do_we_miss.ipynb) | Estimates the seasonal snow area *not* captured by the dataset because Sentinel-1 acquires Extra Wide (not Interferometric Wide) mode over Antarctica and much of the Arctic. Sums seasonal snow area (per the Sturm & Liston 2021 classification, to avoid double-counting ice sheets/glaciers) over the ice-free areas of Greenland, the Canadian Arctic Archipelago, and the Russian Arctic Islands. Feeds the Sect. 3.3 text stat (~1.6 million km² missed). |

## Data

- `data/SnowClass_GL_01km_30.0arcsec_2021_v01.0.tif` — Sturm & Liston (2021) global seasonal snow classification, used to identify genuinely snow-covered area within the excluded polar regions.
- `data/canada_poly.zip` — Canadian Arctic Archipelago boundary shapefile.
- `data/russian_arctic/` — Russian Arctic Islands boundary data.

## Results

`results/` contains the CSV outputs consumed by the manuscript table and text:
- `complete_spatial_coverage_and_temporal_res_per_water_year.csv` / `..._REVISED.csv`
- `modis_coverage_per_water_year.csv`
- `runoff_onset_coverage_and_temporal_res_per_water_year.csv`
- `seasonal_snow_excluded_area_summary.csv`
