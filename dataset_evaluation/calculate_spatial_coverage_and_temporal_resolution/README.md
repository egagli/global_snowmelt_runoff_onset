# Spatial coverage and temporal resolution

Computes the dataset's spatial coverage, global seasonal snow extent, and average temporal resolution per water year and for the 10-year composites — manuscript **Table 1** and the "how much seasonal snow do we miss" text stat in Sect. 3.3.

## Notebooks

| Notebook | Description |
| --- | --- |
| [`calculate_spatial_coverage_and_temporal_resolution.ipynb`](calculate_spatial_coverage_and_temporal_resolution.ipynb) | Loads the MODIS-derived snow phenology dataset and the runoff onset product; computes spatial coverage (km²), global seasonal snow extent, percent coverage, and average temporal resolution for each water year in the config's range and the composites. This is Table 1. **Currently pinned to `global_config_v9.txt`** (WY2015–2024, 10-year composites); a v10 rerun spans WY2015–2025 and yields 11 rows + an 11-year composite. |
| [`how_much_seasonal_snow_do_we_miss.ipynb`](how_much_seasonal_snow_do_we_miss.ipynb) | Estimates the seasonal snow area *not* captured by the dataset because Sentinel-1 acquires Extra Wide (not Interferometric Wide) mode over Antarctica and much of the Arctic. Sums seasonal snow area (per the Sturm & Liston 2021 classification, to avoid double-counting ice sheets/glaciers) over the ice-free areas of Greenland (363,919 km²), the Canadian Arctic Archipelago (1,235,852 km²), and the Russian Arctic Islands (9,080 km²), **plus an upper-bound 54,274 km² for Antarctica's total ice-free area** (Brooks et al. 2019 — used instead of Sturm & Liston there because of artifacting). Feeds the Sect. 3.3 text stat: **1,663,126 km² ≈ 1.6 million km² missed**. |

## Data

- `data/SnowClass_GL_01km_30.0arcsec_2021_v01.0.tif` — Sturm & Liston (2021) global seasonal snow classification, used to identify genuinely snow-covered area within the excluded polar regions.
- `data/canada_poly.zip` — Canadian Arctic Archipelago boundary shapefile.
- `data/russian_arctic/` — **unused leftover**: neither notebook reads it. The Russian Arctic geometry is derived at runtime from Natural Earth (`ne_10m_admin_0_countries.zip` → Russia, exploded, sorted by polygon area); safe to delete.
- Fetched at runtime (not on disk): the Natural Earth admin-0 countries ZIP (supplies both Greenland and Russia) and the MODIS snow phenology store via `config.snow_phenology_store`.
- `analysis.log` (93 MB, untracked/gitignored) — scratch log, safe to delete.

## Results

`results/<version>/` contains the CSV outputs consumed by the manuscript table and
text (scoped by the dataset version the notebook's config names, so a v10 run does
not overwrite the v9 tables):
- `complete_spatial_coverage_and_temporal_res_per_water_year.csv` / `..._REVISED.csv` — the `_REVISED` variant adds `total_seasonal_snow_extent_km2` (MODIS extent **plus** the excluded polar area) and `percent_coverage_of_total_seasonal_snow_extent`, and is the one matching the manuscript's Table 1 percentages
- `modis_coverage_per_water_year.csv`
- `runoff_onset_coverage_and_temporal_res_per_water_year.csv`
- `seasonal_snow_excluded_area_summary.csv`
