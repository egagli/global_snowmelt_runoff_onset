# Comparison to all public snow pillows

Evaluates the global snowmelt runoff onset dataset against daily in-situ SWE
from all public snow station networks archived in
[egagli/global_snow_networks](https://github.com/egagli/global_snow_networks):
SNOTEL (SNTL), SNOTEL Lite (SNTLT), California CCSS/CDEC, BC Snow Survey
(BCSS), NVE Norway, SCAN, COOP, and AWDB's "MSNT" bucket — ~1,500 stations
with daily observations, refreshed daily upstream.

Mirrors the structure of `../compare_to_NorSWE/`, but this is now the primary
in-situ evaluation.

## Why these networks instead of NorSWE

NorSWE has more stations, but two things limit it for evaluating the SAR
record. First, it ends in 2021, so it only overlaps the first ~6 water years
of the SAR archive. The public networks here are ongoing and refreshed daily,
so they cover the full WY 2015–2024 record. Second, a large share of NorSWE
sites are not daily — many are manual snow courses or coarser reports — which
makes it hard to pin down a %-of-max SWE timing to the day. Every station used
here reports daily SWE, so the timing targets are consistent across sites.

---

## Notebooks

### `0_download_and_preprocess_all_snow_pillow_data.ipynb`

1. **Downloads** the station inventory (`all_daily_snow_stations.geojson`) and
   the bulk CSV archive (`all_station_csvs.tar.xz`, ~28 MB) from GitHub.
2. **Extracts** and **combines** all ~1,500 per-station CSVs
   (`date, wteq_cm, snwd_cm`) into a single `(time, station_id)` xarray
   Dataset with station metadata (name, network, client, state, lat/lon,
   elevation) as coordinates.
3. **Writes** it to `data/snow_pillows/snow_pillows.zarr` (~70 MB), deleting
   the tarball and extracted CSVs afterwards.
4. **Cross-validates** against the easysnowdata SNOTEL archive
   (Paradise, WA — 679_WA_SNTL).

**Units:** SWE is converted cm → **mm** (kg m⁻²) to match the NorSWE `snw`
convention, so the downstream notebooks are unit-identical to the NorSWE
ones. Snow depth stays in cm.

### `1_create_snow_pillow_comparison_dataset.ipynb`

1. Adds water year / DOWY coordinates; restricts to WY 2015–2024.
2. De-duplicates AWDB "MSNT" copies of native CCSS/BCSS stations (within
   200 m); drops stations without SWE.
3. QCs the daily SWE series (spike removal, data-density/gap checks per
   station-year, seasonal-snowpack check).
4. Computes per-water-year max SWE value and %-of-max SWE timings
   (100/99/95/90/50%) → `data/comparison_datasets/max_snow_pillow_swe_timing.zarr`.
5. Extracts a 25×25 pixel chip (80 m pixels, ±1 km, relative x_rel/y_rel
   coordinates) from the global SAR runoff onset Zarr around each usable
   station, plus auxiliary layers (fcf, dem, worldcover, snow_class)
   → `data/comparison_datasets/runoff_onset_snow_pillow_station_chips.zarr`.
   Resumable batched loop.

### `2_compare_snow_pillows.ipynb`

Merges chips + timings, computes `SAR − station` timing differences, and
explores performance vs. fcf, temporal resolution, max SWE, radius,
worldcover, snow class, and network: 1D sweeps, 2D binned heatmaps, decision
tree / gradient boosting on |error|, station-year robustness checks, and
"best regime" report cards. Also writes the pixel-level diagnostics
(`figures/pixelwise_performance_analysis*.png`, split WUS / non-WUS) and the
snow-class breakdowns (`figures/residuals_by_snow_class_all_vs_good_pixels.png`,
`figures/performance_vs_prevalence_by_snow_class.png`).

### `3_evaluate_snow_pillows.ipynb`

Applies the chosen filter regime (fcf ≤ 50, max SWE ≥ 200 mm, temporal
resolution ≤ 14 d, excluding built-up/water pixels), aggregates to
per-station-water-year medians, and produces the evaluation figures
(per-year residual histograms, split violins, spatial bias maps — NH polar
plus regional zoom insets, split WUS / non-WUS) in `figures/`.

### `4_snow_pillow_representativeness.ipynb`

Asks how representative the station sample is of global seasonal snow. Places
the stations on the Sturm & Liston (2021) snow classification map (NH polar
stereographic, with Western North America / Alaska / Norway / Nepal zoom
insets) and counts how many sampled 80 m pixels (within 1 km of a station)
fall in each snow class. Compares that sampled distribution against each
class's global area / % of seasonal-snow area from Sturm & Liston (2021,
Table 1) → `figures/snow_pillow_representativeness_snow_class_map_and_pixel_counts.png`.
