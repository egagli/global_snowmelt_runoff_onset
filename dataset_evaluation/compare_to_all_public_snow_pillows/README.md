# Comparison to all public snow pillows

Evaluates the global snowmelt runoff onset dataset against daily in-situ SWE
from all public snow station networks archived in
[egagli/global_snow_networks](https://github.com/egagli/global_snow_networks):
SNOTEL (SNTL), SNOTEL Lite (SNTLT), California CCSS/CDEC, BC Snow Survey
(BCSS), NVE Norway, SCAN, COOP, Yukon Snow Survey (YSS), Yukon ECCC (YKEC),
and AWDB's "MSNT" bucket — ~1,500 stations with daily observations, refreshed
daily upstream (1,542 stations in the written Zarr: SNTL 924, MSNT 178, BCSS
144, CCSS 143, SNTLT 56, NVE 31, SCAN 25, COOP 24, YSS 9, YKEC 8).

Notebooks `0_`/`1_` began as mirrors of `../compare_to_NorSWE/`'s `0_`/`1_`;
`2_`–`4_` have no NorSWE counterpart. This is now the primary in-situ
evaluation — the source of manuscript **Fig. 4, Fig. 5, Fig. A5** and the
headline evaluation stats (v10 run: median residual **−2.0 d**, MAD
**10.0 d**, 8,397 station-water-years across 1,135 stations).

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

1. **Downloads** the station inventory (`all_snow_stations.geojson`, ~34 MB —
   the full 7,007-station inventory; `daily_or_better` is a column filtered
   downstream) and the bulk CSV archive (`all_station_csvs.tar.xz`, ~28 MB)
   from GitHub.
2. **Extracts** and **combines** all ~1,500 per-station CSVs
   (`date, wteq_cm, snwd_cm`) into a single `(time, station_id)` xarray
   Dataset with station metadata (name, network, client, state, lat/lon,
   elevation) as coordinates.
3. **Writes** it to `data/snow_pillows/snow_pillows.zarr` (~27 MB), deleting
   the tarball and extracted CSVs afterwards.
4. **Cross-validates** against the easysnowdata SNOTEL archive
   (Paradise, WA — 679_WA_SNTL).

**Units:** SWE is converted cm → **mm** (kg m⁻²) to match the NorSWE `snw`
convention, so the downstream notebooks are unit-identical to the NorSWE
ones. Snow depth stays in cm.

### `1_create_snow_pillow_comparison_dataset.ipynb`

1. Adds water year / DOWY coordinates; restricts to the config's water years
   (`config.water_years` — v9: WY2015–2024, v10: WY2015–2025).
2. De-duplicates AWDB "MSNT" copies of native CCSS/BCSS stations (within
   200 m); drops stations without SWE.
3. QCs the daily SWE series (spike removal, data-density/gap checks per
   station-year, seasonal-snowpack check).
4. Computes per-water-year max SWE value and %-of-max SWE timings
   (100/99/95/90/50%) → `data/comparison_datasets/<version>/max_snow_pillow_swe_timing.zarr`.
5. Extracts a 25×25 pixel chip (80 m pixels, ±1 km, relative x_rel/y_rel
   coordinates) from the global SAR runoff onset Zarr around each usable
   station, plus auxiliary layers (fcf, dem, worldcover, snow_class)
   → `data/comparison_datasets/<version>/runoff_onset_snow_pillow_station_chips.zarr`.
   Resumable batched loop.
6. Persists the manuscript Sect. 2.3 QC counts →
   `results/<version>/qc_station_counts.csv` (the numbers previously only
   printed by the QC-summary cell).

### `2_compare_snow_pillows.ipynb`

Merges chips + timings, computes `SAR − station` timing differences, and
explores performance vs. fcf, temporal resolution, max SWE, radius,
worldcover, snow class, and network: 1D sweeps, 2D binned heatmaps, decision
tree / gradient boosting on |error|, station-year robustness checks, and
"best regime" report cards. Also writes the pixel-level diagnostics
(`figures/<version>/pixelwise_performance_analysis*.png`, split WUS / non-WUS) and
the snow-class breakdowns
(`figures/<version>/residuals_by_snow_class_all_vs_good_pixels.png`,
`figures/<version>/performance_vs_prevalence_by_snow_class.png`), and persists the
Fig. 4 binned grids (median/MAD/count per fcf × SWE bin × temporal-resolution
group, long format) → `results/<version>/pixelwise_binned_stats.csv`.

### `3_evaluate_snow_pillows.ipynb`

Applies the chosen filter regime (fcf ≤ 50, max SWE ≥ 200 mm, temporal
resolution ≤ 14 d, radius ≤ 1000 m, excluding built-up/water pixels),
aggregates to per-station-water-year medians, and produces the evaluation
figure (**manuscript Fig. 5**) — a per-year residual histogram strip plus an
NH polar bias map with equal-area regional zoom insets — rendered once, for
all stations combined, in `figures/<version>/` (the `_WUS` / `_nonWUS` split
variants were dropped 2026-08-10, as was the earlier split-violin variant;
only `figures/v9/` still has the latter).

v10 outputs: 8,407 station-water-years across 1,136 stations, median residual
−2.0 d, MAD 10.0 d, mean temporal resolution 7.19 d. (Manuscript currently
cites the v9/WY2015–2024 counts — 7,294 across 1,116 of 1,210 — so reconcile
when updating it.)

Persists the headline stats and filter regime →
`results/<version>/evaluation_summary.csv` (single combined row) and the Fig. 5
per-water-year annotation table → `results/<version>/evaluation_per_water_year.csv`.

### `4_snow_pillow_representativeness.ipynb`

Asks how representative the station sample is of global seasonal snow. Places
the stations on the Sturm & Liston (2021) snow classification map (NH polar
stereographic, with Western North America / Alaska / Norway / Nepal zoom
insets) and counts how many sampled 80 m pixels (within 1 km of a station)
fall in each snow class. Compares that sampled distribution against each
class's global area / % of seasonal-snow area from Sturm & Liston (2021,
Table 1) →
`figures/<version>/snow_pillow_representativeness_snow_class_map_and_pixel_counts.png`
(**manuscript Fig. A5**).

> Note: the v10 render was regenerated 2026-08-10 (the notebook's `VERSION`
> constant had sat at `v9` after `1_`–`3_` moved to v10; fixed 2026-08-04).
> The notebook has no markdown title cell; this README is its only description.

### `5_station_density.ipynb`

Computes the manuscript Sect. 5.1 text stat — *"1 station per ~1,280 km² in
mountain regions of the Western U.S."* **Station source: identical to `4_`** —
the usable evaluation stations in
`data/comparison_datasets/<version>/max_snow_pillow_swe_timing.zarr` (from
`1_`), clipped to the WUS box (−125…−66°E, 24…49°N). Keeps the GMBA v2.0
mountain ranges containing at least one station and computes the aggregate
station count, total range area (Albers equal-area), area per station, and
stations per 100 km² → `results/<version>/station_density.csv`. Standalone —
not part of the `0_`–`4_` chip pipeline. Moved from `../compare_to_snotel/` and
rewired to the shared inventory on 2026-08-04; the manuscript's current
~1,280 km² figure came from the old SNOTEL/CCSS-only set (926 stations), so
**rerun and reconcile the manuscript with the CSV value**.

---

## Data & outputs

- `data/snow_pillows/` — `all_snow_stations.geojson` (34 MB) + `snow_pillows.zarr` (27 MB), from `0_`.
- `data/coarse_snow_class_map/SnowClass_GL_05km_2.50arcmin_2021_v01.0.tif` — 5 km snow-class raster read by `4_` (distinct from the 1 km tif in `../calculate_spatial_coverage_and_temporal_resolution/data/`).
- `data/comparison_datasets/{v9,v10}/` — the two Zarrs per version from `1_`.
- `figures/{v9,v10}/` — version-scoped figure outputs (`figures/v10/` currently untracked in git).
- `results/<version>/` — durable homes of the manuscript-cited numbers, written via
  `global_snowmelt_runoff_onset.results.save_result_table` (which stamps `_version`,
  `_git_sha`, `_written_at` columns): `qc_station_counts.csv` (from `1_`),
  `pixelwise_binned_stats.csv` (from `2_`), `evaluation_summary.csv` +
  `evaluation_per_water_year.csv` (from `3_`), `station_density.csv` (from `5_`).
  Added 2026-08-04 — **rerun the notebooks to materialize them**.
