# Comparison to SNOTEL / CCSS

Evaluates the global SAR-derived snowmelt runoff onset dataset against ~969
[SNOTEL](https://www.nrcs.usda.gov/wps/portal/wcc/home/snowClimateMonitoring/snowpack/snowpackNationalMaps/)
and CCSS snow pillow stations across the western US, covering water years 2015–2024.
For each station the SAR-derived onset (minimum C-band VV backscatter within the
MODIS-constrained melt search window) is compared to the snow-pillow-derived onset
(last day SWE ≥ 95 % of the water-year maximum), sweeping over multiple buffer radii
and forest-cover-fraction thresholds to characterize sensitivity.

This analysis is now superseded by the compare_to_all_public_snow_pillows analysis.

> **Git status:** this entire directory is currently **untracked**.

---

## Directory structure

```text
compare_to_snotel/
├── create_all_station_comparison_dataset.ipynb
├── all_station_comparison_analysis.ipynb
├── environmental_conditions_analysis.ipynb
├── inspect_single_station.ipynb
├── supplemental_figure_methodology.ipynb
├── supplemental_figure_methodology_explain_pos_bias_at_low_fcf_high_SWE.ipynb
├── station_plots.py
├── analysis.log                # untracked/gitignored scratch log — safe to delete
├── comparison_datasets/        # NetCDF output files (snotel_sar_differences_*_<version>.nc)
└── figures/<version>/          # output PNG / PDF files, scoped by dataset version
```

There is **no local `config/` directory** — the notebooks' `Config('config/global_config_vN.txt')`
strings resolve against the **repo root** (`Config._resolve_repo_path()` in
`global_snowmelt_runoff_onset/config.py`), i.e. the shared [`../../config/`](../../config/)
(v2–v10 present).

---

## Notebooks

### `create_all_station_comparison_dataset.ipynb`

Builds the comparison datasets in `comparison_datasets/` using Coiled for distributed
computation over all ~969 stations.

1. Loads all SNOTEL/CCSS stations via `easysnowdata`; filters SWE time series
   (removes negative values, jump artifacts > 0.2 m, gaps > 10 days in any 20-day
   window).
2. Identifies snow years meeting a minimum seasonal snow requirement
   (≥ 0.05 m SWE for ≥ 55 days within any 60-day window).
3. Derives the snow-pillow runoff onset as the last day SWE ≥ 95 % of the
   water-year maximum.
4. For each station, sweeps over buffer radii (100–1000 m) and forest-cover-fraction
   thresholds (10–100 %), then extracts the SAR-derived runoff onset (minimum
   C-band VV backscatter within the MODIS-constrained search window).
5. Saves results as NetCDF files with dimensions `(station, WY, buffer_radius, fcf)`.

> **Note:** This notebook loads `config/global_config_v6.txt`, not the current v9/v10.
> Pinning to v6 may be intentional (frozen for reproducibility) or outdated. (Until
> 2026-08-04 the path was written `'../config/global_config_v6.txt'`, which — because
> `Config` resolves relative paths against the repo root — pointed *outside* the repo
> and raised `FileNotFoundError`; the cell had never been re-run since that change.)

---

### `all_station_comparison_analysis.ipynb`

Primary analysis notebook (~25 MB). Loads a precomputed comparison NetCDF and
analyses SAR–snow pillow agreement across all stations.

1. Sweeps bias / MAD metrics over buffer radius and FCF threshold dimensions to
   identify the optimal filtering parameters.
2. Computes per-station and per-water-year statistics
   (`sar_minus_stations`, `sar_minus_95pct`, etc.).
3. Produces scatter plots of agreement versus environmental covariates
   (max SWE, elevation, FCF).
4. Generates geographic comparison maps (dots colored by SAR − snow pillow onset
   difference in days).
5. Identifies best- and worst-agreement stations for further investigation.

---

### `environmental_conditions_analysis.ipynb`

Analyses how environmental factors affect SAR–snow pillow agreement (~15 MB).

1. Loads station metadata (updated locations from
   `~/repos/updated_snotel_locations/`) and the precomputed comparison dataset.
2. Fetches per-station environmental covariates from the SAR Zarr store:
   elevation, slope, aspect, forest cover fraction, and SWE magnitude.
3. Computes correlations between each covariate and the SAR bias / MAD.
4. Produces a multi-panel summary figure (no longer written to disk by the current
   notebook; the `figures/env_analysis.png` on disk is from earlier code).

> **Note:** The notebook's H1 title reads *"Create SAR-derived runoff onset / snow
> pillow-derived runoff onset comparison dataset"* — a copy-paste artifact from
> `create_all_station_comparison_dataset.ipynb`. The actual content is the
> environmental conditions analysis.

---

### `inspect_single_station.ipynb`

Interactive drill-down for a single station.

1. Select a station by code; spatially join to the valid SAR tile grid.
2. Plot the SWE time series with jump-detection filtering applied.
3. Fetch SAR backscatter time series per relative orbit from the Azure Zarr store.
4. Overlay MODIS snow appearance / disappearance dates and computed runoff onset
   timing for each water year.

---

### `supplemental_figure_methodology.ipynb`

Generates manuscript supplemental figures illustrating the detection methodology
at representative sites.

1. Selects one station with clear SAR–SWE agreement and one with an ambiguous signal.
2. Produces a multi-panel figure: filtered SWE time series with the 95 % threshold
   marker, per-orbit SAR backscatter with detected minimum, MODIS snow dates, and
   computed onset offsets for multiple water years.
3. Produces a satellite context map showing the station footprint, forest cover
   fraction, and terrain.
4. Saves figures to `figures/<version>/`.

---

### `supplemental_figure_methodology_explain_pos_bias_at_low_fcf_high_SWE.ipynb`

Targeted investigation of positive bias at low-forest-cover-fraction sites with
high SWE, focusing on three specific problem stations (1222_UT, 314_WY, 626_UT).
**This notebook — not `supplemental_figure_methodology.ipynb` — is what writes
`figures/v9/supplemental_figure_methodology.png`/`.pdf` (manuscript Fig. A1).**

1. Loads the precomputed comparison dataset and filters to the three stations of
   interest.
2. Produces per-station satellite context maps showing forest cover and terrain at
   1 km extent.
3. Produces multi-year time-series panels (SWE + SAR backscatter + onset markers)
   for high-SWE and moderate-SWE water years side-by-side.
4. Saves context and time-series figures to `figures/<version>/`
   (see [Path handling](#path-handling)).

> **Note:** This notebook shares the same opening markdown title as
> `supplemental_figure_methodology.ipynb` — the title is misleading. The actual
> content is the low-FCF positive-bias investigation.

---

### `station_density.ipynb` — moved

Moved to
[`../compare_to_all_public_snow_pillows/5_station_density.ipynb`](../compare_to_all_public_snow_pillows/5_station_density.ipynb)
on 2026-08-04, gaining a title cell and a persisted output
(`results/station_density.csv` — the manuscript Sect. 5.1 station-density stat).

---

## `station_plots.py`

Shared plotting utilities used across the analysis notebooks:

- `create_2d_hist_with_1d_marginals()` — hexbin or 2-D histogram with colormapped
  marginal distributions.
- `median_absolute_deviation()` — statistical helper.

---

## Path handling

Every figure **that is saved** goes through one `FIGURE_DIR = Path('figures') / VERSION`
constant, where `VERSION` is `config.version` — three notebooks define it
(`all_station_comparison_analysis`, `supplemental_figure_methodology`,
`..._explain_pos_bias...`); `environmental_conditions_analysis` saves nothing, and
`inspect_single_station` defines neither. That resolved the
long-standing issue where several `savefig` calls
wrote to the notebook's current working directory instead of `figures/` — fixed
July 2026 alongside the output-versioning change (see
[`../README.md`](../README.md#output-versioning)), so v9 and v10 figures no longer
collide either.

The comparison NetCDFs stay flat in `comparison_datasets/` rather than moving into
per-version subdirectories, because their filenames already carry the version
(`..._v6.nc`, `..._v9.nc`) and several vintages coexist. The version in those names now
comes from `config.version` too, via a `COMPARISON_NC` constant.

Two earlier path fixes, also July 2026, are still worth knowing about:

| Call | Note |
| --- | --- |
| `fig.savefig(FIGURE_DIR / 'station_evaluation.png', ...)` | Previously pointed at the nonexistent `../analysis/manuscript_figures/`. **The fix is unverified**: `figures/v9/` has the sibling `snowmelt_runoff_onset_comparison_map_circles.png` from the same cell but no `station_evaluation.png` — the cell hasn't been re-run since the path change |
| `rxr.open_rasterio('../../visualize/data/global_hillshade_robinson.tif', ...)` | Previously pointed at `../analysis/figures/methods/`, removed in the analysis-repo split |

Which of the ~24 NetCDFs in `comparison_datasets/` are actually live:
`..._andreq10cmSWEfor60days_v9.nc` (analysis / methodology notebooks),
`spatial_..._v9.nc` (`environmental_conditions_analysis`), and
`..._andreq5cmSWEfor60days_v6.nc` (`inspect_single_station`). The rest are v1–v8
vintages kept for provenance.

Two remaining wrinkles:

- `inspect_single_station.ipynb` loads the v9 config but deliberately reads the
  **v6-era** comparison dataset, so its input is not version-scoped off that config.
- `figures/` still holds six PNGs from earlier versions of these notebooks that no
  current `savefig` call reproduces (`env_analysis.png`,
  `supplemental_figure_context.png`, `supplemental_figure_meteorology.png`,
  `supplemental_figure_timeseries.png`, `high_swe_low_swe_comparison_846.png`,
  `supplemental_figure_context_low_fcf_1222_UT_SNTL_highwy_2016_2018_.png`). They were
  left at the top level rather than moved into `figures/v9/`, since their vintage is
  unknown.
