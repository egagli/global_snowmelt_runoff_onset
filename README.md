# Global snowmelt runoff onset from Sentinel-1 SAR, 2015-2024

[![DOI](https://zenodo.org/badge/873255484.svg)](https://doi.org/10.5281/zenodo.19115464)


Eric Gagliano (egagli@uw.edu)

---

## Overview

Global 80-meter resolution dataset of snowmelt runoff onset timing from Sentinel-1 SAR and MODIS snow phenology, spanning water years 2015-2024. Validated against 900+ weather stations with median absolute deviation of 10 days.

## Key features

- **80m global coverage** from 81.1°N to 60°S
- **10-year record** (2015-2024) with annual and composite products  
- **Validated performance** (0 days median bias, 10 days typical uncertainty)
- **Usage guidelines** for optimal performance by environment
- **Cloud-optimized** Zarr format for efficient access

## Methodology

SAR backscatter minima from Sentinel-1 indicate snowmelt runoff onset when liquid water content peaks during the transition from snow ripening to active runoff. We combine multi-orbit Sentinel-1 data with MODIS snow phenology to constrain detection timing and location.

**Key steps:**
1. Create MODIS-derived snow phenology dataset (≥56 days continuous snow)
2. Quality filter Sentinel-1 VV backscatter and relative orbits  
3. Detect backscatter minima within latter half of snow-covered periods
4. Aggregate across orbits using median statistics
5. Generate annual maps and 10-year composites

## Performance

**Optimal conditions** (forest cover <0.5, SWE >25cm, temporal resolution <2 weeks):
- Uncertainty: ~1 week
- Bias: minimal

**Avoid** (dense forest + low SWE + poor temporal resolution):
- Uncertainty: >1 month  
- Systematic early bias

**Limitations:** Unreliable in dense forests (>50% cover), low snow areas (<25cm SWE), and sublimation-dominated regions (>5000m elevation).

## Data products

| Variable | Description | Dimensions | Units |
|----------|-------------|------------|-------|
| `runoff_onset` | Annual runoff onset timing | (water_year, lat, lon) | Day of water year |
| `runoff_onset_median` | 10-year median timing | (lat, lon) | Day of water year |
| `runoff_onset_mad` | 10-year variability | (lat, lon) | Days |
| `temporal_resolution` | Annual sampling frequency | (water_year, lat, lon) | Days |
| `temporal_resolution_median` | 10-year median sampling frequency | (lat, lon) | Days |

**Format:** Cloud-optimized Zarr, 80m resolution, WGS84, -9999 no-data values

## Data access

- **Published dataset:** [Zenodo DOI - to be added]
- **Snow phenology:** [https://zenodo.org/records/15692530](https://zenodo.org/records/15692530)  
- **Source code:** [https://github.com/egagli/global_snowmelt_runoff_onset](https://github.com/egagli/global_snowmelt_runoff_onset)

## Repository structure

This repository is the single home for everything behind the manuscript — dataset creation, dataset evaluation, and figure/table generation. Every folder has its own `README.md` with more detail; this table maps manuscript content to where it lives.

```text
global_snowmelt_runoff_onset/
├── global_snowmelt_runoff_onset/   # core Python package (config, processing, plotting)
├── processing/                     # tile-based dataset creation pipeline + GH Actions scripts
├── dataset/                        # references to the published Zarr store
├── dataset_utils/                  # utilities for accessing/subsetting/exporting the published dataset
├── dataset_evaluation/             # evaluation against snow pillows, NorSWE, passive microwave, etc.
├── visualize/                      # global composite figures (Fig. 2, 3, A2, A3, A4) + methods figure (Fig. 1)
├── config/                         # versioned processing configuration files
└── .github/workflows/              # GitHub Actions tile-processing pipeline
```

| Manuscript item | Where it lives |
| --- | --- |
| Fig. 1 (workflow diagram panels) | [`visualize/methods/`](visualize/methods/) (`create_methods_figure_components.ipynb`, `combine_methods_figure_components.ipynb`) |
| Fig. 2, 3, A2, A3, A4 (global composites) | [`visualize/`](visualize/README.md) |
| Fig. 4, 5 (station residual binning + evaluation) | [`dataset_evaluation/compare_to_all_public_snow_pillows/`](dataset_evaluation/compare_to_all_public_snow_pillows/README.md) |
| Fig. 6 (passive microwave comparison) | [`dataset_evaluation/compare_to_passive/`](dataset_evaluation/compare_to_passive/README.md) |
| Fig. A1 (single-station case study) | [`dataset_evaluation/compare_to_snotel/`](dataset_evaluation/compare_to_snotel/README.md) |
| Fig. A5 (network representativeness) | [`dataset_evaluation/compare_to_all_public_snow_pillows/`](dataset_evaluation/compare_to_all_public_snow_pillows/README.md) |
| Table 1 (spatial coverage / temporal resolution) | [`dataset_evaluation/calculate_spatial_coverage_and_temporal_resolution/`](dataset_evaluation/calculate_spatial_coverage_and_temporal_resolution/README.md) |
| Dataset creation methodology (Sect. 2.2) | [`processing/`](processing/README.md), [`global_snowmelt_runoff_onset/`](global_snowmelt_runoff_onset/README.md) |

Broader science analyses that use this dataset but aren't dataset construction/evaluation (regional case studies, climate correlation, population/basin-scale work) live in the separate [`global_snowmelt_runoff_onset_analysis`](https://github.com/egagli/global_snowmelt_runoff_onset_analysis) repository (split out of this repo's former `analysis/` folder in July 2026; it editable-installs this repo's package from a sibling clone).

## Quick start

```python
import xarray as xr

# Load global dataset 
ds = xr.open_zarr("path/to/dataset.zarr")

# Access 2020 runoff onset
runoff_2020 = ds.runoff_onset.sel(water_year=2020)

# 10-year median patterns
median_onset = ds.runoff_onset_median

# Regional subset (Western US)
western_us = ds.rio.clip_box(-125, 32, -105, 50)
```

## Installation

This repository uses [pixi](https://pixi.sh) for environment management — no conda/mamba required.

```bash
git clone https://github.com/egagli/global_snowmelt_runoff_onset.git
cd global_snowmelt_runoff_onset
pixi install
```

`pixi.toml` defines two environments:

- **`default`** — full development environment (JupyterLab, plotting, notebooks, Coiled)
- **`ci`** — minimal environment used by the GitHub Actions tile-processing workflows

```bash
pixi run lab           # launch JupyterLab (default environment)
pixi shell -e ci        # drop into the minimal CI environment used in GitHub Actions
```

Configure Azure credentials (needed to read/write the Zarr store):

```bash
export AZURE_STORAGE_SAS_TOKEN="your_token"
export AZURE_STORAGE_ACCOUNT="your_account"
```

## Applications

TBD

## Citation

**TBD**

Snow phenology dataset: <https://doi.org/10.5281/zenodo.15692530>

## Contact

Eric Gagliano ([egagli@uw.edu](mailto:egagli@uw.edu))  
University of Washington  
[GitHub Issues](https://github.com/egagli/global_snowmelt_runoff_onset/issues) for bug reports

## Related projects

- [easysnowdata](https://github.com/egagli/easysnowdata): Snow data access tools
- [sar_snowmelt_timing](https://github.com/egagli/sar_snowmelt_timing): Regional SAR methods
- [MODIS_seasonal_snow_mask](https://github.com/egagli/MODIS_seasonal_snow_mask): Snow phenology processing used to build this dataset's published version (configs ≤ v9)
- [MODIS_snow_phenology](https://github.com/egagli/MODIS_snow_phenology): Icechunk/Zarr v3 successor to `MODIS_seasonal_snow_mask`, the snow phenology input from config v10 onward
- [global_snowmelt_runoff_onset_analysis](https://github.com/egagli/global_snowmelt_runoff_onset_analysis): Broader scientific analyses built on this dataset (regional case studies, climate correlation, basin/population-scale work) that go beyond dataset construction and evaluation

## Remaining tasks

Working checklist for finishing the repository reorganization. Completed so far: manuscript-mapped READMEs in every folder, pixi-only environments (conda `environment*.yml` files removed, GitHub Actions already pixi-based), removal of the precomputed snow-mask reprojection path, and migration of the snow phenology input to the `MODIS_snow_phenology` icechunk store (config v10, validated against the legacy store — identical structure, 0-day median timing difference, 99.6% seasonal-snow mask agreement).

### 1. Move broader-science notebooks out of `analysis/`

**Done (July 2026).** The entire `analysis/` folder moved to [`global_snowmelt_runoff_onset_analysis`](https://github.com/egagli/global_snowmelt_runoff_onset_analysis): the parquet aggregation pipeline, continent/basin/regional science, and mountain-range climate notebooks (with `era5_data/`, `aggregated_results/`, `geometries/`, `csvs/`, `figures/`), plus the one-off/superseded notebooks under its `archive/`. The package's `analysis.py` moved there as the `gsro_analysis` package. Legacy figure PNGs produced by `visualize/` code were kept here under `visualize/global/figures/legacy/` and `visualize/methods/figures/` (see the READMEs there for provenance).

- [ ] `rainier_figure.ipynb` — first diff against the methods-figure notebooks in `visualize/methods/` (same Rainier bbox/imagery); move or delete depending on overlap

### 2. Identify superseded/scratch content. Do not delete yourself.

- [ ] Empty stubs: `processing/download_and_compress_zarr.ipynb`, `processing/scripts/consolidate_artifacts_clean.py`, `global_snowmelt_runoff_onset/global_snowmelt_runoff_onset.py` (0-byte file; also remove its import from `__init__.py`)
- [ ] `dataset_evaluation/compare_to_snotel/inspect_single_station copy.ipynb` (byte-identical duplicate)
- [ ] Decide: keep or archive `dataset_evaluation/compare_to_NorSWE/` (superseded by `compare_to_all_public_snow_pillows/`), `compare_to_NH-SWE/` and `compare_to_ucla_reanalysis/` (both parked/unfinished)
- [ ] Decide: track or delete `processing/test_chunking.ipynb` (chunking-tuning scratch, currently untracked)

### 3. Rebuild processing pipeline on icechunk (per-tile-per-water-year, platform-agnostic)

The snow phenology *input* is migrated; the runoff onset *output* store is still the pre-allocated plain Zarr v2 with CSV-based status tracking. Following the `MODIS_snow_phenology` pattern:

- [ ] Decide store strategy: keep the published v9 Zarr v2 store as-is (it has a DOI) and build v10 as a new icechunk/Zarr v3 store
- [ ] Convert the output store to icechunk: writable session per job, region write, `session.commit()` with `icechunk.ConflictDetector()` + randomized retry/backoff for concurrent disjoint-tile writes
- [ ] Granularity: one water year per tile per commit (finer than `MODIS_snow_phenology`'s all-years-per-tile commits), so a single bad year can be reprocessed without redoing all 10
- [ ] Encode per-job stats in commit messages and derive processing status from `repo.ancestry()` at read time (replacing `tile_results_*.csv` + `consolidate_tile_results.yml` artifact consolidation)
- [ ] Embed compute-platform provenance in each commit (new work — `MODIS_snow_phenology` doesn't do this either): e.g. GitHub Actions runner info from `GITHUB_*` env vars, hostname/platform otherwise, so any tile × water year can be traced to where and how it was processed
- [ ] Converge on `processing/scripts/process_single_tile.py` as the single entrypoint runnable locally, on GitHub Actions, and on CryoCloud; retire the Coiled notebooks (`process_tiles.ipynb`, `process_tiles_serverless.ipynb`) and remove `coiled` from dependencies once nothing else needs it
- [ ] Update the GitHub Actions workflows to match (drop the artifact-CSV steps; mirror `MODIS_snow_phenology`'s `process_single_tile.yml` → `process_batch.yml` → `process_all_tiles.yml` chain)
- [ ] If v10 is reprocessed: the evaluation notebooks that open `config.snow_phenology_store` with the legacy `xr.open_zarr(..., consolidated=True)` call need the icechunk-style open (`zarr_format=3, consolidated=False`)

### 4. Antimeridian fix (odc-stac)

`odc.stac.load()` silently drops Sentinel-1 scenes from UTM zones touching ±180°, leaving the westernmost tile column 100% nodata in all water years despite `success=True` (diagnosed in [visualize/global/test_antimeridian.ipynb](visualize/global/test_antimeridian.ipynb)).

- [ ] Until upstream [opendatacube/odc-stac#281](https://github.com/opendatacube/odc-stac/pull/281) is merged and released: apply the `wrapdateline=True` monkeypatch inside `get_sentinel1_rtc()` in `global_snowmelt_runoff_onset/processing.py` **LOOKS LIKE opendatacube/odc-stac#281 IS MERGED, WHAT DOES THIS MEAN FOR INSTALL? NEW VERSION? DEV VERSION?**

- [ ] Reprocess the affected tiles (col 0, partially cols 1–2)
- [ ] Once the fix ships in an odc-stac release: bump the `odc-stac` pin in `pixi.toml` and remove the monkeypatch
- [ ] Remove `.agents/` from the repo (handoff notes/patches from the antimeridian investigation — cross-project scratch, not manuscript code)

### 5. Junk and `.gitignore` cleanup

- [ ] Delete stray files: `=`, `=23.2.0` (shell-redirect accidents at repo root)
- [ ] Add `.gitignore` entries before any broad `git add`: `dataset_evaluation/**/data/`, `dataset_evaluation/**/figures/`, `visualize/**/figures/` (except the tracked legacy READMEs), `*.ovr.tmp`, `.vscode/` (the ~22 GB `analysis/` data moved out with the July 2026 split; its stray `.ovr.tmp` and 5 GB `analysis.log` were deleted then)
- [ ] Reclaim disk: `processing/analysis.log` 557 MB (already gitignored, but on disk)
- [ ] Remove `src/easysnowdata/` (nested git repo inside gitignored `src/`, stray editable install)
- [ ] Decide fate of `dataset/redistribution/` (empty) and whether to track `dataset/global_snowmelt_runoff_onset.zarr.tar.refs.json`

### 6. README/metadata finishing touches

- [ ] Fill in the **Data access** Zenodo DOI (manuscript cites <https://doi.org/10.5281/zenodo.16953614>, v1.1.0) and replace the **Applications** and **Citation** TBD sections with manuscript-consistent text
- [ ] Add a `CITATION.cff` check — make sure it matches the final manuscript author list/DOI

### 7. Sync with remote

- [ ] Commit the completed work in logical chunks (READMEs + pixi consolidation; precompute removal + phenology migration; then each task above as it lands)
- [ ] Triage remaining untracked directories (`dataset_evaluation/compare_to_snotel/`, `compare_to_NH-SWE/`, etc.): track code/READMEs, gitignore data/figures
- [ ] Push and verify GitHub Actions still pass with the pixi `ci` environment (now includes `icechunk`)
