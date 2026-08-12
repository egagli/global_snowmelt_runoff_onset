# Global snowmelt runoff onset from Sentinel-1 SAR

[![DOI](https://zenodo.org/badge/873255484.svg)](https://doi.org/10.5281/zenodo.19115464)


Eric Gagliano (egagli@uw.edu)

---

## Overview

Global 80-meter resolution dataset of snowmelt runoff onset timing from Sentinel-1 SAR and MODIS snow phenology, spanning water years 2015-2024. Evaluated against snow pillows at 1,116 automated weather stations (Western U.S. including Alaska, British Columbia, Norway, Nepal): median timing difference −2.0 days, median absolute deviation 10.0 days.

The dataset is described in the accompanying paper, accepted at *Earth System Science Data*:

> Gagliano, E., Shean, D., and Henderson, S.: A global high-resolution dataset of snowmelt runoff onset timing from Sentinel-1 SAR, 2015–2024, Earth Syst. Sci. Data, in press, <https://doi.org/10.5194/essd-2026-216>, 2026.

## Key features

- **80m global coverage** from 81.1°N to 60°S (data extent; the v10 processing grid spans 84.05°N–63.41°S, reserving room for the high Arctic and the Antarctic Peninsula)
- **10-year record** (2015-2024) with annual and composite products, 9.5-day average temporal resolution
- **Validated performance** (−2.0 days median bias, 10.0 days median absolute deviation)
- **Usage guidelines** for optimal performance by environment
- **Cloud-optimized** Zarr format for efficient access

## Methodology

SAR backscatter minima from Sentinel-1 indicate snowmelt runoff onset when liquid water content peaks during the transition from snow ripening to active runoff. We combine multi-orbit Sentinel-1 data with MODIS snow phenology to constrain detection timing and location.

**Key steps:**
1. Create MODIS-derived snow phenology dataset (≥56 days continuous snow)
2. Quality filter Sentinel-1 VV backscatter and relative orbits  
3. Detect backscatter minima within a constrained temporal search window (midpoint of the snow-covered period through 16 days after snow disappearance)
4. Aggregate across orbits using median statistics
5. Generate annual maps and 10-year composites

## Performance

**Optimal conditions** (forest cover fraction <0.5, max SWE >~20cm, temporal resolution <14 days):
- Near-zero systematic bias
- Spread approaching the temporal resolution of the underlying observations

**Avoid** (dense forest + low SWE + coarse temporal resolution combined):
- Bias and spread up to 30 days
- Systematic early bias

**Limitations:** Unreliable in dense forests (forest cover fraction >0.5), low snow areas (max SWE <~20cm), and sublimation-dominated regions (e.g. >5000m elevation in the tropical Andes and parts of High Mountain Asia). Inherits known MODIS false-positive snow detections near turbid water bodies, over salt flats, and in regions with near-permanent cloud cover (e.g. eastern slopes of the tropical Andes).

## Data products

| Variable | Description | Dimensions | Units |
|----------|-------------|------------|-------|
| `runoff_onset` | Annual runoff onset timing | (water_year, lat, lon) | Day of water year |
| `runoff_onset_median` | 10-year median timing | (lat, lon) | Day of water year |
| `runoff_onset_mad` | 10-year variability | (lat, lon) | Days |
| `temporal_resolution` | Annual sampling frequency | (water_year, lat, lon) | Days |
| `temporal_resolution_median` | 10-year median sampling frequency | (lat, lon) | Days |

**Format:** Cloud-optimized Zarr, 80m resolution, WGS84, int16 storage with scale/offset encoding, -9999 no-data values

## Data access

- **Published dataset:** [10.5281/zenodo.16953614](https://doi.org/10.5281/zenodo.16953614) (concept DOI — always resolves to the latest version; the v1.1.0 record is [19618062](https://zenodo.org/records/19618062))
- The Zenodo record provides the full store (~61 GB `.zarr.tar`), per-water-year stores, the 10-year composite store, and a kerchunk reference file for lazy remote access (also git-tracked at [`dataset/global_snowmelt_runoff_onset.zarr.tar.refs.json`](dataset/global_snowmelt_runoff_onset.zarr.tar.refs.json))
- **Snow phenology dataset:** [10.5281/zenodo.21783366](https://doi.org/10.5281/zenodo.21783366)
- **Source code (this repository):** archived at [10.5281/zenodo.19115464](https://doi.org/10.5281/zenodo.19115464)

### Dataset versions

| Zenodo version | Internal config | Coverage | Status |
| --- | --- | --- | --- |
| [v1.1.0](https://zenodo.org/records/19618062) | v9 | WY2015–2024, 81.1°N–60°S | **Published** — the dataset described in the manuscript |
| next version (in production) | v10 | WY2015–2025, grid extended to 84.05°N–63.41°S | Icechunk/Zarr v3 rebuild in progress; will be published as a **new version** of the same Zenodo record (same concept DOI) |

See [`dataset/README.md`](dataset/README.md) for store locations and details.

## Repository structure

This repository is the single home for everything behind the manuscript — dataset creation, dataset evaluation, and figure/table generation. Every folder has its own `README.md` with more detail; [`docs/results_and_figures.md`](docs/results_and_figures.md) maps manuscript content to where it lives.

```text
global_snowmelt_runoff_onset/
├── global_snowmelt_runoff_onset/   # core Python package (config, processing, plotting)
├── processing/                     # tile-based dataset creation pipeline + GH Actions scripts
├── dataset/                        # references to the published Zarr store
├── dataset_utils/                  # utilities for accessing/subsetting/exporting the published dataset
├── dataset_evaluation/             # evaluation against snow pillows, NorSWE, passive microwave, etc.
├── visualize/                      # manuscript figures (Fig. 1, 2, 3, A2–A4), multiscale pyramid, interactive web map
├── docs/                           # workflow pattern, results & figures map, maintenance runbook
├── config/                         # versioned processing configuration files
└── .github/workflows/              # GitHub Actions tile-processing pipeline
```

### Reproducing manuscript results

Every figure, table, and quoted number in the manuscript can be traced back to code in
this repository. [`docs/results_and_figures.md`](docs/results_and_figures.md) is the
reproducibility map: for each manuscript result it records which notebook creates it,
where every quoted number durably lives, and its current v10-vs-v9 status (including
updated versions as the v10 rebuild lands).

Broader science analyses that use this dataset but aren't dataset construction/evaluation (regional case studies, climate correlation, population/basin-scale work) live in the separate [`global_snowmelt_runoff_onset_analysis`](https://github.com/egagli/global_snowmelt_runoff_onset_analysis) repository.

## Quick start

Lazy remote access straight from Zenodo via the kerchunk reference file — recommended for regional analysis, since it fetches only the chunks your query touches:

```python
import fsspec
import xarray as xr
import rioxarray

REF_JSON_URL = "https://zenodo.org/records/19618062/files/global_snowmelt_runoff_onset.zarr.tar.refs.json"
mapper = fsspec.get_mapper("reference://", fo=REF_JSON_URL, remote_protocol="https")
ds = xr.open_zarr(mapper, consolidated=False, decode_coords="all")

# Clip to a region of interest (Mt. Rainier, WA), then pull the data
rainier = ds.rio.clip_box(minx=-122, miny=46.7, maxx=-121.5, maxy=47, crs="EPSG:4326").compute()

# Annual runoff onset for water year 2020, and the 10-year median
runoff_2020 = rainier.runoff_onset.sel(water_year=2020)
median_onset = rainier.runoff_onset_median
```

> **Note:** Zenodo rate-limits requests per IP — queries that touch more than ~100 Zarr chunks may fail. Subset variables, water years, and extent before calling `.compute()`, or download the full/annual/composite archives for larger analyses. See the [Zenodo record](https://zenodo.org/records/19618062) description for all three access patterns and additional usage notes.

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

The manuscript (Sect. 5.5) surveys applications: retrospective snowmelt timing information for hydrological analysis and streamflow forecasting, climate trend and anomaly analysis, snow–wildfire interactions, effects of forest management on snowmelt timing, snowmelt–phenology relationships, data assimilation, and planning complementary SAR snow retrievals (e.g., NISAR L-band ΔSWE). Broader analyses built on this dataset live in [`global_snowmelt_runoff_onset_analysis`](https://github.com/egagli/global_snowmelt_runoff_onset_analysis).

## Citation

**Paper** (accepted; final volume/pages/DOI pending — update when the typeset article is published):

> Gagliano, E., Shean, D., and Henderson, S.: A global high-resolution dataset of snowmelt runoff onset timing from Sentinel-1 SAR, 2015–2024, Earth Syst. Sci. Data, in press, <https://doi.org/10.5194/essd-2026-216>, 2026.

**Dataset:**

> Gagliano, E., Shean, D., and Henderson, S.: A global high-resolution dataset of snowmelt runoff onset timing from Sentinel-1 SAR, 2015–2024 (1.1.0), Zenodo [data set], <https://doi.org/10.5281/zenodo.16953614>, 2026.

**Software (this repository):**

> Gagliano, E.: Global snowmelt runoff onset from Sentinel-1 SAR, 2015–2024, Zenodo [code], <https://doi.org/10.5281/zenodo.19115464>, 2026.

Snow phenology dataset: <https://doi.org/10.5281/zenodo.21783366>

## Contact

Eric Gagliano ([egagli@uw.edu](mailto:egagli@uw.edu))  
University of Washington  
[GitHub Issues](https://github.com/egagli/global_snowmelt_runoff_onset/issues) for bug reports

## Related projects

- [easysnowdata](https://github.com/egagli/easysnowdata): Snow data access tools
- [sar_snowmelt_timing](https://github.com/egagli/sar_snowmelt_timing): Regional SAR methods
- [MODIS_snow_phenology](https://github.com/egagli/MODIS_snow_phenology): Snow phenology software cited in the manuscript; generates the published snow phenology dataset ([10.5281/zenodo.21783366](https://doi.org/10.5281/zenodo.21783366)) 
- [MODIS_seasonal_snow_mask](https://github.com/egagli/MODIS_seasonal_snow_mask): Predecessor snow phenology processing  
- [global_snowmelt_runoff_onset_analysis](https://github.com/egagli/global_snowmelt_runoff_onset_analysis): Broader scientific analyses built on this dataset (regional case studies, climate correlation, basin/population-scale work) that go beyond dataset construction and evaluation
