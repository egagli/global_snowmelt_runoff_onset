# Global snowmelt runoff onset from Sentinel-1 SAR

[![Paper](https://img.shields.io/badge/paper-10.5194%2Fessd--18--5871--2026-1a7f5a)](https://doi.org/10.5194/essd-18-5871-2026)  
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16953614.svg)](https://doi.org/10.5281/zenodo.16953614)  
[![Repository DOI](https://zenodo.org/badge/873255484.svg)](https://doi.org/10.5281/zenodo.19115464)  


Eric Gagliano (egagli@uw.edu)

---

<p align="center">
  <a href="visualize/global/figures/v9/global_composite_median_polar.png"><img src="visualize/global/figures/v9/global_composite_median_polar.png" alt="10-year median snowmelt runoff onset, Northern Hemisphere polar stereographic view" width="500"></a>
</p>

## Overview

Global 80-meter resolution dataset of snowmelt runoff onset timing from Sentinel-1 SAR and MODIS snow phenology, spanning water years 2015-2024. Evaluated against snow pillows at 1,116 automated weather stations (Western U.S. including Alaska, British Columbia, Norway, Nepal): median timing difference −2.0 days, median absolute deviation 10.0 days.

The dataset is described in the accompanying paper, published in *Earth System Science Data*:

> Gagliano, E., Shean, D., and Henderson, S.: A global high-resolution dataset of snowmelt runoff onset timing from Sentinel-1 SAR, 2015–2024, Earth Syst. Sci. Data, 18, 5871–5894, <https://doi.org/10.5194/essd-18-5871-2026>, 2026.

## Details

- **Coverage:** 80 m resolution, global from 81.1°N to 60°S (data extent; the v10 processing grid spans 84.05°N–63.41°S, reserving room for the high Arctic and the Antarctic Peninsula)
- **Record:** water years 2015–2024, annual and multi-year composite products, 9.5-day average temporal resolution
- **Format:** cloud-optimized Zarr, WGS84 (EPSG:4326), int16 storage with scale/offset encoding, -9999 no-data values
- **Validated:** −2.0 days median bias, 10.0 days median absolute deviation against in-situ snow pillows

### Variables

| Variable | Description | Dimensions | Units |
|----------|-------------|------------|-------|
| `runoff_onset` | Annual runoff onset timing | (water_year, lat, lon) | Day of water year |
| `runoff_onset_median` | 10-year median timing | (lat, lon) | Day of water year |
| `runoff_onset_mad` | 10-year variability | (lat, lon) | Days |
| `temporal_resolution` | Annual sampling frequency | (water_year, lat, lon) | Days |
| `temporal_resolution_median` | 10-year median sampling frequency | (lat, lon) | Days |

### Methodology

We detect snowmelt runoff onset by identifying characteristic minima in Sentinel-1 C-band SAR backscatter time series. As liquid water content rises during snowmelt, absorption of the C-band signal drives backscatter to a minimum, which then recovers as snow surface roughness evolves during the runoff phase. The timing of this minimum is an empirically validated indicator of runoff onset — the transition from the ripening phase to the runoff phase — rather than a direct measurement of meltwater outflow. We combine all available Sentinel-1 relative orbits with a custom MODIS-derived snow phenology dataset that constrains where and when to search for the minimum.

**Key steps:**

1. Create MODIS-derived snow phenology dataset and identify seasonal snow (≥56 days continuous snow cover)
2. Quality filter Sentinel-1 VV backscatter (remove values < −30 dB) and exclude relative orbits with temporal gaps >30 days
3. Detect backscatter minima per relative orbit within a constrained temporal search window (midpoint of the snow-covered period through 16 days after MODIS snow disappearance)
4. Take the median date across relative orbits as the pixel's runoff onset estimate
5. Mosaic tiles into global annual products (water years 2015–2024) and generate 10-year composites (median runoff onset, median absolute deviation, temporal resolution)

### Performance and limitations

Evaluated against snow pillow runoff onset estimates at 1,116 automated weather stations (Western U.S., British Columbia, Norway, Nepal): median difference of −2.0 days, median absolute deviation of 10.0 days across 7,294 station water-years.

**Optimal conditions** (forest cover fraction <0.5, max SWE >~20 cm, temporal resolution <14 days):

- Near-zero systematic bias
- Spread approaching the temporal resolution of the underlying observations

**Avoid** (dense forest + low SWE + coarse temporal resolution combined):

- Systematic early bias, with bias and spread up to 30 days

**Limitations:** Detection degrades in dense forests (forest cover fraction >0.5), shallow snowpacks (max SWE <~20 cm), and with coarse temporal resolution (>14 days). Interpretation is uncertain in sublimation-dominated regions (e.g. >5000 m in the tropical Andes and parts of High Mountain Asia). Coverage excludes ice-free areas of Antarctica, Greenland, the Canadian Arctic Archipelago, and the Russian Arctic Islands, which lack VV-polarized Sentinel-1 IW acquisitions. The dataset also inherits known MODIS false-positive snow detections near turbid water bodies, over salt flats, and in regions with near-permanent cloud cover (e.g. eastern slopes of the tropical Andes).

## Data access and quick start

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

### Quick start

The example below mirrors the one on the [Zenodo dataset page](https://zenodo.org/records/19618062) — check that page for the more detailed usage guide (all three access patterns, chunk-count guidance, and additional usage notes). It opens the dataset lazily straight from Zenodo via the kerchunk reference file — recommended for regional analysis, since it fetches only the chunks your query touches.

#### 1. Lazy remote access via the reference file

```python
import fsspec
import xarray as xr
import rioxarray

REF_JSON_URL = "https://zenodo.org/records/19618062/files/global_snowmelt_runoff_onset.zarr.tar.refs.json"
mapper = fsspec.get_mapper("reference://", fo=REF_JSON_URL, remote_protocol="https")
global_ds = xr.open_zarr(mapper, consolidated=False, decode_coords="all")
```

#### 2. Clip to Mt. Rainier, WA

```python
rainier_ds = global_ds.rio.clip_box(minx=-122, miny=46.7, maxx=-121.5, maxy=47, crs="EPSG:4326").compute()
```

#### 3. Reproject to UTM Zone 10N for equal area visualization

```python
rainier_utm_ds = rainier_ds.rio.reproject("EPSG:32610")
```

#### 4. Plot the 10-year composite products

```python
import matplotlib.pyplot as plt

f, axs = plt.subplots(figsize=(12, 3), ncols=3, nrows=1)
rainier_utm_ds["runoff_onset_median"].plot.imshow(
    ax=axs[0], cmap='viridis', vmin=110, vmax=270,
    cbar_kwargs={'label': 'day of water year'},
)
rainier_utm_ds["runoff_onset_mad"].plot.imshow(
    ax=axs[1], cmap='Reds', vmin=0, vmax=60,
    cbar_kwargs={'label': 'days'},
)
rainier_utm_ds["temporal_resolution_median"].plot.imshow(
    ax=axs[2], cmap='summer', vmin=1, vmax=20,
    cbar_kwargs={'label': 'days'},
)

for ax in axs:
    ax.axis('off')
    ax.set_aspect('equal')

axs[0].set_title("10-year median snowmelt runoff onset")
axs[1].set_title("10-year median absolute deviation")
axs[2].set_title("10-year local median temporal resolution")
f.tight_layout()
```

#### 5. Plot the annual runoff onset and temporal resolution products

```python
rainier_utm_ds["runoff_onset"].plot.imshow(col='water_year', col_wrap=5, cmap='viridis', vmin=110, vmax=270, subplot_kws={'aspect': 'equal'})

rainier_utm_ds["temporal_resolution"].plot.imshow(col='water_year', col_wrap=5, cmap='summer', vmin=1, vmax=20, subplot_kws={'aspect': 'equal'})
```

> **Note:** Zenodo rate-limits requests per IP — queries that touch more than ~100 Zarr chunks may fail. Subset variables, water years, and extent before calling `.compute()`, or download the full/annual/composite archives for larger analyses. See the [Zenodo record](https://zenodo.org/records/19618062) description for all three access patterns and additional usage notes.

## Reproducing manuscript results

This repository is the single home for everything behind the manuscript — dataset creation, dataset evaluation, and figure/table generation. Every figure, table, and quoted number in the manuscript can be traced back to code here: [`docs/results_and_figures.md`](docs/results_and_figures.md) is the reproducibility map, recording for each manuscript result which notebook creates it, where every quoted number durably lives, and its current v10-vs-v9 status.

Broader science analyses that use this dataset but aren't dataset construction/evaluation (regional case studies, climate correlation, population/basin-scale work) live in the separate [`global_snowmelt_runoff_onset_analysis`](https://github.com/egagli/global_snowmelt_runoff_onset_analysis) repository.

## Applications

The manuscript (Sect. 5.5) surveys applications: retrospective snowmelt timing information for hydrological analysis and streamflow forecasting, climate trend and anomaly analysis, snow–wildfire interactions, effects of forest management on snowmelt timing, snowmelt–phenology relationships, data assimilation, and planning complementary SAR snow retrievals (e.g., NISAR L-band ΔSWE). Broader analyses built on this dataset live in [`global_snowmelt_runoff_onset_analysis`](https://github.com/egagli/global_snowmelt_runoff_onset_analysis).

## Repository structure

```text
global_snowmelt_runoff_onset/
├── global_snowmelt_runoff_onset/   # core Python package (config, processing, plotting)
├── processing/                     # tile-based dataset creation pipeline + GH Actions scripts
├── dataset/                        # references to the published Zarr store
├── dataset_utils/                  # utilities for accessing/subsetting/exporting the published dataset
├── dataset_evaluation/             # evaluation against snow pillows, NorSWE, passive microwave, etc.
├── visualize/                      # manuscript figures, multiscale pyramid, interactive web map
├── docs/                           # workflow pattern, results & figures map, maintenance runbook
├── config/                         # versioned processing configuration files
└── .github/workflows/              # GitHub Actions tile-processing pipeline
```

Every folder has its own `README.md` with more detail.

## Maintenance and processing workflow

The dataset is built by a fleet of free GitHub Actions runners writing to a transactional [Icechunk](https://icechunk.io) store on Azure. The global grid is cut into tile × water-year work units that each write one disjoint Zarr shard; every completed unit makes exactly one commit carrying machine-readable stats, and failed units commit nothing — so the store's commit history is the only progress ledger, and running the fleet just means re-dispatching the missing units until none remain. The v10 rebuild processed ~47,000 work units (~4,400 tiles × 11 water years, reading ~60 TB of Sentinel-1 imagery selected from a 2,400 TB catalog) this way — no cluster, no orchestrator, no database, no compute bill. The design, why it works, and its portable rules are written up in [`docs/icechunk-github-actions-pattern.md`](docs/icechunk-github-actions-pattern.md).

Day-to-day operations follow the numbered notebooks in [`processing/`](processing/README.md), in lifecycle order: select tiles (`0_`) → initialize the store (`1_`) → run the fleet (GitHub Actions "Process All Tiles", re-dispatched until nothing is missing) → monitor status (`2_`) → quality-check (`3_`) → finalize, tag, and garbage-collect a release (`4_`) → extend with new water years (`5_`). A monthly **Water Year Watch** workflow opens a GitHub issue when a new water year becomes processable; the extension recipe (phenology store first, append the year, bump the config, re-dispatch, rebuild the visualization pyramid and map) is a checklist, not code changes. The full runbook — credentials, dispatch recipes, failure triage, store maintenance, and known caveats — is [`docs/maintenance_and_processing_workflow.md`](docs/maintenance_and_processing_workflow.md).

## Citation

**Paper:**

> Gagliano, E., Shean, D., and Henderson, S.: A global high-resolution dataset of snowmelt runoff onset timing from Sentinel-1 SAR, 2015–2024, Earth Syst. Sci. Data, 18, 5871–5894, <https://doi.org/10.5194/essd-18-5871-2026>, 2026.

**Dataset:**

> Gagliano, E., Shean, D., and Henderson, S.: A global high-resolution dataset of snowmelt runoff onset timing from Sentinel-1 SAR, 2015–2024 (1.1.0), Zenodo [data set], <https://doi.org/10.5281/zenodo.16953614>, 2026.

**Software (this repository):**

> Gagliano, E.: Global snowmelt runoff onset from Sentinel-1 SAR, Zenodo [code], <https://doi.org/10.5281/zenodo.19115464>, 2026.

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
