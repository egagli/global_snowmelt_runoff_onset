# Global snowmelt runoff onset from Sentinel-1 SAR, 2015-2024

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

```bash
# For development and analysis (includes all dependencies)
conda env create -f environment.yml
conda activate global_snowmelt_runoff_onset

# For GitHub Actions (minimal dependencies)
conda env create -f environment_github_actions.yml
conda activate global_snowmelt_runoff_onset_actions

# Configure Azure credentials
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
- [MODIS_seasonal_snow_mask](https://github.com/egagli/MODIS_seasonal_snow_mask): Snow phenology processing
