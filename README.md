# Global Snowmelt Runoff Onset

Eric Gagliano (egagli@uw.edu)

Last updated: June 26th, 2025


A comprehensive codebase for estimating snowmelt runoff onset timing at global scale using Sentinel-1 SAR data and cloud computing infrastructure.

## Overview

This project implements a novel methodology for detecting the timing of snowmelt runoff onset from Sentinel-1 Synthetic Aperture Radar (SAR) data. The approach...

## Key Features


## Scientific Methodology

### Core Algorithm

The detection methodology is based on identifying minimum backscatter values in Sentinel-1 data, which correspond to peak snowmelt conditions:

*I'll fill this out eventually*

### Processing Pipeline

*graph probably*

### Key Processing Steps

1. **Data Acquisition**: Download Sentinel-1 RTC data from Microsoft Planetary Computer
2. **Spatiotemporal Masking**: Apply MODIS-derived seasonal snow masks to focus on snow-covered areas
3. **Quality Filtering**: Remove scenes with insufficient temporal sampling or data gaps
4. **Minimum Detection**: Identify minimum backscatter timing per satellite orbit and polarization
5. **Statistical Aggregation**: Compute robust statistics (median, MAD) across orbits and years
6. **Output Generation**: Write results to cloud-optimized Zarr store

## Data Products

### Output Variables

| Variable | Description | Units | Type |
|----------|-------------|-------|------|
| `runoff_onset` | Annual runoff onset timing | Day of Water Year (DOWY) | uint16 |
| `runoff_onset_median` | Multi-year median timing | DOWY | uint16 |
| `runoff_onset_mad` | Median absolute deviation | Days | float32 |
| `temporal_resolution` | Effective sampling frequency | Days | float32 |
| `temporal_resolution_median` | Median sampling frequency | Days | float32 |

### Data Specifications

- **Spatial Resolution**: 0.001° (~100m at equator)
- **Spatial Extent**: Global (-180° to 180°, -60° to 81.1°)
- **Temporal Coverage**: Water years 2015-2024
- **Data Format**: Cloud-optimized Zarr with Blosc compression
- **Coordinate System**: WGS84 (EPSG:4326)
- **No-Data Values**: 0 for DOWY, minimum float32 for continuous variables

## Installation and Setup

### Requirements

easysnowdata

### Configuration

1. **Azure Storage Access**: Configure SAS token for Azure Blob Storage
2. **Google Earth Engine**: Set up service account credentials
3. **Coiled Account**: Register for distributed computing access
4. **Configuration File**: Update paths in `config/global_config_v7.txt`

### File Structure

```
global_snowmelt_runoff_onset/
├── config/                          # Configuration files
│   ├── global_config_v7.txt         # Main configuration
│   ├── sas_token.txt                # Azure storage credentials
│   └── ee_key.json                  # Earth Engine credentials
├── global_snowmelt_runoff_onset/    # Python package
│   ├── config.py                    # Configuration management
│   └── processing.py                # Core processing functions
├── processing/                      # Processing notebooks
│   ├── make_zarr_store.ipynb        # Initialize global Zarr store
│   └── runoff_onset_parallel.ipynb  # Parallel tile processing
├── analysis/                        # Analysis and visualization
│   └── view_maps.ipynb              # Mapping and analysis tools
└── README.md                        # This file
```

## Usage

### 1. Initialize Global Dataset

Create the global Zarr store structure:

```python
from global_snowmelt_runoff_onset.config import Config

config = Config('config/global_config_v7.txt')
# Run make_zarr_store.ipynb to create empty store
```

### 2. Process Tiles

Execute parallel processing across spatial tiles:

```python
# Configure distributed computing
cluster = coiled.Cluster(
    n_workers=20,
    worker_memory="32 GB",
    worker_cpu=4
)

# Process tiles in batches
tiles = config.get_list_of_tiles(which='unprocessed')
# Run runoff_onset_parallel.ipynb for processing
```

### 3. Analyze Results

Visualize and analyze the global dataset:

```python
# Load global dataset
global_ds = xr.open_zarr(config.global_runoff_store)

# Create regional maps
regional_ds = global_ds.rio.clip_box(-125, 32, -105, 50)  # Western US
regional_ds['runoff_onset_median'].plot()

# Run view_maps.ipynb for comprehensive analysis
```


### Quality Control Parameters

Key parameters for ensuring data quality:

```python
min_monthly_acquisitions = 4        # Minimum acquisitions per month
max_allowed_days_gap_per_orbit = 24 # Maximum temporal gap
min_years_for_median_std = 3        # Minimum years for statistics
low_backscatter_threshold = 0.001   # Noise removal threshold
```


## Citation

If you use this dataset or methodology, please cite:

*I'll fill this in eventually....*

## Contact

- **Primary Author**: Eric Gagliano (egagli@uw.edu)
- **Institution**: Eric Gagliano
- **Project URL**: https://github.com/egagli/global_snowmelt_runoff_onset
- **Issues**: Submit via GitHub Issues for bug reports and feature requests

## Related Projects

- **easysnowdata**: Snow data access and analysis tools
- **sar_snowmelt_timing**: Regional SAR snowmelt detection
- **MODIS_seasonal_snow_mask**: Global snow cover climatology

---

*Last updated: 26 June 2025*