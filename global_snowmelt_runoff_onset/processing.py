"""
Processing functions for global snowmelt runoff onset detection using Sentinel-1 SAR data.

This module provides the core processing pipeline for detecting snowmelt runoff onset
timing from Sentinel-1 SAR backscatter data. The pipeline includes:

1. Data acquisition from Microsoft Planetary Computer
2. Spatiotemporal snow cover masking using MODIS-derived seasonal snow masks
3. Optional mountain region filtering using GMBA mountain inventory
4. Quality filtering and gap analysis
5. Runoff onset detection using backscatter minima
6. Statistical aggregation and uncertainty quantification
7. Output formatting and storage in Zarr format
"""

import easysnowdata
import pystac_client
import planetary_computer
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
import odc.stac
import flox
from typing import List, Tuple, Dict, Any, Union, Optional, Callable


def get_sentinel1_rtc(
    geobox: 'odc.geo.GeoBox', 
    bands: List[str] = ["vv","vh"], 
    start_date: str = '2014-01-01', 
    end_date: str = pd.Timestamp.today().strftime('%Y-%m-%d'), 
    chunks_read: Dict[str, Union[int, str]] = {}, 
    fail_on_error: bool = True
) -> xr.Dataset:
    """
    Retrieve Sentinel-1 RTC (Radiometric Terrain Corrected) data for a geographic region.
    
    Downloads and processes Sentinel-1 SAR data from Microsoft Planetary Computer,
    organizing it by orbit and adding water year and day-of-water-year coordinates
    based on the hemisphere.
    
    Args:
        geobox: Geographic bounding box defining the area of interest
        bands: SAR polarization bands to retrieve (e.g., ['vv', 'vh'])
        start_date: Start date for data retrieval (YYYY-MM-DD format)
        end_date: End date for data retrieval (YYYY-MM-DD format)
        chunks_read: Dask chunking configuration for data loading
        fail_on_error: Whether to raise exceptions on data loading errors
        
    Returns:
        xarray.Dataset containing:
        
        **Data variables:**
        - SAR backscatter data with dimensions ('time', 'latitude', 'longitude') for each band
        
        **Coordinates:**
        - time: Acquisition timestamps (datetime64[ns])
        - latitude: Latitude coordinates (float64)
        - longitude: Longitude coordinates (float64)
        - sat:orbit_state: Orbit direction ('ascending'/'descending') with dimension ('time',)
        - sat:relative_orbit: Orbit number (int16) with dimension ('time',)
        - water_year: Water year for each acquisition (int) with dimension ('time',)
        - DOWY: Day of water year (int) with dimension ('time',)
        
        **Attributes:**
        - hemisphere: 'northern' or 'southern' based on location
        
    Raises:
        Various exceptions from odc.stac.load if fail_on_error=True
    """
    items = (
        pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace)
        .search(
            intersects=geobox.geographic_extent,
            collections=["sentinel-1-rtc"],
            datetime=(start_date, end_date),
        )
        .item_collection()
    )

    load_params = {
        "items": items,
        "bands": bands,
        "nodata": -32768,
        "chunks": chunks_read, 
        "groupby": "sat:absolute_orbit",
        "geobox": geobox,
        "resampling": "bilinear",
        "fail_on_error": fail_on_error,
    }

    s1_rtc_ds = odc.stac.load(**load_params).sortby("time")
    metadata = gpd.GeoDataFrame.from_features(items, "epsg:4326")
    metadata_groupby_gdf = (
        metadata.groupby(["sat:absolute_orbit"]).first().sort_values("datetime")
    )

    s1_rtc_ds = s1_rtc_ds.assign_coords({
        "sat:orbit_state": ("time", metadata_groupby_gdf["sat:orbit_state"]),
        "sat:relative_orbit": ("time", metadata_groupby_gdf["sat:relative_orbit"].astype("int16"))
    })

    epsg = s1_rtc_ds.rio.estimate_utm_crs().to_epsg()
    hemisphere = 'northern' if epsg < 32700 else 'southern'
    s1_rtc_ds.attrs['hemisphere'] = hemisphere

    s1_rtc_ds = s1_rtc_ds.assign_coords({
        "water_year": ("time", pd.to_datetime(s1_rtc_ds.time).map(lambda x: easysnowdata.utils.datetime_to_WY(x, hemisphere=hemisphere))),
        "DOWY": ("time", pd.to_datetime(s1_rtc_ds.time).map(lambda x: easysnowdata.utils.datetime_to_DOWY(x, hemisphere=hemisphere)))
    })
      
    return s1_rtc_ds


def get_spatiotemporal_snow_cover_mask(
    ds: xr.Dataset, 
    bbox_gdf: gpd.GeoDataFrame, 
    seasonal_snow_mask_store: Any, 
    extend_search_window_beyond_SDD_days: int = 16, 
    min_consec_snow_days_for_seasonal_snow: int = 56,
    reproject_method: str = 'rasterio'
) -> xr.Dataset:
    """
    Generate spatiotemporal snow cover mask from MODIS-derived seasonal snow data.
    
    Creates masks that define:
    1. Spatial regions with seasonal snow (based on consecutive snow days threshold)
    2. Temporal windows for runoff onset detection (from snow accumulation to snow disappearance + buffer)
    
    The search window extends from the middle of the snow accumulation period to
    several days after the snow disappearance date to capture late-season melt events.
    
    Args:
        ds: Sentinel-1 dataset with dimensions ('time', 'latitude', 'longitude') used for spatial matching
        bbox_gdf: Bounding box as GeoDataFrame for clipping snow mask
        seasonal_snow_mask_store: Zarr store containing MODIS seasonal snow data
        extend_search_window_beyond_SDD_days: Days to extend search window past snow disappearance
        min_consec_snow_days_for_seasonal_snow: Minimum consecutive snow days to define seasonal snow
        reproject_method: Method for reprojection ('rasterio' or 'odc')
        
    Returns:
        xarray.Dataset containing:
        
        **Data variables (all with dimensions ('water_year', 'latitude', 'longitude')):**
        - SAD_DOWY: Snow accumulation date (day of water year, float32)
        - SDD_DOWY: Snow disappearance date (day of water year, float32)  
        - max_consec_snow_days: Maximum consecutive snow days (float32)
        - search_window_start_DOWY: Start of runoff detection window (float32)
        - search_window_end_DOWY: End of runoff detection window (float32)
        - binary_seasonal_snow_cover_presence: Boolean mask for seasonal snow areas (bool)
        - search_window_length: Length of detection window in days (float32)
        
        **Coordinates:**
        - water_year: Water years available in the seasonal snow mask (int)
        - latitude: Latitude coordinates matching input ds (float64)
        - longitude: Longitude coordinates matching input ds (float64)
        
    Raises:
        ValueError: If reproject_method is not 'rasterio' or 'odc'
    """
    seasonal_snow_mask = xr.open_zarr(seasonal_snow_mask_store, consolidated=True, decode_coords='all') 
    seasonal_snow_mask_clip_ds = seasonal_snow_mask.rio.clip_box(*bbox_gdf.total_bounds, crs='EPSG:4326')
    if reproject_method == 'rasterio':
        spatiotemporal_snow_cover_mask_ds = seasonal_snow_mask_clip_ds.rio.reproject_match(
            ds.isel(time=0), resampling=rasterio.enums.Resampling.bilinear
        ).rename({'x':'longitude','y':'latitude'})
    elif reproject_method == 'odc':
        spatiotemporal_snow_cover_mask_ds = seasonal_snow_mask_clip_ds.odc.reproject(ds.odc.geobox,resampling='bilinear')#.rename({'x':'longitude','y':'latitude'})
    else:
        raise ValueError("reproject_method must be either 'rasterio' or 'odc'.")

    # Create search window variables
    spatiotemporal_snow_cover_mask_ds['search_window_start_DOWY'] = (
        spatiotemporal_snow_cover_mask_ds['SAD_DOWY'] + 
        spatiotemporal_snow_cover_mask_ds['max_consec_snow_days']/2
    )
    spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'] = (
        spatiotemporal_snow_cover_mask_ds['SDD_DOWY'] + extend_search_window_beyond_SDD_days
    )
    spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'] = spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'].where(
        spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'] <= 366, 366
    )
    spatiotemporal_snow_cover_mask_ds['binary_seasonal_snow_cover_presence'] = (
        spatiotemporal_snow_cover_mask_ds['max_consec_snow_days'] >= min_consec_snow_days_for_seasonal_snow
    )
    spatiotemporal_snow_cover_mask_ds['search_window_length'] = (
        spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'] - 
        spatiotemporal_snow_cover_mask_ds['search_window_start_DOWY']
    )
    
    return spatiotemporal_snow_cover_mask_ds


def get_gmba_mountain_inventory(bbox_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Retrieve GMBA (Global Mountain Biodiversity Assessment) mountain inventory data.
    
    Downloads and clips mountain polygon data to the specified bounding box.
    Used when mountain_snow_only=True to restrict processing to mountain regions.
    
    Args:
        bbox_gdf: Bounding box as GeoDataFrame for clipping
        
    Returns:
        GeoDataFrame containing mountain polygons within the bounding box with columns:
        - geometry: Mountain polygon geometries
        - Additional GMBA attribute columns (mountain names, elevations, etc.)
    """
    url = "https://data.earthenv.org/mountains/standard/GMBA_Inventory_v2.0_standard_300.zip"
    gmba_gdf = gpd.read_file("zip+" + url)
    gmba_clipped_gdf = gpd.clip(gmba_gdf, bbox_gdf)
    return gmba_clipped_gdf


def apply_all_masks(
    s1_rtc_ds: xr.Dataset, 
    gmba_clipped_gdf: Optional[gpd.GeoDataFrame], 
    spatiotemporal_snow_cover_mask_ds: xr.Dataset, 
    water_years: np.ndarray
) -> xr.Dataset:
    """
    Apply all spatial and temporal masks to Sentinel-1 data.
    
    Combines multiple filtering steps:
    1. Temporal filtering to specified water years
    2. Equator crossing removal (for near-equatorial tiles)
    3. Mountain region masking (if specified)
    4. Spatiotemporal snow cover masking
    
    Args:
        s1_rtc_ds: Sentinel-1 RTC dataset with dimensions ('time', 'latitude', 'longitude')
                   and coordinates (water_year, DOWY, sat:relative_orbit)
        gmba_clipped_gdf: Mountain polygons (None if not using mountain masking)
        spatiotemporal_snow_cover_mask_ds: Snow cover mask dataset with dimensions 
                                          ('water_year', 'latitude', 'longitude')
        water_years: Array of water years to include
        
    Returns:
        Masked Sentinel-1 dataset with same structure as input but with:
        - Only specified water years retained in time dimension
        - Pixels outside seasonal snow regions set to NaN
        - Temporal observations outside detection windows set to NaN
        - Mountain masking applied if gmba_clipped_gdf provided
        
        **Dimensions:** ('time', 'latitude', 'longitude')
        **Coordinates:** Same as input s1_rtc_ds but filtered
    """
    s1_rtc_ds = remove_unwanted_water_years(s1_rtc_ds, water_years)

    center_lat = (s1_rtc_ds.rio.bounds()[1] + s1_rtc_ds.rio.bounds()[3]) / 2
    if np.absolute(center_lat) < 3:
        s1_rtc_ds = remove_equator_crossing(s1_rtc_ds)

    if gmba_clipped_gdf is not None:
        s1_rtc_ds = s1_rtc_ds.rio.clip_box(*gmba_clipped_gdf.total_bounds, crs=gmba_clipped_gdf.crs)
        s1_rtc_ds = s1_rtc_ds.rio.clip(gmba_clipped_gdf.geometry, drop=True)

    s1_rtc_masked_ds = apply_seasonal_snow_spatial_and_temporal_mask(s1_rtc_ds, spatiotemporal_snow_cover_mask_ds)
    return s1_rtc_masked_ds


def remove_unwanted_water_years(s1_rtc_ds: xr.Dataset, water_years: np.ndarray) -> xr.Dataset:
    """
    Filter dataset to include only specified water years.
    
    Args:
        s1_rtc_ds: Sentinel-1 dataset with 'water_year' coordinate on time dimension
        water_years: Array of water years to retain
        
    Returns:
        Dataset filtered to specified water years, maintaining same dimensions but 
        with reduced time dimension length
        
        **Input dimensions:** ('time', 'latitude', 'longitude')
        **Output dimensions:** ('time', 'latitude', 'longitude') - time dimension filtered
    """
    s1_rtc_ds = s1_rtc_ds.sel(time=s1_rtc_ds.water_year.isin(water_years))
    return s1_rtc_ds


def remove_equator_crossing(s1_rtc_ds: xr.Dataset) -> xr.Dataset:
    """
    Remove pixels that cross the equator to avoid water year definition conflicts.
    
    For tiles near the equator, ensures all pixels use the same hemisphere-based
    water year definition by masking pixels on the opposite side of the equator.
    
    Args:
        s1_rtc_ds: Sentinel-1 dataset with hemisphere attribute and dimensions 
                   ('time', 'latitude', 'longitude')
        
    Returns:
        Dataset with equator-crossing pixels masked (set to NaN)
        
        **Input/Output dimensions:** ('time', 'latitude', 'longitude')
        **Masking:** Applied across latitude dimension based on hemisphere attribute
    """
    if s1_rtc_ds.attrs['hemisphere'] == 'northern':
        mask = s1_rtc_ds.latitude >= 0
    else:
        mask = s1_rtc_ds.latitude < 0
    s1_rtc_ds = s1_rtc_ds.where(mask)
    return s1_rtc_ds


def apply_seasonal_snow_spatial_and_temporal_mask(
    s1_rtc_ds: xr.Dataset, 
    spatiotemporal_snow_cover_mask_ds: xr.Dataset
) -> xr.Dataset:
    """
    Apply seasonal snow spatial and temporal masks to Sentinel-1 data.
    
    For each water year, masks data to:
    1. Pixels with seasonal snow coverage
    2. Time periods within the runoff detection window
    
    Args:
        s1_rtc_ds: Sentinel-1 dataset with dimensions ('time', 'latitude', 'longitude')
                   and coordinates including water_year and DOWY on time dimension
        spatiotemporal_snow_cover_mask_ds: Snow cover mask dataset with dimensions 
                                          ('water_year', 'latitude', 'longitude')
        
    Returns:
        Masked dataset with same structure as input where:
        - Pixels without seasonal snow coverage are set to NaN
        - Time periods outside detection windows are set to NaN
        - Grouping preserves water_year coordinate structure
        
        **Input/Output dimensions:** ('time', 'latitude', 'longitude')
    """
    s1_rtc_masked_ds = s1_rtc_ds.groupby("water_year").map(
        lambda group: apply_mask_for_year(group, spatiotemporal_snow_cover_mask_ds)
    )
    s1_rtc_masked_ds.rio.write_crs(s1_rtc_ds.rio.crs, inplace=True)
    return s1_rtc_masked_ds


def apply_mask_for_year(
    group: xr.Dataset, 
    spatiotemporal_snow_cover_mask_ds: xr.Dataset
) -> xr.Dataset:
    """
    Apply spatiotemporal mask for a specific water year.
    
    Args:
        group: Sentinel-1 data for a single water year with dimensions 
               ('time', 'latitude', 'longitude') and DOWY coordinate
        spatiotemporal_snow_cover_mask_ds: Snow cover mask dataset with dimensions 
                                          ('water_year', 'latitude', 'longitude')
        
    Returns:
        Masked data for the water year, or all-NaN data if year not available
        
        **Input/Output dimensions:** ('time', 'latitude', 'longitude')
        **Masking logic:** 
        - Temporal: DOWY within [search_window_start_DOWY, search_window_end_DOWY]
        - Spatial: binary_seasonal_snow_cover_presence == True
    """
    year = group.water_year.values[0]

    if year not in spatiotemporal_snow_cover_mask_ds.water_year:
        print(f"Warning: water_year {year} not found in spatiotemporal_snow_cover_mask_ds")
        return group.where(False) 

    sad_mask = group['DOWY'] >= spatiotemporal_snow_cover_mask_ds['search_window_start_DOWY'].sel(water_year=year)
    sdd_mask = group['DOWY'] <= spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'].sel(water_year=year)
    consec_mask = spatiotemporal_snow_cover_mask_ds['binary_seasonal_snow_cover_presence'].sel(water_year=year)
    combined_mask = sad_mask & sdd_mask & consec_mask
    return group.where(combined_mask)


def remove_bad_scenes_and_border_noise(
    da: xr.DataArray, 
    threshold: float = 0.001
) -> xr.DataArray:
    """
    Remove bad scenes and border noise from SAR data.
    
    Applies different thresholds based on acquisition date to account for
    processing changes in the Sentinel-1 archive. Pre-2018 data uses a
    higher threshold to remove border noise.
    
    Args:
        da: SAR backscatter data array with dimensions ('time', 'latitude', 'longitude')
            and time coordinate as datetime64
        threshold: Threshold for removing very low backscatter values
        
    Returns:
        Filtered data array with bad scenes and noise removed (set to NaN)
        
        **Input/Output dimensions:** ('time', 'latitude', 'longitude')
        **Filtering:** Applied element-wise with date-dependent thresholds
    """
    cutoff_date = np.datetime64('2018-03-14')
    original_crs = da.rio.crs
    
    result = xr.where(
        da.time < cutoff_date,
        da.where(da > threshold),
        da.where(da > 0)
    )
    
    result.rio.write_crs(original_crs, inplace=True)
    return result


def calc_max_gap_pixelwise(
    group: xr.Dataset, 
    spatiotemporal_snow_cover_mask_ds: xr.Dataset
) -> xr.Dataset:
    """
    Calculate maximum temporal gap in acquisitions for each pixel.
    
    Computes the maximum time gap between consecutive valid acquisitions
    within the runoff detection window for each pixel. Used for quality
    control to ensure adequate temporal sampling.
    
    Args:
        group: Sentinel-1 data for a single water year and orbit with dimensions 
               ('time', 'latitude', 'longitude') and DOWY coordinate
        spatiotemporal_snow_cover_mask_ds: Snow cover mask with search window boundaries,
                                          dimensions ('water_year', 'latitude', 'longitude')
        
    Returns:
        Dataset with maximum gap in days for each pixel
        
        **Input dimensions:** ('time', 'latitude', 'longitude')
        **Output dimensions:** ('latitude', 'longitude') - time dimension reduced via max operation
        **Values:** Maximum consecutive gap in DOWY units between valid observations
    """
    group_is_null_ds = group.isnull()
    group_valid_pixels_DOWY_ds = xr.where(
        group_is_null_ds, group_is_null_ds.DOWY, np.nan
    )

    window_start_da = spatiotemporal_snow_cover_mask_ds["search_window_start_DOWY"]
    window_end_da = spatiotemporal_snow_cover_mask_ds["search_window_end_DOWY"]
    window_start_da["time"] = group_valid_pixels_DOWY_ds.time[0] - 1
    window_end_da["time"] = group_valid_pixels_DOWY_ds.time[-1] + 1
    window_start_da = window_start_da.expand_dims('time')
    window_end_da = window_end_da.expand_dims('time')

    window_start_ds = window_start_da.to_dataset(name='vv')
    window_end_ds = window_end_da.to_dataset(name='vv')

    for data_var in group_valid_pixels_DOWY_ds.data_vars:
        if data_var not in window_start_ds.data_vars:
            window_start_ds[data_var] = window_start_da
        if data_var not in window_end_ds.data_vars:
            window_end_ds[data_var] = window_end_da

    group_valid_pixels_DOWY_ds = xr.concat([
        window_start_ds,
        group_valid_pixels_DOWY_ds,
        window_end_ds,
    ], dim='time')

    group_max_gap_days_ds = group_valid_pixels_DOWY_ds.diff(dim="time").max(dim="time")
    return group_max_gap_days_ds


def filter_insufficient_pixels_per_orbit(
    s1_rtc_masked_ds: xr.Dataset, 
    spatiotemporal_snow_cover_mask_ds: xr.Dataset, 
    min_monthly_acquisitions: int, 
    max_allowed_days_gap_per_orbit: int
) -> xr.Dataset:
    """
    Filter pixels with insufficient temporal sampling per orbit.
    
    Removes pixels that don't meet minimum data quality requirements:
    1. Minimum acquisition frequency (scaled by detection window length)
    2. Maximum temporal gaps between acquisitions
    3. Presence of at least some valid data
    
    This ensures that runoff onset detection is based on adequate temporal sampling.
    
    Args:
        s1_rtc_masked_ds: Masked Sentinel-1 dataset for a single water year with dimensions 
                         ('time', 'latitude', 'longitude') and sat:relative_orbit coordinate
        spatiotemporal_snow_cover_mask_ds: Snow cover mask dataset with dimensions 
                                          ('water_year', 'latitude', 'longitude')
        min_monthly_acquisitions: Minimum acquisitions per 30-day period
        max_allowed_days_gap_per_orbit: Maximum allowed gap between acquisitions
        
    Returns:
        Dataset with only adequately sampled pixels per orbit (others set to NaN)
        
        **Input/Output dimensions:** ('time', 'latitude', 'longitude')
        **Grouping:** Applied per sat:relative_orbit, filtering within each orbit
        **Quality criteria:** Applied per pixel based on temporal sampling adequacy
    """
    water_year = s1_rtc_masked_ds.water_year.values[0]
    spatiotemporal_snow_cover_mask_slice_ds = spatiotemporal_snow_cover_mask_ds.sel(water_year=water_year)

    # Filter scenes by minimum acquisitions and max gaps
    pixelwise_counts_per_orbit_and_polarization_ds = s1_rtc_masked_ds.groupby(
        "sat:relative_orbit"
    ).count(dim="time", engine="flox")

    pixelwise_max_gaps_per_orbit_ds = s1_rtc_masked_ds.groupby("sat:relative_orbit").map(
        calc_max_gap_pixelwise, 
        spatiotemporal_snow_cover_mask_ds=spatiotemporal_snow_cover_mask_slice_ds
    )

    sufficient_acquisitions_mask = pixelwise_counts_per_orbit_and_polarization_ds >= (
        min_monthly_acquisitions * (spatiotemporal_snow_cover_mask_slice_ds["search_window_length"] / 30)
    )
    acceptable_gaps_mask = pixelwise_max_gaps_per_orbit_ds <= max_allowed_days_gap_per_orbit
    has_data_mask = pixelwise_counts_per_orbit_and_polarization_ds > 0
    
    # Combine all criteria
    sufficient_mask = sufficient_acquisitions_mask & acceptable_gaps_mask & has_data_mask
    s1_rtc_masked_filtered_ds = s1_rtc_masked_ds.groupby("sat:relative_orbit").where(sufficient_mask)

    return s1_rtc_masked_filtered_ds


def get_temporal_resolution(
    s1_rtc_masked_filtered_ds: xr.Dataset, 
    spatiotemporal_snow_cover_mask_ds: xr.Dataset
) -> xr.DataArray:
    """
    Calculate temporal resolution of the dataset.
    
    Computes the effective temporal resolution as the ratio of the detection
    window length to the number of valid acquisitions. Provides a measure
    of data density for each pixel and water year.
    
    Args:
        s1_rtc_masked_filtered_ds: Filtered Sentinel-1 dataset with dimensions 
                                  ('time', 'latitude', 'longitude') and water_year coordinate
        spatiotemporal_snow_cover_mask_ds: Snow cover mask dataset with dimensions 
                                          ('water_year', 'latitude', 'longitude')
        
    Returns:
        DataArray with temporal resolution in days for each pixel and water year
        
        **Dimensions:** ('water_year', 'latitude', 'longitude')
        **Values:** search_window_length / count_of_valid_observations (float32)
        **Coordinates:** water_year, latitude, longitude
    """
    temporal_resolution_da = (
        spatiotemporal_snow_cover_mask_ds["search_window_length"] / 
        s1_rtc_masked_filtered_ds['vv'].groupby("water_year").count(dim="time").where(lambda x: x > 0)
    )
    return temporal_resolution_da


def xr_datetime_to_DOWY(
    date_da: xr.DataArray, 
    hemisphere: str = "northern"
) -> xr.DataArray:
    """
    Convert xarray DataArray of datetime objects to Day of Water Year (DOWY).
    
    Converts datetime coordinates to day-of-water-year values, which provide
    a consistent temporal reference for comparing dates across different
    calendar years within the same water year.
    
    Args:
        date_da: DataArray containing datetime objects with any dimensions
        hemisphere: 'northern' or 'southern' for water year definition
        
    Returns:
        DataArray with DOWY values (1-366), with NaT values converted to 0
        
        **Input dimensions:** Same as date_da (typically includes 'latitude', 'longitude')
        **Output dimensions:** Same as input
        **Data type:** uint16 with 0 as nodata value
        **Values:** 1-366 representing day within water year
    """
    if date_da.attrs.get("any_valid_date") is not None:
        any_valid_date = pd.to_datetime(date_da.attrs["any_valid_date"])
    else:
        any_valid_date = pd.to_datetime(date_da.isel(latitude=0, longitude=0).values)

    start_of_water_year = easysnowdata.utils.get_water_year_start(
        any_valid_date, hemisphere=hemisphere
    )
    
    start_of_water_year_np = np.datetime64(start_of_water_year)
    
    def vectorized_dowy_calc(x):
        """Vectorized function that works efficiently with dask chunks and handles NaT"""
        nodata_uint16 = 0
        x_days = x.astype('datetime64[D]')
        
        days_diff = (x_days - start_of_water_year_np).astype('timedelta64[D]').astype('int64') + 1
        
        result = days_diff.astype('uint16')
        result[pd.isna(x)] = nodata_uint16

        return result
    
    return xr.apply_ufunc(
        vectorized_dowy_calc,
        date_da,
        vectorize=False,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {}},
    )


def calculate_backscatter_min_per_orbit(s1_rtc_masked_filtered_ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate timing of minimum backscatter per orbit and polarization.
    
    Finds the date of minimum backscatter for each pixel, orbit, and polarization.
    
    Args:
        s1_rtc_masked_filtered_ds: Quality-filtered Sentinel-1 dataset with dimensions 
                                  ('time', 'latitude', 'longitude') and sat:relative_orbit coordinate
        
    Returns:
        Dataset with datetime of minimum backscatter for each orbit and polarization
        
        **Input dimensions:** ('time', 'latitude', 'longitude')
        **Output dimensions:** ('sat:relative_orbit', 'latitude', 'longitude')
        **Values:** datetime64[ns] of minimum backscatter occurrence
        **Data variables:** One per band (e.g., 'vv', 'vh')
    """
    backscatter_min_timing_per_orbit_and_polarization_ds = s1_rtc_masked_filtered_ds.groupby(
        "sat:relative_orbit"
    ).map(lambda c: c.idxmin(dim="time"))
    return backscatter_min_timing_per_orbit_and_polarization_ds


def calculate_runoff_onset_from_constituent_runoff_onsets(
    constituent_runoff_onsets_da: xr.DataArray
) -> xr.DataArray:
    """
    Calculate final runoff onset from multiple orbit/polarization estimates.
    
    Combines runoff onset estimates from different satellite orbits and
    polarizations by taking the median value. This reduces the impact of
    outliers and provides a more robust estimate.
    
    Args:
        constituent_runoff_onsets_da: DataArray with onset dates by orbit and polarization,
                                     dimensions ('sat:relative_orbit', 'polarization', 'latitude', 'longitude')
        
    Returns:
        DataArray with median runoff onset date for each pixel
        
        **Input dimensions:** ('sat:relative_orbit', 'polarization', 'latitude', 'longitude')
        **Output dimensions:** ('latitude', 'longitude')
        **Values:** datetime64[ns] - median across orbits and polarizations
        **Processing:** Excludes zero values, computes median, converts back to datetime
    """
    runoff_onset_da = (
        constituent_runoff_onsets_da.astype("int64")
        .where(lambda x: x > 0)
        .median(dim=["sat:relative_orbit", "polarization"], skipna=True)
        .astype("datetime64[ns]")
    )
    return runoff_onset_da


def calculate_runoff_onset(
    s1_rtc_masked_filtered_ds: xr.Dataset, 
    return_constituent_runoff_onsets: bool = False, 
    returned_dates_format: str = "dowy"
) -> xr.DataArray:
    """
    Calculate snowmelt runoff onset dates from filtered Sentinel-1 data.
    
    Main function for runoff onset detection. Identifies minimum backscatter
    timing per orbit/polarization, then optionally aggregates to a single
    estimate per pixel. Output format can be customized.
    
    Args:
        s1_rtc_masked_filtered_ds: Quality-filtered Sentinel-1 dataset with dimensions 
                                  ('time', 'latitude', 'longitude') and coordinates:
                                  water_year, sat:relative_orbit
        return_constituent_runoff_onsets: Whether to return individual orbit/polarization estimates
        returned_dates_format: Output format ('dowy', 'doy', or 'datetime64')
        
    Returns:
        DataArray with runoff onset dates in the specified format
        
        **If return_constituent_runoff_onsets=False:**
        - **Dimensions:** ('latitude', 'longitude')
        - **Values:** Aggregated onset estimate per pixel
        
        **If return_constituent_runoff_onsets=True:**
        - **Dimensions:** ('sat:relative_orbit', 'polarization', 'latitude', 'longitude')
        - **Values:** Individual onset estimates per orbit/polarization
        
        **Value types by format:**
        - 'dowy': uint16 (1-366, 0=nodata)
        - 'doy': int (1-366)
        - 'datetime64': datetime64[ns]
        
    Raises:
        ValueError: If returned_dates_format is not recognized
    """
    backscatter_min_timing_per_orbit_and_polarization_ds = calculate_backscatter_min_per_orbit(s1_rtc_masked_filtered_ds)
    constituent_runoff_onsets_da = backscatter_min_timing_per_orbit_and_polarization_ds.to_dataarray(dim="polarization")

    if return_constituent_runoff_onsets == False:
        runoff_onset_da = calculate_runoff_onset_from_constituent_runoff_onsets(constituent_runoff_onsets_da)
    else:
        runoff_onset_da = constituent_runoff_onsets_da

    if returned_dates_format == "dowy":
        hemisphere = (
            "northern"
            if s1_rtc_masked_filtered_ds.rio.estimate_utm_crs().to_epsg() < 32700
            else "southern"
        )
        month_start = 10 if hemisphere == "northern" else 4
        print(f"Area is in the {hemisphere} hemisphere. Water year starts in month {month_start}.")
        runoff_onset_da.attrs["any_valid_date"] = s1_rtc_masked_filtered_ds.time[0].values
        runoff_onset_da = xr_datetime_to_DOWY(runoff_onset_da, hemisphere=hemisphere)

    elif returned_dates_format == "doy":
        runoff_onset_da = runoff_onset_da.dt.dayofyear
    elif returned_dates_format == "datetime64":
        runoff_onset_da = runoff_onset_da
    else:
        raise ValueError('returned_dates_format must be either "doy", "dowy", or "datetime64".')

    return runoff_onset_da


def median_and_mad_with_min_obs(
    da: xr.DataArray, 
    dim: str, 
    min_count: int
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate median and median absolute deviation with minimum observation requirement.
    
    Computes robust statistics for runoff onset timing across multiple years.
    Only calculates statistics for pixels with sufficient observations to
    ensure statistical reliability.
    
    Args:
        da: DataArray with runoff onset dates, dimensions typically include the aggregation 
            dimension (e.g., 'water_year') plus spatial dimensions ('latitude', 'longitude')
            Zero values are excluded from calculations
        dim: Dimension along which to calculate statistics (typically 'water_year')
        min_count: Minimum number of valid observations required
        
    Returns:
        Tuple of (median, mad) DataArrays with statistics where sufficient data exists
        
        **Input dimensions:** (dim, 'latitude', 'longitude') - e.g., ('water_year', 'latitude', 'longitude')
        **Output dimensions:** ('latitude', 'longitude') - aggregation dimension removed
        **Data type:** Same as input for median, float for MAD
        **Values:** NaN where min_count not met, valid statistics elsewhere
    """
    da = da.where(lambda x: x > 0)  # Exclude zero values
    count_mask = da.notnull().sum(dim=dim) >= min_count
    median = da.where(count_mask).median(dim=dim)
    abs_dev = np.abs(da - median)
    mad = abs_dev.where(count_mask).median(dim=dim)
    return median, mad


def median_with_min_obs(
    da: xr.DataArray, 
    dim: str, 
    min_count: int
) -> xr.DataArray:
    """
    Calculate median with minimum observation requirement.
    
    Args:
        da: DataArray with values, dimensions typically include the aggregation 
            dimension plus spatial dimensions. Zero values are excluded from calculations
        dim: Dimension along which to calculate median
        min_count: Minimum number of valid observations required
        
    Returns:
        DataArray with median values where sufficient data exists
        
        **Input dimensions:** (dim, ...) - includes aggregation dimension
        **Output dimensions:** Same as input minus the aggregation dimension
        **Values:** NaN where min_count not met, median elsewhere
    """
    da = da.where(lambda x: x > 0)  # Exclude zero values
    count_mask = da.notnull().sum(dim=dim) >= min_count
    median = da.where(count_mask).median(dim=dim)
    return median


def dataarrays_to_dataset(
    runoff_onsets_da: xr.DataArray, 
    median_da: xr.DataArray, 
    mad_da: xr.DataArray, 
    water_years: np.ndarray, 
    temporal_resolution_da: Optional[xr.DataArray] = None, 
    median_temporal_resolution_da: Optional[xr.DataArray] = None
) -> xr.Dataset:
    """
    Combine individual DataArrays into a comprehensive output Dataset.
    
    Creates the final output dataset containing runoff onset estimates,
    statistical summaries, and data quality metrics. Ensures proper
    data types and coordinates for Zarr storage.
    
    Args:
        runoff_onsets_da: Runoff onset dates by water year with dimensions 
                         ('water_year', 'latitude', 'longitude'), dtype compatible with uint16
        median_da: Median runoff onset across years with dimensions 
                   ('latitude', 'longitude'), dtype compatible with uint16
        mad_da: Median absolute deviation across years with dimensions 
                ('latitude', 'longitude'), dtype float32
        water_years: Array of all water years for coordinate alignment
        temporal_resolution_da: Temporal resolution by water year with dimensions 
                               ('water_year', 'latitude', 'longitude'), optional
        median_temporal_resolution_da: Median temporal resolution with dimensions 
                                      ('latitude', 'longitude'), optional
        
    Returns:
        xarray.Dataset with all variables properly formatted for output
        
        **Dataset structure:**
        - **runoff_onset**: dimensions ('water_year', 'latitude', 'longitude'), dtype uint16
        - **runoff_onset_median**: dimensions ('latitude', 'longitude'), dtype uint16  
        - **runoff_onset_mad**: dimensions ('latitude', 'longitude'), dtype float32
        - **temporal_resolution**: dimensions ('water_year', 'latitude', 'longitude'), dtype float32 (if provided)
        - **temporal_resolution_median**: dimensions ('latitude', 'longitude'), dtype float32 (if provided)
        
        **Coordinates:**
        - water_year: Complete range from water_years array
        - latitude: Spatial coordinates from input arrays
        - longitude: Spatial coordinates from input arrays
    """
    runoff_onsets_ds = runoff_onsets_da.to_dataset(name='runoff_onset').round().astype('uint16')
    runoff_onsets_ds = runoff_onsets_ds.reindex(water_year=water_years)
    runoff_onsets_ds['runoff_onset_median'] = median_da.round().astype('uint16')
    runoff_onsets_ds['runoff_onset_mad'] = mad_da

    if temporal_resolution_da is not None:
        runoff_onsets_ds['temporal_resolution'] = temporal_resolution_da
    if median_temporal_resolution_da is not None:
        runoff_onsets_ds['temporal_resolution_median'] = median_temporal_resolution_da

    return runoff_onsets_ds