import easysnowdata
import pystac_client
import tqdm
import planetary_computer
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
import odc.stac
import time
import dask
import dask.distributed
import coiled
import matplotlib.pyplot as plt
import traceback
from global_snowmelt_runoff_onset.config import Config, Tile



def get_sentinel1_rtc(geobox, bands=["vv","vh"], start_date='2014-01-01', end_date=pd.Timestamp.today().strftime('%Y-%m-%d'), chunks_read={}):

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
        "geobox":geobox,
        "resampling": "bilinear",
        "fail_on_error":False
    }


    s1_rtc_ds = odc.stac.load(**load_params).sortby("time")

    metadata = gpd.GeoDataFrame.from_features(items, "epsg:4326")

    metadata_groupby_gdf = (
        metadata.groupby(["sat:absolute_orbit"]).first().sort_values("datetime")
    )


    s1_rtc_ds = s1_rtc_ds.assign_coords(
    {
        "sat:orbit_state": ("time", metadata_groupby_gdf["sat:orbit_state"]),
        "sat:relative_orbit": ("time", metadata_groupby_gdf["sat:relative_orbit"].astype("int16"))
    })

    epsg = s1_rtc_ds.rio.estimate_utm_crs().to_epsg()
    hemisphere = 'northern' if epsg < 32700 else 'southern'

    s1_rtc_ds.attrs['hemisphere'] = hemisphere

    s1_rtc_ds = s1_rtc_ds.assign_coords(
    {
        "water_year": ("time", pd.to_datetime(s1_rtc_ds.time).map(lambda x: easysnowdata.utils.datetime_to_WY(x, hemisphere=hemisphere))),
        "DOWY": ("time", pd.to_datetime(s1_rtc_ds.time).map(lambda x: easysnowdata.utils.datetime_to_DOWY(x, hemisphere=hemisphere)))
    })
      

    return s1_rtc_ds



# def apply_all_masks(s1_rtc_ds,gmba_clipped_gdf,seasonal_snow_mask_matched_ds,water_years):

#     s1_rtc_ds = remove_unwanted_water_years(s1_rtc_ds,water_years)

#     center_lat = (s1_rtc_ds.rio.bounds()[1]+s1_rtc_ds.rio.bounds()[3])/2
#     if np.absolute(center_lat) < 3:
#         s1_rtc_ds = remove_equator_crossing(s1_rtc_ds)
        
#     if gmba_clipped_gdf is not None:
#         s1_rtc_ds = s1_rtc_ds.rio.clip_box(*gmba_clipped_gdf.total_bounds,crs=gmba_clipped_gdf.crs)
#         s1_rtc_ds = s1_rtc_ds.rio.clip(gmba_clipped_gdf.geometry,drop=True) # does this compute?

#     s1_rtc_masked_ds = apply_seasonal_snow_spatial_and_temporal_mask(s1_rtc_ds, seasonal_snow_mask_matched_ds)
    
#     return s1_rtc_masked_ds

def apply_all_masks(s1_rtc_ds,gmba_clipped_gdf,spatiotemporal_snow_cover_mask_ds,water_years):

    s1_rtc_ds = remove_unwanted_water_years(s1_rtc_ds,water_years)

    center_lat = (s1_rtc_ds.rio.bounds()[1]+s1_rtc_ds.rio.bounds()[3])/2
    if np.absolute(center_lat) < 3:
        s1_rtc_ds = remove_equator_crossing(s1_rtc_ds)

    if gmba_clipped_gdf is not None:
        s1_rtc_ds = s1_rtc_ds.rio.clip_box(*gmba_clipped_gdf.total_bounds,crs=gmba_clipped_gdf.crs)
        s1_rtc_ds = s1_rtc_ds.rio.clip(gmba_clipped_gdf.geometry,drop=True) # does this compute?

    s1_rtc_masked_ds = apply_seasonal_snow_spatial_and_temporal_mask(s1_rtc_ds, spatiotemporal_snow_cover_mask_ds)

    return s1_rtc_masked_ds

def remove_unwanted_water_years(s1_rtc_ds, water_years):
    s1_rtc_ds = s1_rtc_ds.sel(time=s1_rtc_ds.water_year.isin(water_years))
    return s1_rtc_ds


def remove_equator_crossing(s1_rtc_ds):
    if s1_rtc_ds.attrs['hemisphere'] == 'northern':
        mask = s1_rtc_ds.latitude >= 0
    else:
        mask = s1_rtc_ds.latitude < 0

    s1_rtc_ds = s1_rtc_ds.where(mask)
    return s1_rtc_ds

def get_gmba_mountain_inventory(bbox_gdf):
    url = (f"https://data.earthenv.org/mountains/standard/GMBA_Inventory_v2.0_standard_300.zip")
    gmba_gdf = gpd.read_file("zip+" + url)
    gmba_clipped_gdf = gpd.clip(gmba_gdf, bbox_gdf)
    return gmba_clipped_gdf

def get_custom_seasonal_snow_mask(s1_rtc_ds,bbox_gdf,seasonal_snow_mask_store):
    seasonal_snow_mask = xr.open_zarr(seasonal_snow_mask_store, consolidated=True, decode_coords='all') 
    seasonal_snow_mask_clip_ds = seasonal_snow_mask.rio.clip_box(*bbox_gdf.total_bounds,crs='EPSG:4326') # clip to correct box, maybe use total_bounds and then use crs 
    seasonal_snow_mask_matched_ds = seasonal_snow_mask_clip_ds.rio.reproject_match(s1_rtc_ds.isel(time=slice(0,10)).max(dim='time'), resampling=rasterio.enums.Resampling.bilinear).rename({'x':'longitude','y':'latitude'}) # if S1 scene at t=0 isn't full, does this get rid of mask values?
    #seasonal_snow_mask_matched_ds = seasonal_snow_mask_clip_ds.odc.reproject(geobox)#.rename({'x':'longitude','y':'latitude'})
    return seasonal_snow_mask_matched_ds

def get_spatiotemporal_snow_cover_mask(ds,bbox_gdf,seasonal_snow_mask_store,extend_search_window_beyond_SDD_days=16, min_consec_snow_days_for_seasonal_snow=56):
    """
    Get the snow cover mask for the given dataset and bounding box.
    """
    spatiotemporal_snow_cover_mask_ds = get_custom_seasonal_snow_mask(
        ds, bbox_gdf, seasonal_snow_mask_store
    )
    
    # Create a new dataset with the search window and snow cover presence
    spatiotemporal_snow_cover_mask_ds['search_window_start_DOWY'] = spatiotemporal_snow_cover_mask_ds['SAD_DOWY'] +  spatiotemporal_snow_cover_mask_ds['max_consec_snow_days']/2
    spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'] = spatiotemporal_snow_cover_mask_ds['SDD_DOWY'] + extend_search_window_beyond_SDD_days
    # where search window end is greater than 366, set it to 366
    spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'] = spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'].where(spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'] <= 366, 366)
    spatiotemporal_snow_cover_mask_ds['binary_seasonal_snow_cover_presence'] = spatiotemporal_snow_cover_mask_ds['max_consec_snow_days'] >= min_consec_snow_days_for_seasonal_snow
    spatiotemporal_snow_cover_mask_ds['search_window_length'] = spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'] - spatiotemporal_snow_cover_mask_ds['search_window_start_DOWY']
    
    return spatiotemporal_snow_cover_mask_ds

# def apply_seasonal_snow_spatial_and_temporal_mask(s1_rtc_ds, seasonal_snow_mask_matched_ds):
#     s1_rtc_masked_ds = s1_rtc_ds.groupby(s1_rtc_ds.water_year.compute()).map(lambda group: apply_mask_for_year(group, seasonal_snow_mask_matched_ds))
#     s1_rtc_masked_ds.rio.write_crs(s1_rtc_ds.rio.crs,inplace=True)
#     return s1_rtc_masked_ds

def apply_seasonal_snow_spatial_and_temporal_mask(s1_rtc_ds, spatiotemporal_snow_cover_mask_ds):
    s1_rtc_masked_ds = s1_rtc_ds.groupby("water_year").map(lambda group: apply_mask_for_year(group, spatiotemporal_snow_cover_mask_ds))
    s1_rtc_masked_ds.rio.write_crs(s1_rtc_ds.rio.crs,inplace=True)
    return s1_rtc_masked_ds

# def apply_mask_for_year(group, seasonal_snow_mask_matched_ds):

#     year = group.water_year.values[0]


#     if year not in seasonal_snow_mask_matched_ds.water_year:
#         print(f"Warning: water_year {year} not found in seasonal_snow_mask_matched_ds")
#         return group.where(False) 

#     sad_mask = group['DOWY'] >= seasonal_snow_mask_matched_ds['SAD_DOWY'].sel(water_year=year)
#     sdd_mask = group['DOWY'] <= seasonal_snow_mask_matched_ds['SDD_DOWY'].sel(water_year=year)
#     consec_mask = seasonal_snow_mask_matched_ds['max_consec_snow_days'].sel(water_year=year) >= 56
#     combined_mask = sad_mask & sdd_mask & consec_mask
#     return group.where(combined_mask)


# def apply_mask_for_year(group, seasonal_snow_mask_matched_ds):

#     year = group.water_year.values[0]


#     if year not in seasonal_snow_mask_matched_ds.water_year:
#         print(f"Warning: water_year {year} not found in seasonal_snow_mask_matched_ds")
#         return group.where(False) 
    
#     #sad_mask = group['DOWY'] >= seasonal_snow_mask_matched_ds['SAD_DOWY'].sel(water_year=year)
#     sad_mask = group['DOWY'] >= (seasonal_snow_mask_matched_ds['SAD_DOWY'].sel(water_year=year) + seasonal_snow_mask_matched_ds['max_consec_snow_days'].sel(water_year=year)/2)
#     # CHANGED THIS -- MULTIPLY BY 2 TO ACCOUNT FOR HALFING SNOW MAX TEMPORAL SEARCH WINDOW
#     sdd_mask = group['DOWY'] <= seasonal_snow_mask_matched_ds['SDD_DOWY'].sel(water_year=year)+16 # changed from 8 to 16 june 22 2025 for v6
#     consec_mask = seasonal_snow_mask_matched_ds['max_consec_snow_days'].sel(water_year=year) >= 56
#     combined_mask = sad_mask & sdd_mask & consec_mask
#     return group.where(combined_mask)

def apply_mask_for_year(group, spatiotemporal_snow_cover_mask_ds):

    year = group.water_year.values[0]


    if year not in spatiotemporal_snow_cover_mask_ds.water_year:
        print(f"Warning: water_year {year} not found in spatiotemporal_snow_cover_mask_ds")
        return group.where(False) 

    sad_mask = group['DOWY'] >= spatiotemporal_snow_cover_mask_ds['search_window_start_DOWY'].sel(water_year=year)
    sdd_mask = group['DOWY'] <= spatiotemporal_snow_cover_mask_ds['search_window_end_DOWY'].sel(water_year=year)
    consec_mask = spatiotemporal_snow_cover_mask_ds['binary_seasonal_snow_cover_presence'].sel(water_year=year)
    combined_mask = sad_mask & sdd_mask & consec_mask
    return group.where(combined_mask)

def xr_datetime_to_DOWY(date_da, hemisphere="northern"):
    """
    Converts an xarray DataArray containing datetime objects to the Day of Water Year (DOWY).

    Parameters:
    date (xr.DataArray): An xarray DataArray with datetime64 data type.
    hemisphere (str): 'northern' or 'southern'

    Returns:
    xr.DataArray: An xarray DataArray containing the DOWY for each datetime in the input DataArray.
    """

    if date_da.attrs.get("any_valid_date") is not None:
        any_valid_date = pd.to_datetime(date_da.attrs["any_valid_date"])
    else:
        any_valid_date = pd.to_datetime(date_da.sel(x=0, y=0, method="nearest").values)

    start_of_water_year = easysnowdata.utils.get_water_year_start(
        any_valid_date, hemisphere=hemisphere
    )

    return xr.apply_ufunc(
        lambda x: (x - start_of_water_year).days + 1,  # dt accessor?
        date_da,
        input_core_dims=[[]],
        vectorize=True,
        dask="parallelized",  # try allowed also
        output_dtypes=[float],
    )


def calculate_runoff_onset(
    s1_rtc_ds: xr.Dataset,
    min_monthly_acquisitions: int,
    max_allowed_days_gap_per_orbit: int,
    consec_snow_days_da: xr.DataArray,
    return_constituent_runoff_onsets: bool = False,
    returned_dates_format: str = "dowy",
    report_temporal_res: bool = False,
):



    #pixelwise_counts_per_orbit_and_polarization_ds = (count_acquisitions_per_orbit_and_polarization(s1_rtc_ds))  # this should be for melt, keeping general for now to integrate modis data
    pixelwise_counts_per_orbit_and_polarization_ds, max_days_gap_per_orbit_da = count_acquisitions_and_max_gap_per_orbit_and_polarization(s1_rtc_ds)

    backscatter_min_timing_per_orbit_and_polarization_ds = (calculate_backscatter_min_per_orbit(s1_rtc_ds))


    if report_temporal_res:
        constituent_runoff_onsets_da, temporal_resolution, pixel_count = ( # , temporal_resolution, pixel_count
            filter_insufficient_pixels_per_orbit_and_polarization(
                backscatter_min_timing_per_orbit_and_polarization_ds,
                pixelwise_counts_per_orbit_and_polarization_ds,
                max_days_gap_per_orbit_da,
                consec_snow_days_da,
                min_monthly_acquisitions,
                max_allowed_days_gap_per_orbit,
                report_temporal_res
            )
        )
    else:
        constituent_runoff_onsets_da = ( # , temporal_resolution, pixel_count
            filter_insufficient_pixels_per_orbit_and_polarization(
                backscatter_min_timing_per_orbit_and_polarization_ds,
                pixelwise_counts_per_orbit_and_polarization_ds,
                max_days_gap_per_orbit_da,
                consec_snow_days_da,
                min_monthly_acquisitions,
                max_allowed_days_gap_per_orbit,
                report_temporal_res
            )
        )

    if return_constituent_runoff_onsets == False:
        runoff_onset_da = calculate_runoff_onset_from_constituent_runoff_onsets(constituent_runoff_onsets_da)
    else:
        runoff_onset_da = constituent_runoff_onsets_da


    if returned_dates_format == "dowy":

        hemisphere = (
            "northern"
            if s1_rtc_ds.rio.estimate_utm_crs().to_epsg() < 32700
            else "southern"
        )
        month_start = 10 if hemisphere == "northern" else 4
        print(
            f"Area is in the {hemisphere} hemisphere. Water year starts in month {month_start}."
        )
        runoff_onset_da.attrs["any_valid_date"] = s1_rtc_ds.time[0].values
        runoff_onset_da = xr_datetime_to_DOWY(runoff_onset_da, hemisphere=hemisphere)

    elif returned_dates_format == "doy":
        runoff_onset_da = runoff_onset_da.dt.dayofyear
    elif returned_dates_format == "datetime64":
        runoff_onset_da = runoff_onset_da
    else:
        raise ValueError(
            'returned_dates_format must be either "doy", "dowy", or "datetime64".'
        )

    if report_temporal_res:
        return runoff_onset_da, temporal_resolution, pixel_count  
    else:
        return runoff_onset_da

def remove_bad_scenes_and_border_noise(da, threshold=0.001):
    cutoff_date = np.datetime64('2018-03-14')
    
    original_crs = da.rio.crs
    
    result = xr.where(
        da.time < cutoff_date,
        da.where(da > threshold),
        da.where(da > 0)
    )
    
    result.rio.write_crs(original_crs, inplace=True)
    
    return result


# def count_acquisitions_per_orbit_and_polarization(s1_rtc_ds: xr.Dataset):
#     print("Calculating pixelwise counts per orbit and polarization...")
#     pixelwise_counts_per_orbit_and_polarization = s1_rtc_ds.groupby(
#         "sat:relative_orbit"
#     ).count(dim="time", engine='flox')
#     return pixelwise_counts_per_orbit_and_polarization

def count_acquisitions_and_max_gap_per_orbit_and_polarization(s1_rtc_ds: xr.Dataset):
    print("Calculating pixelwise counts and maximum gaps per orbit and polarization...")
    pixelwise_counts_per_orbit_and_polarization_ds = s1_rtc_ds.groupby("sat:relative_orbit").count(dim="time", engine='flox') #, engine='flox'
    
    def calc_max_gap(group):
            times = group.time.sortby('time')
            if len(times) == 1: # if only one scene in this group, set gap to very large number so it won't be calculated
                return times.count()*9999
            gaps = times.diff(dim='time').max().dt.days
            return gaps

    max_time_gap_per_orbit_days_da = s1_rtc_ds.groupby("sat:relative_orbit").map(calc_max_gap)
    
    return pixelwise_counts_per_orbit_and_polarization_ds, max_time_gap_per_orbit_days_da


def calculate_backscatter_min_per_orbit(s1_rtc_ds: xr.Dataset):
    print("Calculating backscatter min per orbit...")
    backscatter_min_timing_per_orbit_and_polarization_ds = s1_rtc_ds.groupby(
        "sat:relative_orbit"
    ).map(lambda c: c.idxmin(dim="time")) # maybe only map if if max-min > -1dB
    return backscatter_min_timing_per_orbit_and_polarization_ds


# def filter_insufficient_pixels_per_orbit_and_polarization(
#     backscatter_min_timing_per_orbit_and_polarization_ds: xr.Dataset,
#     pixelwise_counts_per_orbit: xr.Dataset,
#     consec_snow_days_da: xr.DataArray,
#     min_monthly_acquisitions: int,
# ):
    
#     print(f"Filtering insufficient pixels per orbit and polarization, must have at least {min_monthly_acquisitions} per month...")
#     constituent_runoff_onsets_ds = (
#         backscatter_min_timing_per_orbit_and_polarization_ds.where(
#             pixelwise_counts_per_orbit >= (min_monthly_acquisitions*(consec_snow_days_da/30))
#         )
#     )
#     return constituent_runoff_onsets_ds

def filter_insufficient_pixels_per_orbit_and_polarization(
    backscatter_min_timing_per_orbit_and_polarization_ds: xr.Dataset,
    pixelwise_counts_per_orbit_and_polarization_ds: xr.Dataset,
    max_days_gap_per_orbit_da: xr.DataArray,
    consec_snow_days_da: xr.DataArray,
    min_monthly_acquisitions: int,
    max_allowed_days_gap_per_orbit: int,
    report_temporal_res: bool,
):
    print(f"Filtering insufficient pixels per orbit and polarization...")

    modified_consec_snow_days_da = (consec_snow_days_da/2)+16 # changed june 22 2025 for v6, this is to account for the fact that we are halving the max temporal search window for snow days and extending 16 days beyond SDD

    #pixelwise_counts_per_orbit_and_polarization_ds = pixelwise_counts_per_orbit_and_polarization_ds.persist()
    #insufficient_mask = (pixelwise_counts_per_orbit_and_polarization_ds >= (min_monthly_acquisitions*(consec_snow_days_da/30))) & (max_days_gap_per_orbit_da <= max_allowed_days_gap_per_orbit) & (pixelwise_counts_per_orbit_and_polarization_ds>0)
    insufficient_mask = (pixelwise_counts_per_orbit_and_polarization_ds >= (min_monthly_acquisitions*(modified_consec_snow_days_da/30))) & (max_days_gap_per_orbit_da <= max_allowed_days_gap_per_orbit) & (pixelwise_counts_per_orbit_and_polarization_ds>0)
    # CHANGED THIS -- MULTIPLY BY 2 TO ACCOUNT FOR HALFING SNOW MAX TEMPORAL SEARCH WINDOW

    constituent_runoff_onsets_ds = backscatter_min_timing_per_orbit_and_polarization_ds.where(insufficient_mask)
    constituent_runoff_onsets_da = constituent_runoff_onsets_ds.to_dataarray(dim="polarization")

    if not report_temporal_res:
        return constituent_runoff_onsets_da
    else:
        #temporal_resolution_da = consec_snow_days_da / (pixelwise_counts_per_orbit_and_polarization_ds.where(insufficient_mask)['vv'].sum(dim='sat:relative_orbit').where(lambda x: x>0))
        temporal_resolution_da = modified_consec_snow_days_da / (pixelwise_counts_per_orbit_and_polarization_ds.where(insufficient_mask)['vv'].sum(dim='sat:relative_orbit').where(lambda x: x>0))
        # CHANGED THIS -- MULTIPLY BY 2 TO ACCOUNT FOR HALFING SNOW MAX TEMPORAL SEARCH WINDOW
        temporal_resolution = temporal_resolution_da.mean(dim=['latitude','longitude'],skipna=True)
        pixel_count = temporal_resolution_da.count(dim=['latitude','longitude'])
        return constituent_runoff_onsets_da, temporal_resolution, pixel_count


def calculate_runoff_onset_from_constituent_runoff_onsets(constituent_runoff_onsets_da: xr.DataArray,):
    print("Calculating runoff onset from constituent runoff onsets...")
    runoff_onset_da = (
        constituent_runoff_onsets_da.astype("int64")
        .where(lambda x: x > 0)
        .median(dim=["sat:relative_orbit", "polarization"], skipna=True)
        .astype("datetime64[ns]")
    )  #
    return runoff_onset_da


def calculate_runoff_onset_wrapper(ds, consec_snow_days_da, min_monthly_acquisitions, max_allowed_days_gap_per_orbit, returned_dates_format, return_constituent_runoff_onsets, report_temporal_res, tile):
    
    water_year = ds.water_year.values[0]

    print(f'calculating for WY {water_year}...')

    if water_year not in consec_snow_days_da.water_year:
        print(f"Warning: water_year {water_year} not found in consec_snow_days_da")
        consec_snow_days_slice = consec_snow_days_da.sel(water_year=water_year, method='nearest').where(False,other=9999) # if water year does not exist in the consec_snow_days_da, set to 9999 so no values are calculated in calculate_runoff_onset...
    else:
        consec_snow_days_slice = consec_snow_days_da.sel(water_year=water_year)
    
    if report_temporal_res:
        runoff_onset_da, temporal_resolution, pixel_count = calculate_runoff_onset( #, temporal_resolution, pixel_count
            ds,
            consec_snow_days_da=consec_snow_days_slice,
            min_monthly_acquisitions=min_monthly_acquisitions,
            max_allowed_days_gap_per_orbit=max_allowed_days_gap_per_orbit,
            returned_dates_format=returned_dates_format,
            return_constituent_runoff_onsets=return_constituent_runoff_onsets,
            report_temporal_res=report_temporal_res,
        )

        setattr(tile, f'tr_{water_year}', round(float(temporal_resolution),3))
        setattr(tile, f'pix_ct_{water_year}', int(pixel_count))
    else:
        runoff_onset_da = calculate_runoff_onset( #, temporal_resolution, pixel_count
            ds,
            consec_snow_days_da=consec_snow_days_slice,
            min_monthly_acquisitions=min_monthly_acquisitions,
            max_allowed_days_gap_per_orbit=max_allowed_days_gap_per_orbit,
            returned_dates_format=returned_dates_format,
            return_constituent_runoff_onsets=return_constituent_runoff_onsets,
            report_temporal_res=report_temporal_res,
        )
    
    return runoff_onset_da


def dataarrays_to_dataset(runoff_onsets_da, median_da, mad_da, water_years):

    runoff_onsets_ds = runoff_onsets_da.to_dataset(name='runoff_onset').round().astype('uint16')
    runoff_onsets_ds = runoff_onsets_ds.reindex(water_year=water_years)
    runoff_onsets_ds['runoff_onset_median'] = median_da.round().astype('uint16')
    runoff_onsets_ds['runoff_onset_mad'] = mad_da
    
    return runoff_onsets_ds

def median_and_std_with_min_obs(da, dim, min_count):
    count_mask = da.notnull().sum(dim=dim) >= min_count
    median = da.where(count_mask).median(dim=dim)
    std = da.where((count_mask) & (median>0)).std(dim=dim)
    
    return median, std


def median_and_mad_with_min_obs(da, dim, min_count):
    count_mask = da.notnull().sum(dim=dim) >= min_count
    median = da.where(count_mask).median(dim=dim)
    abs_dev = np.abs(da - median)
    mad = abs_dev.where(count_mask).median(dim=dim)

    return median, mad




# batch_size = 5
# tile_batches = [tiles[i:i + batch_size] for i in range(0, len(tiles), batch_size)]


# for tile_batch in tqdm.tqdm(tile_batches, total=len(tile_batches)):

#     batch_results = [dask.delayed(process_tile)(tile) for tile in tile_batch]

#     try:    
#         computed_results = dask.compute(*batch_results)
#     except:
#         computed_results = tile_batch
    

#     df = pd.DataFrame(
#         [[getattr(r, f) for f in fields] for r in computed_results if r is not None],
#         columns=fields,
#     )

#     df.to_csv(config.tile_results_path, mode='a', header=False, index=False) # header=True if starting over
    
#     client.restart()   
# 
# 
# # batch_size = 5
# tile_batches = [tiles[i:i + batch_size] for i in range(0, len(tiles), batch_size)]


# for tile_batch in tqdm.tqdm(tile_batches, total=len(tile_batches)):

#     futures = []
#     for tile in tile_batch:
#         futures.append(client.submit(process_tile, tile, retries=0))


#     try:
#         dask.distributed.wait(futures, timeout=1000, return_when="ALL_COMPLETED")
#     except Exception as e:
#         print(e)
#         print('Error waiting for futures to complete')
#         for future in futures:
#             if future.status != 'finished':
#                 future.cancel()

#     computed_results = client.gather(futures,errors='skip')

#     for tile in tile_batch:
#         if tile.index not in [computed_tile.index for computed_tile in computed_results]:
#             computed_results.append(tile)
    

#     df = pd.DataFrame(
#         [[getattr(r, f) for f in fields] for r in computed_results if r is not None],
#         columns=fields,
#     )

#     df.to_csv(config.tile_results_path, mode='a', header=False, index=False) # header=True if starting over
    
#     client.restart()
# 
# 
# # from dask.distributed import as_completed

# fields = ("row","col","percent_valid_snow_pixels","s1_rtc_ds_dims","runoff_onsets_dims",
#             "tr_2015", "tr_2016", "tr_2017", "tr_2018", "tr_2019", "tr_2020", "tr_2021", "tr_2022", "tr_2023","tr_2024",
#             "pix_ct_2015","pix_ct_2016","pix_ct_2017","pix_ct_2018","pix_ct_2019","pix_ct_2020","pix_ct_2021","pix_ct_2022","pix_ct_2023","pix_ct_2024",
#             "start_time","total_time","success","error_messages")

# futures = [client.submit(process_tile, tile, retries=0) for tile in tiles]

# with tqdm.tqdm(total=len(tiles)) as pbar:
#     for future in as_completed(futures):
#         try:
#             result = future.result()
#             df = pd.DataFrame([[getattr(result, f) for f in fields]], columns=fields)
#         except Exception as e:
#             # Handle the error, possibly by adding the original tile to the results
#             error_tile = next(tile for tile in tiles if tile.index == future.key.split('-')[1])
#             error_tile.error_messages = str(e)
#             df = pd.DataFrame([[getattr(error_tile, f) for f in fields]], columns=fields)

#         df.to_csv('tile_results.csv', mode='a', header=not header_written, index=False)
#         header_written = True
#         pbar.update(1)

#     client.restart()
# 
# 
# # global_store = adlfs.AzureBlobFileSystem(account_name="snowmelt", credential=sas_token).get_mapper("snowmelt/snowmelt_runoff_onset/global.zarr")
# global_ds = xr.open_zarr(global_store, consolidated=True,decode_coords='all')
# 
# 
# # tile = Tile(16,204)
# tile = Tile(16,203)
# futures = []
# futures.append(client.submit(process_tile, tile, retries=1))
# futures
# computed_results = client.gather(futures,errors='skip')
# computed_results[0].error_messages
# computed_results[0].runoff_onsets_dims
# test_ds = global_ds.rio.clip_box(*tile.get_geobox().boundingbox,crs='EPSG:4326')
# test_ds
# test_ds['runoff_onset'].sum(dim=['latitude','longitude']).values
# view_tile(tile)

#global_ds['runoff_onset_median'].odc.explore()

# seasonal_snow_mask_clip_ds = get_custom_seasonal_snow_mask(bbox_gdf)
# seasonal_snow_mask_matched_ds = seasonal_snow_mask_clip_ds.rio.reproject_match(s1_rtc_ds.isel(time=0)).rename({'x':'longitude','y':'latitude'})
# #seasonal_snow_mask_matched_ds = seasonal_snow_mask_clip_ds.odc.reproject(geobox).persist()#.rename({'x':'longitude','y':'latitude'})
# seasonal_snow_mask_matched_ds

# gmba_clipped_gdf = get_gmba_mountain_inventory(bbox_gdf)
# s1_rtc_ds = s1_rtc_ds.rio.clip(gmba_clipped_gdf.geometry)
# s1_rtc_ds

# s1_rtc_ds = get_sentinel1_rtc(geobox)
# s1_rtc_ds

# tiles = [Tile(row,col) for row,col in zip(valid_tiles_gdf.row,valid_tiles_gdf.col)]
# tile = tiles[6]
# geobox = tile.get_geobox()
# geobox.explore()
# bbox = geobox.boundingbox
# bbox_geometry = shapely.geometry.box(bbox.left, bbox.bottom, bbox.right, bbox.top)
# bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geometry], crs=geobox.crs)


# s1_rtc_masked_ds = apply_seasonal_snow_spatial_and_temporal_mask(s1_rtc_ds, seasonal_snow_mask_matched_ds)
# s1_rtc_masked_ds
# s1_rtc_masked_vv_WY2019_ds = s1_rtc_masked_ds['vv'].sel(time=slice('2018-10-01','2019-09-30')).compute()
# s1_rtc_masked_vv_WY2019_ds
#s1_rtc_masked_vv_WY2019_ds.plot.imshow(col='time',col_wrap=6,vmin=0,vmax=1)
#s1_rtc_masked_vv_WY2019_ds.where(lambda x:x>0.001).plot.imshow(col='time',col_wrap=6,vmin=0,vmax=1)


# runoff_onsets = (
# s1_rtc_masked_ds.groupby("water_year")
# .apply(
#     calculate_runoff_onset_wrapper,
#     consec_snow_days_da=seasonal_snow_mask_matched_ds['max_consec_snow_days'],
#     min_monthly_acquisitions=1, #one or two
#     returned_dates_format="dowy",
#     return_constituent_runoff_onsets=True,
#     low_backscatter_threshold=0.001#0.001
# ))
#runoff_onsets

#getsizeof(futures[0].result())






# runoff_onsets
# runoff_onsets_computed = runoff_onsets.to_dataset(name='runoff_onset').persist()# add .persist here?
# runoff_onsets_computed
# runoff_onsets_computed['runoff_onset_median'] = median_with_min_obs(runoff_onsets_computed['runoff_onset'], 'water_year', 3)
# runoff_onsets_computed
# runoff_onsets_computed['runoff_onset'].plot.imshow(col='water_year',col_wrap=3)
# f,ax=plt.subplots(figsize=(12,12))
# runoff_onsets_computed['runoff_onset_median'].plot.imshow(ax=ax)
# global_store = adlfs.AzureBlobFileSystem(account_name="snowmelt", credential=sas_token).get_mapper("snowmelt/snowmelt_runoff_onset/global.zarr")
# global_ds = xr.open_zarr(global_store, consolidated=True) #consolidated=False if processed tiles not showing up
# global_ds
# global_subset = global_ds.sel(latitude=runoff_onsets_computed.latitude,longitude=runoff_onsets_computed.longitude,method='nearest')
# global_subset
# runoff_onsets_computed_reindexed = runoff_onsets_computed.round().astype('uint16').assign_coords(latitude=global_subset.latitude,longitude=global_subset.longitude)
# runoff_onsets_computed_reindexed
# runoff_onsets_computed_reindexed.drop_vars('spatial_ref').chunk({"longitude": 2048, "latitude": 2048})#.to_zarr(global_store, region="auto", mode="r+", consolidated=True)
# cluster.scale(1)


# #testing equator
# # 
# # client.restart()

# test_tiles_gdf = valid_tiles_gdf[(valid_tiles_gdf['row'] == 54) | (valid_tiles_gdf['row'] == 55)]

# test_tiles = [Tile(row,col) for row,col in zip(test_tiles_gdf.row,test_tiles_gdf.col)]

# futures = []

# for tile in test_tiles:
#     future = client.submit(process_tile,tile)
#     #future = process_tile.submit(tile)
#     futures.append(future)

#     test_tiles_gdf.explore()

#     results = [f.result() for f in futures]

# for result in results:
#     print(f'tile {result.index} success: {result.success}, time: {result.total_time}, errors: {result.error_messages}, s1_rtc_ds_dims: {result.s1_rtc_ds_dims}, s1_rtc_masked_ds_dims: {result.s1_rtc_masked_ds_dims}, runoff_onsets_dims: {result.runoff_onsets_dims}')

# global_store = adlfs.AzureBlobFileSystem(account_name="snowmelt", credential=sas_token).get_mapper("snowmelt/snowmelt_runoff_onset/global.zarr")
# global_ds = xr.open_zarr(global_store, consolidated=True,decode_coords='all')


# def view_tile(tile: Tile, global_ds: xr.Dataset):


#     test_ds = global_ds.rio.clip_box(*tile.get_geobox().boundingbox,crs='EPSG:4326')

#     f,ax=plt.subplots(1,1,figsize=(10,10))
#     test_ds['runoff_onset_median'].plot.imshow(ax=ax)

#     test_ds['runoff_onset'].plot.imshow(col='water_year',col_wrap=3)


# #@coiled.function(cpu=4, memory='32 GB', spot_policy="spot", region="westeurope", environ={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}, keepalive="5m", workspace="azure", threads_per_worker=-1)
# #@coiled.function(cpu=4, spot_policy="spot", region="westeurope", environ={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}, keepalive="5m", workspace="azure")
# #, name=f"process_tile_batch_{batch_number}"
# #@dask.delayed#, threads_per_worker=-1
# #odc.stac.configure_rio(cloud_defaults=True)
# #@coiled.function(n_workers=80, cpu=4, memory='32 GB', spot_policy="spot", region="westeurope", environ={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}, keepalive="5m", workspace="azure", name='mem_test')
# def process_tile(tile : Tile):


#     tile.start_time = time.time()

#     geobox = tile.geobox
#     bbox_gdf = tile.bbox_gdf
    
#     #odc.stac.configure_rio(cloud_defaults=True)

#     try:


#         #s1_rtc_ds = get_sentinel1_rtc(geobox)
#         s1_rtc_ds = dask.delayed(get_sentinel1_rtc)(geobox)


#         #tile.s1_rtc_ds = s1_rtc_ds
        
#         #tile.s1_rtc_ds_dims = dict(s1_rtc_ds.sizes)

#         #seasonal_snow_mask_matched_ds = get_custom_seasonal_snow_mask(s1_rtc_ds,bbox_gdf)
#         seasonal_snow_mask_matched_ds = dask.delayed(get_custom_seasonal_snow_mask)(s1_rtc_ds,bbox_gdf)

#         #gmba_clipped_gdf = get_gmba_mountain_inventory(bbox_gdf)
#         gmba_clipped_gdf = dask.delayed(get_gmba_mountain_inventory)(bbox_gdf)


#         #s1_rtc_masked_ds = apply_all_masks(s1_rtc_ds,gmba_clipped_gdf,seasonal_snow_mask_matched_ds)
#         s1_rtc_masked_ds = dask.delayed(apply_all_masks)(s1_rtc_ds,gmba_clipped_gdf,seasonal_snow_mask_matched_ds)

        
#         #tile.s1_rtc_masked_ds_dims = dict(s1_rtc_masked_ds.sizes)

#         runoff_onsets_da = (
#         s1_rtc_masked_ds.groupby("water_year")
#         .apply(
#             calculate_runoff_onset_wrapper,
#             consec_snow_days_da=seasonal_snow_mask_matched_ds['max_consec_snow_days'],
#             min_monthly_acquisitions=2, #one or two
#             returned_dates_format="dowy",
#             return_constituent_runoff_onsets=False,
#             low_backscatter_threshold=0.001,
#         ))
        
#         #tile.runoff_onsets = runoff_onsets
        
#         #runoff_onsets_ds = runoff_onsets_da.to_dataset(name='runoff_onset')# add .persist here?


#         median_da, std_da = median_and_std_with_min_obs(runoff_onsets_da, 'water_year', 5)

#         runoff_onsets_ds = dask.delayed(dataarrays_to_dataset)(runoff_onsets_da, median_da, std_da)

#         #runoff_onsets_ds['runoff_onset_median'] = median_da
        
#         #runoff_onsets_ds = runoff_onsets_ds.round().astype('uint16')

#         #runoff_onsets_ds['runoff_onset_std'] = std_da.astype('float32')

#         #tile.runoff_onsets_dims = dict(runoff_onsets_ds.sizes)

#         #with dask.config.set(pool=ThreadPoolExecutor(16), scheduler="threads"):
#         #    runoff_onsets_computed = runoff_onsets_computed.compute() # use compute instead, then if stilll sloe remove thread pool saturation
        
#         #del s1_rtc_ds, seasonal_snow_mask_clip_ds, seasonal_snow_mask_matched_ds, s1_rtc_masked_ds, runoff_onsets #gmba_clipped_gdf,
#         #gc.collect()

#         global_subset_ds = global_ds.sel(latitude=runoff_onsets_ds.latitude,longitude=runoff_onsets_ds.longitude,method='nearest')

#         runoff_onsets_reindexed_ds = runoff_onsets_ds.assign_coords(latitude=global_subset_ds.latitude,longitude=global_subset_ds.longitude)
        
#         #with dask.config.set(pool=ThreadPoolExecutor(16), scheduler="threads"):
#         runoff_onsets_reindexed_ds.drop_vars('spatial_ref').chunk({"longitude": 2048, "latitude": 2048}).to_zarr(global_store, region="auto", mode="r+", consolidated=True, compute=True)

#         #del runoff_onsets_computed, global_store, global_ds, global_subset, runoff_onsets_computed_reindexed
#         #gc.collect()

#         #del seasonal_snow_mask_matched_ds

#         tile.total_time = time.time() - tile.start_time
#         tile.success = True

#     except Exception as e:
#         #gc.collect()
#         tile.error_messages.append(str(e))
#         tile.error_messages.append(traceback.format_exc())
#         tile.total_time = time.time() - tile.start_time
#         tile.success = False

#     return tile



# def apply_all_masks(s1_rtc_ds,bbox_gdf):

#     s1_rtc_ds = remove_unwanted_water_years(s1_rtc_ds)

#     center_lat = (s1_rtc_ds.rio.bounds()[1]+s1_rtc_ds.rio.bounds()[3])/2

#     if np.absolute(center_lat) < 3:
#         s1_rtc_ds = remove_equator_crossing(s1_rtc_ds)
        
#     gmba_clipped_gdf = get_gmba_mountain_inventory(bbox_gdf)
#     s1_rtc_ds = s1_rtc_ds.rio.clip(gmba_clipped_gdf.geometry)
    
#     seasonal_snow_mask_clip_ds = get_custom_seasonal_snow_mask(bbox_gdf)
#     seasonal_snow_mask_matched_ds = seasonal_snow_mask_clip_ds.rio.reproject_match(s1_rtc_ds.isel(time=0)).rename({'x':'longitude','y':'latitude'})
#     #seasonal_snow_mask_matched_ds = seasonal_snow_mask_clip_ds.odc.reproject(geobox)


#     s1_rtc_masked_ds = apply_seasonal_snow_spatial_and_temporal_mask(s1_rtc_ds, seasonal_snow_mask_matched_ds)
    
#     return s1_rtc_masked_ds, seasonal_snow_mask_matched_ds

# def remove_unwanted_water_years(s1_rtc_ds):
#     s1_rtc_ds = s1_rtc_ds.sel(time=s1_rtc_ds.water_year.isin(water_years))
#     return s1_rtc_ds


# def remove_equator_crossing(s1_rtc_ds):
#     if s1_rtc_ds.attrs['hemisphere'] == 'northern':
#         mask = s1_rtc_ds.latitude >= 0
#     else:
#         mask = s1_rtc_ds.latitude < 0

#     s1_rtc_ds = s1_rtc_ds.where(mask)
#     return s1_rtc_ds

# def get_gmba_mountain_inventory(bbox_gdf):
#     url = (f"https://data.earthenv.org/mountains/standard/GMBA_Inventory_v2.0_standard_300.zip")
#     gmba_gdf = gpd.read_file("zip+" + url)
#     return gpd.clip(gmba_gdf, bbox_gdf)

# def get_custom_seasonal_snow_mask(bbox_gdf):
#     #xmin, ymin, xmax, ymax = bbox_gdf.total_bounds
#     mask_store = adlfs.AzureBlobFileSystem(account_name="snowmelt", credential=sas_token).get_mapper("snowmelt/snow_mask/global_modis_snow_mask.zarr")
#     seasonal_snow_mask = xr.open_zarr(mask_store, consolidated=True, decode_coords='all') 
#     seasonal_snow_mask_clip = seasonal_snow_mask.rio.clip_box(*bbox_gdf.total_bounds,crs='EPSG:4326') # clip to correct box, maybe use total_bounds and then use crs 
#     return seasonal_snow_mask_clip


# def apply_seasonal_snow_spatial_and_temporal_mask(s1_rtc_ds, seasonal_snow_mask_matched_ds):
#     s1_rtc_masked_ds = s1_rtc_ds.groupby('water_year').map(lambda group: apply_mask_for_year(group, seasonal_snow_mask_matched_ds))
#     s1_rtc_masked_ds.rio.write_crs(s1_rtc_ds.rio.crs,inplace=True)
#     return s1_rtc_masked_ds

# def apply_mask_for_year(group, seasonal_snow_mask_matched_ds):

#     year = group.water_year.values[0]


#     if year not in seasonal_snow_mask_matched_ds.water_year:
#         print(f"Warning: water_year {year} not found in seasonal_snow_mask_matched_ds")
#         return group.where(False) 

#     sad_mask = group['DOWY'] >= seasonal_snow_mask_matched_ds['SAD_DOWY'].sel(water_year=year)
#     sdd_mask = group['DOWY'] <= seasonal_snow_mask_matched_ds['SDD_DOWY'].sel(water_year=year)
#     consec_mask = seasonal_snow_mask_matched_ds['max_consec_snow_days'].sel(water_year=year) >= 56
#     combined_mask = sad_mask & sdd_mask & consec_mask
#     return group.where(combined_mask)




# def xr_datetime_to_DOWY_map_blocks(date_da, hemisphere="northern"):
#     """
#     Converts an xarray DataArray containing datetime objects to the Day of Water Year (DOWY).

#     Parameters:
#     date (xr.DataArray): An xarray DataArray with datetime64 data type.
#     hemisphere (str): 'northern' or 'southern'

#     Returns:
#     xr.DataArray: An xarray DataArray containing the DOWY for each datetime in the input DataArray.
#     """

#     # Determine any valid date
#     if date_da.attrs.get("any_valid_date") is not None:
#         any_valid_date = pd.to_datetime(date_da.attrs["any_valid_date"])
#     else:
#         any_valid_date = pd.to_datetime(date_da.sel(x=0, y=0, method="nearest").values)

#     # Calculate the start of the water year
#     start_of_water_year = easysnowdata.utils.get_water_year_start(
#         any_valid_date, hemisphere=hemisphere
#     )

#     # Define the function to calculate DOWY for a block
#     def calculate_dowy_block(block, start_of_water_year):
#         # Calculate DOWY for the block
#         dowy_block = (block - np.datetime64(start_of_water_year)).astype('timedelta64[D]').astype(int) + 1
#         return dowy_block

#     # Apply the function using map_blocks
#     return date_da.map_blocks(
#         calculate_dowy_block,
#         args=(start_of_water_year,),
#         template=date_da.astype(int)
#     )

# tile = tiles[0]
# geobox = tile.geobox
# bbox_gdf = tile.bbox_gdf

# s1_rtc_ds = get_sentinel1_rtc(geobox)
# s1_rtc_ds = dask.delayed(get_sentinel1_rtc)(geobox)
# s1_rtc_ds


# seasonal_snow_mask_matched_ds = dask.delayed(get_custom_seasonal_snow_mask)(s1_rtc_ds,bbox_gdf)
# seasonal_snow_mask_matched_ds


# gmba_clipped_gdf = dask.delayed(get_gmba_mountain_inventory)(bbox_gdf)
# gmba_clipped_gdf


# s1_rtc_masked_ds = dask.delayed(apply_all_masks)(s1_rtc_ds,gmba_clipped_gdf,seasonal_snow_mask_matched_ds)
# s1_rtc_masked_ds


# runoff_onsets_da = (
# s1_rtc_masked_ds.groupby("water_year")
# .apply(
# calculate_runoff_onset_wrapper,
# consec_snow_days_da=seasonal_snow_mask_matched_ds['max_consec_snow_days'],
# min_monthly_acquisitions=2, #one or two
# returned_dates_format="doy",
# return_constituent_runoff_onsets=False,
# low_backscatter_threshold=0.001,
# ))

# runoff_onsets_da


# median_da, std_da = median_and_std_with_min_obs(runoff_onsets_da, 'water_year', 5)

# def dataarrays_to_dataset(runoff_onsets_da, median_da, std_da):

#     runoff_onsets_ds = runoff_onsets_da.to_dataset(name='runoff_onset').round().astype('uint16')
#     runoff_onsets_ds['runoff_onset_median'] = median_da.round().astype('uint16')
#     runoff_onsets_ds['runoff_onset_std'] = std_da
    
#     return runoff_onsets_ds

# runoff_onsets_ds = dask.delayed(dataarrays_to_dataset)(runoff_onsets_da, median_da, std_da)

# runoff_onsets_ds

# runoff_onsets_computed_ds = dask.compute(runoff_onsets_ds)

# runoff_onsets_computed_ds

# global_subset_ds = global_ds.sel(latitude=runoff_onsets_ds.latitude,longitude=runoff_onsets_ds.longitude,method='nearest')
# global_subset_ds

# runoff_onsets_reindexed_ds = runoff_onsets_ds.assign_coords(latitude=global_subset_ds.latitude,longitude=global_subset_ds.longitude)
# runoff_onsets_reindexed_ds

# dask.compute(runoff_onsets_reindexed_ds.drop_vars('spatial_ref').chunk({"longitude": 2048, "latitude": 2048}).to_zarr(global_store, region="auto", mode="r+", consolidated=True))


# runoff_onsets_ds['runoff_onset_median'] = median_da

# runoff_onsets_ds = runoff_onsets_ds.round().astype('uint16')

# runoff_onsets_ds['runoff_onset_std'] = std_da.astype('float32')

# computed_results = dask.compute(*results)


# results = computed_results


# results = []

# for i,tile_batch in tqdm.tqdm(enumerate(tile_batches),total=len(tile_batches)):
#     for tile in tile_batch:
#         result = process_tile(tile)
#         results.append(result)
    
#     if i == 0:
#         break
    
    
#     for i,tile_batch in tqdm.tqdm(enumerate(tile_batches),total=len(tile_batches)):
#     futures = []
#     for tile in tile_batch:
#         future = client.submit(process_tile,tile)
#         #future = process_tile.submit(tile) # when trying serverless coiled.function
#         futures.append(future)

#     fields = ("row","col","percent_valid_snow_pixels","s1_rtc_ds_dims","runoff_onsets_dims","start_time","total_time","success","error_messages")

#     results = [f.result() for f in futures]
#     #results 

#     df = pd.DataFrame(
#         [[getattr(r, f) for f in fields] for r in results if r is not None],
#         columns=fields,
#     )
#     df.to_csv('tile_results.csv', mode='a', header=not header_written, index=False)
#     header_written = True
    
#     client.restart()
    
#     if i == 1:
#         break


# full_results_list = []

# batch_size = 10
# tile_batches = [tiles[i:i + batch_size] for i in range(0, len(tiles), batch_size)]

# process_tile_serverless = coiled.function(n_workers=11, cpu=8, memory='64 GB', spot_policy="spot", region="westeurope", environ={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}, keepalive="5m", workspace="azure", name='mem_test')(process_tile)


# for i,tile_batch in tqdm.tqdm(enumerate(tile_batches),total=len(tile_batches)):
#     futures = []
#     for tile in tile_batch:
#         #future = client.submit(process_tile,tile)
#         future = process_tile_serverless.submit(tile) # when trying serverless coiled.function
#         futures.append(future)

#     results = [f.result() for f in futures]
    
#     full_results_list.extend(results)

#     fields = ("row","col","percent_valid_snow_pixels","s1_rtc_ds_dims","runoff_onsets_dims","start_time","total_time","success","error_messages")

    
#     #results = client.gather(futures) 

#     df = pd.DataFrame(
#         [[getattr(r, f) for f in fields] for r in results if r is not None],
#         columns=fields,
#     )
#     df.to_csv('tile_results.csv', mode='a', header=not header_written, index=False)
#     header_written = True
    
    
#     if i == 1:
#         break


# def count_acquisitions_and_max_gap_per_orbit_and_polarization(s1_rtc_ds: xr.Dataset):
#     print("Calculating pixelwise counts and maximum gaps per orbit and polarization...")
#     pixelwise_counts_per_orbit_ds = s1_rtc_ds.groupby("sat:relative_orbit").count(dim="time", engine='flox')
    
#     def calc_max_gap(group):
#         times = group.time.sortby('time')
#         gaps = times.diff(dim='time').max()
#         return gaps

#     max_time_gap_per_orbit_days_da = s1_rtc_ds.groupby("sat:relative_orbit").map(calc_max_gap).dt.days
    
#     return pixelwise_counts_per_orbit_ds, max_time_gap_per_orbit_days_da


# def filter_insufficient_pixels_per_orbit_and_polarization(
#     backscatter_min_timing_per_orbit_and_polarization_ds: xr.Dataset,
#     pixelwise_counts_per_orbit_ds: xr.Dataset,
#     max_days_gap_per_orbit_da: xr.Dataset,
#     consec_snow_days_da: xr.DataArray,
#     min_monthly_acquisitions: int,
#     max_allowed_days_gap_per_orbit: int
# ):
#     print(f"Filtering insufficient pixels per orbit and polarization...")
#     constituent_runoff_onsets_ds = (
#         backscatter_min_timing_per_orbit_and_polarization_ds.where(
#             (pixelwise_counts_per_orbit_ds >= (min_monthly_acquisitions*(consec_snow_days_da/30))) &
#             (max_days_gap_per_orbit_da <= max_allowed_days_gap_per_orbit)
#         )
#     )
#     return constituent_runoff_onsets_ds


# constituent_runoff_onsets_ds = (
#     filter_insufficient_pixels_per_orbit_and_polarization(
#         backscatter_min_timing_per_orbit_and_polarization_ds,
#         pixelwise_counts_per_orbit_ds,
#         max_days_gap_per_orbit_da,
#         seasonal_snow_mask_matched_ds['max_consec_snow_days'].sel(water_year=2019),
#         min_monthly_acquisitions,
#         max_allowed_days_gap_per_orbit
#     )
# )

# constituent_runoff_onsets_da = constituent_runoff_onsets_ds.to_dataarray(dim="polarization")

# def process_tile(tile: Tile):
#     @dask.delayed
#     def _process():
#         tile.start_time = time.time()
#         geobox = tile.geobox
#         bbox_gdf = tile.bbox_gdf
        
#         try:

#             s1_rtc_ds = get_sentinel1_rtc(geobox)
#             tile.s1_rtc_ds_dims = dict(s1_rtc_ds.sizes)

#             seasonal_snow_mask_matched_ds = get_custom_seasonal_snow_mask(s1_rtc_ds, bbox_gdf)

#             gmba_clipped_gdf = get_gmba_mountain_inventory(bbox_gdf)

#             s1_rtc_masked_ds = apply_all_masks(s1_rtc_ds, gmba_clipped_gdf, seasonal_snow_mask_matched_ds)

#             runoff_onsets_da = (
#                 s1_rtc_masked_ds.groupby("water_year")
#                 .apply(
#                     calculate_runoff_onset_wrapper,
#                     consec_snow_days_da=seasonal_snow_mask_matched_ds['max_consec_snow_days'],
#                     min_monthly_acquisitions=min_monthly_acquisitions,
#                     max_allowed_days_gap_per_orbit=max_allowed_days_gap_per_orbit,
#                     returned_dates_format="dowy",
#                     return_constituent_runoff_onsets=False,
#                     low_backscatter_threshold=low_backscatter_threshold,
#                     report_temporal_res=False,
#                     tile=tile,
#                 )
#             )
            
#             tile.runoff_onsets_dims = dict(runoff_onsets_da.sizes)
#             median_da, std_da = median_and_std_with_min_obs(runoff_onsets_da, 'water_year', min_years_for_median_std)
#             runoff_onsets_ds = dataarrays_to_dataset(runoff_onsets_da, median_da, std_da)

#             global_store = get_global_runoff_store()
            
#             global_ds = xr.open_zarr(global_store, consolidated=True)
#             global_subset_ds = global_ds.sel(latitude=runoff_onsets_ds.latitude, longitude=runoff_onsets_ds.longitude, method='nearest')
            
#             runoff_onsets_reindexed_ds = runoff_onsets_ds.assign_coords(latitude=global_subset_ds.latitude, longitude=global_subset_ds.longitude)
            
#             # Write to Zarr
#             runoff_onsets_reindexed_ds.drop_vars('spatial_ref').chunk({"longitude": 2048, "latitude": 2048}).to_zarr(
#                 global_store, region="auto", mode="r+", consolidated=True
#             )
            
#             tile.total_time = time.time() - tile.start_time
#             tile.success = True
#         except Exception as e:
#             tile.error_messages.append(str(e))
#             tile.error_messages.append(traceback.format_exc())
#             tile.total_time = time.time() - tile.start_time
#             tile.success = False
        
#         return tile
    
#     return _process()



# def get_sentinel1_rtc(geobox):

#     items = (
#         pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace)
#         .search(
#             intersects=geobox.geographic_extent,
#             collections=["sentinel-1-rtc"],
#             datetime=(config.start_date, config.end_date),
#         )
#         .item_collection()
#     )

#     load_params = {
#         "items": items,
#         "bands": ["vv"],
#         "nodata": -32768,
#         "chunks": config.chunks_read, 
#         "groupby": "sat:absolute_orbit",
#         "geobox":geobox,
#         "resampling": "bilinear",
#         "fail_on_error":False
#     }


#     s1_rtc_ds = odc.stac.load(**load_params).sortby("time")

#     metadata = gpd.GeoDataFrame.from_features(items, "epsg:4326")

#     metadata_groupby_gdf = (
#         metadata.groupby(["sat:absolute_orbit"]).first().sort_values("datetime")
#     )


#     s1_rtc_ds = s1_rtc_ds.assign_coords(
#     {
#         "sat:orbit_state": ("time", metadata_groupby_gdf["sat:orbit_state"]),
#         "sat:relative_orbit": ("time", metadata_groupby_gdf["sat:relative_orbit"].astype("int16"))
#     })

#     #s1_rtc_ds = s1_rtc_ds.drop_vars(['hh','hv'],errors='ignore')

#     epsg = s1_rtc_ds.rio.estimate_utm_crs().to_epsg()
#     hemisphere = 'northern' if epsg < 32700 else 'southern'

#     s1_rtc_ds.attrs['hemisphere'] = hemisphere

#     s1_rtc_ds = s1_rtc_ds.assign_coords(
#     {
#         "water_year": ("time", pd.to_datetime(s1_rtc_ds.time).map(lambda x: easysnowdata.utils.datetime_to_WY(x, hemisphere=hemisphere))),
#         "DOWY": ("time", pd.to_datetime(s1_rtc_ds.time).map(lambda x: easysnowdata.utils.datetime_to_DOWY(x, hemisphere=hemisphere)))
#     })        

#     return s1_rtc_ds

# def apply_all_masks(s1_rtc_ds,gmba_clipped_gdf,seasonal_snow_mask_matched_ds):

#     s1_rtc_ds = remove_unwanted_water_years(s1_rtc_ds)

#     center_lat = (s1_rtc_ds.rio.bounds()[1]+s1_rtc_ds.rio.bounds()[3])/2
#     if np.absolute(center_lat) < 3:
#         s1_rtc_ds = remove_equator_crossing(s1_rtc_ds)
        
#     s1_rtc_ds = s1_rtc_ds.rio.clip_box(*gmba_clipped_gdf.total_bounds,crs=gmba_clipped_gdf.crs)
#     s1_rtc_ds = s1_rtc_ds.rio.clip(gmba_clipped_gdf.geometry,drop=True) # does this compute?

#     s1_rtc_masked_ds = apply_seasonal_snow_spatial_and_temporal_mask(s1_rtc_ds, seasonal_snow_mask_matched_ds)
    
#     return s1_rtc_masked_ds

# def remove_unwanted_water_years(s1_rtc_ds):
#     s1_rtc_ds = s1_rtc_ds.sel(time=s1_rtc_ds.water_year.isin(config.water_years))
#     return s1_rtc_ds


# def remove_equator_crossing(s1_rtc_ds):
#     if s1_rtc_ds.attrs['hemisphere'] == 'northern':
#         mask = s1_rtc_ds.latitude >= 0
#     else:
#         mask = s1_rtc_ds.latitude < 0

#     s1_rtc_ds = s1_rtc_ds.where(mask)
#     return s1_rtc_ds

# def get_gmba_mountain_inventory(bbox_gdf):
#     url = (f"https://data.earthenv.org/mountains/standard/GMBA_Inventory_v2.0_standard_300.zip")
#     gmba_gdf = gpd.read_file("zip+" + url)
#     gmba_clipped_gdf = gpd.clip(gmba_gdf, bbox_gdf)
#     return gmba_clipped_gdf

# def get_custom_seasonal_snow_mask(s1_rtc_ds,bbox_gdf,geobox):
#     seasonal_snow_mask = xr.open_zarr(config.seasonal_snow_mask_store, consolidated=True, decode_coords='all') 
#     seasonal_snow_mask_clip_ds = seasonal_snow_mask.rio.clip_box(*bbox_gdf.total_bounds,crs='EPSG:4326') # clip to correct box, maybe use total_bounds and then use crs 
#     seasonal_snow_mask_matched_ds = seasonal_snow_mask_clip_ds.rio.reproject_match(s1_rtc_ds.isel(time=slice(0,5)).max(dim='time')).rename({'x':'longitude','y':'latitude'}) # if S1 scene at t=0 isn't full, does this get rid of mask values?
#     #seasonal_snow_mask_matched_ds = seasonal_snow_mask_clip_ds.odc.reproject(geobox)#.rename({'x':'longitude','y':'latitude'})
#     return seasonal_snow_mask_matched_ds


# def apply_seasonal_snow_spatial_and_temporal_mask(s1_rtc_ds, seasonal_snow_mask_matched_ds):
#     s1_rtc_masked_ds = s1_rtc_ds.groupby('water_year').map(lambda group: apply_mask_for_year(group, seasonal_snow_mask_matched_ds))
#     s1_rtc_masked_ds.rio.write_crs(s1_rtc_ds.rio.crs,inplace=True)
#     return s1_rtc_masked_ds

# def apply_mask_for_year(group, seasonal_snow_mask_matched_ds):

#     year = group.water_year.values[0]


#     if year not in seasonal_snow_mask_matched_ds.water_year:
#         print(f"Warning: water_year {year} not found in seasonal_snow_mask_matched_ds")
#         return group.where(False) 

#     sad_mask = group['DOWY'] >= seasonal_snow_mask_matched_ds['SAD_DOWY'].sel(water_year=year)
#     sdd_mask = group['DOWY'] <= seasonal_snow_mask_matched_ds['SDD_DOWY'].sel(water_year=year)
#     consec_mask = seasonal_snow_mask_matched_ds['max_consec_snow_days'].sel(water_year=year) >= 56
#     combined_mask = sad_mask & sdd_mask & consec_mask
#     return group.where(combined_mask)

# def xr_datetime_to_DOWY(date_da, hemisphere="northern"):
#     """
#     Converts an xarray DataArray containing datetime objects to the Day of Water Year (DOWY).

#     Parameters:
#     date (xr.DataArray): An xarray DataArray with datetime64 data type.
#     hemisphere (str): 'northern' or 'southern'

#     Returns:
#     xr.DataArray: An xarray DataArray containing the DOWY for each datetime in the input DataArray.
#     """

#     if date_da.attrs.get("any_valid_date") is not None:
#         any_valid_date = pd.to_datetime(date_da.attrs["any_valid_date"])
#     else:
#         any_valid_date = pd.to_datetime(date_da.sel(x=0, y=0, method="nearest").values)

#     start_of_water_year = easysnowdata.utils.get_water_year_start(
#         any_valid_date, hemisphere=hemisphere
#     )

#     return xr.apply_ufunc(
#         lambda x: (x - start_of_water_year).days + 1,  # dt accessor?
#         date_da,
#         input_core_dims=[[]],
#         vectorize=True,
#         dask="parallelized",  # try allowed also
#         output_dtypes=[float],
#     )


# def calculate_runoff_onset(
#     s1_rtc_ds: xr.Dataset,
#     min_monthly_acquisitions: int,
#     max_allowed_days_gap_per_orbit: int,
#     consec_snow_days_da: xr.DataArray,
#     return_constituent_runoff_onsets: bool = False,
#     returned_dates_format: str = "dowy",
#     low_backscatter_threshold: float = 0.001,
#     report_temporal_res: bool = False,
# ):


#     s1_rtc_ds = remove_bad_scenes_and_border_noise(s1_rtc_ds, low_backscatter_threshold)

#     #pixelwise_counts_per_orbit_and_polarization_ds = (count_acquisitions_per_orbit_and_polarization(s1_rtc_ds))  # this should be for melt, keeping general for now to integrate modis data
#     pixelwise_counts_per_orbit_and_polarization_ds, max_days_gap_per_orbit_da = count_acquisitions_and_max_gap_per_orbit_and_polarization(s1_rtc_ds)

#     backscatter_min_timing_per_orbit_and_polarization_ds = (calculate_backscatter_min_per_orbit(s1_rtc_ds))


#     if report_temporal_res:
#         constituent_runoff_onsets_da, temporal_resolution, pixel_count = ( # , temporal_resolution, pixel_count
#             filter_insufficient_pixels_per_orbit_and_polarization(
#                 backscatter_min_timing_per_orbit_and_polarization_ds,
#                 pixelwise_counts_per_orbit_and_polarization_ds,
#                 max_days_gap_per_orbit_da,
#                 consec_snow_days_da,
#                 min_monthly_acquisitions,
#                 max_allowed_days_gap_per_orbit,
#                 report_temporal_res
#             )
#         )
#     else:
#         constituent_runoff_onsets_da = ( # , temporal_resolution, pixel_count
#             filter_insufficient_pixels_per_orbit_and_polarization(
#                 backscatter_min_timing_per_orbit_and_polarization_ds,
#                 pixelwise_counts_per_orbit_and_polarization_ds,
#                 max_days_gap_per_orbit_da,
#                 consec_snow_days_da,
#                 min_monthly_acquisitions,
#                 max_allowed_days_gap_per_orbit,
#                 report_temporal_res
#             )
#         )

#     if return_constituent_runoff_onsets == False:
#         runoff_onset_da = calculate_runoff_onset_from_constituent_runoff_onsets(constituent_runoff_onsets_da)
#     else:
#         runoff_onset_da = constituent_runoff_onsets_da


#     if returned_dates_format == "dowy":

#         hemisphere = (
#             "northern"
#             if s1_rtc_ds.rio.estimate_utm_crs().to_epsg() < 32700
#             else "southern"
#         )
#         month_start = 10 if hemisphere == "northern" else 4
#         print(
#             f"Area is in the {hemisphere} hemisphere. Water year starts in month {month_start}."
#         )
#         runoff_onset_da.attrs["any_valid_date"] = s1_rtc_ds.time[0].values
#         runoff_onset_da = xr_datetime_to_DOWY(runoff_onset_da, hemisphere=hemisphere)

#     elif returned_dates_format == "doy":
#         runoff_onset_da = runoff_onset_da.dt.dayofyear
#     elif returned_dates_format == "datetime64":
#         runoff_onset_da = runoff_onset_da
#     else:
#         raise ValueError(
#             'returned_dates_format must be either "doy", "dowy", or "datetime64".'
#         )

#     if report_temporal_res:
#         return runoff_onset_da, temporal_resolution, pixel_count  
#     else:
#         return runoff_onset_da

# def remove_bad_scenes_and_border_noise(da, threshold):
#     cutoff_date = np.datetime64('2018-03-14')
    
#     original_crs = da.rio.crs
    
#     result = xr.where(
#         da.time < cutoff_date,
#         da.where(da > threshold),
#         da.where(da > 0)
#     )
    
#     result.rio.write_crs(original_crs, inplace=True)
    
#     return result


# # def count_acquisitions_per_orbit_and_polarization(s1_rtc_ds: xr.Dataset):
# #     print("Calculating pixelwise counts per orbit and polarization...")
# #     pixelwise_counts_per_orbit_and_polarization = s1_rtc_ds.groupby(
# #         "sat:relative_orbit"
# #     ).count(dim="time", engine='flox')
# #     return pixelwise_counts_per_orbit_and_polarization

# def count_acquisitions_and_max_gap_per_orbit_and_polarization(s1_rtc_ds: xr.Dataset):
#     print("Calculating pixelwise counts and maximum gaps per orbit and polarization...")
#     pixelwise_counts_per_orbit_and_polarization_ds = s1_rtc_ds.groupby("sat:relative_orbit").count(dim="time", engine='flox') #, engine='flox'
    
#     def calc_max_gap(group):
#             times = group.time.sortby('time')
#             if len(times) == 1: # if only one scene in this group, set gap to very large number so it won't be calculated
#                 return times.count()*9999
#             gaps = times.diff(dim='time').max().dt.days
#             return gaps

#     max_time_gap_per_orbit_days_da = s1_rtc_ds.groupby("sat:relative_orbit").map(calc_max_gap)
    
#     return pixelwise_counts_per_orbit_and_polarization_ds, max_time_gap_per_orbit_days_da


# def calculate_backscatter_min_per_orbit(s1_rtc_ds: xr.Dataset):
#     print("Calculating backscatter min per orbit...")
#     backscatter_min_timing_per_orbit_and_polarization_ds = s1_rtc_ds.groupby(
#         "sat:relative_orbit"
#     ).map(lambda c: c.idxmin(dim="time")) # maybe only map if if max-min > -1dB
#     return backscatter_min_timing_per_orbit_and_polarization_ds


# # def filter_insufficient_pixels_per_orbit_and_polarization(
# #     backscatter_min_timing_per_orbit_and_polarization_ds: xr.Dataset,
# #     pixelwise_counts_per_orbit: xr.Dataset,
# #     consec_snow_days_da: xr.DataArray,
# #     min_monthly_acquisitions: int,
# # ):
    
# #     print(f"Filtering insufficient pixels per orbit and polarization, must have at least {min_monthly_acquisitions} per month...")
# #     constituent_runoff_onsets_ds = (
# #         backscatter_min_timing_per_orbit_and_polarization_ds.where(
# #             pixelwise_counts_per_orbit >= (min_monthly_acquisitions*(consec_snow_days_da/30))
# #         )
# #     )
# #     return constituent_runoff_onsets_ds

# def filter_insufficient_pixels_per_orbit_and_polarization(
#     backscatter_min_timing_per_orbit_and_polarization_ds: xr.Dataset,
#     pixelwise_counts_per_orbit_and_polarization_ds: xr.Dataset,
#     max_days_gap_per_orbit_da: xr.DataArray,
#     consec_snow_days_da: xr.DataArray,
#     min_monthly_acquisitions: int,
#     max_allowed_days_gap_per_orbit: int,
#     report_temporal_res: bool,
# ):
#     print(f"Filtering insufficient pixels per orbit and polarization...")

#     #pixelwise_counts_per_orbit_and_polarization_ds = pixelwise_counts_per_orbit_and_polarization_ds.persist()
#     insufficient_mask = (pixelwise_counts_per_orbit_and_polarization_ds >= (min_monthly_acquisitions*(consec_snow_days_da/30))) & (max_days_gap_per_orbit_da <= max_allowed_days_gap_per_orbit) & (pixelwise_counts_per_orbit_and_polarization_ds>0)
    

#     constituent_runoff_onsets_ds = backscatter_min_timing_per_orbit_and_polarization_ds.where(insufficient_mask)
#     constituent_runoff_onsets_da = constituent_runoff_onsets_ds.to_dataarray(dim="polarization")

#     if not report_temporal_res:
#         return constituent_runoff_onsets_da
#     else:
#         temporal_resolution_da = consec_snow_days_da / (pixelwise_counts_per_orbit_and_polarization_ds.where(insufficient_mask)['vv'].sum(dim='sat:relative_orbit').where(lambda x: x>0))
#         temporal_resolution = temporal_resolution_da.mean(dim=['latitude','longitude'],skipna=True)
#         pixel_count = temporal_resolution_da.count(dim=['latitude','longitude'])
#         return constituent_runoff_onsets_da, temporal_resolution, pixel_count


# def calculate_runoff_onset_from_constituent_runoff_onsets(constituent_runoff_onsets_da: xr.DataArray,):
#     print("Calculating runoff onset from constituent runoff onsets...")
#     runoff_onset_da = (
#         constituent_runoff_onsets_da.astype("int64")
#         .where(lambda x: x > 0)
#         .median(dim=["sat:relative_orbit", "polarization"], skipna=True)
#         .astype("datetime64[ns]")
#     )  #
#     return runoff_onset_da


# def calculate_runoff_onset_wrapper(ds, consec_snow_days_da, min_monthly_acquisitions, max_allowed_days_gap_per_orbit, returned_dates_format, return_constituent_runoff_onsets, low_backscatter_threshold, report_temporal_res, tile):
    
#     water_year = ds.water_year.values[0]

#     print(f'calculating for WY {water_year}...')

#     if water_year not in consec_snow_days_da.water_year:
#         print(f"Warning: water_year {water_year} not found in consec_snow_days_da")
#         consec_snow_days_slice = consec_snow_days_da.sel(water_year=water_year, method='nearest').where(False,other=9999) # if water year does not exist in the consec_snow_days_da, set to 9999 so no values are calculated in calculate_runoff_onset...
#     else:
#         consec_snow_days_slice = consec_snow_days_da.sel(water_year=water_year)
    
#     if report_temporal_res:
#         runoff_onset_da, temporal_resolution, pixel_count = calculate_runoff_onset( #, temporal_resolution, pixel_count
#             ds,
#             consec_snow_days_da=consec_snow_days_slice,
#             min_monthly_acquisitions=min_monthly_acquisitions,
#             max_allowed_days_gap_per_orbit=max_allowed_days_gap_per_orbit,
#             returned_dates_format=returned_dates_format,
#             return_constituent_runoff_onsets=return_constituent_runoff_onsets,
#             low_backscatter_threshold=low_backscatter_threshold,
#             report_temporal_res=report_temporal_res,
#         )

#         setattr(tile, f'tr_{water_year}', round(float(temporal_resolution),3))
#         setattr(tile, f'pix_ct_{water_year}', int(pixel_count))
#     else:
#         runoff_onset_da = calculate_runoff_onset( #, temporal_resolution, pixel_count
#             ds,
#             consec_snow_days_da=consec_snow_days_slice,
#             min_monthly_acquisitions=min_monthly_acquisitions,
#             max_allowed_days_gap_per_orbit=max_allowed_days_gap_per_orbit,
#             returned_dates_format=returned_dates_format,
#             return_constituent_runoff_onsets=return_constituent_runoff_onsets,
#             low_backscatter_threshold=low_backscatter_threshold,
#             report_temporal_res=report_temporal_res,
#         )
    
#     return runoff_onset_da

# def dataarrays_to_dataset(runoff_onsets_da, median_da, mad_da):

#     runoff_onsets_ds = runoff_onsets_da.to_dataset(name='runoff_onset').round().astype('uint16')
#     runoff_onsets_ds = runoff_onsets_ds.reindex(water_year=config.water_years)
#     runoff_onsets_ds['runoff_onset_median'] = median_da.round().astype('uint16')
#     runoff_onsets_ds['runoff_onset_mad'] = mad_da
    
#     return runoff_onsets_ds

# def median_and_std_with_min_obs(da, dim, min_count):
#     count_mask = da.notnull().sum(dim=dim) >= min_count
#     median = da.where(count_mask).median(dim=dim)
#     std = da.where((count_mask) & (median>0)).std(dim=dim)
    
#     return median, std


# def median_and_mad_with_min_obs(da, dim, min_count):
#     count_mask = da.notnull().sum(dim=dim) >= min_count
#     median = da.where(count_mask).median(dim=dim)
#     abs_dev = np.abs(da - median)
#     mad = abs_dev.where(count_mask).median(dim=dim)

#     return median, mad

