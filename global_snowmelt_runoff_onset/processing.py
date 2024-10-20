














# def get_sentinel1_rtc(geobox):

#     chunks_read = {"x": 2048, "y": 2048, "time": 1}

#     items = (
#         pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace)
#         .search(
#             intersects=geobox.geographic_extent,
#             collections=["sentinel-1-rtc"],
#             datetime=(start_date, end_date),
#         )
#         .item_collection()
#     )

#     load_params = {
#         "items": items,
#         "nodata": -32768,
#         "chunks": chunks_read,
#         "groupby": "sat:absolute_orbit",
#         "geobox":geobox,
#         "resampling": "bilinear",
#         #"fail_on_error":False
#     }


#     s1_rtc_ds = odc.stac.load(**load_params).sortby("time")#.chunk(chunks_compute) # rechunk?

#     metadata = gpd.GeoDataFrame.from_features(items, "epsg:4326")

#     metadata_groupby_gdf = (
#         metadata.groupby(["sat:absolute_orbit"]).first().sort_values("datetime")
#     )


#     s1_rtc_ds = s1_rtc_ds.assign_coords(
#     {
#         "sat:orbit_state": ("time", metadata_groupby_gdf["sat:orbit_state"]),
#         "sat:relative_orbit": ("time", metadata_groupby_gdf["sat:relative_orbit"].astype("int16"))
#     })

#     s1_rtc_ds = s1_rtc_ds.drop_vars(['hh','hv'],errors='ignore')

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
        
#     s1_rtc_ds = s1_rtc_ds.rio.clip(gmba_clipped_gdf.geometry) # does this compute?

#     s1_rtc_masked_ds = apply_seasonal_snow_spatial_and_temporal_mask(s1_rtc_ds, seasonal_snow_mask_matched_ds)
    
#     return s1_rtc_masked_ds

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
#     gmba_clipped_gdf = gpd.clip(gmba_gdf, bbox_gdf)
#     return gmba_clipped_gdf

# def get_custom_seasonal_snow_mask(s1_rtc_ds,bbox_gdf):
#     mask_store = adlfs.AzureBlobFileSystem(account_name="snowmelt", credential=sas_token).get_mapper("snowmelt/snow_mask_v2/global_modis_snow_mask.zarr")
#     seasonal_snow_mask = xr.open_zarr(mask_store, consolidated=True, decode_coords='all') 
#     seasonal_snow_mask_clip_ds = seasonal_snow_mask.rio.clip_box(*bbox_gdf.total_bounds,crs='EPSG:4326') # clip to correct box, maybe use total_bounds and then use crs 
#     seasonal_snow_mask_matched_ds = seasonal_snow_mask_clip_ds.rio.reproject_match(s1_rtc_ds.isel(time=0)).rename({'x':'longitude','y':'latitude'})
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
#     pixelwise_counts_per_orbit_and_polarization_ds = s1_rtc_ds.groupby("sat:relative_orbit").count(dim="time", engine='flox')
    
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

#     pixelwise_counts_per_orbit_and_polarization_ds = pixelwise_counts_per_orbit_and_polarization_ds.persist()
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

# def dataarrays_to_dataset(runoff_onsets_da, median_da, std_da):

#     runoff_onsets_ds = runoff_onsets_da.to_dataset(name='runoff_onset').round().astype('uint16')
#     runoff_onsets_ds = runoff_onsets_ds.reindex(water_year=water_years)
#     runoff_onsets_ds['runoff_onset_median'] = median_da.round().astype('uint16')
#     runoff_onsets_ds['runoff_onset_std'] = std_da
    
#     return runoff_onsets_ds

# def median_and_std_with_min_obs(da, dim, min_count):
#     count_mask = da.notnull().sum(dim=dim) >= min_count
    
#     median = da.where(count_mask).median(dim=dim)
#     std = da.where((count_mask) & (median>0)).std(dim=dim)
    
#     return median, std