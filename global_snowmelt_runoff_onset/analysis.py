import numpy as np
import pandas as pd
import xarray as xr
import pystac_client
import xdem
import easysnowdata
import rasterio
import odc.stac
import planetary_computer
import logging
import traceback
import ee
import dask
import dask.array as da
import dask.dataframe as dd


# Configure logging
logging.basicConfig(filename='analysis.log',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_coordinate_arrays(ds):
    """
    Create additional arrays to store original lat/lon coordinates.
    """
    lat, lon = np.meshgrid(ds.latitude, ds.longitude)
    ds['original_lat'] = xr.DataArray(lat.T, dims=('latitude', 'longitude'))
    ds['original_lon'] = xr.DataArray(lon.T, dims=('latitude', 'longitude'))
    return ds


def convert_water_year_dim_to_var(ds):
    for year in ds.water_year.values:
        ds[f'runoff_onset_WY{year}'] = ds['runoff_onset'].sel(water_year=year)

    ds = ds.drop_vars('runoff_onset').drop_vars('water_year')
    return ds

def add_topography(tile,ds):
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=[f"cop-dem-glo-30"],intersects=tile.geobox.geographic_extent)
    dem_da = odc.stac.load(items=search.items(),like=ds,chunks={},resampling='bilinear')['data'].squeeze()
    dem_da = dem_da.rio.write_nodata(-32767,encoded=True).drop_vars('time') # compute for xdem stuff

    ds['dem'] = dem_da.compute()

    # [xDEM](https://xdem.readthedocs.io/en/stable/index.html) to calculate slope and aspect and topographic position index

    attributes = xdem.terrain.get_terrain_attribute(
        ds['dem'],
        resolution=ds['dem'].rio.resolution()[0],
        attribute=["aspect", "slope", "topographic_position_index"],
    )

    ds['aspect'] = xr.DataArray(attributes[0], dims=ds['dem'].dims, coords=ds['dem'].coords)
    ds['slope'] = xr.DataArray(attributes[1], dims=ds['dem'].dims, coords=ds['dem'].coords)
    # TPI? https://xdem.readthedocs.io/en/stable/gen_modules/xdem.DEM.topographic_position_index.html, https://tc.copernicus.org/articles/8/1989/2014/tc-8-1989-2014.pdf
    # maybe incorrect radius...
    ds['tpi'] = xr.DataArray(attributes[2], dims=ds['dem'].dims, coords=ds['dem'].coords)

    # DAH?
    # alpha_max = 202.5 #only in northern hemisphere at specific latitude?
    # DAH_da = np.cos(np.deg2rad(alpha_max-aspect_da))*np.arctan(np.deg2rad(slope_da))

    return ds

def add_chili(tile,ds):

    chili_da = easysnowdata.topography.get_chili(tile.bbox_gdf, initialize_ee=False)
    ds['chili'] = chili_da.rio.reproject_match(ds['dem'],resampling=rasterio.enums.Resampling.bilinear)

    return ds

def add_snow_class(tile,ds):

    snow_classification = easysnowdata.remote_sensing.get_seasonal_snow_classification(tile.bbox_gdf, mask_nodata=True)
    ds['snow_classification'] = snow_classification.rio.reproject_match(ds['dem'],resampling=rasterio.enums.Resampling.mode)

    return ds

def add_esa_worldcover(tile,ds):

    esa_worldcover = easysnowdata.remote_sensing.get_esa_worldcover(tile.bbox_gdf, mask_nodata=True)
    ds['esa_worldcover'] = esa_worldcover.rio.reproject_match(ds['dem'],resampling=rasterio.enums.Resampling.mode)

    return ds

def add_forest_cover(tile,ds):

    forest_cover_fraction = easysnowdata.remote_sensing.get_forest_cover_fraction(tile.bbox_gdf, mask_nodata=True)
    ds['forest_cover_fraction'] = forest_cover_fraction.rio.reproject_match(ds['dem'],resampling=rasterio.enums.Resampling.bilinear)

    return ds


def dataset_to_dataframe(tile,ds):
    # only drop row values / a pixel if there are no runoff_onset_predictions for any of the years
    # drop_subset = [f'runoff_onset_WY{water_year}' for water_year in water_years]

    #df = ds.to_dataframe().reset_index().dropna(subset=drop_subset,how='all').drop_vars('spatial_ref', axis=1) use this if we want to keep pixels with runoff_onset_predictions for two or less years, for three or more use next line
    df = ds.to_dataframe().reset_index().dropna(subset='runoff_onset_median').drop('spatial_ref', axis=1)

    df['tile_row'] = tile.row
    df['tile_col'] = tile.col

    # change these as we include/exclude new vars
    df = df[["tile_row","tile_col","original_lat","original_lon","runoff_onset_WY2015","runoff_onset_WY2016","runoff_onset_WY2017","runoff_onset_WY2018","runoff_onset_WY2019","runoff_onset_WY2020","runoff_onset_WY2021","runoff_onset_WY2022","runoff_onset_WY2023","runoff_onset_WY2024","runoff_onset_median","runoff_onset_std","dem","aspect","slope","tpi","chili","snow_classification","esa_worldcover","forest_cover_fraction"]]
    
    columns_to_round = [col for col in df.columns if col not in ['original_lat', 'original_lon', 'runoff_onset_std', 'chili']]
    df[columns_to_round] = df[columns_to_round].replace([np.inf, -np.inf, np.nan], -9999)
    df[columns_to_round] = df[columns_to_round].round().astype(np.int16)

    df['original_lat'] = df['original_lat'].round(4).astype(np.float32)
    df['original_lon'] = df['original_lon'].round(4).astype(np.float32)
    df['chili'] = df['chili'].round(4).astype(np.float32)
    df['runoff_onset_std'] = df['runoff_onset_std'].round(2).astype(np.float32)

    # for col in df.columns:
    #     if df[col].dtype == 'int64':
    #         df[col] = pd.to_numeric(df[col], downcast='integer')
    #     elif df[col].dtype == 'float64':
    #         df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


def create_utm_datacube(tile, global_ds):
    tile_ds = global_ds.rio.clip_box(*tile.get_geobox().boundingbox,crs='EPSG:4326')
    tile_ds = add_coordinate_arrays(tile_ds)
    utm_crs = tile_ds.rio.estimate_utm_crs()
    tile_utm_ds = tile_ds.rio.reproject(utm_crs,resolution=80,resampling=rasterio.enums.Resampling.bilinear)
    tile_utm_ds = convert_water_year_dim_to_var(tile_utm_ds)
    tile_utm_ds = add_topography(tile,tile_utm_ds)
    tile_utm_ds = add_chili(tile,tile_utm_ds)
    tile_utm_ds = add_snow_class(tile,tile_utm_ds)
    tile_utm_ds = add_esa_worldcover(tile,tile_utm_ds)
    tile_utm_ds = add_forest_cover(tile,tile_utm_ds)
    return tile_utm_ds


def create_and_save_analysis_parquet(tile, filename, filesystem, global_ds, ee_credentials):
    logger.info(f"Processing {filename}")
    ee.Initialize(ee_credentials, opt_url='https://earthengine-highvolume.googleapis.com')

    try:
        tile_utm_ds = create_utm_datacube(tile, global_ds)
        tile_utm_df = dataset_to_dataframe(tile, tile_utm_ds)
        tile_utm_df.to_parquet(f"snowmelt/analysis/tiles/{filename}",filesystem=filesystem)
        logger.info(f"Saved {filename}")
        return filename, True
    except Exception as e:
        error_message = f"Error processing {filename}: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        return filename, False, error_message, traceback.format_exc()



def create_lat_and_elev_binned_ds(results_ddf, lat_bin_low=-80,lat_bin_high=80,lat_bin_interval=1,dem_bin_low=0,dem_bin_high=8000,dem_bin_interval=100):

    dem_bins = np.arange(dem_bin_low,dem_bin_high+dem_bin_interval,dem_bin_interval)
    lat_bins = np.arange(lat_bin_low,lat_bin_high+lat_bin_interval,lat_bin_interval)

    results_ddf['lat_bin'] = results_ddf['original_lat'].map_partitions(pd.cut, lat_bins)
    results_ddf['dem_bin'] = results_ddf['dem'].map_partitions(pd.cut, dem_bins)
    results_ddf = results_ddf.dropna(subset=['lat_bin','dem_bin'])

    results_ddf['lat_bin'] = results_ddf['lat_bin'].apply(lambda x: x.left).astype(int)
    results_ddf['dem_bin'] = results_ddf['dem_bin'].apply(lambda x: x.left).astype(int)

    stat_coords = ["mean", "median", "count"]
    lat_coords = lat_bins[:-1]
    elev_coords = dem_bins[:-1]

    # Initialize empty arrays with NaN values
    shape = (len(lat_coords), len(elev_coords), len(stat_coords))
    runoff_onset_array = np.full(shape, np.nan)
    std_array = np.full(shape, np.nan)
    mad_array = np.full(shape, np.nan)
    chili_cool_array = np.full(shape, np.nan)
    chili_neutral_array = np.full(shape, np.nan)
    chili_warm_array = np.full(shape, np.nan)
    chili_corr_array = np.full((len(lat_coords), len(elev_coords)), np.nan)

    lat_idx = {lat: i for i, lat in enumerate(lat_coords)}
    elev_idx = {elev: i for i, elev in enumerate(elev_coords)}
    stat_idx = {stat: i for i, stat in enumerate(stat_coords)}

    def calc_mad_efficient(df):
        data = da.from_array(df.values)
        data = da.where(data > 0, data, np.nan)
        med = da.nanmedian(data, axis=1)
        abs_dev = da.abs(data - med[:, None])
        mad = da.nanmedian(abs_dev, axis=1)
        return pd.Series(mad, index=df.index, name='runoff_onset_mad', dtype='float32')

    runoff_onset_cols = [col for col in results_ddf.columns if col.startswith('runoff_onset_WY')]
    results_runoff_onset_cols_ddf=results_ddf[runoff_onset_cols]

    results_ddf['runoff_onset_mad'] = results_runoff_onset_cols_ddf.map_partitions(
        calc_mad_efficient,
        meta=('runoff_onset_mad', 'float32')
    )

    results_ddf["chili_class"] = "neutral"
    results_ddf["chili_class"] = results_ddf["chili_class"].where(
        (results_ddf["chili"] >= 0.448) & (results_ddf["chili"] <= 0.767),
        other=results_ddf["chili"].map(
            lambda x: "warm" if x > 0.767 else "cool" if x < 0.448 else "neutral"
        ),
    )

    results_ddf = results_ddf.persist()

    chili_and_median_runoff_onset_groupby_latitude_and_elevation_df = (
        results_ddf[["lat_bin", "dem_bin", "chili_class", "runoff_onset_median"]]
        .dropna()
        .groupby(["lat_bin", "dem_bin", "chili_class"])["runoff_onset_median"]
        .agg(
            [
                "mean",
                "median",
                "count",
            ]
        )
        .compute()
    )

    chili_and_median_runoff_onset_corr_groupby_latitude_and_elevation_df = (
        results_ddf[["lat_bin", "dem_bin", "chili", "runoff_onset_median"]]
        .dropna()
        .groupby(["lat_bin", "dem_bin"])
        .apply(lambda x: x["chili"].corr(x["runoff_onset_median"]))
        .compute()
    )  

    with dask.config.set({"dataframe.shuffle.method": "tasks"}):
        agg_groupby_latitude_and_elevation_df = (
            results_ddf.groupby(["lat_bin", "dem_bin"])
            .agg(
                {
                    "runoff_onset_median": ["mean", "median", "count"],
                    "runoff_onset_std": ["mean", "median", "count"],
                    "runoff_onset_mad": ["mean", "median", "count"],
                }
            )
            .compute()
        )


    # Fill runoff_onset_median array
    for (
        lat,
        elev,
    ), row in agg_groupby_latitude_and_elevation_df.iterrows():
        if lat in lat_idx and elev in elev_idx:
            i, j = lat_idx[lat], elev_idx[elev]
            for stat in stat_coords:
                k = stat_idx[stat]
                runoff_onset_array[i, j, k] = row[("runoff_onset_median", stat)]
                std_array[i, j, k] = row[("runoff_onset_std", stat)]
                mad_array[i, j, k] = row[("runoff_onset_mad", stat)]



    # Fill chili class arrays
    for (
        lat,
        elev,
        chili_class,
    ), row in chili_and_median_runoff_onset_groupby_latitude_and_elevation_df.iterrows():
        if lat in lat_idx and elev in elev_idx:
            i, j = lat_idx[lat], elev_idx[elev]
            target_array = {
                "cool": chili_cool_array,
                "neutral": chili_neutral_array,
                "warm": chili_warm_array,
            }[chili_class]
            for stat in stat_coords:
                k = stat_idx[stat]
                target_array[i, j, k] = row[("runoff_onset_median", stat)]

        # Fill correlation array
        for (
            lat,
            elev,
        ), corr in (
            chili_and_median_runoff_onset_corr_groupby_latitude_and_elevation_df.items()
        ):
            if lat in lat_idx and elev in elev_idx:
                i, j = lat_idx[lat], elev_idx[elev]
                chili_corr_array[i, j] = corr

    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "runoff_onset_median": (
                ("latitude", "elevation", "statistic"),
                runoff_onset_array,
            ),
            "runoff_onset_std": (("latitude", "elevation", "statistic"), std_array),
            "runoff_onset_mad": (("latitude", "elevation", "statistic"), mad_array),
            "chili_cool": (("latitude", "elevation", "statistic"), chili_cool_array),
            "chili_neutral": (("latitude", "elevation", "statistic"), chili_neutral_array),
            "chili_warm": (("latitude", "elevation", "statistic"), chili_warm_array),
            "chili_corr": (("latitude", "elevation"), chili_corr_array),
        },
        coords={
            "latitude": lat_coords + lat_bin_interval / 2,
            "elevation": elev_coords + dem_bin_interval / 2,
            "statistic": stat_coords,
        },
    )

    ds["chili_warm_cool_ratio"] = ds["chili_warm"] / ds["chili_cool"]
    ds["chili_warm_cool_ratio"].loc[{"statistic": "count"}] = ds["chili_warm"].sel(statistic="count")

    ds.latitude.attrs['units'] = 'degrees'
    ds.elevation.attrs['units'] = 'meters'

    return ds