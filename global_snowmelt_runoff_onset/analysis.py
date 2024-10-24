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
