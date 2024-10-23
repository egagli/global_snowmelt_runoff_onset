import numpy as np
import pandas as pd
import geopandas as gpd
import pathlib
import configparser
import shapely
import odc.geo
import odc.stac
import adlfs
import os
from typing import List, Tuple, Dict, Any

class Config:
    _instance = None

    def __new__(cls, config_file=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            if config_file:
                cls._instance._init_config(config_file)
        return cls._instance

    def _init_config(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self._load_values()
        self._init_derived_values()
        self._print_config()

    def _load_values(self):
        self.resolution = self.config.getfloat('VALUES', 'resolution')
        self.spatial_chunk_dim = self.config.getint('VALUES', 'spatial_chunk_dim')
        self.bbox_left = self.config.getfloat('VALUES', 'bbox_left')
        self.bbox_right = self.config.getfloat('VALUES', 'bbox_right')
        self.bbox_top = self.config.getfloat('VALUES', 'bbox_top')
        self.bbox_bottom = self.config.getfloat('VALUES', 'bbox_bottom')
        self.WY_start = self.config.getint('VALUES', 'WY_start')
        self.WY_end = self.config.getint('VALUES', 'WY_end')
        self.min_years_for_median_std = self.config.getint('VALUES', 'min_years_for_median_std')
        self.min_monthly_acquisitions = self.config.getint('VALUES', 'min_monthly_acquisitions')
        self.max_allowed_days_gap_per_orbit = self.config.getint('VALUES', 'max_allowed_days_gap_per_orbit')
        self.low_backscatter_threshold = self.config.getfloat('VALUES', 'low_backscatter_threshold')
        self.valid_tiles_geojson_path = self.config.get('VALUES', 'valid_tiles_geojson_path')
        self.tile_results_path = self.config.get('VALUES', 'tile_results_path')
        self.global_runoff_zarr_store_azure_path = self.config.get('VALUES', 'global_runoff_zarr_store_azure_path')
        self.seasonal_snow_mask_zarr_store_azure_path = self.config.get('VALUES', 'seasonal_snow_mask_zarr_store_azure_path')

        self.fields = ("row","col","percent_valid_snow_pixels","s1_rtc_ds_dims","runoff_onsets_dims",
        "tr_2015", "tr_2016", "tr_2017", "tr_2018", "tr_2019", "tr_2020", "tr_2021", "tr_2022", "tr_2023","tr_2024",
        "pix_ct_2015","pix_ct_2016","pix_ct_2017","pix_ct_2018","pix_ct_2019","pix_ct_2020","pix_ct_2021","pix_ct_2022","pix_ct_2023","pix_ct_2024",
        "start_time","total_time","success","error_messages")

    def _init_derived_values(self):
        self.water_years = np.arange(self.WY_start, self.WY_end + 1)
        self.start_date = '2014-01-01'
        self.end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        self.spatial_chunk_dims = (self.spatial_chunk_dim, self.spatial_chunk_dim)
        self.chunks_read = {"x": self.spatial_chunk_dim, "y": self.spatial_chunk_dim, "time": 1}
        self.chunks_write = {"longitude": self.spatial_chunk_dim, "latitude": self.spatial_chunk_dim}
        self.global_geobox = odc.geo.geobox.GeoBox.from_bbox((self.bbox_left, self.bbox_bottom,
            self.bbox_right, self.bbox_top), crs="epsg:4326", resolution=self.resolution)
        self.geobox_tiles = odc.geo.geobox.GeoboxTiles(self.global_geobox, self.spatial_chunk_dims)
        self.sas_token = pathlib.Path('../config/sas_token.txt').read_text()
        self.azure_blob_fs = adlfs.AzureBlobFileSystem(account_name="snowmelt", credential=self.sas_token)
        self.global_runoff_store = self.azure_blob_fs.get_mapper(self.global_runoff_zarr_store_azure_path)
        self.seasonal_snow_mask_store = self.azure_blob_fs.get_mapper(self.seasonal_snow_mask_zarr_store_azure_path)
        self._load_valid_tiles()

    def _load_valid_tiles(self):
        self.valid_tiles_gdf = gpd.read_file(self.valid_tiles_geojson_path).drop(columns=['tile'])
        self.valid_tiles_gdf = self.valid_tiles_gdf.sort_values(by='percent_valid_snow_pixels', ascending=False)
        if os.path.exists(self.tile_results_path):
            processed_tiles_df = pd.read_csv(self.tile_results_path).drop_duplicates(subset=['row', 'col'], keep='last')
            self.valid_tiles_gdf = self.valid_tiles_gdf.merge(processed_tiles_df.drop(columns=['percent_valid_snow_pixels']), on=['row', 'col'], how='outer')
        else:
            df = pd.DataFrame(columns=self.fields)
            df.to_csv(self.tile_results_path, mode='a', header=True, index=False)

    def _print_config(self):
        print("Configuration loaded:")
        for section in self.config.sections():
            for key, value in self.config[section].items():
                print(f"{key} = {value}")

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            'resolution': self.resolution,
            'zarr_chunk_size': self.zarr_chunk_size,
            'bbox_left': self.bbox_left,
            'bbox_right': self.bbox_right,
            'bbox_top': self.bbox_top,
            'bbox_bottom': self.bbox_bottom,
            'WY_start': self.WY_start,
            'WY_end': self.WY_end,
            'water_years': self.water_years.tolist(),
            'min_years_for_median_std': self.min_years_for_median_std,
            'min_monthly_acquisitions': self.min_monthly_acquisitions,
            'max_allowed_days_gap_per_orbit': self.max_allowed_days_gap_per_orbit,
            'low_backscatter_threshold': self.low_backscatter_threshold,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'valid_tiles_geojson_path': self.valid_tiles_geojson_path,
            'tile_results_path': self.tile_results_path,
            'global_runoff_zarr_store_azure_path': self.global_runoff_zarr_store_azure_path,
            'seasonal_snow_mask_zarr_store_azure_path': self.seasonal_snow_mask_zarr_store_azure_path,
        }

    def get_tile(self, row, col):
        return Tile(row,col,self)


    def get_list_of_tiles(self, which='all'):
        if which == 'all':
            tiles = [Tile(row, col, self) for row, col, success in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success)]
        elif which == 'processed':
            tiles = [Tile(row, col, self) for row, col, success in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success) if success==True]
        elif which == 'failed':
            tiles = [Tile(row, col, self) for row, col, success in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success) if success==False]
        elif which == 'unprocessed':
            tiles = [Tile(row, col, self) for row, col, success in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success) if success is np.nan]
        elif which == 'unprocessed_and_failed':
            tiles = [Tile(row, col, self) for row, col, success in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success) if success is np.nan or success==False]
        else:
            raise ValueError("Must choose one of ['all', 'processed', 'failed', 'unprocessed', 'unprocessed_and_failed']")        
        return tiles


class Tile:
    def __init__(self, row, col, config):
        self.row = row
        self.col = col
        self.config = config
        self.index = row, col
        self.percent_valid_snow_pixels = self.get_percent_valid_snow_pixels()
        self.geobox = self.get_geobox()
        self.bbox_gdf = self.get_bbox_gdf()
        self.start_time = None
        self.total_time = None
        self.s1_rtc_ds = None
        self.s1_rtc_ds_dims = None
        self.s1_rtc_masked_ds_dims = None
        self.runoff_onsets = None
        self.runoff_onsets_dims = None
        self.tr_2015 = None
        self.tr_2016 = None
        self.tr_2017 = None
        self.tr_2018 = None
        self.tr_2019 = None
        self.tr_2020 = None
        self.tr_2021 = None
        self.tr_2022 = None
        self.tr_2023 = None
        self.tr_2024 = None
        self.pix_ct_2015 = None
        self.pix_ct_2016 = None
        self.pix_ct_2017 = None
        self.pix_ct_2018 = None
        self.pix_ct_2019 = None
        self.pix_ct_2020 = None
        self.pix_ct_2021 = None
        self.pix_ct_2022 = None
        self.pix_ct_2023 = None
        self.pix_ct_2024 = None
        self.error_messages = []
        self.success = False

    def get_geobox(self):
        return self.config.geobox_tiles[self.index]
    
    def get_bbox_gdf(self):
        bbox = self.geobox.boundingbox
        bbox_geometry = shapely.geometry.box(bbox.left, bbox.bottom, bbox.right, bbox.top)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geometry], crs=self.geobox.crs)
        return bbox_gdf
    
    def get_percent_valid_snow_pixels(self):
        return float(self.config.valid_tiles_gdf['percent_valid_snow_pixels'].loc[(self.config.valid_tiles_gdf['row'] == self.row) & (self.config.valid_tiles_gdf['col'] == self.col)].values[0])

