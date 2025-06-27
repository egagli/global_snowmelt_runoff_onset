"""
Configuration management for global snowmelt runoff onset detection.

This module provides configuration management for processing Sentinel-1 SAR data
to detect snowmelt runoff onset timing globally. It handles spatial tiling, processing parameters,
chunking strategies, data and file management, and Azure storage integration.
"""

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
import ee
from typing import List, Tuple, Dict, Any, Union, Optional


class Config:
    """
    Configuration manager for global snowmelt runoff onset processing.
    
    This class handles loading configuration from files, setting up spatial tiling,
    managing chunking strategies for different processing stages, and providing
    access to Azure storage resources.
    
    Attributes:
        resolution (float): Spatial resolution in degrees
        bands (List[str]): SAR polarization bands to process (e.g., ['vv'])
        mountain_snow_only (bool): Whether to restrict processing to mountain regions
        spatial_chunk_dim_s1_read (int): Chunk size for reading S1 data
        spatial_chunk_dim_s1_process (int): Chunk size for processing operations
        spatial_chunk_dim_zarr_output (int): Chunk size for Zarr output
        water_years (np.ndarray): Array of water years to process
        global_geobox (odc.geo.GeoBox): Global geographic bounding box
        chunks_s1_read (Dict[str, int]): Dask chunks for reading S1 data
        chunks_s1_process (Dict[str, int]): Dask chunks for processing
        chunks_zarr_output (Dict[str, int]): Dask chunks for Zarr output
    """
    
    def __init__(self, config_file: Optional[str] = None) -> None:
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file. If None, creates empty config.
        """
        if config_file:
            self._init_config(config_file)

    def _init_config(self, config_file: str) -> None:
        """
        Initialize configuration from file.
        
        Args:
            config_file: Path to configuration file
        """
        self.config = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        self.config.read(config_file)
        self._load_values()
        self._init_derived_values()
        self._print_config()

    def _load_values(self) -> None:
        """
        Load configuration values from the config file.
        
        Handles backward compatibility between old single-chunk configs and new
        multi-stage chunking configs.
        """
        self.resolution: float = self.config.getfloat('VALUES', 'resolution')
        self.bands: List[str] = self.config.getlist('VALUES', 'bands')
        self.mountain_snow_only: bool = self.config.getboolean('VALUES', 'mountain_snow_only', fallback=True)
        
        # Handle backward compatibility for chunking configuration
        if self.config.has_option('VALUES', 'spatial_chunk_dim_s1_read'):
            # New format with separate chunk dimensions
            self.spatial_chunk_dim_s1_read: int = self.config.getint('VALUES', 'spatial_chunk_dim_s1_read')
            self.spatial_chunk_dim_s1_process: int = self.config.getint('VALUES', 'spatial_chunk_dim_s1_process')
            self.spatial_chunk_dim_zarr_output: int = self.config.getint('VALUES', 'spatial_chunk_dim_zarr_output')
        else:
            # Old format - use single spatial_chunk_dim for all purposes
            spatial_chunk_dim = self.config.getint('VALUES', 'spatial_chunk_dim')
            self.spatial_chunk_dim_s1_read: int = spatial_chunk_dim
            self.spatial_chunk_dim_s1_process: int = 512  # Use smaller chunks for processing
            self.spatial_chunk_dim_zarr_output: int = spatial_chunk_dim
        
        # Geographic bounds
        self.bbox_left: float = self.config.getfloat('VALUES', 'bbox_left')
        self.bbox_right: float = self.config.getfloat('VALUES', 'bbox_right')
        self.bbox_top: float = self.config.getfloat('VALUES', 'bbox_top')
        self.bbox_bottom: float = self.config.getfloat('VALUES', 'bbox_bottom')
        
        # Temporal parameters
        self.WY_start: int = self.config.getint('VALUES', 'WY_start')
        self.WY_end: int = self.config.getint('VALUES', 'WY_end')
        
        # Processing parameters
        self.min_years_for_median_std: int = self.config.getint('VALUES', 'min_years_for_median_std')
        self.min_monthly_acquisitions: int = self.config.getint('VALUES', 'min_monthly_acquisitions')
        self.max_allowed_days_gap_per_orbit: int = self.config.getint('VALUES', 'max_allowed_days_gap_per_orbit')
        self.low_backscatter_threshold: float = self.config.getfloat('VALUES', 'low_backscatter_threshold')
        self.extend_search_window_beyond_SDD_days: int = self.config.getint('VALUES', 'extend_search_window_beyond_SDD_days', fallback=16)
        self.min_consec_snow_days_for_seasonal_snow: int = self.config.getint('VALUES', 'min_consec_snow_days_for_seasonal_snow', fallback=56)
        
        # File paths
        self.valid_tiles_geojson_path: str = self.config.get('VALUES', 'valid_tiles_geojson_path')
        self.tile_results_path: str = self.config.get('VALUES', 'tile_results_path')
        self.global_runoff_zarr_store_azure_path: str = self.config.get('VALUES', 'global_runoff_zarr_store_azure_path')
        self.seasonal_snow_mask_zarr_store_azure_path: str = self.config.get('VALUES', 'seasonal_snow_mask_zarr_store_azure_path')

        # Output fields for tile processing results
        self.fields: Tuple[str, ...] = ("row","col","percent_valid_snow_pixels","s1_rtc_ds_dims","runoff_onsets_dims",
        "tr_2015", "tr_2016", "tr_2017", "tr_2018", "tr_2019", "tr_2020", "tr_2021", "tr_2022", "tr_2023","tr_2024",
        "pix_ct_2015","pix_ct_2016","pix_ct_2017","pix_ct_2018","pix_ct_2019","pix_ct_2020","pix_ct_2021","pix_ct_2022","pix_ct_2023","pix_ct_2024",
        "start_time","total_time","success","error_messages")

    def _init_derived_values(self) -> None:
        """
        Initialize derived configuration values.
        
        Sets up temporal ranges, chunking configurations, geographic transforms,
        and cloud storage connections.
        """
        # Temporal configuration
        self.water_years: np.ndarray = np.arange(self.WY_start, self.WY_end + 1)
        self.start_date: str = f'{self.WY_start-1}-10-01'
        self.end_date: str = f'{self.WY_end+1}-03-31'
        
        # Chunking configurations for different processing stages
        self.spatial_chunk_dims_zarr: Tuple[int, int] = (self.spatial_chunk_dim_zarr_output, self.spatial_chunk_dim_zarr_output)
        self.chunks_s1_read: Dict[str, int] = {"x": self.spatial_chunk_dim_s1_read, "y": self.spatial_chunk_dim_s1_read, "time": 1}
        self.chunks_s1_process: Dict[str, Union[int, str]] = {"latitude": self.spatial_chunk_dim_s1_process, "longitude": self.spatial_chunk_dim_s1_process, "time": -1}
        self.chunks_zarr_output: Dict[str, int] = {"longitude": self.spatial_chunk_dim_zarr_output, "latitude": self.spatial_chunk_dim_zarr_output}
        
        # Backward compatibility aliases
        self.chunks_read: Dict[str, int] = self.chunks_s1_read
        self.chunks_write: Dict[str, int] = self.chunks_zarr_output
        self.spatial_chunk_dims: Tuple[int, int] = self.spatial_chunk_dims_zarr
        
        # Geographic setup
        self.global_geobox: odc.geo.GeoBox = odc.geo.geobox.GeoBox.from_bbox((self.bbox_left, self.bbox_bottom,
            self.bbox_right, self.bbox_top), crs="epsg:4326", resolution=self.resolution)
        self.geobox_tiles: odc.geo.GeoboxTiles = odc.geo.geobox.GeoboxTiles(self.global_geobox, self.spatial_chunk_dims_zarr)
        
        # Cloud storage setup
        self.sas_token: str = pathlib.Path('../config/sas_token.txt').read_text()
        self.ee_credentials = ee.ServiceAccountCredentials(email='coiled@buoyant-aileron-352100.iam.gserviceaccount.com',key_file='../config/ee_key.json')
        self._azure_blob_fs: adlfs.AzureBlobFileSystem = adlfs.AzureBlobFileSystem(account_name="snowmelt", credential=self.sas_token, skip_instance_cache=True)
        self.global_runoff_store = self.azure_blob_fs.get_mapper(self.global_runoff_zarr_store_azure_path)
        self.seasonal_snow_mask_store = self.azure_blob_fs.get_mapper(self.seasonal_snow_mask_zarr_store_azure_path)
        self._load_valid_tiles()

    def _load_valid_tiles(self) -> None:
        """
        Load valid tiles from GeoJSON and merge with processing results.
        
        Creates or loads the tile results CSV file for tracking processing progress.
        """
        self.valid_tiles_gdf: gpd.GeoDataFrame = gpd.read_file(self.valid_tiles_geojson_path).drop(columns=['tile'])
        self.valid_tiles_gdf = self.valid_tiles_gdf.sort_values(by='percent_valid_snow_pixels', ascending=False)
        if os.path.exists(self.tile_results_path):
            processed_tiles_df = pd.read_csv(self.tile_results_path).drop_duplicates(subset=['row', 'col'], keep='last')
            self.valid_tiles_gdf = self.valid_tiles_gdf.merge(processed_tiles_df.drop(columns=['percent_valid_snow_pixels']), on=['row', 'col'], how='outer').sort_values(by='percent_valid_snow_pixels', ascending=False)
        else:
            df = pd.DataFrame(columns=self.fields)
            df.to_csv(self.tile_results_path, mode='a', header=True, index=False)

    def _print_config(self) -> None:
        """Print configuration summary."""
        print("Configuration loaded:")
        for section in self.config.sections():
            for key, value in self.config[section].items():
                print(f"{key} = {value}")

    @property
    def azure_blob_fs(self) -> adlfs.AzureBlobFileSystem:
        """
        Get Azure Blob File System with cache invalidation.
        
        Returns:
            Azure Blob File System instance with fresh cache
        """
        self._azure_blob_fs.invalidate_cache()
        return self._azure_blob_fs

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'resolution': self.resolution,
            'bands': self.bands,
            'mountain_snow_only': self.mountain_snow_only,
            'spatial_chunk_dim_s1_read': self.spatial_chunk_dim_s1_read,
            'spatial_chunk_dim_s1_process': self.spatial_chunk_dim_s1_process,
            'spatial_chunk_dim_zarr_output': self.spatial_chunk_dim_zarr_output,
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
            'extend_search_window_beyond_SDD_days': self.extend_search_window_beyond_SDD_days,
            'min_consec_snow_days_for_seasonal_snow': self.min_consec_snow_days_for_seasonal_snow,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'valid_tiles_geojson_path': self.valid_tiles_geojson_path,
            'tile_results_path': self.tile_results_path,
            'global_runoff_zarr_store_azure_path': self.global_runoff_zarr_store_azure_path,
            'seasonal_snow_mask_zarr_store_azure_path': self.seasonal_snow_mask_zarr_store_azure_path,
        }

    def get_tile(self, row: int, col: int) -> 'Tile':
        """
        Get a specific tile by row and column indices.
        
        Args:
            row: Tile row index
            col: Tile column index
            
        Returns:
            Tile object for the specified location
        """
        return Tile(row, col, self)

    def get_list_of_tiles(self, which: str = 'all') -> List['Tile']:
        """
        Get list of tiles based on processing status.
        
        Args:
            which: Filter criterion. Options:
                - 'all': All tiles regardless of processing status
                - 'processed': Successfully completed tiles  
                - 'failed': Tiles that encountered errors
                - 'unprocessed': Tiles not yet attempted
                - 'unprocessed_and_failed': Tiles needing processing or reprocessing
                - 'unprocessed_and_failed_weather_stations': Unprocessed/failed tiles that contain weather stations
            
        Returns:
            List of Tile objects matching the filter criterion
            
        Raises:
            ValueError: If 'which' parameter is not recognized
        """
        # Get base tile list based on processing status
        if which in ['all', 'unprocessed_and_failed_weather_stations']:
            base_tiles = [(row, col, success) for row, col, success in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success)]
        elif which == 'processed':
            base_tiles = [(row, col, success) for row, col, success in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success) if success==True]
        elif which == 'failed':
            base_tiles = [(row, col, success) for row, col, success in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success) if success==False]
        elif which == 'unprocessed':
            base_tiles = [(row, col, success) for row, col, success in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success) if success is np.nan]
        elif which == 'unprocessed_and_failed':
            base_tiles = [(row, col, success) for row, col, success in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success) if success is np.nan or success==False]
        else:
            raise ValueError("Must choose one of ['all', 'processed', 'failed', 'unprocessed', 'unprocessed_and_failed', 'unprocessed_and_failed_weather_stations']")
        
        # Apply weather station filtering if requested
        if which == 'unprocessed_and_failed_weather_stations':
            import easysnowdata
            
            # Get weather stations
            StationsWUS = easysnowdata.automatic_weather_stations.StationCollection()
            
            # Find tiles that contain weather stations
            tiles_with_stations_gdf = gpd.sjoin(
                self.valid_tiles_gdf,
                StationsWUS.all_stations,
                how='inner',
                predicate='contains'
            )
            tiles_with_stations_gdf = tiles_with_stations_gdf.drop_duplicates(subset=['row','col'])
            station_tiles = set(zip(tiles_with_stations_gdf.row, tiles_with_stations_gdf.col))
            
            # Filter to unprocessed/failed tiles with stations
            base_tiles = [(row, col, success) for row, col, success in base_tiles 
                         if (success is np.nan or success==False) and (row, col) in station_tiles]
        
        # Create Tile objects
        tiles = [Tile(row, col, self) for row, col, success in base_tiles]
        return tiles


class Tile:
    """
    Represents a spatial tile for processing snowmelt runoff onset.
    
    Each tile corresponds to a spatial chunk of the global grid and contains
    all necessary information for processing Sentinel-1 data within that region.
    
    Attributes:
        row (int): Tile row index in global grid
        col (int): Tile column index in global grid
        config (Config): Configuration object
        index (Tuple[int, int]): (row, col) tuple for indexing
        geobox (odc.geo.GeoBox): Geographic bounding box for this tile
        bbox_gdf (gpd.GeoDataFrame): Bounding box as GeoDataFrame
        percent_valid_snow_pixels (float): Percentage of pixels with seasonal snow
        success (bool): Whether processing completed successfully
        error_messages (List[str]): List of error messages if processing failed
    """
    
    def __init__(self, row: int, col: int, config: Config) -> None:
        """
        Initialize a tile.
        
        Args:
            row: Tile row index
            col: Tile column index
            config: Configuration object
        """
        self.row: int = row
        self.col: int = col
        self.config: Config = config
        self.index: Tuple[int, int] = row, col
        self.percent_valid_snow_pixels: float = self.get_percent_valid_snow_pixels()
        self.geobox: odc.geo.GeoBox = self.get_geobox()
        self.bbox_gdf: gpd.GeoDataFrame = self.get_bbox_gdf()
        
        # Processing timing
        self.start_time: Optional[float] = None
        self.total_time: Optional[float] = None
        
        # Data containers
        self.s1_rtc_ds = None
        self.s1_rtc_ds_dims: Optional[Dict[str, int]] = None
        self.s1_rtc_masked_ds_dims: Optional[Dict[str, int]] = None
        self.runoff_onsets = None
        self.runoff_onsets_dims: Optional[Dict[str, int]] = None
        
        # Temporal resolution metrics by water year
        self.tr_2015: Optional[float] = None
        self.tr_2016: Optional[float] = None
        self.tr_2017: Optional[float] = None
        self.tr_2018: Optional[float] = None
        self.tr_2019: Optional[float] = None
        self.tr_2020: Optional[float] = None
        self.tr_2021: Optional[float] = None
        self.tr_2022: Optional[float] = None
        self.tr_2023: Optional[float] = None
        self.tr_2024: Optional[float] = None
        
        # Pixel count metrics by water year
        self.pix_ct_2015: Optional[int] = None
        self.pix_ct_2016: Optional[int] = None
        self.pix_ct_2017: Optional[int] = None
        self.pix_ct_2018: Optional[int] = None
        self.pix_ct_2019: Optional[int] = None
        self.pix_ct_2020: Optional[int] = None
        self.pix_ct_2021: Optional[int] = None
        self.pix_ct_2022: Optional[int] = None
        self.pix_ct_2023: Optional[int] = None
        self.pix_ct_2024: Optional[int] = None
        
        # Processing status
        self.error_messages: List[str] = []
        self.success: bool = False

    def get_geobox(self) -> odc.geo.GeoBox:
        """
        Get the odc.geo.GeoBox for this tile.

        Returns:
            GeoBox object defining the spatial extent of this tile
        """
        return self.config.geobox_tiles[self.index]
    
    def get_bbox_gdf(self) -> gpd.GeoDataFrame:
        """
        Get bounding box as a GeoDataFrame.
        
        Returns:
            GeoDataFrame containing the tile boundary geometry
        """
        bbox = self.geobox.boundingbox
        bbox_geometry = shapely.geometry.box(bbox.left, bbox.bottom, bbox.right, bbox.top)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geometry], crs=self.geobox.crs)
        return bbox_gdf
    
    def get_percent_valid_snow_pixels(self) -> float:
        """
        Get percentage of pixels with valid seasonal snow.
        
        Returns:
            Percentage of pixels in this tile that have seasonal snow coverage
        """
        return float(self.config.valid_tiles_gdf['percent_valid_snow_pixels'].loc[(self.config.valid_tiles_gdf['row'] == self.row) & (self.config.valid_tiles_gdf['col'] == self.col)].values[0])
