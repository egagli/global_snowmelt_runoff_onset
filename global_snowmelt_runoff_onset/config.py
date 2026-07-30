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
import icechunk
import os
from typing import List, Tuple, Dict, Any, Union, Optional
import warnings
from datetime import datetime, timezone
from urllib.parse import parse_qs, unquote


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
        snow_phenology_store: Store for the MODIS-derived snow phenology
            dataset. An icechunk session store (Zarr v3) for configs >= v10,
            or a legacy fsspec mapper (consolidated Zarr v2) for configs <= v9
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
        self.config_file_path = self._resolve_repo_path(config_file)
        if not pathlib.Path(self.config_file_path).exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_file_path!r} "
                f"(from {config_file!r}; relative paths resolve against the repo root, "
                f"not the current working directory)"
            )
        self.config = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        self.config.read(self.config_file_path)
        self._load_metadata()
        self._load_values()
        self._init_derived_values()
        self._print_config()

    def _load_metadata(self) -> None:
        """Load metadata from the config file."""
        if self.config.has_section('METADATA'):
            self.config_name: str = self.config.get('METADATA', 'config_name', fallback='unknown')
            self.version: str = self.config.get('METADATA', 'version', fallback='unknown')
        else:
            # Extract from filename as fallback
            config_path = pathlib.Path(self.config_file_path)
            self.config_name = config_path.stem
            if 'v' in self.config_name:
                self.version = f"v{self.config_name.split('v')[1]}"
            else:
                self.version = 'unknown'

    def _resolve_repo_path(self, path: str) -> str:
        """
        Resolve a file path relative to the repository root.
        
        This ensures that paths work correctly regardless of where the script
        is executed from (e.g., GitHub Actions vs local development).
        
        Args:
            path: Path from config file (may be relative)
            
        Returns:
            str: Absolute path resolved relative to repository root
        """
        if pathlib.Path(path).is_absolute():
            return path
            
        # Find the repository root by looking for setup.py or .git
        current_dir = pathlib.Path(__file__).parent
        # Go up from global_snowmelt_runoff_onset/ to repo root
        repo_root = current_dir.parent
        
        # Verify we found the right directory
        setup_exists = (repo_root / 'setup.py').exists()
        git_exists = (repo_root / '.git').exists()
        if not setup_exists and not git_exists:
            # If not found, try going up one more level
            repo_root = repo_root.parent
            
        resolved_path = repo_root / path
        return str(resolved_path)

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

        # Optional grid tripwires (v10+). GeoBox.from_bbox snaps the bbox OUTWARD to
        # the resolution lattice, so a bbox edit of less than one pixel can silently
        # add or drop a row -- which would renumber every tile. When present, these
        # are asserted against the realized geobox in _validate_grid().
        self.expected_grid_shape: Optional[Tuple[int, int]] = None
        self.expected_tile_grid: Optional[Tuple[int, int]] = None
        if self.config.has_option('VALUES', 'expected_grid_shape'):
            self.expected_grid_shape = tuple(
                int(v) for v in self.config.getlist('VALUES', 'expected_grid_shape'))
        if self.config.has_option('VALUES', 'expected_tile_grid'):
            self.expected_tile_grid = tuple(
                int(v) for v in self.config.getlist('VALUES', 'expected_tile_grid'))
        
        # Temporal parameters
        self.WY_start: int = self.config.getint('VALUES', 'WY_start')
        self.WY_end: int = self.config.getint('VALUES', 'WY_end')
        
        # Processing parameters
        self.min_years_for_median_std: int = self.config.getint('VALUES', 'min_years_for_median_std')
        # (min_monthly_acquisitions was removed in v10: the edge-anchored max-gap
        # criterion implies the 1/month density floor; old configs may still
        # contain the key, which is simply ignored)
        self.max_allowed_days_gap_per_orbit: int = self.config.getint('VALUES', 'max_allowed_days_gap_per_orbit')
        self.low_backscatter_threshold: float = self.config.getfloat('VALUES', 'low_backscatter_threshold')
        self.extend_search_window_beyond_SDD_days: int = self.config.getint('VALUES', 'extend_search_window_beyond_SDD_days', fallback=16)
        self.min_consec_snow_days_for_seasonal_snow: int = self.config.getint('VALUES', 'min_consec_snow_days_for_seasonal_snow', fallback=56)
        # Days past a hemisphere's season end before a water year is eligible for
        # dispatch/processing (see status.wy_eligible): phenology's 90d trailing
        # buffer + grace for its fleet to run. Gates both status.get_remaining_work
        # and the per-tile processor so a half-elapsed season is never committed.
        self.trailing_buffer_days: int = self.config.getint('VALUES', 'trailing_buffer_days', fallback=120)

        # File paths (resolve relative to repository root)
        self.valid_tiles_geojson_path: str = self._resolve_repo_path(
            self.config.get('VALUES', 'valid_tiles_geojson_path'))

        # Output store. Configs >= v10 use 'global_runoff_icechunk_azure_prefix'
        # (an icechunk repository; processing status is derived from its commit
        # history -- see status.py). Configs <= v9 use the legacy
        # 'global_runoff_zarr_store_azure_path' (a pre-allocated plain Zarr v2
        # store) plus 'tile_results_path' CSV-based status tracking.
        if self.config.has_option('VALUES', 'global_runoff_icechunk_azure_prefix'):
            self.output_store_is_icechunk: bool = True
            self.global_runoff_icechunk_azure_prefix: str = self.config.get(
                'VALUES', 'global_runoff_icechunk_azure_prefix')
            self.tile_results_path = None
            # Zarr v3 sharding: shard spatial dims == tile dims (one shard per
            # tile per water year, so concurrent tile jobs never share a shard);
            # inner chunks keep point/subset reads small.
            self.inner_chunk_dim: int = self.config.getint('VALUES', 'inner_chunk_dim', fallback=256)
        else:
            self.output_store_is_icechunk: bool = False
            self.global_runoff_zarr_store_azure_path: str = self.config.get(
                'VALUES', 'global_runoff_zarr_store_azure_path')
            self.tile_results_path: str = self._resolve_repo_path(
                self.config.get('VALUES', 'tile_results_path'))

        # Snow phenology input. Configs >= v10 use 'snow_phenology_zarr_store_azure_path'
        # (an icechunk repo from MODIS_snow_phenology); configs <= v9 use the legacy
        # 'seasonal_snow_mask_zarr_store_azure_path' key (a plain consolidated Zarr v2
        # store from MODIS_seasonal_snow_mask).
        if self.config.has_option('VALUES', 'snow_phenology_zarr_store_azure_path'):
            self.snow_phenology_zarr_store_azure_path: str = self.config.get(
                'VALUES', 'snow_phenology_zarr_store_azure_path')
            self.snow_phenology_store_is_icechunk: bool = True
        else:
            self.snow_phenology_zarr_store_azure_path: str = self.config.get(
                'VALUES', 'seasonal_snow_mask_zarr_store_azure_path')
            self.snow_phenology_store_is_icechunk: bool = False

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
        self._validate_grid()
        
        # Cloud storage setup
        # Try to get credentials from environment variables first
        # (for GitHub Actions) Fall back to local files for development
        sas_token_env = os.getenv('AZURE_STORAGE_SAS_TOKEN')
        if sas_token_env:
            self.sas_token: str = sas_token_env
        else:
            # Fallback to local file for development
            sas_token_file = pathlib.Path(
                self._resolve_repo_path('config/sas_token.txt'))
            if sas_token_file.exists():
                self.sas_token: str = sas_token_file.read_text().strip()
            else:
                raise ValueError("Azure SAS token not found in environment "
                                 "variable AZURE_STORAGE_SAS_TOKEN or "
                                 "config/sas_token.txt")
                
        self._check_sas_token_expiration()
        
        # Azure storage account name from environment or default
        self.azure_storage_account: str = os.getenv('AZURE_STORAGE_ACCOUNT', 'uwcryo')
        account_name = self.azure_storage_account
        
        # Earth Engine credentials (optional - only used if available).
        # Imported lazily so the minimal CI environment doesn't need earthengine-api.
        ee_key_path = self._resolve_repo_path('config/ee_key.json')
        ee_key_file = pathlib.Path(ee_key_path)
        if ee_key_file.exists():
            import ee
            self.ee_credentials = ee.ServiceAccountCredentials(
                email='coiled@buoyant-aileron-352100.iam.gserviceaccount.com',
                key_file=str(ee_key_file)
            )
        else:
            self.ee_credentials = None
        
        # adlfs/fsspec is only needed for the legacy (<= v9) plain-Zarr stores;
        # icechunk does its own (Rust object-store) I/O, and Sentinel-1 COGs are
        # read via rasterio/GDAL. Constructed lazily so icechunk-era runs never
        # build it at all.
        self._azure_blob_fs: Optional[adlfs.AzureBlobFileSystem] = None
        if self.output_store_is_icechunk:
            self.global_runoff_store = None
        else:
            self.global_runoff_store = self.azure_blob_fs.get_mapper(
                self.global_runoff_zarr_store_azure_path)
        self._snow_phenology_store = None
        self._output_repo = None
        self._load_valid_tiles()
        
    def _validate_grid(self) -> None:
        """
        Assert the realized global grid matches what the config declares.

        The tile grid is the dataset's coordinate system: tile (row, col) indices are
        baked into the icechunk commit history (see status.py) and into every published
        tile-wise comparison, so a grid that silently shifts is unrecoverable without a
        full rebuild. Two things are checked:

        1. expected_grid_shape / expected_tile_grid (v10+ configs): the realized geobox
           must have exactly the declared pixel and tile dimensions. GeoBox.from_bbox
           expands the bbox outward to the resolution lattice, so a sub-pixel edit to
           bbox_top -- e.g. 84.048 -> 84.0486 -- adds one row and renumbers every tile.
        2. Latitude must be a whole number of tiles (v10+ / icechunk configs only).
           With no partial row at the south edge, extending the grid southward later is
           a pure append (zarr can only grow a dimension at its end), and every shard
           is full-height. Legacy <= v9 configs had a 1410-row partial last tile row
           and are left alone.

        Raises:
            ValueError: if the realized grid contradicts the config.
        """
        shape_yx = tuple(int(v) for v in self.global_geobox.shape.yx)
        tile_grid_yx = tuple(int(v) for v in self.geobox_tiles.shape.yx)

        if self.expected_grid_shape is not None and shape_yx != tuple(self.expected_grid_shape):
            raise ValueError(
                f"Grid shape mismatch in {self.config_name}: expected_grid_shape="
                f"{tuple(self.expected_grid_shape)} but the bbox "
                f"({self.bbox_left}, {self.bbox_bottom}, {self.bbox_right}, {self.bbox_top}) "
                f"at resolution {self.resolution} realizes {shape_yx} (lat, lon) pixels. "
                "The bbox edges snap outward to the resolution lattice -- a sub-pixel "
                "change to bbox_top renumbers every tile row."
            )
        if self.expected_tile_grid is not None and tile_grid_yx != tuple(self.expected_tile_grid):
            raise ValueError(
                f"Tile grid mismatch in {self.config_name}: expected_tile_grid="
                f"{tuple(self.expected_tile_grid)} but realized {tile_grid_yx} "
                f"(rows, cols) at tile size {self.spatial_chunk_dim_zarr_output}."
            )
        if self.output_store_is_icechunk:
            tile_dim = self.spatial_chunk_dim_zarr_output
            if shape_yx[0] % tile_dim != 0:
                raise ValueError(
                    f"Latitude extent {shape_yx[0]} is not a whole number of "
                    f"{tile_dim}-pixel tiles ({shape_yx[0] % tile_dim} px over). "
                    "Adjust bbox_bottom so the south edge lands on a tile boundary: "
                    "a partial last tile row means any future southward extension "
                    "changes that row's footprint instead of being a pure append."
                )

    def _check_sas_token_expiration(self) -> None:
        """
        Check if the SAS token is expired or about to expire.
        
        Parses the SAS token to extract the expiration date and warns the user
        if the token is expired or expires within 24 hours.
        """
        try:
            # Parse the SAS token parameters
            # Remove any leading '?' if present
            token = self.sas_token.lstrip('?')
            params = parse_qs(token)
            
            if 'se' not in params:
                warnings.warn("SAS token does not contain expiration date (se parameter)")
                return
            
            # Get expiration date string and decode URL encoding
            expiration_str = unquote(params['se'][0])
            
            # Parse the expiration date (format: 2025-09-15T17:18Z)
            expiration_dt = datetime.strptime(expiration_str, '%Y-%m-%dT%H:%MZ')
            expiration_dt = expiration_dt.replace(tzinfo=timezone.utc)
            
            # Get current time in UTC
            current_dt = datetime.now(timezone.utc)
            
            # Calculate time difference
            time_until_expiry = expiration_dt - current_dt
            hours_until_expiry = time_until_expiry.total_seconds() / 3600
            
            if hours_until_expiry < 0:
                # Token is expired
                raise ValueError(f"SAS token is EXPIRED! Expired {abs(hours_until_expiry):.1f} hours ago on {expiration_dt.strftime('%Y-%m-%d %H:%M UTC')}")
            elif hours_until_expiry < 24:
                # Token expires soon
                warnings.warn(f"SAS token expires in {hours_until_expiry:.1f} hours on {expiration_dt.strftime('%Y-%m-%d %H:%M UTC')}. "
                             "Consider renewing it soon.")
            else:
                # Token is valid for a while
                print(f"SAS token is valid until {expiration_dt.strftime('%Y-%m-%d %H:%M UTC')} ({hours_until_expiry:.1f} hours)")
                
        except Exception as e:
            warnings.warn(f"Could not parse SAS token expiration date: {e}")

    def _load_valid_tiles(self) -> None:
        """
        Load valid tiles from GeoJSON and merge with processing results.

        For legacy (<= v9) configs this also creates/merges the tile results CSV.
        For icechunk (>= v10) configs the registry is loaded as-is; processing
        status lives in the icechunk commit history (see status.py).
        """
        # Full registry, including any to_process == False rows (v10+ registries keep
        # excluded tiles for documentation) so per-tile lookups (Tile, QC) always work;
        # the work universe is filtered on to_process in status.get_tile_status_gdf.
        self.valid_tiles_gdf: gpd.GeoDataFrame = gpd.read_file(self.valid_tiles_geojson_path).drop(columns=['tile'], errors='ignore')
        self.valid_tiles_gdf = self.valid_tiles_gdf.sort_values(by='percent_valid_snow_pixels', ascending=False)
        if self.output_store_is_icechunk:
            return
        if os.path.exists(self.tile_results_path):
            processed_tiles_df = pd.read_csv(self.tile_results_path).drop_duplicates(subset=['row', 'col'], keep='last')
            self.valid_tiles_gdf = self.valid_tiles_gdf.merge(processed_tiles_df.drop(columns=['percent_valid_snow_pixels']), on=['row', 'col'], how='outer').sort_values(by='percent_valid_snow_pixels', ascending=False)
        else:
            df = pd.DataFrame(columns=self.fields)
            df.to_csv(self.tile_results_path, mode='a', header=True, index=False)

    def _print_config(self) -> None:
        """Print configuration summary."""
        print("-" * 40)
        print("Configuration loaded:")
        for section in self.config.sections():
            for key, value in self.config[section].items():
                print(f"{key} = {value}")
        print("-" * 40)

    @property
    def azure_blob_fs(self) -> adlfs.AzureBlobFileSystem:
        """
        Get Azure Blob File System with cache invalidation (lazily created;
        only used for legacy <= v9 plain-Zarr store access).

        Returns:
            Azure Blob File System instance with fresh cache
        """
        if self._azure_blob_fs is None:
            self._azure_blob_fs = adlfs.AzureBlobFileSystem(
                account_name=self.azure_storage_account,
                credential=self.sas_token,
                skip_instance_cache=True,
            )
        self._azure_blob_fs.invalidate_cache()
        return self._azure_blob_fs

    @property
    def snow_phenology_store(self):
        """
        Get the store for the MODIS-derived snow phenology dataset (lazily opened).

        For configs >= v10 this opens the MODIS_snow_phenology icechunk repository
        and returns a read-only session store (Zarr v3). For configs <= v9 this
        returns a legacy fsspec mapper onto the plain consolidated Zarr v2 store
        from MODIS_seasonal_snow_mask.
        """
        if self._snow_phenology_store is None:
            if self.snow_phenology_store_is_icechunk:
                container, prefix = self.snow_phenology_zarr_store_azure_path.split('/', 1)
                storage = icechunk.azure_storage(
                    account=self.azure_storage_account,
                    container=container,
                    prefix=prefix,
                    sas_token=self.sas_token,
                )
                repo = icechunk.Repository.open(storage)
                self._snow_phenology_store = repo.readonly_session("main").store
            else:
                self._snow_phenology_store = self.azure_blob_fs.get_mapper(
                    self.snow_phenology_zarr_store_azure_path)
        return self._snow_phenology_store

    def output_repo_storage(self) -> 'icechunk.Storage':
        """Icechunk storage handle for the (v10+) output repository on Azure."""
        if not self.output_store_is_icechunk:
            raise ValueError(
                f"Config {self.config_name} uses the legacy pre-allocated Zarr v2 "
                "output store, not icechunk (configs >= v10)."
            )
        container, prefix = self.global_runoff_icechunk_azure_prefix.split('/', 1)
        return icechunk.azure_storage(
            account=self.azure_storage_account,
            container=container,
            prefix=prefix,
            sas_token=self.sas_token,
        )

    def output_repo_config(self) -> 'icechunk.RepositoryConfig':
        """
        RepositoryConfig for the output repository.

        - Storage retries: tolerate transient Azure errors from hundreds of
          concurrent GitHub Actions writers (same settings as MODIS_snow_phenology).
        - Manifest splitting: one manifest per water year per array, so a
          tile x water_year commit rewrites only that year's manifest
          (<= ~4,700 chunk refs) instead of the full ~500k-ref manifest.
          Persisted in the repo at creation via save_config; passing it at
          open time too keeps behavior identical for repos created before
          any future config change.
        """
        repo_config = icechunk.RepositoryConfig.default()
        repo_config.storage = icechunk.StorageSettings(
            retries=icechunk.StorageRetriesSettings(
                max_tries=20,
                initial_backoff_ms=200,
                max_backoff_ms=60_000,
            )
        )
        split_config = icechunk.ManifestSplittingConfig.from_dict({
            icechunk.ManifestSplitCondition.AnyArray(): {
                icechunk.ManifestSplitDimCondition.DimensionName("water_year"): 1,
                icechunk.ManifestSplitDimCondition.Any(): 1_000_000,
            }
        })
        repo_config.manifest = icechunk.ManifestConfig(splitting=split_config)
        return repo_config

    def open_output_repo(self) -> 'icechunk.Repository':
        """Open the icechunk output repository (cached on this Config)."""
        if self._output_repo is None:
            self._output_repo = icechunk.Repository.open(
                self.output_repo_storage(), config=self.output_repo_config()
            )
        return self._output_repo

    def create_output_repo(self) -> 'icechunk.Repository':
        """
        Create the icechunk output repository (one-time, from the store init
        notebook) with the manifest splitting + retry config persisted on-disk.
        """
        repo = icechunk.Repository.create(
            self.output_repo_storage(), config=self.output_repo_config()
        )
        repo.save_config()
        self._output_repo = repo
        return repo

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
            'max_allowed_days_gap_per_orbit': self.max_allowed_days_gap_per_orbit,
            'low_backscatter_threshold': self.low_backscatter_threshold,
            'extend_search_window_beyond_SDD_days': self.extend_search_window_beyond_SDD_days,
            'trailing_buffer_days': self.trailing_buffer_days,
            'min_consec_snow_days_for_seasonal_snow': self.min_consec_snow_days_for_seasonal_snow,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'valid_tiles_geojson_path': self.valid_tiles_geojson_path,
            'snow_phenology_zarr_store_azure_path': self.snow_phenology_zarr_store_azure_path,
            **({'global_runoff_icechunk_azure_prefix': self.global_runoff_icechunk_azure_prefix,
                'inner_chunk_dim': self.inner_chunk_dim}
               if self.output_store_is_icechunk else
               {'global_runoff_zarr_store_azure_path': self.global_runoff_zarr_store_azure_path,
                'tile_results_path': self.tile_results_path}),
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
                - 'unprocessed_and_failed_skip_empty_tiles': Unprocessed/failed tiles that do not have the error message "No such band/alias"
                - 'unprocessed_and_failed_weather_stations': Unprocessed/failed tiles that contain weather stations
            
        Returns:
            List of Tile objects matching the filter criterion
            
        Raises:
            ValueError: If 'which' parameter is not recognized
        """
        if self.output_store_is_icechunk:
            raise ValueError(
                "Configs >= v10 derive processing status from the icechunk commit "
                "history. Use global_snowmelt_runoff_onset.status.get_remaining_work() "
                "or get_tile_status_gdf() instead of get_list_of_tiles()."
            )
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
        elif which == 'unprocessed_and_failed_skip_empty_tiles':
            base_tiles = [(row, col, success) for row, col, success, error_messages in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success, self.valid_tiles_gdf.error_messages) if ((success is np.nan or success==False) and (('No such band/alias' not in str(error_messages)) and ('empty sequence' not in str(error_messages))))]
            #base_tiles = [(row, col, success) for row, col, success, error_messages in zip(self.valid_tiles_gdf.row, self.valid_tiles_gdf.col, self.valid_tiles_gdf.success, self.valid_tiles_gdf.error_messages) if (success is np.nan or success==False) and ('No such band/alias' not in str(error_messages))]

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
