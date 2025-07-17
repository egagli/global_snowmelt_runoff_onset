#!/usr/bin/env python3
"""
Process a single tile using GitHub Actions.

This script adapts the Coiled serverless tile processing function to run
within GitHub Actions infrastructure. It processes a single spatial tile
through the complete snowmelt runoff onset detection pipeline.
"""

import argparse
import sys
import time
import traceback
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import gc
import psutil
import xarray as xr
import odc.stac
import dask
import dask.array

# Add the parent directory to Python path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))


def dask_or_computed(variable):
    """
    Check if a variable is dask-backed and return status.
    
    Args:
        variable: Any variable (DataArray, Dataset, etc.)
        
    Returns:
        str: "DASK" or "COMPUTED"
    """
    # Check if it's dask-backed
    if hasattr(variable, 'data'):
        # Single DataArray
        is_dask = isinstance(variable.data, dask.array.Array)
    elif hasattr(variable, 'data_vars'):
        # Dataset - check if any data variables are dask-backed
        is_dask = any([isinstance(variable[var].data, dask.array.Array)
                      for var in variable.data_vars])
    else:
        # Not a dask-compatible object
        is_dask = False

    return (f"[DASK: {variable.nbytes * 1e-9:.2f}GB]" if is_dask
            else f"[COMPUTED: {variable.nbytes * 1e-9:.2f}GB]")


def setup_logging(tile_row: int, tile_col: int) -> None:
    """Set up logging for the tile processing."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"tile_{tile_row}_{tile_col}.log"
    
    # Configure root logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress verbose logging from cloud storage and HTTP libraries
    verbose_loggers = [
        'azure.storage.blob',
        'azure.core.pipeline.policies.http_logging_policy',
        'azure.storage.blob._base_client',
        'azure.storage.blob._blob_client',
        'azure.storage.blob._container_client',
        'azure.storage.blob._download',
        'azure.storage.blob._upload_helpers',
        'azure.identity',
        'urllib3.connectionpool',
        'urllib3.util.retry',
        'requests.packages.urllib3.connectionpool',
        's3fs',
        'fsspec',
        'aiohttp.access',
    ]
    
    for logger_name in verbose_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Keep important zarr and xarray logs but reduce verbosity
    logging.getLogger('zarr').setLevel(logging.WARNING)
    logging.getLogger('xarray').setLevel(logging.INFO)
    
    logging.info(f"Logging configured for tile ({tile_row}, {tile_col})")
    logging.info("Suppressed verbose cloud storage HTTP request/response logging")

def setup_modules():
    """Set up the required processing modules (now included in repository)."""
    # The modules are now included directly in the repository
    # so we just need to verify they exist
    module_files = [
        "global_snowmelt_runoff_onset/__init__.py",
        "global_snowmelt_runoff_onset/config.py", 
        "global_snowmelt_runoff_onset/processing.py"
    ]
    
    for file_path in module_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Required module file not found: {file_path}")
    
    logging.info("Processing modules are available")

def get_config_file(config_filename: str) -> str:
    """Get the path to an existing configuration file."""
    config_dir = Path("config")
    
    # If user provides just version (e.g., "v9"), construct filename
    if not config_filename.endswith('.txt'):
        config_file = config_dir / f"global_config_{config_filename}.txt"
    else:
        # User provided full filename
        config_file = config_dir / config_filename
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    return str(config_file)

def monitor_memory_and_cleanup():
    """Monitor memory usage and trigger cleanup if needed."""
    # Get current memory usage
    memory_percent = psutil.virtual_memory().percent
    logging.info(f"Current memory usage: {memory_percent:.1f}%")
    
    if memory_percent > 85:
        logging.warning(f"High memory usage: {memory_percent:.1f}%. Running cleanup...")
        gc.collect()
        
        # Force garbage collection for specific types
        for obj in gc.get_objects():
            if hasattr(obj, 'close') and callable(obj.close):
                try:
                    obj.close()
                except:
                    pass
        
        memory_after = psutil.virtual_memory().percent
        logging.info(f"Memory usage after cleanup: {memory_after:.1f}%")
    
    return memory_percent


def process_tile_github_actions(tile_row: int, tile_col: int, config):
    """
    Process a single tile within GitHub Actions infrastructure.
    
    This function adapts the original Coiled serverless processing function
    to work within GitHub Actions constraints.
    """
    try:
        # Import after modules are available
        import global_snowmelt_runoff_onset.processing as processing

        start_time = time.time()

        # Get the specific tile
        tile = config.get_tile(tile_row, tile_col)
        tile.start_time = start_time

        logging.info(f"Processing tile ({tile_row}, {tile_col})")
        monitor_memory_and_cleanup()

        # Configure ODC for cloud access
        odc.stac.configure_rio(cloud_defaults=True)

        # Get Sentinel-1 data
        logging.info("Retrieving Sentinel-1 data...")
        s1_rtc_ds = processing.get_sentinel1_rtc(
            geobox=tile.geobox,
            bands=config.bands,
            start_date=config.start_date,
            end_date=config.end_date,
            chunks_read=config.chunks_s1_read,
            fail_on_error=True,
        )
        # Check if lazily loaded
        logging.info(f"Retrieved Sentinel-1 RTC dataset (s1_rtc_ds) - {dask_or_computed(s1_rtc_ds)}")
        monitor_memory_and_cleanup()

        tile.s1_rtc_ds_dims = dict(s1_rtc_ds.sizes)
        logging.info(f"Sentinel-1 RTC dataset dimensions: {tile.s1_rtc_ds_dims}")

        # Get spatiotemporal snow cover mask
        logging.info("Getting spatiotemporal snow cover mask...")
        spatiotemporal_snow_cover_mask_ds = processing.get_spatiotemporal_snow_cover_mask(
            ds=s1_rtc_ds,
            bbox_gdf=tile.bbox_gdf,
            seasonal_snow_mask_store=config.seasonal_snow_mask_store,
            extend_search_window_beyond_SDD_days=config.extend_search_window_beyond_SDD_days,
            min_consec_snow_days_for_seasonal_snow=config.min_consec_snow_days_for_seasonal_snow,
            reproject_method=config.seasonal_snow_mask_reproject_method,
        ).compute()
        # Check if lazily loaded (should be computed/eager after .compute())
        logging.info(f"Retrieved spatiotemporal snow cover mask dataset "
                     f"(spatiotemporal_snow_cover_mask_ds) - {dask_or_computed(spatiotemporal_snow_cover_mask_ds)}")
        monitor_memory_and_cleanup()

        # Get mountain inventory if needed
        if config.mountain_snow_only:
            gmba_clipped_gdf = processing.get_gmba_mountain_inventory(tile.bbox_gdf)
        else:
            gmba_clipped_gdf = None

        # Apply masks
        logging.info("Applying masks...")
        s1_rtc_masked_ds = processing.apply_all_masks(
            s1_rtc_ds=s1_rtc_ds,
            gmba_clipped_gdf=gmba_clipped_gdf,
            spatiotemporal_snow_cover_mask_ds=spatiotemporal_snow_cover_mask_ds,
            water_years=config.water_years,
        )
        # Check if lazily loaded
        logging.info(f"Applied all masks to S1 RTC dataset "
                     f"(s1_rtc_masked_ds) - {dask_or_computed(s1_rtc_masked_ds)}")
        monitor_memory_and_cleanup()

        # Remove bad scenes and border noise
        logging.info("Removing bad scenes and border noise...")
        s1_rtc_masked_ds = processing.remove_bad_scenes_and_border_noise(
            s1_rtc_masked_ds, config.low_backscatter_threshold
        )
        # Check if lazily loaded
        logging.info(f"Removed bad scenes and border noise from S1 RTC "
                     f"dataset (s1_rtc_masked_ds) - {dask_or_computed(s1_rtc_masked_ds)}")
        monitor_memory_and_cleanup()

        # Filter by acquisitions and gaps
        logging.info("Filtering by acquisitions and gaps...")
        s1_rtc_masked_filtered_ds = s1_rtc_masked_ds.groupby("water_year").map(
            lambda group: processing.filter_insufficient_pixels_per_orbit(
                s1_rtc_masked_ds=group,
                spatiotemporal_snow_cover_mask_ds=spatiotemporal_snow_cover_mask_ds,
                min_monthly_acquisitions=config.min_monthly_acquisitions,
                max_allowed_days_gap_per_orbit=config.max_allowed_days_gap_per_orbit,
            )
        )
        # Check if lazily loaded
        logging.info(f"Filtered S1 RTC dataset by acquisitions and gaps "
                     f"(s1_rtc_masked_filtered_ds) - {dask_or_computed(s1_rtc_masked_filtered_ds)}")
        monitor_memory_and_cleanup()

        # Calculate temporal resolution
        logging.info("Calculating temporal resolution...")
        monitor_memory_and_cleanup()
        temporal_resolution_da = processing.get_temporal_resolution(
            s1_rtc_masked_filtered_ds, spatiotemporal_snow_cover_mask_ds
        )
        # Check if lazily loaded
        logging.info(f"Calculated temporal resolution data array "
                     f"(temporal_resolution_da) - {dask_or_computed(temporal_resolution_da)}")

        tile_median_temporal_resolution = temporal_resolution_da.median(
            dim=["latitude", "longitude"]
        )
        tile_pixel_count = temporal_resolution_da.count(
            dim=["latitude", "longitude"]
        )

        # Log if these are lazy
        logging.info(f"Tile median temporal resolution (tile_median_temporal_resolution) - "
                     f"{dask_or_computed(tile_median_temporal_resolution)}")
        logging.info(f"Tile pixel count (tile_pixel_count) - "
                     f"{dask_or_computed(tile_pixel_count)}")

        with dask.config.set(pool=ThreadPoolExecutor(16), scheduler="threads"):
            tile_median_temporal_resolution, tile_pixel_count = dask.compute(
                tile_median_temporal_resolution,
                tile_pixel_count,
            )

        # Store temporal resolution metrics
        for water_year in config.water_years:
            if water_year in tile_median_temporal_resolution.water_year:
                temporal_resolution = tile_median_temporal_resolution.sel(
                    water_year=water_year
                ).values
                setattr(tile, f"tr_{water_year}", round(float(temporal_resolution), 3))

            if water_year in tile_pixel_count.water_year:
                pixel_count = tile_pixel_count.sel(water_year=water_year).values
                setattr(tile, f"pix_ct_{water_year}", int(pixel_count))


        logging.info("Computed runoff_onsets_da, median temporal resolution, and pixel count")
        logging.info(f"Tile median temporal resolution (tile_median_temporal_resolution) - "
                     f"{dask_or_computed(tile_median_temporal_resolution)}")
        logging.info(f"Tile pixel count (tile_pixel_count) - "
                     f"{dask_or_computed(tile_pixel_count)}")

        # Calculate runoff onsets
        logging.info("Calculating runoff onsets...")
        monitor_memory_and_cleanup()
        runoff_onsets_da = s1_rtc_masked_filtered_ds.groupby("water_year").apply(
            processing.calculate_runoff_onset,
            returned_dates_format="dowy",
            return_constituent_runoff_onsets=False,
        )
        # Check if lazily loaded
        logging.info(f"Calculated runoff onsets data array "
                     f"(runoff_onsets_da) - {dask_or_computed(runoff_onsets_da)}")
        monitor_memory_and_cleanup()

        tile.runoff_onsets_dims = dict(runoff_onsets_da.sizes)

        # Calculate median and MAD
        logging.info("Calculating statistics...")
        monitor_memory_and_cleanup()
        median_da, mad_da = processing.median_and_mad_with_min_obs(
            da=runoff_onsets_da,
            dim="water_year",
            min_count=config.min_years_for_median_std
        )
        # Check if lazily loaded
        logging.info(f"Calculated median data array (median_da) - {dask_or_computed(median_da)}")
        logging.info(f"Calculated MAD data array (mad_da) - {dask_or_computed(mad_da)}")

        # Calculate median temporal resolution
        median_temporal_resolution_da = processing.median_with_min_obs(
            da=temporal_resolution_da,
            dim="water_year",
            min_count=config.min_years_for_median_std
        )
        # Check if lazily loaded
        logging.info(f"Calculated median temporal resolution data array "
                     f"(median_temporal_resolution_da) - {dask_or_computed(median_temporal_resolution_da)}")

        # Create dataset
        runoff_onsets_ds = processing.dataarrays_to_dataset(
            runoff_onsets_da=runoff_onsets_da,
            median_da=median_da,
            mad_da=mad_da,
            water_years=config.water_years,
            temporal_resolution_da=temporal_resolution_da,
            median_temporal_resolution_da=median_temporal_resolution_da,
        )
        # Check if dataset is lazy (handle chunking inconsistencies)
        try:
            status = dask_or_computed(runoff_onsets_ds)
        except ValueError as e:
            if "inconsistent chunks" in str(e):
                logging.warning(f"Dataset has inconsistent chunks: {e}")
                # Fix chunking inconsistencies
                runoff_onsets_ds = runoff_onsets_ds.unify_chunks()
                logging.info("Applied unify_chunks() to fix inconsistent chunking")
                # Try again to check lazy loading
                status = dask_or_computed(runoff_onsets_ds)
            else:
                raise

        logging.info(f"Created runoff onsets dataset (runoff_onsets_ds) - {status}")

        # Reindex to global coordinates
        global_ds = xr.open_zarr(config.global_runoff_store, consolidated=True)
        global_subset_ds = global_ds.sel(
            latitude=runoff_onsets_ds.latitude,
            longitude=runoff_onsets_ds.longitude,
            method="nearest",
        )

        runoff_onsets_reindexed_ds = runoff_onsets_ds.assign_coords(
            latitude=global_subset_ds.latitude, 
            longitude=global_subset_ds.longitude
        )

        logging.info(f"Reindexed to global coordinates (runoff_onsets_reindexed_ds) - {dask_or_computed(runoff_onsets_reindexed_ds)}")

        # Write to Zarr
        with dask.config.set(pool=ThreadPoolExecutor(16), scheduler="threads"):
            runoff_onsets_reindexed_ds.drop_vars("spatial_ref").chunk(
                config.chunks_zarr_output
            ).to_zarr(
                config.global_runoff_store, region="auto", mode="r+", consolidated=True
            )
        logging.info("Results written to global zarr store")

        # Record success
        tile.total_time = time.time() - start_time
        tile.success = True

        logging.info(f"Tile ({tile_row}, {tile_col}) processed successfully in {tile.total_time:.2f} seconds")

    except Exception as e:
        error_msg = str(e)
        traceback_msg = traceback.format_exc()

        logging.error(f"Error processing tile ({tile_row}, {tile_col}): {error_msg}")
        logging.error(f"Traceback: {traceback_msg}")

        # Make sure tile is defined even if error occurs early
        if 'tile' not in locals():
            tile = config.get_tile(tile_row, tile_col)
            tile.start_time = time.time()

        tile.error_messages.append(error_msg)
        tile.error_messages.append(traceback_msg)
        tile.total_time = time.time() - tile.start_time
        tile.success = False

    return tile


def save_results_csv(tile, config) -> None:
    """Save tile results directly to main CSV file."""
    save_results_to_main_csv(tile, config)


def save_results_to_main_csv(tile, config) -> None:
    """Append results directly to the main CSV file."""
    import csv
    import time
    import random
    
    main_csv_path = Path(config.tile_results_path)
    main_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create row data
    row_data = [getattr(tile, field, None) for field in config.fields]
    
    # Retry logic for handling concurrent access
    max_retries = 5
    base_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            # Check if file exists and needs header
            needs_header = not main_csv_path.exists()
            
            # Use context manager for atomic operations
            with open(main_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if needs_header:
                    writer.writerow(config.fields)
                
                writer.writerow(row_data)
                f.flush()  # Ensure data is written immediately
            
            logging.info(f"Results appended to main CSV: {main_csv_path}")
            return  # Success!
            
        except (IOError, OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                # Random jitter to avoid thundering herd
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                logging.debug(f"CSV write attempt {attempt + 1} failed, "
                             f"retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
            else:
                # Final attempt failed, re-raise
                raise


def main():
    parser = argparse.ArgumentParser(
        description="Process a single tile for snowmelt runoff onset")
    parser.add_argument("--tile-row", type=int, required=True,
                        help="Tile row coordinate")
    parser.add_argument("--tile-col", type=int, required=True,
                        help="Tile column coordinate")
    parser.add_argument("--config-file", type=str,
                        default="global_config_v9.txt",
                        help="Config file (e.g., global_config_v9.txt)")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.tile_row, args.tile_col)
    
    # Set up required modules
    try:
        setup_modules()
    except Exception as e:
        logging.error(f"Failed to set up required modules: {e}")
        sys.exit(1)
    
    # Get configuration file and load config
    config_file = get_config_file(args.config_file)
    
    # Load configuration once
    from global_snowmelt_runoff_onset.config import Config
    config = Config(config_file)
    
    # Process the tile
    result = process_tile_github_actions(
        args.tile_row,
        args.tile_col,
        config
    )
    
    # result is always a tile object (matching original repository pattern)
    # Always save results CSV, whether success or failure
    save_results_csv(result, config)
    
    if result.success:
        logging.info("Tile processing completed successfully")
        sys.exit(0)
    else:
        logging.error("Tile processing failed")
        if result.error_messages:
            for error in result.error_messages:
                logging.error(f"Error: {error}")
        logging.info("Failed tile results saved to CSV")
        sys.exit(1)


if __name__ == "__main__":
    main()
