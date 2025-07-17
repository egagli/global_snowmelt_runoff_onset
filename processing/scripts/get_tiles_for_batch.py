#!/usr/bin/env python3
"""
Generate tile lists for batch processing based on config parameters.

This script uses the Config class to get filtered lists of tiles for batch processing
in GitHub Actions workflows.
"""

import argparse
import json
import sys
import logging
from pathlib import Path

# Add the parent directory to Python path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

def get_config_file(config_file: str) -> str:
    """
    Get the full path to the configuration file.
    
    Args:
        config_file: Name of the configuration file (with or without path)
        
    Returns:
        Full path to the configuration file
    """
    # If the config_file already starts with "config/", use it as is
    if config_file.startswith("config/"):
        return config_file
    
    # Otherwise, prepend "config/" to the filename
    return f"config/{config_file}"


def get_tiles_for_batch(config_file: str, which_tiles: str, how_many: int,
                        output_format: str = 'list', batch_size: int = 256,
                        batch_index: int = 0) -> list:
    """
    Get a list of tiles for batch processing.
    
    Args:
        config_file: Path to configuration file
        which_tiles: Filter criterion for tiles to process
        how_many: Maximum number of tiles to return
        output_format: Output format ('json', 'list', 'count') - suppresses
                      config logging for 'json' and 'count'
        batch_size: Number of tiles per batch (default 256 for GitHub Actions limit)
        batch_index: Which batch to return (0-based index)
        
    Returns:
        List of dictionaries with tile coordinates
    """
    # Get the full config file path
    config_file = get_config_file(config_file)
    
    # Suppress configuration logging for JSON and count outputs
    if output_format in ['json', 'count']:
        # Redirect stdout to suppress print statements during config loading
        import io
        import contextlib
        
        # Capture stdout during config loading
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            # Also disable logging during config loading
            logging.disable(logging.CRITICAL)
            
            # Load configuration
            from global_snowmelt_runoff_onset.config import Config
            config = Config(config_file)
            
            # Re-enable logging after config loading
            logging.disable(logging.NOTSET)
    else:
        # Load configuration normally for 'list' output
        from global_snowmelt_runoff_onset.config import Config
        config = Config(config_file)
    
    # Get filtered list of tiles
    tiles = config.get_list_of_tiles(which_tiles)
    
    # Limit to requested number
    if how_many > 0:
        tiles = tiles[:how_many]
    
    # If batch_index is specified, slice the tiles for this batch
    if batch_index >= 0:
        start_idx = batch_index * batch_size
        end_idx = start_idx + batch_size
        tiles = tiles[start_idx:end_idx]
    
    # Convert to list of dictionaries for JSON serialization
    tile_list = [{"row": tile.row, "col": tile.col} for tile in tiles]
    
    return tile_list


def get_batch_info(config_file: str, which_tiles: str, how_many: int,
                   batch_size: int = 256) -> dict:
    """
    Get information about how many batches are needed.
    
    Args:
        config_file: Path to configuration file
        which_tiles: Filter criterion for tiles to process
        how_many: Maximum number of tiles to return
        batch_size: Number of tiles per batch
        
    Returns:
        Dictionary with batch information
    """
    # Get the full config file path
    config_file = get_config_file(config_file)
    
    # Load configuration (suppress output)
    import io
    import contextlib
    
    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        logging.disable(logging.CRITICAL)
        from global_snowmelt_runoff_onset.config import Config
        config = Config(config_file)
        logging.disable(logging.NOTSET)
    
    # Get filtered list of tiles
    tiles = config.get_list_of_tiles(which_tiles)
    
    # Limit to requested number
    if how_many > 0:
        tiles = tiles[:how_many]
    
    total_tiles = len(tiles)
    num_batches = (total_tiles + batch_size - 1) // batch_size  # Ceiling division
    
    return {
        "total_tiles": total_tiles,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "batches": [{"batch_index": i} for i in range(num_batches)]
    }

def main():
    parser = argparse.ArgumentParser(
        description="Generate tile lists for batch processing")
    parser.add_argument("--config-file", type=str,
                        default="config/global_config_v9.txt",
                        help="Config file (e.g., config/global_config_v9.txt)")
    parser.add_argument("--which-tiles", type=str,
                        default="unprocessed",
                        choices=['all', 'processed', 'failed', 'unprocessed',
                                'unprocessed_and_failed',
                                'unprocessed_and_failed_weather_stations'],
                        help="Which tiles to process")
    parser.add_argument("--how-many", type=int,
                        default=10,
                        help="Maximum number of tiles to return (0 for all)")
    parser.add_argument("--output", type=str,
                        choices=['json', 'list', 'count', 'batch-info'],
                        default='json',
                        help="Output format")
    parser.add_argument("--batch-size", type=int,
                        default=256,
                        help="Number of tiles per batch")
    parser.add_argument("--batch-index", type=int,
                        default=-1,
                        help="Which batch to return (0-based, -1 for all)")
    
    args = parser.parse_args()
    
    try:
        if args.output == 'batch-info':
            # Get batch information
            batch_info = get_batch_info(args.config_file, args.which_tiles,
                                       args.how_many, args.batch_size)
            print(json.dumps(batch_info, separators=(',', ':')))
        else:
            # Get tile list
            tile_list = get_tiles_for_batch(
                args.config_file, args.which_tiles, args.how_many,
                args.output, args.batch_size, args.batch_index)
            
            if args.output == 'json':
                # Output JSON for GitHub Actions matrix
                json_output = json.dumps(tile_list, separators=(',', ':'))
                print(json_output)
            elif args.output == 'list':
                # Output human-readable list
                print(f"Found {len(tile_list)} tiles matching criteria "
                      f"'{args.which_tiles}':")
                for i, tile in enumerate(tile_list, 1):
                    print(f"{i:3d}: tile ({tile['row']}, {tile['col']})")
            elif args.output == 'count':
                # Output just the count
                print(len(tile_list))
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
