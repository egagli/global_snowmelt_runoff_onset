#!/usr/bin/env python3
"""
Synchronize individual tile CSV files with the main tile_results_vX.csv file.

This script reads the latest results from all individual tile CSV files in
processing/tile_data/github_workflow_results/ and appends them to the main
tile_results_vX.csv file, avoiding duplicates.
"""

import argparse
import logging
import sys
from pathlib import Path
from contextlib import redirect_stdout
import io

import pandas as pd

# Add the parent directory to Python path to import our modules
sys.path.append(str(Path(__file__).parent.parent))


def setup_logging():
    """Set up logging for the sync process."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "sync_tile_results.log"
    
    # Configure root logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("Starting tile results synchronization")


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


def extract_version_from_config(config_filename: str) -> str:
    """Extract version string from config filename."""
    # Extract version from config filename
    # (e.g., "global_config_v9.txt" -> "v9")
    if 'v' in config_filename:
        version = f"v{config_filename.split('v')[1].split('.')[0]}"
    else:
        # Fallback - use filename without extension
        version = Path(config_filename).stem.replace('global_config_', '')
    
    return version


def find_individual_tile_csvs(version: str) -> list[Path]:
    """Find all individual tile CSV files for the given version."""
    results_dir = Path("processing/tile_data/github_workflow_results")
    
    if not results_dir.exists():
        logging.warning(f"Results directory does not exist: {results_dir}")
        return []
    
    # Pattern: tile_row_col_vX.csv
    pattern = f"tile_*_{version}.csv"
    csv_files = list(results_dir.glob(pattern))
    
    logging.info(f"Found {len(csv_files)} individual tile CSV files "
                 f"for {version}")
    return csv_files


def get_main_csv_path(version: str) -> Path:
    """Get the path to the main tile_results_vX.csv file."""
    tile_data_dir = Path("processing/tile_data")
    main_csv = tile_data_dir / f"tile_results_{version}.csv"
    return main_csv


def read_latest_results_from_individual_csvs(
        csv_files: list[Path]) -> pd.DataFrame:
    """Read the last line from each individual CSV file."""
    all_results = []
    
    for csv_file in csv_files:
        try:
            # Read the entire CSV file
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                logging.warning(f"Empty CSV file: {csv_file}")
                continue
            
            # Get the last row (most recent result for this tile)
            latest_result = df.iloc[-1:].copy()
            
            logging.debug(f"Read latest result from {csv_file.name}: "
                          f"row={latest_result.iloc[0].get('row', 'N/A')}, "
                          f"col={latest_result.iloc[0].get('col', 'N/A')}, "
                          f"success={latest_result.iloc[0].get('success')}")
            
            all_results.append(latest_result)
            
        except Exception as e:
            logging.error(f"Error reading {csv_file}: {e}")
            continue
    
    if not all_results:
        logging.warning("No valid results found in individual CSV files")
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    logging.info(f"Combined {len(combined_df)} tile results "
                 f"from individual CSV files")
    
    return combined_df


def sync_to_main_csv(new_results: pd.DataFrame, main_csv_path: Path,
                     config) -> bool:
    """Synchronize new results to the main CSV file, avoiding duplicates."""
    if len(new_results) == 0:
        logging.info("No new results to sync")
        return False
    
    # Ensure the new_results DataFrame has the correct column order
    # matching config.fields (which matches the main CSV)
    try:
        # Reorder columns to match config.fields
        ordered_results = new_results.reindex(columns=config.fields)
        logging.info(f"Reordered {len(config.fields)} columns to match "
                     f"main CSV format")
    except KeyError as e:
        logging.error(f"Missing columns in individual CSV data: {e}")
        # Fill missing columns with None/NaN
        for field in config.fields:
            if field not in new_results.columns:
                new_results[field] = None
                logging.warning(f"Added missing field '{field}' "
                                f"with None values")
        ordered_results = new_results.reindex(columns=config.fields)
    
    changes_made = False
    
    # Check if main CSV exists
    if main_csv_path.exists():
        existing_df = pd.read_csv(main_csv_path)
        logging.info(f"Main CSV exists with {len(existing_df)} records")
        
        # Identify unique tiles by row/col combination
        if 'row' in existing_df.columns and 'col' in existing_df.columns:
            existing_tiles = set(zip(existing_df['row'], existing_df['col']))
            
            # Find tiles that are new or have changed since last sync
            tiles_to_add = []
            for _, row in ordered_results.iterrows():
                tile_key = (row['row'], row['col'])
                
                if tile_key not in existing_tiles:
                    # Completely new tile
                    tiles_to_add.append(row)
                    logging.info(f"New tile: ({row['row']}, {row['col']})")
                else:
                    # Tile exists - compare with most recent entry in main CSV
                    # Get all entries for this tile (there might be multiple)
                    tile_entries = existing_df[
                        (existing_df['row'] == row['row']) &
                        (existing_df['col'] == row['col'])
                    ]
                    
                    # Get the most recent entry (last one in the CSV)
                    most_recent_entry = tile_entries.iloc[-1]
                    
                    # Compare key fields to see if this is actually different
                    # We'll compare success, total_time, and error_messages
                    # (these are the most likely to change between runs)
                    key_fields_to_compare = ['success', 'total_time',
                                             'error_messages']
                    
                    is_different = False
                    for field in key_fields_to_compare:
                        if field in row and field in most_recent_entry:
                            new_val = row[field]
                            old_val = most_recent_entry[field]
                            
                            # Handle NaN/None comparisons
                            if pd.isna(new_val) and pd.isna(old_val):
                                continue  # Both are NaN, consider them equal
                            elif pd.isna(new_val) or pd.isna(old_val):
                                is_different = True
                                break
                            elif str(new_val) != str(old_val):
                                is_different = True
                                logging.debug(f"Field '{field}' changed for "
                                              f"tile ({row['row']}, "
                                              f"{row['col']}): '{old_val}' -> "
                                              f"'{new_val}'")
                                break
                    
                    if is_different:
                        tiles_to_add.append(row)
                        logging.info(f"Updated tile (content changed): "
                                     f"({row['row']}, {row['col']})")
                    else:
                        logging.info(f"Skipping unchanged tile: "
                                     f"({row['row']}, {row['col']})")
            
            if tiles_to_add:
                new_rows_df = pd.DataFrame(tiles_to_add)
                # Append to existing file (columns already in correct order)
                new_rows_df.to_csv(main_csv_path, mode='a', header=False,
                                   index=False)
                logging.info(f"Appended {len(new_rows_df)} new/updated "
                             f"records to {main_csv_path}")
                changes_made = True
            else:
                logging.info("No new or updated records to add")
        else:
            # If no row/col columns, just append everything
            ordered_results.to_csv(main_csv_path, mode='a', header=False,
                                   index=False)
            logging.info(f"Appended {len(ordered_results)} records "
                         f"to {main_csv_path}")
            changes_made = True
    else:
        # Create new main CSV file with correct column order
        main_csv_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_results.to_csv(main_csv_path, index=False)
        logging.info(f"Created new main CSV file {main_csv_path} "
                     f"with {len(ordered_results)} records")
        logging.info(f"CSV column order: {list(ordered_results.columns)}")
        changes_made = True
    
    return changes_made


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize individual tile CSV files "
                    "with main tile_results_vX.csv")
    parser.add_argument("--config-file", type=str,
                        default="global_config_v9.txt",
                        help="Config file to determine version "
                             "(e.g., global_config_v9.txt)")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    try:
        # Get configuration file and extract version
        config_file = get_config_file(args.config_file)
        version = extract_version_from_config(args.config_file)
        
        logging.info(f"Synchronizing results for config: {args.config_file}, "
                     f"version: {version}")
        
        # Load configuration to get field names
        # (suppress config printing to stdout)
        f = io.StringIO()
        with redirect_stdout(f):
            from global_snowmelt_runoff_onset.config import Config
            config = Config(config_file)
        
        # Find all individual tile CSV files
        individual_csvs = find_individual_tile_csvs(version)
        
        if not individual_csvs:
            logging.warning("No individual CSV files found to sync")
            return
        
        # Read latest results from each individual CSV
        new_results = read_latest_results_from_individual_csvs(individual_csvs)
        
        if len(new_results) == 0:
            logging.warning("No valid results found to sync")
            return
        
        # Get main CSV path
        main_csv_path = get_main_csv_path(version)
        
        # Sync to main CSV
        changes_made = sync_to_main_csv(new_results, main_csv_path, config)
        
        if changes_made:
            logging.info(f"Successfully synchronized {len(new_results)} "
                         f"tile results")
            logging.info(f"Main CSV file: {main_csv_path}")
        else:
            logging.info("No changes made during synchronization")
            
    except Exception as e:
        logging.error(f"Error during synchronization: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
