#!/usr/bin/env python3
"""
Manual script to consolidate GitHub Actions artifacts into the main CSV.

This script can be run locally to consolidate artifacts without waiting for
the scheduled workflow.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import argparse

# Add the parent directory to Python path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

def consolidate_local_artifacts(artifacts_dir: str, config_version: str = 'v9') -> None:
    """
    Consolidate CSV files from a local artifacts directory.
    
    Args:
        artifacts_dir: Directory containing extracted artifact CSV files
        config_version: Config version (e.g., 'v9')
    """
    artifacts_path = Path(artifacts_dir)
    
    if not artifacts_path.exists():
        print(f"❌ Artifacts directory not found: {artifacts_dir}")
        return
    
    # Find all CSV files in the artifacts directory
    csv_files = list(artifacts_path.rglob('*.csv'))
    
    if not csv_files:
        print(f"❌ No CSV files found in {artifacts_dir}")
        return
    
    print(f"📦 Found {len(csv_files)} CSV files to consolidate")
    
    consolidated_data = []
    processed_tiles = set()
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                for _, row in df.iterrows():
                    tile_key = (row.get('row'), row.get('col'))
                    if tile_key not in processed_tiles:
                        consolidated_data.append(row.to_dict())
                        processed_tiles.add(tile_key)
                print(f"  ✅ Processed {len(df)} records from {csv_file.name}")
        except Exception as e:
            print(f"  ⚠️  Error processing {csv_file}: {e}")
    
    if not consolidated_data:
        print("✅ No new data to consolidate")
        return
    
    print(f"📊 Consolidated {len(consolidated_data)} unique tile results")
    
    # Create consolidated DataFrame
    new_df = pd.DataFrame(consolidated_data)
    
    # Path to main results CSV
    main_csv_path = f"processing/tile_data/tile_results_{config_version}.csv"
    
    # Load existing data if it exists
    if os.path.exists(main_csv_path):
        try:
            existing_df = pd.read_csv(main_csv_path)
            print(f"📋 Loaded existing CSV with {len(existing_df)} records")
            
            # Remove duplicates - keep the newest version of each tile
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Sort by processing time if available, otherwise by row order
            if 'total_time' in combined_df.columns:
                combined_df = combined_df.sort_values('total_time', ascending=False)
            
            # Keep only the latest result for each tile
            final_df = combined_df.drop_duplicates(subset=['row', 'col'], keep='first')
            
            print(f"🔄 After deduplication: {len(final_df)} total records")
        except Exception as e:
            print(f"⚠️  Error reading existing CSV: {e}")
            final_df = new_df
    else:
        print("📄 Creating new main CSV file")
        final_df = new_df
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(main_csv_path), exist_ok=True)
    
    # Save consolidated results
    final_df.to_csv(main_csv_path, index=False)
    print(f"💾 Saved consolidated results to {main_csv_path}")
    
    # Create summary
    success_count = len(final_df[final_df.get('success', False) == True])
    failure_count = len(final_df[final_df.get('success', False) == False])
    
    print(f"""
📈 CONSOLIDATION SUMMARY:
========================
• Total tiles: {len(final_df)}
• Successful: {success_count}
• Failed: {failure_count}
• Success rate: {success_count/len(final_df)*100:.1f}%
• New records added: {len(consolidated_data)}
""")

def main():
    parser = argparse.ArgumentParser(
        description="Consolidate tile result artifacts into main CSV")
    parser.add_argument("--artifacts-dir", type=str, required=True,
                        help="Directory containing extracted artifact CSV files")
    parser.add_argument("--config-version", type=str, default="v9",
                        help="Config version (e.g., v9)")
    
    args = parser.parse_args()
    
    try:
        consolidate_local_artifacts(args.artifacts_dir, args.config_version)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
