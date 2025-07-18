#!/usr/bin/env python3
"""
Script to consolidate GitHub Actions artifacts into the main CSV.

This script can be run locally with downloaded artifacts or in GitHub Actions
to automatically download and consolidate artifacts via the GitHub API.
"""

import os
import sys
import io
from pathlib import Path
import pandas as pd
import argparse
import requests
import zipfile
from datetime import datetime, timedelta, timezone


def consolidate_local_artifacts(artifacts_dir: str,
                                config_version: str = 'v9') -> None:
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
    
    # Use shared consolidation logic
    _save_consolidated_data(consolidated_data, config_version)


def consolidate_github_artifacts(repo: str, token: str, days_back: int = 7,
                                 config_version: str = 'v9') -> None:
    """
    Consolidate artifacts directly from GitHub API.
    
    Args:
        repo: Repository in format 'owner/repo'
        token: GitHub API token
        days_back: How many days back to look for artifacts
        config_version: Config version (e.g., 'v9')
    """
    print(f"🔍 Fetching artifacts from GitHub repo: {repo}")
    
    # Set up GitHub API headers
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Calculate date threshold
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    try:
        # Get workflow runs with pagination
        all_artifacts_data = []
        runs_page = 1
        
        while True:
            runs_url = (f"https://api.github.com/repos/{repo}/actions/runs"
                        f"?per_page=100&page={runs_page}")
            runs_response = requests.get(runs_url, headers=headers)
            runs_response.raise_for_status()
            runs_data = runs_response.json()
            
            if not runs_data['workflow_runs']:
                break  # No more runs
                
            print(f"📄 Processing workflow runs page {runs_page} "
                  f"({len(runs_data['workflow_runs'])} runs)")
            
            for run in runs_data['workflow_runs']:
                # Skip old runs
                run_date = datetime.fromisoformat(
                    run['created_at'].replace('Z', '+00:00'))
                if run_date < cutoff_date:
                    continue
                    
                # Get artifacts for this run with pagination
                artifacts_page = 1
                while True:
                    artifacts_url = (
                        f"https://api.github.com/repos/{repo}/actions/runs/"
                        f"{run['id']}/artifacts?per_page=100&page="
                        f"{artifacts_page}")
                    artifacts_response = requests.get(
                        artifacts_url, headers=headers)
                    artifacts_response.raise_for_status()
                    artifacts_data = artifacts_response.json()
                    
                    if not artifacts_data['artifacts']:
                        break  # No more artifacts for this run
                    
                    if artifacts_page == 1:
                        print(f"  📦 Run {run['id']}: found "
                              f"{artifacts_data['total_count']} artifacts")
                    
                    for artifact in artifacts_data['artifacts']:
                        if 'tile-result' in artifact['name']:
                            print(f"    ✅ Processing artifact: "
                                  f"{artifact['name']}")
                            
                            # Download artifact
                            download_url = artifact['archive_download_url']
                            download_response = requests.get(
                                download_url, headers=headers)
                            download_response.raise_for_status()
                            
                            # Extract and process the zip content
                            with zipfile.ZipFile(
                                    io.BytesIO(download_response.content)
                                    ) as zip_ref:
                                for file_name in zip_ref.namelist():
                                    if file_name.endswith('.csv'):
                                        with zip_ref.open(
                                                file_name) as csv_file:
                                            df = pd.read_csv(csv_file)
                                            all_artifacts_data.append(df)
                    
                    artifacts_page += 1
            
            runs_page += 1
        
        # Consolidate all artifact data
        if all_artifacts_data:
            print(f"📊 Processing {len(all_artifacts_data)} artifact files...")
            consolidated_df = pd.concat(all_artifacts_data, ignore_index=True)
            
            # Convert to list of dictionaries for consistency with local function
            consolidated_data = []
            processed_tiles = set()
            
            for _, row in consolidated_df.iterrows():
                tile_key = (row.get('row'), row.get('col'))
                if tile_key not in processed_tiles:
                    consolidated_data.append(row.to_dict())
                    processed_tiles.add(tile_key)
            
            # Use the shared consolidation logic
            _save_consolidated_data(consolidated_data, config_version)
        else:
            print("⚠️ No artifacts found")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ GitHub API error: {e}")
        raise
    except Exception as e:
        print(f"❌ Error processing artifacts: {e}")
        raise

def to_unix_timestamp(value):
    if isinstance(value, str):
        # Handle ISO datetime strings - convert to datetime first, then to Unix timestamp
        try:
            dt = pd.to_datetime(value)
            return dt.timestamp()
        except:
            # If it's a string that looks like a number, convert to float
            try:
                return float(value)
            except:
                return None
    else:
        # Already a numeric Unix timestamp
        try:
            return float(value)
        except:
            return None       


def _save_consolidated_data(consolidated_data: list, config_version: str) -> None:
    """
    Save consolidated data to main CSV file.
    
    Args:
        consolidated_data: List of tile result dictionaries
        config_version: Config version (e.g., 'v9')
    """
    # Create consolidated DataFrame
    new_df = pd.DataFrame(consolidated_data)
    
    # Path to main results CSV
    main_csv_path = f"processing/tile_data/tile_results_{config_version}.csv"
    
    # Load existing data if it exists
    if os.path.exists(main_csv_path):
        print(f"📄 Loading existing data from {main_csv_path}")
        existing_df = pd.read_csv(main_csv_path)
        
        # Combine with new data
        final_df = pd.concat([existing_df, new_df], ignore_index=True)

        final_df['start_time'] = final_df['start_time'].apply(to_unix_timestamp)
        
        # Remove duplicates based on row/col
        final_df = final_df.sort_values('start_time').drop_duplicates(subset=['row', 'col','success'], keep='last')
        
        print(f"📊 Combined dataset: {len(existing_df)} existing + {len(new_df)} new = {len(final_df)} total")
    else:
        print(f"📄 Creating new file: {main_csv_path}")
        final_df = new_df
    
    # Sort by row, then col
    final_df = final_df.sort_values('start_time').reset_index(drop=True)
    
    # Save to file
    os.makedirs(os.path.dirname(main_csv_path), exist_ok=True)
    final_df.to_csv(main_csv_path, index=False)
    
    # Calculate statistics
    success_count = len(final_df[final_df['success'] == True])
    failure_count = len(final_df[final_df['success'] != True])
    
    print(f"""
✅ Consolidation complete!
• Total tiles: {len(final_df)}
• Successful: {success_count}
• Failed: {failure_count}
• Success rate: {success_count/len(final_df)*100:.1f}%
• New records added: {len(consolidated_data)}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate tile result artifacts into main CSV")
    subparsers = parser.add_subparsers(dest='mode', help='Consolidation mode')
    
    # Local mode
    local_parser = subparsers.add_parser('local', help='Consolidate from local artifacts directory')
    local_parser.add_argument("--artifacts-dir", type=str, required=True,
                             help="Directory containing extracted artifact CSV files")
    local_parser.add_argument("--config-version", type=str, default="v9",
                             help="Config version (e.g., v9)")
    
    # GitHub mode
    github_parser = subparsers.add_parser('github', help='Consolidate from GitHub API')
    github_parser.add_argument("--repo", type=str, required=True,
                              help="GitHub repository in format 'owner/repo'")
    github_parser.add_argument("--token", type=str, required=True,
                              help="GitHub API token")
    github_parser.add_argument("--days-back", type=int, default=7,
                              help="How many days back to look for artifacts")
    github_parser.add_argument("--config-version", type=str, default="v9",
                              help="Config version (e.g., v9)")
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.mode == 'github':
            consolidate_github_artifacts(args.repo, args.token, 
                                        args.days_back, args.config_version)
        elif args.mode == 'local':
            consolidate_local_artifacts(args.artifacts_dir, args.config_version)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
