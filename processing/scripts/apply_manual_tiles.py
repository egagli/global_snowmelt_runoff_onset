#!/usr/bin/env python3
"""
Force-select tiles in the v10 tile registry from a manual tile list.

Flips ``to_process`` to True in the registry geojson for every "row,col" line
in tile_data/manual_tiles_v10.txt and appends a marker to ``tile_notes`` —
registry-level manual additions on top of the catalog rule
(has_seasonal_snow & n_vv_items > 40). Because the whole work universe
(status.get_tile_status_gdf / get_remaining_work / the dispatchers) is derived
from ``to_process``, this is all it takes: the tiles appear as 'unprocessed'
work items on the next dispatch, with normal redispatch-on-failure and
composite tracking.

Safe with respect to processing progress: the registry is a static input that
status derivation JOINS against the icechunk commit history at read time —
nothing in the store or its commits is touched, and already-committed tile
years are never redone.

Run this after editing the manual list instead of rerunning
0_select_tiles_to_process.ipynb (a full registry rebuild re-probes the S1
catalog — slow, and borderline tiles near the 40-item threshold can flip
either way at a new probe date). The notebook applies the same list itself at
registry-build time, so manual additions survive a legitimate rebuild.

Idempotent: re-running changes nothing once the flags are set.
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd

REPO_ROOT = Path(__file__).parent.parent.parent
MANUAL_NOTE = "manually added (manual_tiles_v10.txt)"


def load_manual_tiles(path: Path) -> set:
    tiles = set()
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        try:
            r, c = line.split(",")
            tiles.add((int(r), int(c)))
        except ValueError:
            sys.exit(f"{path}:{lineno}: expected 'row,col', got {line!r}")
    return tiles


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--registry", type=str,
                        default="processing/tile_data/global_tiles_with_seasonal_snow_v10.geojson",
                        help="Registry geojson, relative to the repo root")
    parser.add_argument("--manual-file", type=str,
                        default="processing/tile_data/manual_tiles_v10.txt",
                        help="Manual tile list ('row,col' lines, # comments), relative to the repo root")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing the registry")
    args = parser.parse_args()

    registry_path = REPO_ROOT / args.registry
    manual_path = REPO_ROOT / args.manual_file
    manual = load_manual_tiles(manual_path)
    print(f"{len(manual)} tiles listed in {manual_path.name}")

    gdf = gpd.read_file(registry_path)
    registry_keys = set(zip(gdf["row"].astype(int), gdf["col"].astype(int)))
    unknown = manual - registry_keys
    if unknown:
        sys.exit(f"ERROR: {len(unknown)} manual tiles not in the registry grid: {sorted(unknown)[:10]}")

    is_manual = [(int(r), int(c)) in manual for r, c in zip(gdf["row"], gdf["col"])]
    is_manual = gdf.assign(_m=is_manual)["_m"].to_numpy()
    newly = is_manual & ~gdf["to_process"].to_numpy()
    needs_note = newly & ~gdf["tile_notes"].str.contains(MANUAL_NOTE, regex=False).to_numpy()

    print(f"already to_process: {int(is_manual.sum() - newly.sum())}; "
          f"newly flipped: {int(newly.sum())}")
    if args.dry_run:
        print(f"dry run -- registry not written "
              f"(would have {int((gdf['to_process'] | is_manual).sum()):,} to_process tiles)")
        return

    gdf.loc[needs_note, "tile_notes"] = gdf.loc[needs_note, "tile_notes"] + "; " + MANUAL_NOTE
    gdf.loc[is_manual, "to_process"] = True
    gdf.to_file(registry_path, driver="GeoJSON")
    print(f"wrote {registry_path.name}: {int(gdf['to_process'].sum()):,} to_process tiles")


if __name__ == "__main__":
    main()
