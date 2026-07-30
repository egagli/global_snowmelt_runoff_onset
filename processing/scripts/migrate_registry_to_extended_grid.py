"""
Migrate the v10 tile registry from the pre-2026-07-30 grid to the extended grid.

The v10 grid gained 2 tile rows in the north (top 81.099 -> 84.048) and extended
south to a whole tile boundary (bottom -59.999 -> -63.4074), so:

  * every evaluated tile keeps its footprint but its row index shifts by +2;
  * tile rows 0-1 (82.57-84.05N) and 98-99 (-60.46 to -63.41) are new;
  * old row 95 -> new row 97 was a PARTIAL row (1410 of 2048 px) and is now full,
    so its footprint grew and its per-tile statistics no longer describe it.

This does the mechanical part only: it shifts row indices for the evaluated tiles
and inserts placeholder rows (to_process = False) for the rows that have never been
evaluated, so the registry is self-consistent and safe to dispatch from in the
meantime. It does NOT compute seasonal-snow percentages or probe the S1 catalog for
the new rows -- rerun processing/select_tiles_to_process.ipynb for that.

Usage:
    python processing/scripts/migrate_registry_to_extended_grid.py [--dry-run]

Idempotent: refuses to run twice (detects an already-migrated registry).
"""

import argparse
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from global_snowmelt_runoff_onset.config import Config

ROW_SHIFT = 2
NEW_NORTH_ROWS = (0, 1)
GREW_ROW = 97  # old 95, was partial in y
PENDING_NOTE = ("new grid row (2026-07-30 north/south extension); "
                "not yet evaluated -- rerun select_tiles_to_process.ipynb")
REGREW_NOTE = ("row grew from 1410 to 2048 px when the south edge moved to a tile "
               "boundary (2026-07-30); statistics describe only the northern 1410 "
               "px -- rerun select_tiles_to_process.ipynb")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing the file")
    args = ap.parse_args()

    config = Config("config/global_config_v10.txt")
    path = config.valid_tiles_geojson_path
    n_tile_rows, n_tile_cols = (int(v) for v in config.geobox_tiles.shape.yx)
    new_south_rows = tuple(range(n_tile_rows - ROW_SHIFT, n_tile_rows))

    gdf = gpd.read_file(path)
    print(f"registry: {path}\n  {len(gdf)} rows, tile rows {gdf.row.min()}-{gdf.row.max()}, "
          f"cols {gdf.col.min()}-{gdf.col.max()}, to_process={int(gdf.to_process.sum())}")

    # --- guard: is this the pre-extension registry? ---
    if gdf.row.max() == n_tile_rows - 1:
        print(f"\nrow range already spans the extended grid (0-{n_tile_rows-1}); "
              "nothing to do. Refusing to shift a second time.")
        return 0
    expected_old_max = n_tile_rows - 1 - ROW_SHIFT * 2
    if gdf.row.max() != expected_old_max:
        print(f"\nERROR: expected the old registry to have max row {expected_old_max}, "
              f"found {gdf.row.max()}. Not migrating -- inspect by hand.")
        return 1

    # --- shift, and verify each shifted tile's stored footprint against the new grid ---
    gdf = gdf.copy()
    gdf["row"] = gdf["row"] + ROW_SHIFT
    mismatch, checked = 0, 0
    for record in gdf.itertuples():
        if record.row == GREW_ROW:
            continue  # footprint legitimately grew; replaced below
        footprint = config.geobox_tiles[record.row, record.col].extent.geom
        checked += 1
        # 1e-9 deg ~ 0.1 mm: catches a real misalignment, ignores serialization noise
        if not record.geometry.equals_exact(footprint, 1e-9):
            mismatch += 1
            if mismatch <= 3:
                print(f"  footprint mismatch at new tile ({record.row},{record.col})")
    print(f"\nfootprint check: {checked} shifted tiles verified against the new grid, "
          f"{mismatch} mismatches")
    if mismatch:
        print("ERROR: shifted tiles do not land on their new-grid footprints. Not writing.")
        return 1

    # --- the formerly-partial row now covers more ground: reset it for re-evaluation ---
    to_process_before = int(gdf.to_process.sum())
    grew = gdf.row == GREW_ROW
    grew_to_process = int(gdf.loc[grew, "to_process"].sum())
    print(f"\nrow {GREW_ROW} (was partial): {int(grew.sum())} tiles, "
          f"{grew_to_process} of them to_process -> reset to pending")
    gdf.loc[grew, "geometry"] = [
        config.geobox_tiles[GREW_ROW, c].extent.geom for c in gdf.loc[grew, "col"]
    ]
    gdf.loc[grew, "to_process"] = False
    gdf.loc[grew, "tile_notes"] = REGREW_NOTE

    # --- placeholder rows for the never-evaluated new rows ---
    stat_columns = [c for c in gdf.columns
                    if c not in ("row", "col", "to_process", "tile_notes", "geometry")]
    placeholders = []
    for row in (*NEW_NORTH_ROWS, *new_south_rows):
        for col in range(n_tile_cols):
            record = {"row": row, "col": col, "to_process": False,
                      "tile_notes": PENDING_NOTE,
                      "geometry": config.geobox_tiles[row, col].extent.geom}
            for column in stat_columns:
                record[column] = pd.NaT if "date" in column else np.nan
            placeholders.append(record)
    new_gdf = gpd.GeoDataFrame(placeholders, crs=gdf.crs)[gdf.columns]
    print(f"placeholders: {len(new_gdf)} tiles across rows "
          f"{sorted({*NEW_NORTH_ROWS, *new_south_rows})}")

    out = pd.concat([gdf, new_gdf], ignore_index=True)
    out = gpd.GeoDataFrame(out, crs=gdf.crs).sort_values(["row", "col"]).reset_index(drop=True)

    # --- invariants ---
    assert len(out) == n_tile_rows * n_tile_cols, (len(out), n_tile_rows * n_tile_cols)
    assert not out.duplicated(subset=["row", "col"]).any(), "duplicate (row, col)"
    assert out.row.min() == 0 and out.row.max() == n_tile_rows - 1
    # The work universe must be exactly what it was, minus any tile in the row whose
    # footprint grew (none today, but assert rather than assume).
    assert int(out.to_process.sum()) == to_process_before - grew_to_process, (
        f"to_process went {to_process_before} -> {int(out.to_process.sum())}, "
        f"expected -{grew_to_process}")
    print(f"\nresult: {len(out)} rows ({n_tile_rows} x {n_tile_cols}), "
          f"to_process={int(out.to_process.sum())} (was {to_process_before}, "
          f"-{grew_to_process} from the grown row)")

    if args.dry_run:
        print("\n--dry-run: not written")
        return 0
    out.to_file(path, driver="GeoJSON")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
