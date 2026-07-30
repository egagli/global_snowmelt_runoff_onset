"""
Verify that two config versions describe the same pixel lattice, and report the
constant (row, col) offset between their tile grids.

The v10 grid was extended north and south relative to <= v9 (2026-07-30):

    top     81.099   -> 84.048     (+4096 rows = +2 whole tile rows)
    bottom -59.999   -> -63.4074   (latitude now an exact multiple of 2048)

Both edges snap to the same lattice of `resolution` multiples, so no pixel center
moves: v9 pixel (i, j) is v10 pixel (i + 4096, j), and v9 tile (r, c) is v10 tile
(r + 2, c) with a byte-identical geobox. That is what makes tile-wise comparison
against the published v9 store exact rather than approximate.

Run after ANY change to a config's bbox/resolution/tile size:

    python processing/scripts/verify_grid_alignment.py
    python processing/scripts/verify_grid_alignment.py --old v9 --new v10

Reads the config files directly (configparser, no credentials) so it works in CI.
Exits non-zero if the grids are not lattice-compatible.
"""

import argparse
import configparser
import pathlib
import sys

import numpy as np
from odc.geo.geobox import GeoBox, GeoboxTiles

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_grid(version: str):
    """Build (geobox, tiles, tile_dim) for a config version, without credentials."""
    path = REPO_ROOT / "config" / f"global_config_{version}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    parser = configparser.ConfigParser()
    parser.read(path)
    values = parser["VALUES"]
    resolution = values.getfloat("resolution")
    bbox = (
        values.getfloat("bbox_left"),
        values.getfloat("bbox_bottom"),
        values.getfloat("bbox_right"),
        values.getfloat("bbox_top"),
    )
    if "spatial_chunk_dim_zarr_output" in values:
        tile_dim = values.getint("spatial_chunk_dim_zarr_output")
    else:
        tile_dim = values.getint("spatial_chunk_dim")
    geobox = GeoBox.from_bbox(bbox, crs="epsg:4326", resolution=resolution)
    return geobox, GeoboxTiles(geobox, (tile_dim, tile_dim)), tile_dim, resolution


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="v9", help="older config version (default: v9)")
    ap.add_argument("--new", default="v10", help="newer config version (default: v10)")
    args = ap.parse_args()

    old_gb, old_tiles, old_dim, old_res = load_grid(args.old)
    new_gb, new_tiles, new_dim, new_res = load_grid(args.new)
    failures = []

    print(f"{args.old}: {tuple(old_gb.shape.yx)} px, {tuple(old_tiles.shape.yx)} tiles of {old_dim}")
    print(f"     lat {old_gb.extent.boundingbox.bottom!r} .. {old_gb.transform.f!r}")
    print(f"{args.new}: {tuple(new_gb.shape.yx)} px, {tuple(new_tiles.shape.yx)} tiles of {new_dim}")
    print(f"     lat {new_gb.extent.boundingbox.bottom!r} .. {new_gb.transform.f!r}")

    if old_res != new_res:
        failures.append(f"resolutions differ ({old_res} vs {new_res}) -- grids are not comparable")
        print("\n".join(failures))
        return 1
    if old_dim != new_dim:
        failures.append(f"tile sizes differ ({old_dim} vs {new_dim}); tile indices cannot map 1:1")

    # --- pixel offset between the two origins, in pixels ---
    # transform.f is the NORTH edge with a -res step, so old row i lands on new row
    # i + (new.f - old.f)/res; transform.c is the WEST edge with a +res step, hence
    # the opposite sign for columns.
    row_off_f = (new_gb.transform.f - old_gb.transform.f) / new_res
    col_off_f = (old_gb.transform.c - new_gb.transform.c) / new_res
    row_off, col_off = round(row_off_f), round(col_off_f)
    print(f"\npixel offset {args.old} -> {args.new}: row {row_off_f:+.9f}, col {col_off_f:+.9f}")
    for name, exact, rounded in (("row", row_off_f, row_off), ("col", col_off_f, col_off)):
        if abs(exact - rounded) > 1e-6:
            failures.append(
                f"{name} offset {exact} is not an integer number of pixels -- the two "
                "grids are on DIFFERENT lattices and no exact pixel mapping exists"
            )
    print(f"  => {args.old} pixel (i, j) == {args.new} pixel (i{row_off:+d}, j{col_off:+d})")

    if row_off % new_dim or col_off % new_dim:
        failures.append(
            f"pixel offset ({row_off}, {col_off}) is not a whole number of {new_dim}-px "
            "tiles, so tile footprints do not correspond 1:1 (data would need resampling "
            "or re-tiling to compare tile-wise)"
        )
    tile_row_off, tile_col_off = row_off // new_dim, col_off // new_dim
    print(f"  => {args.old} tile (r, c) == {args.new} tile (r{tile_row_off:+d}, c{tile_col_off:+d})")

    # --- exhaustive tile-by-tile geobox identity ---
    n_checked, mismatched = 0, []
    for r in range(old_tiles.shape[0]):
        for c in range(old_tiles.shape[1]):
            nr, nc = r + tile_row_off, c + tile_col_off
            if not (0 <= nr < new_tiles.shape[0] and 0 <= nc < new_tiles.shape[1]):
                mismatched.append((r, c, "outside new grid"))
                continue
            a, b = old_tiles[r, c], new_tiles[nr, nc]
            n_checked += 1
            if not a.transform.almost_equals(b.transform, precision=1e-12):
                mismatched.append((r, c, "transform"))
            elif a.shape != b.shape:
                # An edge tile that was partial in the old grid and is full in the new
                # one covers the same ground over its overlapping part; flag, don't fail.
                mismatched.append((r, c, f"shape {tuple(a.shape.yx)} -> {tuple(b.shape.yx)}"))
    print(f"\nexhaustive tile check: {n_checked} tiles compared")
    hard = [m for m in mismatched if m[2] in ("transform", "outside new grid")]
    soft = [m for m in mismatched if m not in hard]
    if hard:
        failures.append(f"{len(hard)} tiles have a different origin/are missing, e.g. {hard[:3]}")
    else:
        print("  origins: all identical (same ground, byte-identical transforms)")
    if soft:
        print(f"  shape changes (same origin, edge tile grew): {len(soft)}")
        for r, c, what in soft[:4]:
            print(f"    {args.old} tile ({r},{c}) -> {args.new} ({r+tile_row_off},{c+tile_col_off}): {what}")
        if len(soft) > 4:
            print(f"    ... and {len(soft)-4} more (same row)")

    # --- coordinate arrays must match element-for-element under the offset ---
    # The transforms above are the authoritative comparison (bit-equal). Materialized
    # coordinate VALUES are computed as origin + (i + 0.5) * step, so a different
    # origin can leave a 1-ULP difference (~3e-14 deg = nanometres at these
    # latitudes). Anyone aligning v9 and v10 by coordinate rather than by integer
    # index must therefore use a tolerance; xr.align/assert_array_equal would fail.
    COORD_TOL = 1e-9  # deg; ~0.1 mm, far below one 80 m pixel
    for dim, off in (("latitude", row_off), ("longitude", col_off)):
        o = old_gb.coordinates[dim].values
        n = new_gb.coordinates[dim].values
        window = n[off:off + len(o)]
        if len(window) != len(o):
            failures.append(f"{dim}: {args.new} does not fully contain {args.old}")
            continue
        max_diff = float(np.abs(o - window).max())
        if max_diff > COORD_TOL:
            failures.append(f"{dim}: coordinate values differ under the offset "
                            f"(max |diff| {max_diff:g} deg > {COORD_TOL:g})")
        else:
            exact = " (bit-exact)" if np.array_equal(o, window) else \
                    f" (max |diff| {max_diff:g} deg = float noise)"
            print(f"  {dim}: {args.old}[i] == {args.new}[i{off:+d}] over all "
                  f"{len(o)} values{exact}")

    # --- v10-era invariant: whole tile rows in y (southward growth stays an append) ---
    if new_gb.shape[0] % new_dim:
        failures.append(f"{args.new} latitude extent {new_gb.shape[0]} is not a whole "
                        f"number of {new_dim}-px tiles ({new_gb.shape[0] % new_dim} px over)")
    else:
        print(f"\n{args.new} latitude = {new_gb.shape[0] // new_dim} whole tile rows "
              "(no partial row; future southward extension is a pure append)")
    print(f"{args.new} longitude = {new_gb.shape[1] / new_dim:.4f} tile cols "
          f"(last col {new_gb.shape[1] % new_dim} px -- unchanged from {args.old})")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: lattice-compatible; the tile offset above is exact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
