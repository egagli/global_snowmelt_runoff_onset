"""Build the station-tile list for prioritized processing.

Spatial-joins the public snow-pillow station set against the v10 tile registry
and writes processing/tile_data/station_tiles_v10.txt -- one 'row,col' per
line. Pass that file as the 'tiles_file' input of the Process All Tiles /
Process Batch workflows (or --tiles-file on get_tiles_for_batch.py) to process
only tiles containing stations, e.g. to run the snow-pillow evaluation on v10
before committing to the full fleet.

The station set comes from the station_id coord of the Zarr built by notebook 0
of dataset_evaluation/compare_to_all_public_snow_pillows -- i.e. exactly the
stations the evaluation can actually score. Deliberately not the upstream
inventory GeoJSON: all_snow_stations.geojson is the combined inventory, which
carries periodic snow courses (no daily record) plus ~200 daily_or_better
stations that have no archived CSV yet, and those pull in tiles with nothing to
evaluate against.

Stations are padded by ~1.1 km so a station on a tile boundary also pulls
the neighboring tile (the evaluation samples a pixel window around each
station, which can straddle tile edges). The pad is applied in degrees on
unprojected coords, so the longitude half-width is divided by cos(latitude)
-- otherwise it collapses toward the poles (0.01 deg is ~1.1 km of longitude
at the equator but only ~390 m at 69N, where the Yukon stations sit).

Usage:
    pixi run python processing/scripts/make_station_tile_list.py \
        [--stations-zarr dataset_evaluation/compare_to_all_public_snow_pillows/data/snow_pillows/snow_pillows.zarr]
"""
import argparse
import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely import affinity

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_STATIONS = ("dataset_evaluation/compare_to_all_public_snow_pillows/"
                    "data/snow_pillows/snow_pillows.zarr")
OUT_PATH = REPO_ROOT / "processing/tile_data/station_tiles_v10.txt"
BUFFER_M = 1100  # pad radius; sampling windows can straddle tile edges
M_PER_DEG_LAT = 111_320  # spherical approximation, ~0.5% error


def pad_stations(points, buffer_m=BUFFER_M):
    """Grow each station point into a ~buffer_m ellipse in lon/lat degrees.

    Buffering unprojected coords directly would under-pad in longitude by
    cos(latitude), so the circle is stretched east-west by 1/cos(latitude).
    """
    pad_lat = buffer_m / M_PER_DEG_LAT
    return [
        affinity.scale(point.buffer(pad_lat),
                       xfact=1.0 / np.cos(np.radians(point.y)),
                       yfact=1.0,
                       origin=point)
        for point in points
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations-zarr", default=DEFAULT_STATIONS,
                        help="Snow-pillow Zarr store (build via notebook 0 of "
                             "compare_to_all_public_snow_pillows)")
    parser.add_argument("--registry", default="processing/tile_data/global_tiles_with_seasonal_snow_v10.geojson")
    args = parser.parse_args()

    snow_pillows_ds = xr.open_zarr(REPO_ROOT / args.stations_zarr)
    stations = gpd.GeoDataFrame(
        {
            "station_id": snow_pillows_ds["station_id"].values,
            "network": snow_pillows_ds["network"].values,
        },
        geometry=gpd.points_from_xy(snow_pillows_ds["longitude"].values,
                                    snow_pillows_ds["latitude"].values),
        crs="EPSG:4326",
    )

    registry = gpd.read_file(REPO_ROOT / args.registry)
    tiles = registry[registry.to_process].copy()

    padded = stations.copy()
    padded["geometry"] = pad_stations(stations.geometry)
    hit = gpd.sjoin(tiles, padded, how="inner", predicate="intersects")
    pairs = sorted(set(zip(hit.row.astype(int), hit.col.astype(int))))

    n_inside = stations.sjoin(tiles, how="inner", predicate="within").index.nunique()
    header = (
        f"# Tiles containing (or within ~1.1 km of) public snow-pillow stations.\n"
        f"# Generated {datetime.date.today().isoformat()} by make_station_tile_list.py: "
        f"{len(pairs)} tiles cover {n_inside}/{len(stations)} stations "
        f"({stations['network'].nunique()} networks).\n"
        f"# Usage: 'tiles_file' input of Process All Tiles, or "
        f"--tiles-file on get_tiles_for_batch.py.\n"
    )
    OUT_PATH.write_text(header + "".join(f"{r},{c}\n" for r, c in pairs))
    print(f"wrote {len(pairs)} tiles -> {OUT_PATH}")
    print(f"stations covered: {n_inside}/{len(stations)}")


if __name__ == "__main__":
    main()
