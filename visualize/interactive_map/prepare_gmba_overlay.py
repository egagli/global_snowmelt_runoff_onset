"""Publish the GMBA Mountain Inventory v2.0 overlay for the interactive map.

Downloads the GMBA Mountain Inventory v2.0 "standard" 300-selection (the 291
polygons GMBA curates for ~1:300M cartography; Snethlage et al. 2022,
doi:10.1038/s41597-022-01256-y, https://www.earthenv.org/mountains), drops the
one Ocean feature, keeps only the attributes the map's hover card shows,
simplifies geometry for web display, and uploads the result to the public
Azure container as a gzip-encoded GeoJSON blob:

    https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/gmba_v2_standard_300_1.geojson

The map (map/components/map.tsx, GMBA_URL in map/lib/store.ts) lazy-loads
this URL the first time the "GMBA mountain ranges" overlay is toggled on
(issue #13). The blob is immutable-cached like the pyramid stores: to change
the overlay, bump the trailing _1 in --dest-blob AND in GMBA_URL together.

Content-Encoding is set to gzip at upload, so browsers receive ~2 MB and
decompress transparently; Cache-Control is set in the same call (no separate
header pass needed for a single blob).

    # rebuild + upload (needs config/sas_token.txt or AZURE_STORAGE_SAS_TOKEN)
    pixi run python visualize/interactive_map/prepare_gmba_overlay.py

    # inspect locally without touching Azure
    pixi run python visualize/interactive_map/prepare_gmba_overlay.py \
        --local-out /tmp/gmba.geojson --skip-upload
"""

import argparse
import gzip
import pathlib
import sys
import tempfile

GMBA_ZIP_URL = (
    "https://data.earthenv.org/mountains/standard/"
    "GMBA_Inventory_v2.0_standard_300.zip"
)
DEST_BLOB_DEFAULT = "snowmelt_runoff_onset/gmba_v2_standard_300_1.geojson"
CONTAINER = "snowmelt"
CACHE = "public, max-age=31536000, immutable"

# Attributes kept for the hover card, renamed to compact GeoJSON keys.
FIELD_MAP = {
    "GMBA_V2_ID": "id",
    "MapName": "name",
    "Feature": "feature",
    "Countries": "countries",
    "Area": "area_km2",
    "Elev_Low": "elev_low",
    "Elev_High": "elev_high",
}


def build_geojson(src: str, tolerance: float, out_path: pathlib.Path) -> int:
    """Read the GMBA shapefile (zip URL or local path), simplify, write GeoJSON.

    Returns the feature count. RFC7946 output splits any antimeridian-crossing
    polygons, which MapLibre needs; 3-decimal coordinates (~110 m) are well
    below the simplify tolerance so they cost nothing visually.
    """
    import geopandas as gpd

    gdf = gpd.read_file(src)
    gdf = gdf[gdf["Feature"] != "Ocean"]  # one polygon: the Arctic Ocean
    gdf = gdf[list(FIELD_MAP) + ["geometry"]].rename(columns=FIELD_MAP)
    gdf["area_km2"] = gdf["area_km2"].round(0)
    gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)
    gdf.to_file(out_path, driver="GeoJSON", COORDINATE_PRECISION=3,
                RFC7946="YES")
    return len(gdf)


def upload(geojson_path: pathlib.Path, dest_blob: str) -> None:
    """Gzip the GeoJSON and upload it with cache/encoding headers set."""
    from azure.storage.blob import ContainerClient, ContentSettings

    from global_snowmelt_runoff_onset.config import Config

    config = Config("config/global_config_v10.txt")
    account_url = f"https://{config.azure_storage_account}.blob.core.windows.net"
    client = ContainerClient(account_url, CONTAINER,
                             credential=config.sas_token)

    compressed = gzip.compress(geojson_path.read_bytes(), compresslevel=9)
    client.upload_blob(
        dest_blob,
        compressed,
        overwrite=True,
        content_settings=ContentSettings(
            content_type="application/geo+json",
            content_encoding="gzip",
            cache_control=CACHE,
        ),
    )
    print(f"uploaded {len(compressed) / 1e6:.2f} MB (gzip) -> "
          f"{account_url}/{CONTAINER}/{dest_blob}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", default=None,
                        help="Local GMBA shapefile/zip instead of downloading "
                             f"(default: {GMBA_ZIP_URL})")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Douglas-Peucker simplify tolerance in degrees "
                             "(default 0.02 ~ 2 km; raw output ~8 MB, ~2 MB "
                             "gzipped)")
    parser.add_argument("--dest-blob", default=DEST_BLOB_DEFAULT,
                        help="Destination blob within the container; bump the "
                             "_N suffix together with GMBA_URL in "
                             "map/lib/store.ts (immutable-cache convention)")
    parser.add_argument("--local-out", default=None,
                        help="Also keep the uncompressed GeoJSON at this path")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Build only; do not touch Azure")
    args = parser.parse_args()

    src = args.src or f"zip+{GMBA_ZIP_URL}"
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(args.local_out or
                           pathlib.Path(tmp) / "gmba_overlay.geojson")
        n = build_geojson(src, args.tolerance, out)
        print(f"{n} polygons, {out.stat().st_size / 1e6:.2f} MB raw "
              f"(tolerance {args.tolerance}) -> {out}")
        if not args.skip_upload:
            upload(out, args.dest_blob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
