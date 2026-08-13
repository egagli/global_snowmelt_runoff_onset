"""
Publish the Liston & Sturm (2021) 300 m global seasonal snow classification
(NSIDC-0768) as a small plain Zarr v3 store for interactive-map point queries
(the "snow class" row of the point-query card, GitHub issue #9).

The source GeoTIFF (64800 x 129600 uint8, EPSG:4326, 10 arcsec, nodata 9) is
streamed in chunk-row strips (~130 MB each; the full raw array is 8.4 GB and
is never materialized) into a single 2-D uint8 array named ``snow_class`` at
the store ROOT, zstd-compressed, plain (1024, 1024) chunks, no consolidated
metadata (house style). Plain chunks instead of shards: the map fetches one
chunk blob per point query at an explicit path with a single HTTP request;
sharding would add a shard-index read (suffix byte-range) per query for a
store this small (~8.1k blobs, tens of MB total). All-fill chunks are not
written (``write_empty_chunks=False``) -- a 404 on a chunk path means
all-fill (9).

Final public URLs (production prefix, immutable-generation convention --
bump the ``_1`` suffix to regenerate):

    root metadata   https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/snow_classification_300m_1/zarr.json
    ARRAY metadata  https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/snow_classification_300m_1/snow_class/zarr.json
    chunk blobs     https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/snow_classification_300m_1/snow_class/c/{row_chunk}/{col_chunk}

so ``{prefix}/snow_class`` opens directly as a zarr v3 array (zarrita
FetchStore / ``zarr.open_array``). Map-side pixel math (constants match the
``spatial:transform`` attr written on both the root group and the array;
pixel/edge registration, row 0 = north edge):

    row = floor((89.99999999994958 - lat) / 0.0027777777777770003)   # 0..64799
    col = floor((lon - -180.0)          / 0.0027777777777770003)     # 0..129599
    row_chunk, col_chunk = row // 1024, col // 1024
    byte offset in decoded chunk = (row % 1024) * 1024 + (col % 1024)

Classes: 1 Tundra, 2 Boreal Forest, 3 Maritime, 4 Ephemeral (includes no
snow), 5 Prairie, 6 Montane Forest, 7 Ice (glaciers and ice sheets),
8 Ocean, 9 Fill. The map treats 8/9, missing chunks, and off-grid
coordinates as "no class".

Citation: Liston, G. E. and M. Sturm. (2021). Global Seasonal-Snow
Classification, Version 1. NSIDC-0768. NASA NSIDC DAAC.
doi:10.5067/99FTCYYYLAQ0

Usage:
    # production build (full grid -> Azure; needs config/sas_token.txt or
    # AZURE_STORAGE_SAS_TOKEN, like build_pyramid.py)
    pixi run python visualize/interactive_map/build_snow_class_store.py

    # then the separate cache-header pass (mirrors 2_verify_pyramid.ipynb §5)
    pixi run python visualize/interactive_map/build_snow_class_store.py --set-cache-headers

    # plan only
    pixi run python visualize/interactive_map/build_snow_class_store.py --dry-run

    # local test on a subset (--bbox is for testing ONLY; production runs
    # omit it so the array covers the full global grid)
    pixi run python visualize/interactive_map/build_snow_class_store.py \\
        --source processing/tile_data/SnowClass_GL_300m_10.0arcsec_2021_v01.0_tiled.tif \\
        --local-dest /tmp/snow_class_wa --bbox -125.5 45.0 -116.0 49.5
"""

import argparse
import datetime
import logging
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
import zarr
from rasterio.windows import Window
from zarr.codecs import ZstdCodec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_snow_class_store")

DEFAULT_SOURCE = ("https://uwcryo.blob.core.windows.net/snowmelt/eric/"
                  "snow_classification/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif")
DEFAULT_DEST_PREFIX = "snowmelt/snowmelt_runoff_onset/snow_classification_300m_1"
ARRAY_NAME = "snow_class"
CHUNKS = (1024, 1024)
FILL_VALUE = 9
CACHE = "public, max-age=31536000, immutable"  # same header as the pyramid pass

CLASS_INFO = {
    "1": "Tundra",
    "2": "Boreal Forest",
    "3": "Maritime",
    "4": "Ephemeral (includes no snow)",
    "5": "Prairie",
    "6": "Montane Forest",
    "7": "Ice (glaciers and ice sheets)",
    "8": "Ocean",
    "9": "Fill",
}
CITATION = ("Liston, G. E. and M. Sturm. (2021). Global Seasonal-Snow "
            "Classification, Version 1. NSIDC-0768. NASA National Snow and "
            "Ice Data Center DAAC. doi:10.5067/99FTCYYYLAQ0")

# Same convention identifiers the pyramid store carries in its root zarr.json
# (global_runoff_onset_v10_multiscale_1) minus multiscales, so both stores
# self-describe georeferencing the same way.
ZARR_CONVENTIONS = [
    {
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/geo-proj/refs/tags/v1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/geo-proj/blob/v1/README.md",
        "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
        "name": "proj",
        "description": "Coordinate reference system information for geospatial data",
    },
    {
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
        "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
        "name": "spatial",
        "description": "Spatial coordinate information",
    },
]

GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}  # HTTPS (vsicurl) reads


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help="Source GeoTIFF: HTTPS URL (default, read via GDAL "
                        "vsicurl) or a local path (e.g. the gitignored copy in "
                        "processing/tile_data/)")
    p.add_argument("--dest-prefix", default=DEFAULT_DEST_PREFIX,
                   help="Azure container/prefix destination (immutable-prefix "
                        "generation convention: bump the trailing _N to "
                        "regenerate rather than mutating in place)")
    p.add_argument("--local-dest", default=None, metavar="PATH",
                   help="Write to a local directory store instead of Azure "
                        "(testing; no credentials needed)")
    p.add_argument("--set-cache-headers", action="store_true",
                   help="Do NOT build; only set Cache-Control "
                        f"'{CACHE}' on every blob under --dest-prefix "
                        "(the separate post-step, mirroring "
                        "visualize/pyramid/2_verify_pyramid.ipynb §5). Run it "
                        "after a successful Azure build.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan (shapes, chunks, estimated sizes) "
                        "without writing; with --set-cache-headers, only "
                        "count blobs needing the header")
    p.add_argument("--bbox", nargs=4, type=float, default=None,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   help="TESTING ONLY: window the source to this box before "
                        "writing (the subset store carries its own shifted "
                        "spatial:transform). Production runs omit it.")
    p.add_argument("--config-file", default="global_config_v10.txt",
                   help="Config file name in config/ supplying the Azure "
                        "account + SAS token (Azure destinations only; "
                        "resolved like build_pyramid.py)")
    p.add_argument("--zstd-level", type=int, default=3,
                   help="zstd compression level (one-time build of a small "
                        "store, so a mid level is fine)")
    return p.parse_args()


def load_config(config_file):
    """Resolve the repo Config the way build_pyramid.py / the fleet scripts do."""
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from global_snowmelt_runoff_onset.config import Config
    name = config_file
    if "/" in name:
        return Config(name)
    if not name.endswith(".txt"):
        name = f"global_config_{name}.txt"
    return Config(str(repo_root / "config" / name))


def azure_dest_store(config, dest_prefix):
    """zarr store on Azure via obstore -- same pattern as build_pyramid.py."""
    from obstore.store import AzureStore
    container, prefix = dest_prefix.split("/", 1)
    return zarr.storage.ObjectStore(
        AzureStore(account_name=config.azure_storage_account,
                   container_name=container, prefix=prefix,
                   sas_key=config.sas_token))


def window_from_bbox(src, bbox):
    """Integer pixel window covering bbox (clamped to the grid)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    t = src.transform  # a=x_res, c=x_origin, e=-y_res, f=y_origin
    col0 = max(0, math.floor((lon_min - t.c) / t.a))
    col1 = min(src.width, math.ceil((lon_max - t.c) / t.a))
    row0 = max(0, math.floor((lat_max - t.f) / t.e))  # t.e < 0
    row1 = min(src.height, math.ceil((lat_min - t.f) / t.e))
    if col1 <= col0 or row1 <= row0:
        raise SystemExit(f"--bbox {bbox} does not intersect the source grid")
    return Window(col0, row0, col1 - col0, row1 - row0)


def spatial_attrs(transform, shape, crs):
    """proj:/spatial: convention attrs, mirroring the pyramid store."""
    h, w = shape
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    return {
        "proj:code": f"EPSG:{crs.to_epsg()}",
        "proj:wkt2": crs.to_wkt(),
        "spatial:dimensions": ["latitude", "longitude"],
        "spatial:registration": "pixel",
        "spatial:transform": [a, b, c, d, e, f],
        "spatial:bbox": [c, f + e * h, c + a * w, f],  # W, S, E, N
        "spatial:shape": [h, w],
    }


def plan(src, window, args):
    h, w = int(window.height), int(window.width)
    n_chunk_rows = math.ceil(h / CHUNKS[0])
    n_chunk_cols = math.ceil(w / CHUNKS[1])
    dest = args.local_dest or f"azure://{args.dest_prefix}"
    log.info("Source: %s (%d x %d uint8, nodata %s)",
             args.source, src.height, src.width, src.nodata)
    if args.bbox:
        log.info("TEST subset --bbox %s -> window rows %d:%d cols %d:%d",
                 args.bbox, window.row_off, window.row_off + h,
                 window.col_off, window.col_off + w)
    log.info("Dest: %s ; array '%s' shape (%d, %d) uint8, fill %d, "
             "chunks %s (no shards), zstd level %d, no consolidated metadata",
             dest, ARRAY_NAME, h, w, FILL_VALUE, CHUNKS, args.zstd_level)
    log.info("Chunk grid: %d x %d = %d chunk blobs max (all-fill chunks "
             "skipped) + 2 coord blobs + 4 zarr.json",
             n_chunk_rows, n_chunk_cols, n_chunk_rows * n_chunk_cols)
    log.info("Raw array %.2f GB streamed in %d strips of %d rows (~%.0f MB "
             "each); compressed store expected at the source GeoTIFF's scale "
             "(~67 MB zstd for the full grid)",
             h * w / 1e9, n_chunk_rows, CHUNKS[0], CHUNKS[0] * w / 1e6)
    log.info("Chunk URL scheme: {dest}/%s/c/{row//%d}/{col//%d}",
             ARRAY_NAME, CHUNKS[0], CHUNKS[1])


def build(args):
    with rasterio.Env(**GDAL_ENV), rasterio.open(args.source) as src:
        if src.dtypes[0] != "uint8":
            raise SystemExit(f"expected uint8 source, got {src.dtypes[0]}")
        if src.nodata is not None and int(src.nodata) != FILL_VALUE:
            log.warning("source nodata %s != %d; keeping %d as fill",
                        src.nodata, FILL_VALUE, FILL_VALUE)

        window = (window_from_bbox(src, args.bbox) if args.bbox
                  else Window(0, 0, src.width, src.height))
        transform = src.window_transform(window)
        h, w = int(window.height), int(window.width)

        plan(src, window, args)
        if args.dry_run:
            log.info("Dry run: no changes made")
            return

        if args.local_dest:
            store = args.local_dest
        else:
            config = load_config(args.config_file)
            store = azure_dest_store(config, args.dest_prefix)

        geo = spatial_attrs(transform, (h, w), src.crs)
        root = zarr.open_group(store, mode="a", zarr_format=3)
        root_attrs = {
            "title": "Global Seasonal-Snow Classification v1 (300 m / 10 arcsec)",
            "citation": CITATION,
            "zarr_conventions": ZARR_CONVENTIONS,
            **geo,
            "provenance": {
                "source": args.source,
                "builder": "visualize/interactive_map/build_snow_class_store.py",
                "created": datetime.date.today().isoformat(),
                "purpose": ("point-query layer for the interactive map "
                            "(snow class row of the point-query card)"),
            },
        }
        if args.bbox:
            root_attrs["subset_bbox"] = list(args.bbox)
            root_attrs["note"] = "TEST SUBSET (--bbox); not a production store"
        root.attrs.update(root_attrs)

        compressors = ZstdCodec(level=args.zstd_level)
        arr = root.create_array(
            ARRAY_NAME, shape=(h, w), chunks=CHUNKS, dtype="uint8",
            fill_value=FILL_VALUE, compressors=compressors,
            dimension_names=["latitude", "longitude"],
            config={"write_empty_chunks": False}, overwrite=True,
            attributes={
                "description": ("Seasonal snow classification (Sturm & "
                                "Liston 2021); classes 8 (Ocean) and 9 "
                                "(Fill) mean 'no class'"),
                "class_info": CLASS_INFO,
                "citation": CITATION,
                "_FillValue": FILL_VALUE,
                # duplicated from the root group so the array's zarr.json is
                # self-sufficient for the map's single-fetch point queries
                **{k: geo[k] for k in ("proj:code", "spatial:transform",
                                       "spatial:registration", "spatial:shape")},
            })

        # pixel-center coordinates (float64; ~1.5 MB raw for the full grid --
        # the map uses the affine constants, these are for xarray consumers)
        t = transform
        lat = root.create_array("latitude", shape=(h,), chunks=(h,),
                                dtype="float64", compressors=compressors,
                                dimension_names=["latitude"], overwrite=True,
                                attributes={"standard_name": "latitude",
                                            "units": "degrees_north"})
        lat[:] = t.f + t.e * (np.arange(h) + 0.5)
        lon = root.create_array("longitude", shape=(w,), chunks=(w,),
                                dtype="float64", compressors=compressors,
                                dimension_names=["longitude"], overwrite=True,
                                attributes={"standard_name": "longitude",
                                            "units": "degrees_east"})
        lon[:] = t.c + t.a * (np.arange(w) + 0.5)

        # stream chunk-row strips: strip height == chunk height and strips
        # start at array row 0, so every zarr chunk is written exactly once
        # (no read-modify-write) and memory stays ~1 strip (<200 MB globally)
        n_strips = math.ceil(h / CHUNKS[0])
        for i in range(n_strips):
            r0 = i * CHUNKS[0]
            r1 = min(r0 + CHUNKS[0], h)
            strip = src.read(1, window=Window(window.col_off,
                                              window.row_off + r0,
                                              w, r1 - r0))
            arr[r0:r1, :] = strip
            log.info("strip %d/%d rows %d:%d written", i + 1, n_strips, r0, r1)

    dest = args.local_dest or f"azure://{args.dest_prefix}"
    log.info("Done: %s/%s (open as a zarr v3 array at that path)", dest, ARRAY_NAME)
    if not args.local_dest:
        log.info("Now run the cache-header pass: build_snow_class_store.py "
                 "--set-cache-headers --dest-prefix %s", args.dest_prefix)


def set_cache_headers(args):
    """Cache-Control pass over every blob under the prefix.

    Same header, listing, and threaded set_http_headers pattern as
    visualize/pyramid/2_verify_pyramid.ipynb §5 -- safe because cache-busting
    is by prefix generation, not by mutation. Re-run after any rebuild.
    """
    from concurrent.futures import ThreadPoolExecutor

    from azure.storage.blob import ContainerClient, ContentSettings

    logging.getLogger("azure").setLevel(logging.WARNING)  # per-request noise
    if args.local_dest:
        raise SystemExit("--set-cache-headers applies to Azure destinations only")
    config = load_config(args.config_file)
    container, prefix = args.dest_prefix.split("/", 1)
    account_url = f"https://{config.azure_storage_account}.blob.core.windows.net"
    client = ContainerClient(account_url, container, credential=config.sas_token)

    blobs = list(client.list_blobs(name_starts_with=prefix + "/"))
    todo = [b for b in blobs if (b.content_settings.cache_control or "") != CACHE]
    log.info("%s: %d blobs, %d need Cache-Control '%s'",
             args.dest_prefix, len(blobs), len(todo), CACHE)
    if args.dry_run:
        log.info("Dry run: no changes made")
        return

    def _set_header(blob):
        settings = ContentSettings(cache_control=CACHE,
                                   content_type=blob.content_settings.content_type)
        client.get_blob_client(blob.name).set_http_headers(content_settings=settings)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_set_header, todo))
    log.info("done")


def main():
    args = parse_args()
    if args.set_cache_headers:
        set_cache_headers(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
