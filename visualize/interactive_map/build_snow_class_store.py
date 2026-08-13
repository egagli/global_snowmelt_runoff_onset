"""
Publish the Liston & Sturm (2021) 300 m global seasonal snow classification
(NSIDC-0768) as a multiscale Zarr v3 pyramid for the interactive map.

The map uses it two ways (GitHub issue #9):

  * the "snow class" row of the point-query card, read from level 0, and
  * the "snow class" basemap, a categorical layer rendered by zarr-layer,
    which needs the coarse levels -- a single-level 300 m array would make a
    world-view render read every chunk (8.4 GB decoded).

Built with topozarr using ``method="nearest"``: class codes are categorical,
so coarser levels decimate (keep each window's top-left cell) instead of
averaging, which would invent codes that mean nothing. Documented caveat of
nearest: a class present only away from window corners can vanish at coarse
zoom (a majority ``mode`` would fix that; topozarr does not implement one).

The source GeoTIFF (64800 x 129600 uint8, EPSG:4326, 10 arcsec, nodata 9) is
never materialized -- level 0 is streamed from a lazy dask source by windowed
reads, and each coarser level is computed from the one below it in the store.
Level 0 keeps the source grid exactly, so the map's point-query pixel math is
plain affine arithmetic. Plain chunks, no shards: a point query fetches one
chunk blob in one HTTP request (sharding would add a shard-index read).
All-fill chunks are skipped, so a 404 on a chunk path means all-fill (9).

Final public URLs (production prefix, immutable-generation convention --
bump the ``_1`` suffix to regenerate rather than mutating in place):

    root metadata   https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/snow_classification_300m_multiscale_1/zarr.json
    level 0 array   https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/snow_classification_300m_multiscale_1/0/snow_class/zarr.json
    chunk blobs     https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/snow_classification_300m_multiscale_1/0/snow_class/c/{row_chunk}/{col_chunk}

The root carries topozarr's ``multiscales`` + ``proj:``/``spatial:``
convention attrs, so zarr-layer self-describes the layer's CRS and extent
with no client-side georeferencing. Map-side point-query math against
level 0 (constants match the ``spatial:transform`` attr; pixel/edge
registration, row 0 = north edge):

    row = floor((89.99999999994958 - lat) / 0.0027777777777770003)   # 0..64799
    col = floor((lon - -180.0)          / 0.0027777777777770003)     # 0..129599
    row_chunk, col_chunk = row // 1024, col // 1024

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

    # plan only (level shapes/chunks, no writes)
    pixi run python visualize/interactive_map/build_snow_class_store.py --dry-run

    # local test on a subset (--bbox is for testing ONLY; production runs
    # omit it so the pyramid covers the full global grid)
    pixi run python visualize/interactive_map/build_snow_class_store.py \\
        --source processing/tile_data/SnowClass_GL_300m_10.0arcsec_2021_v01.0_tiled.tif \\
        --local-dest /tmp/snow_class_wa --bbox -125.5 45.0 -116.0 49.5 --levels 4
"""

import argparse
import datetime
import logging
import math
import sys
import threading
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
import xproj  # noqa: F401 -- registers the .proj accessor
import zarr
from rasterio.windows import Window
from topozarr import create_pyramid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_snow_class_store")

DEFAULT_SOURCE = ("https://uwcryo.blob.core.windows.net/snowmelt/eric/"
                  "snow_classification/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif")
DEFAULT_DEST_PREFIX = ("snowmelt/snowmelt_runoff_onset/"
                       "snow_classification_300m_multiscale_1")
ARRAY_NAME = "snow_class"
DEFAULT_LEVELS = 10          # 64800x129600 -> 126x253 at level 9
SOURCE_CHUNK = (1024, 1024)  # also the level-0 chunk shape (1 MB uint8)
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

# windowed reads: https:// routes through GDAL /vsicurl/; don't probe the
# "directory", and retry transient blob-storage hiccups (same as build_pyramid)
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "GDAL_HTTP_MAX_RETRY": "5",
            "GDAL_HTTP_RETRY_DELAY": "1"}


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
    p.add_argument("--levels", type=int, default=DEFAULT_LEVELS,
                   help=f"Number of pyramid levels (default {DEFAULT_LEVELS}: "
                        "level 0 native 300 m, each level halved)")
    p.add_argument("--set-cache-headers", action="store_true",
                   help="Do NOT build; only set Cache-Control "
                        f"'{CACHE}' on every blob under --dest-prefix "
                        "(the separate post-step, mirroring "
                        "visualize/pyramid/2_verify_pyramid.ipynb §5). Run it "
                        "after a successful Azure build.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan (level shapes, chunks, estimated "
                        "sizes) without writing; with --set-cache-headers, "
                        "only count blobs needing the header")
    p.add_argument("--bbox", nargs=4, type=float, default=None,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   help="TESTING ONLY: window the source to this box before "
                        "writing (the subset store carries its own shifted "
                        "spatial:transform). Production runs omit it.")
    p.add_argument("--config-file", default="global_config_v10.txt",
                   help="Config file name in config/ supplying the Azure "
                        "account + SAS token (Azure destinations only; "
                        "resolved like build_pyramid.py)")
    p.add_argument("--max-workers", type=int, default=8,
                   help="Region-write threads per level")
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


def snow_class_dataset(tif_path, bbox=None, chunk=SOURCE_CHUNK):
    """Lazy (latitude, longitude) uint8 Dataset on the source's own grid.

    Level 0 of the pyramid is this grid unchanged -- no reprojection, no
    resampling -- so each dask block is one windowed read of the GeoTIFF at
    the same pixel indices. One GDAL handle per reader thread.
    """
    import dask.array as dska

    with rasterio.Env(**GDAL_ENV), rasterio.open(tif_path) as src:
        if src.dtypes[0] != "uint8":
            raise SystemExit(f"expected uint8 source, got {src.dtypes[0]}")
        if src.nodata is not None and int(src.nodata) != FILL_VALUE:
            log.warning("source nodata %s != %d; keeping %d as fill",
                        src.nodata, FILL_VALUE, FILL_VALUE)
        t0 = src.transform
        if t0.b != 0 or t0.d != 0 or t0.a <= 0 or t0.e >= 0:
            raise SystemExit(
                f"snow-class GeoTIFF is not a north-up axis-aligned grid: {t0!r}")
        window = (window_from_bbox(src, bbox) if bbox
                  else Window(0, 0, src.width, src.height))
        transform = src.window_transform(window)
        crs_wkt = src.crs.to_wkt()
        epsg = src.crs.to_epsg()

    h, w = int(window.height), int(window.width)
    row_off, col_off = int(window.row_off), int(window.col_off)
    handles = threading.local()

    def _read_block(block, block_info=None):
        (i0, i1), (j0, j1) = block_info[None]["array-location"]
        tif = getattr(handles, "tif", None)
        if tif is None:
            tif = handles.tif = rasterio.open(tif_path)
        return tif.read(1, window=Window(col_off + j0, row_off + i0,
                                         j1 - j0, i1 - i0))

    template = dska.zeros((h, w), chunks=chunk, dtype=np.uint8)
    data = template.map_blocks(_read_block, dtype=np.uint8)

    t = transform
    da = xr.DataArray(
        data, dims=("latitude", "longitude"),
        coords={"latitude": t.f + t.e * (np.arange(h) + 0.5),
                "longitude": t.c + t.a * (np.arange(w) + 0.5)},
        name=ARRAY_NAME,
        attrs={
            "long_name": "Seasonal snow classification (Sturm & Liston 2021)",
            "description": (
                "Global seasonal-snow classification class codes (Sturm & "
                "Liston 2021; NSIDC-0768, 10 arcsec, "
                "doi:10.5067/99FTCYYYLAQ0). Categorical: coarser levels "
                "decimate (nearest), never average. Classes 8 (Ocean) and 9 "
                "(Fill) mean 'no class'."),
            "class_info": CLASS_INFO,
            "citation": CITATION,
            "_FillValue": FILL_VALUE,
            "grid_mapping": "spatial_ref",
        })
    da.encoding["_FillValue"] = FILL_VALUE

    ds = xr.Dataset({ARRAY_NAME: da})
    ds = ds.proj.assign_crs(spatial_ref=f"EPSG:{epsg}")
    for coord in ds.coords:
        ds[coord].attrs.pop("_FillValue", None)
    return ds, transform, crs_wkt


def build(args):
    ds, transform, _ = snow_class_dataset(args.source, args.bbox)
    h, w = ds[ARRAY_NAME].shape

    pyramid = create_pyramid(
        ds,
        levels=args.levels,
        x_dim="longitude",
        y_dim="latitude",
        # categorical class codes: decimate, never average
        method="nearest",
        # 1 MB uint8 chunks == the source read blocks
        target_chunk_bytes=SOURCE_CHUNK[0] * SOURCE_CHUNK[1],
        # no shards: one HTTP request per point query (see module docstring)
        chunks_per_shard=None,
    )
    pyramid.attrs = {
        **pyramid.attrs,
        "title": "Global Seasonal-Snow Classification v1 (300 m / 10 arcsec)",
        "citation": CITATION,
        "provenance": {
            "source": args.source,
            "builder": "visualize/interactive_map/build_snow_class_store.py",
            "created": datetime.date.today().isoformat(),
            "method": "nearest",
            "purpose": ("interactive-map snow class basemap + point-query "
                        "card row (issue #9)"),
        },
    }
    if args.bbox:
        pyramid.attrs["subset_bbox"] = list(args.bbox)
        pyramid.attrs["note"] = "TEST SUBSET (--bbox); not a production store"

    dest = args.local_dest or f"azure://{args.dest_prefix}"
    log.info("Source: %s -> level 0 %d x %d uint8 (fill %d), grid unchanged",
             args.source, h, w, FILL_VALUE)
    log.info("Dest: %s ; %d levels, method=nearest, no shards, "
             "no consolidated metadata", dest, args.levels)
    for lv in range(args.levels):
        enc = pyramid.encoding[f"/{lv}"][ARRAY_NAME]
        shape = pyramid.level_templates[lv][ARRAY_NAME].shape
        log.info("  level %d: shape %s, chunks %s", lv, tuple(shape),
                 tuple(enc["chunks"]))
    log.info("Chunk URL scheme: %s/{level}/%s/c/{row_chunk}/{col_chunk}",
             dest, ARRAY_NAME)
    if args.dry_run:
        log.info("Dry run: no changes made")
        return

    if args.local_dest:
        store = args.local_dest
    else:
        config = load_config(args.config_file)
        store = azure_dest_store(config, args.dest_prefix)

    # one level at a time: level 0 streams from the lazy source, each coarser
    # level reads the one below it back out of the store
    for lv in range(args.levels):
        log.info("writing level %d ...", lv)
        pyramid.write(store, mode="a", levels=[lv],
                      max_workers=args.max_workers, progress=True, stats=True)

    log.info("Done: %s (level 0 at %s/0/%s)", dest, dest, ARRAY_NAME)
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
