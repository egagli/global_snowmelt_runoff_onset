"""
Build the multiscale visualization pyramid (plain Zarr v3) from the icechunk store.

topozarr does the heavy lifting end-to-end: level 0 is streamed region by
region from the source, each coarser level is block-reduced from the
previously written level (shard-sized regions on a thread pool, bounded
memory), and the multiscales + proj: + spatial: convention attrs and
zarr-layer hints are emitted for us. Verified semantics (topozarr 0.1.4, see
README.md next to this script): the mean kernel is fill-aware on raw int16 --
windows average valid values only, all-fill windows stay _FillValue and are
skipped on write -- so the source is read RAW (mask_and_scale=False): level 0
is a value-exact copy, and the cascade is mean-of-valid in encoded-integer
space (identical to decode->mean->re-encode up to truncation toward zero,
i.e. <= 0.1 day on scaled variables, <= 1 day on runoff_onset).

Written with obstore (Rust object_store handles the Azure byte-range patterns
for Zarr v3 shards that adlfs gets wrong). Naming and provenance are fully
config-driven (config/global_config_v10.txt): the destination is
`global_runoff_multiscale_azure_prefix` (versioned by dataset VERSION +
`multiscale_generation`, e.g. ..._v10_multiscale_1), and the source snapshot
is `release_tag`. Appending a water year to v10 = one config edit bumping
WY_end, release_tag, multiscale_generation, and the prefix suffix together;
the map and figure notebooks follow the config automatically. The icechunk
repo stays the only source of truth; the pyramid is disposable and
regenerable -- bump the generation (cache headers are immutable) to
regenerate.

Jobs (one topozarr pass per variable group; a job owns whole arrays, so jobs
never write the same zarr object and can run concurrently once the store
exists):

    composites           runoff_onset_median, runoff_onset_mad, temporal_resolution_median
    runoff_onset         runoff_onset (water_year, latitude, longitude)
    temporal_resolution  temporal_resolution (water_year, latitude, longitude)
    seasonal_snow        seasonal_snow_pct (latitude, longitude), synthesized
                         from the Sturm & Liston (2021) snow classification
                         GeoTIFF (NSIDC-0768) instead of the icechunk source

Run `--job composites` FIRST (alone -- it creates the root group, the level
groups, and the convention attrs), then the other jobs in parallel.

The seasonal_snow job still opens the icechunk source: that pins the snapshot
and supplies the exact latitude/longitude coordinate arrays, so the coord
blobs it rewrites at every level are byte-identical to the live ones (the
immutable-cache convention). Its variable is a nearest-neighbor reclassify of
the 10 arcsec snow classification onto the level-0 grid -- accepted classes
{1,2,3,5,6,7} -> 100, class 4 (Ephemeral) -> 0, everything else -> -9999 --
and the standard fill-aware mean cascade turns that into percent of
seasonal-snow area at coarser levels. Because pyramid.write unconditionally
updates the root group attrs and this job's 2-D dataset would compute
different `multiscales`/`zarr-layer` values than the live 3-D ones, the job
neutralizes its root-attr payload and skips the provenance rewrite: the root
zarr.json is left untouched (see `--check-attrs`). Provenance for the mask
lives in the variable attrs and the `_build/seasonal_snow.json` marker
(GeoTIFF URL + ETag).

Usage:
    python visualize/pyramid/build_pyramid.py --job seasonal_snow --check-attrs
    python visualize/pyramid/build_pyramid.py --job seasonal_snow \\
        --snow-class-tif /tmp/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif

Levels are written one at a time, and a progress marker (`_build/<job>.json`
under the pyramid prefix, outside the zarr metadata) records each completed
level plus the source snapshot id. `--mode resume` reads the marker and
rewrites from the first incomplete level (a partially-written level is
rewritten deterministically -- the source is tag-pinned); `--mode fresh`
(default) resets the marker and writes everything. Resuming against a marker
from a different snapshot fails loudly: that's a new dataset, not a resume --
bump multiscale_generation in the config and build fresh.

Usage:
    python visualize/pyramid/build_pyramid.py --job composites
    python visualize/pyramid/build_pyramid.py --job all                 # sequential local run
    python visualize/pyramid/build_pyramid.py --job runoff_onset --mode resume
    python visualize/pyramid/build_pyramid.py --job composites --plan-only
    # shakedown: one small variable to a scratch prefix
    python visualize/pyramid/build_pyramid.py --variables runoff_onset_median \\
        --dest-prefix snowmelt/snowmelt_runoff_onset/scratch_pyramid_shakedown
"""

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import obstore
import xarray as xr
import xproj  # noqa: F401 -- registers the .proj accessor
import zarr
from obstore.store import AzureStore
import topozarr
from topozarr import ZarrLayerVarConfig, create_pyramid

TOPOZARR_VERSION = topozarr.__version__

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from global_snowmelt_runoff_onset.config import Config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_pyramid")

DEFAULT_LEVELS = 10  # /0 native (80 m) ... /9 (~41 km); coarsest ~976 x 400 px


JOBS = {
    "composites": ["runoff_onset_median", "runoff_onset_mad", "temporal_resolution_median"],
    "runoff_onset": ["runoff_onset"],
    "temporal_resolution": ["temporal_resolution"],
    "seasonal_snow": ["seasonal_snow_pct"],
}

# --- seasonal_snow job: Sturm & Liston (2021) snow classification mask -------
# NSIDC-0768 (doi:10.5067/99FTCYYYLAQ0), 10 arcsec global GeoTIFF, mirrored on
# uwcryo (public read; tiled/ZSTD, so windowed HTTPS range reads are cheap).
# Same source and accepted-class rule as the tile registry
# (processing/0_select_tiles_to_process.ipynb).
SEASONAL_SNOW_VAR = "seasonal_snow_pct"
DEFAULT_SNOW_CLASS_TIF = ("https://uwcryo.blob.core.windows.net/snowmelt/eric/"
                          "snow_classification/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif")
SNOW_CLASSES_SEASONAL = (1, 2, 3, 5, 6, 7)  # Tundra, Boreal Forest, Maritime, Prairie, Montane Forest, Ice
SNOW_CLASS_EPHEMERAL = 4                    # excluded from "seasonal" by design (issue #9)
SNOW_CLASS_FILL = -9999                     # ocean (8), fill (9), anything unexpected

# Defaults for generic zarr-layer viewers, in DECODED units (zarr-layer applies
# scale_factor before clim). The map app sets its own colormaps/clims; DOWY
# 110-270 matches the plot_utils month-colorbar convention.
LAYER_HINTS = {
    "runoff_onset": ZarrLayerVarConfig(colormap="viridis", clim=[110, 270]),
    "runoff_onset_median": ZarrLayerVarConfig(colormap="viridis", clim=[110, 270]),
    "runoff_onset_mad": ZarrLayerVarConfig(colormap="magma", clim=[0, 30]),
    "temporal_resolution": ZarrLayerVarConfig(colormap="magma", clim=[0, 24]),
    "temporal_resolution_median": ZarrLayerVarConfig(colormap="magma", clim=[0, 24]),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="global_config_v10.txt",
                   help="Config file name in config/ (e.g. global_config_v10.txt, "
                        "matching the fleet scripts); values containing '/' are "
                        "treated as explicit paths. Supplies the source tag "
                        "(release_tag) and the destination prefix "
                        "(global_runoff_multiscale_azure_prefix).")
    p.add_argument("--source-tag", default=None,
                   help="Override the icechunk tag to build from (default: the "
                        "config's release_tag; never a branch)")
    p.add_argument("--mode", choices=["fresh", "resume"], default="fresh",
                   help="fresh: write all levels from scratch (resets the "
                        "progress marker). resume: continue from the first "
                        "level the marker doesn't record as complete.")
    p.add_argument("--job", choices=[*JOBS, "all"], default=None,
                   help="Variable group to build ('all' runs the groups sequentially)")
    p.add_argument("--variables", default=None,
                   help="Comma-separated variable override (instead of --job); "
                        "for shakedowns and partial rebuilds")
    p.add_argument("--dest-prefix", default=None,
                   help="Override the container/prefix for the pyramid store "
                        "(default: the config's "
                        "global_runoff_multiscale_azure_prefix)")
    p.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    p.add_argument("--max-workers", type=int, default=None,
                   help="topozarr region thread pool size (default: derived "
                        "from CPU count and available memory)")
    p.add_argument("--plan-only", action="store_true",
                   help="Print per-level shapes without writing")
    p.add_argument("--snow-class-tif", default=DEFAULT_SNOW_CLASS_TIF,
                   help="Sturm & Liston (2021) snow classification GeoTIFF "
                        "for the seasonal_snow job: an HTTPS URL (default, "
                        "read via GDAL /vsicurl/) or a local file path")
    p.add_argument("--check-attrs", action="store_true",
                   help="Read-only: fetch the live store's root zarr.json "
                        "over public HTTPS, diff it against the root attrs "
                        "this job would leave behind, and exit (nonzero on "
                        "any difference). Run before any real write against "
                        "a live prefix.")
    return p.parse_args()


def load_config(args):
    """Resolve the config file the way the fleet scripts do."""
    name = args.config_file
    if "/" in name:
        return Config(name)
    if not name.endswith(".txt"):
        name = f"global_config_{name}.txt"
    return Config(str(Path(__file__).parent.parent.parent / "config" / name))


def open_source(config, tag):
    """Open the output store at ``tag``, raw and lazy (no dask, no decoding)."""
    repo = config.open_output_repo()
    snapshot_id = repo.lookup_tag(tag)
    session = repo.readonly_session(tag=tag)
    ds = xr.open_zarr(session.store, zarr_format=3, consolidated=False,
                      mask_and_scale=False, chunks=None)
    # fresh CRS assignment: the pyramid encoding is topozarr's business, and
    # the grid is EPSG:4326 by construction
    ds = ds.drop_encoding().drop_vars("spatial_ref", errors="ignore")
    ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")
    # the raw open leaves _FillValue in coordinate attrs, which collides with
    # xarray's CF encoder at write time; it's meaningless on coords
    for coord in ds.coords:
        ds[coord].attrs.pop("_FillValue", None)
    return ds, snapshot_id


def seasonal_snow_pct_dataarray(ds, tif_path, chunk=2048):
    """Lazy (latitude, longitude) int16 percent-seasonal-snow mask on ``ds``'s grid.

    Nearest-neighbor lookup from the Sturm & Liston (2021) snow classification
    GeoTIFF, reclassified: accepted classes -> 100, Ephemeral (4) -> 0,
    everything else (ocean 8, fill 9, unexpected) -> -9999. Both grids are
    regular EPSG:4326 lattices, so nearest-neighbor is exact integer affine
    index arithmetic: the source cell containing each target cell center is
    ``floor((center - origin) / pixel_size)``, precomputed per axis as two
    monotonic index vectors. Each dask block then does one windowed rasterio
    read (a 2048-px block spans ~530 source px, ~0.3 MB) and a LUT remap --
    the full 204800 x 499998 array is never materialized.

    The coords are taken from ``ds`` (the icechunk source) so the coordinate
    blobs topozarr rewrites at every level stay byte-identical to the live
    store's.
    """
    import threading

    import dask.array as dska
    import rasterio
    from rasterio.windows import Window

    # windowed HTTPS reads: https:// routes through GDAL /vsicurl/; don't
    # probe the "directory", and retry transient blob-storage hiccups
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")

    with rasterio.open(tif_path) as src:
        t = src.transform
        src_h, src_w = src.height, src.width
        src_dtype = src.dtypes[0]
    if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
        raise SystemExit(f"snow-class GeoTIFF is not a north-up axis-aligned grid: {t!r}")
    if src_dtype != "uint8":
        raise SystemExit(f"snow-class GeoTIFF dtype {src_dtype!r}, expected uint8")

    lat = ds["latitude"].values   # descending cell centers
    lon = ds["longitude"].values  # ascending cell centers
    src_row = np.floor((t.f - lat) / -t.e).astype(np.int64)
    src_col = np.floor((lon - t.c) / t.a).astype(np.int64)
    if (src_row.min() < 0 or src_row.max() >= src_h
            or src_col.min() < 0 or src_col.max() >= src_w):
        # the target grid (84.05 to -63.41) is a strict subset of the source
        # (90 to -90), so out-of-range indices mean a wrong file, not roundoff
        raise SystemExit(
            f"target grid falls outside the snow-class GeoTIFF: rows "
            f"{src_row.min()}..{src_row.max()} of {src_h}, cols "
            f"{src_col.min()}..{src_col.max()} of {src_w}")

    lut = np.full(256, SNOW_CLASS_FILL, dtype=np.int16)
    lut[list(SNOW_CLASSES_SEASONAL)] = 100
    lut[SNOW_CLASS_EPHEMERAL] = 0

    handles = threading.local()  # one GDAL dataset handle per reader thread

    def _read_block(block, block_info=None):
        (i0, i1), (j0, j1) = block_info[None]["array-location"]
        rows, cols = src_row[i0:i1], src_col[j0:j1]
        r0, c0 = int(rows[0]), int(cols[0])
        tif = getattr(handles, "tif", None)
        if tif is None:
            tif = handles.tif = rasterio.open(tif_path)
        window = tif.read(1, window=Window(c0, r0, int(cols[-1]) + 1 - c0,
                                           int(rows[-1]) + 1 - r0))
        return lut[window[np.ix_(rows - r0, cols - c0)]]

    template = dska.zeros((lat.size, lon.size), chunks=chunk, dtype=np.int16)
    data = template.map_blocks(_read_block, dtype=np.int16)

    return xr.DataArray(
        data, dims=("latitude", "longitude"),
        coords={"latitude": ds["latitude"], "longitude": ds["longitude"]},
        name=SEASONAL_SNOW_VAR,
        attrs={
            "long_name": "Percent of cell area with seasonal snow (Sturm & Liston 2021)",
            "description": (
                "Percent (0-100) of cell area classified as seasonal snow in the "
                "Sturm & Liston (2021) global seasonal snow classification "
                "(NSIDC-0768, 10 arcsec, doi:10.5067/99FTCYYYLAQ0). Level 0 is a "
                "nearest-neighbor reclassification: snow classes 1-3 and 5-7 "
                "(Tundra, Boreal Forest, Maritime, Prairie, Montane Forest, Ice) "
                "= 100; class 4 (Ephemeral snow) = 0 (excluded from 'seasonal' "
                "by design); ocean/fill = -9999. Coarser levels are the "
                "fill-aware integer mean, i.e. the percent of seasonal-snow "
                "area among classified land in each cell."),
            "units": "percent",
            "grid_mapping": "spatial_ref",
            "_FillValue": SNOW_CLASS_FILL,
        })


def blob_etag(url):
    """ETag of an HTTP(S) blob via a HEAD request, or None (best-effort)."""
    if not url.startswith(("http://", "https://")):
        return None
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.headers.get("ETag")
    except Exception as err:  # marker metadata only -- never fail the build
        log.warning("HEAD %s failed (%s); marker gets no ETag", url, err)
        return None


def azure_prefix_store(config, dest_prefix):
    container, prefix = dest_prefix.split("/", 1)
    return AzureStore(account_name=config.azure_storage_account,
                      container_name=container, prefix=prefix,
                      sas_key=config.sas_token)


def dest_store(config, dest_prefix, read_only=False):
    return zarr.storage.ObjectStore(azure_prefix_store(config, dest_prefix),
                                    read_only=read_only)


def open_pyramid_level(config, level, dest_prefix=None, decode=True,
                       chunks=None):
    """
    Open one pyramid level as an xarray Dataset (read-only, via obstore).

    The consumer-side counterpart of this builder — used by the global figure
    notebooks (visualize/global/) in place of the retired v9 coarsened store.
    Resolution at level n is ~80 m * 2**n (level 4 ~1.3 km, level 5 ~2.6 km,
    level 7 ~10 km). Open levels individually: xr.open_datatree on a
    multiscale hierarchy tries to align same-named dims with different
    coordinates across levels and fails.

    Args:
        config: Config (supplies the Azure account + SAS token AND the
            pyramid prefix via global_runoff_multiscale_azure_prefix).
        level: Pyramid level number (0 = native ~80 m).
        dest_prefix: Override the container/prefix.
        decode: Decode CF metadata (fill -> NaN, scale_factor applied).
        chunks: xarray chunking. None (default) = lazy backend arrays, no
            dask -- right for figure-scale levels (>= 4). Pass 'auto' for
            dask-backed reads of the big fine levels (0-3).
    """
    if dest_prefix is None:
        dest_prefix = config.global_runoff_multiscale_azure_prefix
    return xr.open_zarr(dest_store(config, dest_prefix, read_only=True),
                        group=str(level), zarr_format=3, consolidated=False,
                        mask_and_scale=decode, decode_coords="all",
                        chunks=chunks)


def read_progress(az, job_name):
    """Progress marker for a job, or None if absent/unreadable."""
    try:
        return json.loads(bytes(obstore.get(az, f"_build/{job_name}.json").bytes()))
    except Exception:
        return None


def write_progress(az, job_name, snapshot_id, completed, extra=None):
    payload = {"job": job_name, "source_snapshot_id": snapshot_id,
               "completed_levels": sorted(completed),
               "topozarr_version": TOPOZARR_VERSION, **(extra or {})}
    obstore.put(az, f"_build/{job_name}.json", json.dumps(payload).encode())


def fetch_live_root_attrs(config, dest_prefix):
    """Live root zarr.json attrs via anonymous public HTTPS (read-only, no SAS)."""
    url = (f"https://{config.azure_storage_account}.blob.core.windows.net/"
           f"{dest_prefix}/zarr.json")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read()).get("attributes", {}), url
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None, url
        raise


def check_root_attrs(config, args, job_name, computed_attrs, effective_attrs,
                     provenance):
    """Diff the root attrs this job would leave against the live store's.

    ``effective_attrs`` is what ``pyramid.write`` will actually apply via
    ``root.attrs.update`` ({} for the seasonal_snow job -- a literal no-op);
    ``computed_attrs`` is what topozarr computed before any neutralization
    (reported for context when they differ from live). Exits nonzero if the
    job would change or add any root-attr key.
    """
    live, url = fetch_live_root_attrs(config, args.dest_prefix)
    if live is None:
        log.info("no live root zarr.json at %s (fresh store) -- nothing to diff", url)
        return
    would_be = dict(live)
    would_be.update(effective_attrs)
    if provenance is not None:
        would_be["provenance"] = provenance
    added = sorted(set(would_be) - set(live))
    changed = sorted(k for k in set(would_be) & set(live)
                     if would_be[k] != live[k])
    for k in sorted(set(live) - set(added) - set(changed)):
        log.info("root attr %-22s unchanged", k)
    for k in added:
        log.warning("root attr %-22s ADDED: %s", k,
                    json.dumps(would_be[k], default=str)[:300])
    for k in changed:
        log.warning("root attr %-22s CHANGED\n  live: %s\n  new:  %s", k,
                    json.dumps(live[k], default=str)[:400],
                    json.dumps(would_be[k], default=str)[:400])
    suppressed = sorted(k for k in computed_attrs
                        if k not in effective_attrs
                        and computed_attrs[k] != live.get(k))
    if suppressed:
        log.info("neutralized computed attrs that differ from live (never "
                 "written): %s", suppressed)
    if added or changed:
        raise SystemExit(
            f"{job_name}: this job would rewrite root attrs {added + changed} "
            f"on {url} -- refusing is the point of --check-attrs")
    log.info("%s: root zarr.json stays byte-identical -- no root-attr changes",
             job_name)


def build_job(ds, job_name, job_vars, args, config, snapshot_id):
    seasonal_job = SEASONAL_SNOW_VAR in job_vars
    if seasonal_job:
        ds = ds.assign(
            {SEASONAL_SNOW_VAR: seasonal_snow_pct_dataarray(ds, args.snow_class_tif)})
    missing = [v for v in job_vars if v not in ds.data_vars]
    if missing:
        raise SystemExit(f"variables not in source store: {missing}")

    job_ds = ds[job_vars]
    pyramid = create_pyramid(
        job_ds,
        levels=args.levels,
        x_dim="longitude",
        y_dim="latitude",
        method="mean",
        # None (not {}) when no var has hints: an empty dict would still emit
        # a "zarr-layer": {} root attr
        layer_hints={k: v for k, v in LAYER_HINTS.items() if k in job_vars} or None,
    )

    # Root-attr byte stability: pyramid.write unconditionally runs
    # root.attrs.update(pyramid.attrs), and the live root attrs were computed
    # from a 3-D (water_year, latitude, longitude) job dataset -- the
    # seasonal_snow job's 2-D one would rewrite `multiscales` (2- vs 3-element
    # scale/translation lists) and `zarr-layer` with different values,
    # changing the immutable-cached root zarr.json. Neutralizing the payload
    # makes the update a literal no-op (MutableMapping.update({}) never
    # writes), so the root zarr.json is untouched; the mask's provenance
    # lives in its variable attrs and the _build marker instead, and the
    # cross-job `provenance` root attr below is skipped for the same reason.
    computed_attrs = dict(pyramid.attrs)
    if seasonal_job:
        pyramid.attrs = {}

    # Deterministic provenance (identical across the icechunk-sourced jobs of
    # one build, so the last-writer-wins race between parallel jobs is benign).
    provenance = None if seasonal_job else {
        "source_store": config.global_runoff_icechunk_azure_prefix,
        "source_tag": args.source_tag,
        "source_snapshot_id": snapshot_id,
        "levels": args.levels,
        "method": ("fill-aware integer mean-of-valid, cascaded level-from-level "
                   "on raw int16 (topozarr)"),
        "topozarr_version": TOPOZARR_VERSION,
        "builder": "visualize/pyramid/build_pyramid.py",
    }

    if args.check_attrs:
        check_root_attrs(config, args, job_name, computed_attrs,
                         pyramid.attrs, provenance)
        return

    # NOTE: pyramid.as_datatree() eagerly coarsens the (lazy) source -- fine
    # for small data, 191 GiB here. Level shapes are iterated floor-halving
    # of the spatial dims (topozarr trims trailing partial windows).
    sizes = {d: int(s) for d, s in job_ds.sizes.items()}
    for i in range(args.levels):
        log.info("level %d: %s", i, sizes)
        sizes = {d: (s // 2 if d in ("latitude", "longitude") else s)
                 for d, s in sizes.items()}
    if args.plan_only:
        log.info("Plan only: no changes made")
        return

    marker_extra = None
    if seasonal_job:
        marker_extra = {"snow_class_tif": args.snow_class_tif,
                        "snow_class_tif_etag": blob_etag(args.snow_class_tif)}

    az = azure_prefix_store(config, args.dest_prefix)
    completed: set[int] = set()
    if args.mode == "resume":
        marker = read_progress(az, job_name)
        if marker:
            if marker["source_snapshot_id"] != snapshot_id:
                raise SystemExit(
                    f"{job_name}: progress marker is from snapshot "
                    f"{marker['source_snapshot_id']}, source tag now resolves to "
                    f"{snapshot_id}. That's a different dataset, not a resume -- "
                    "bump multiscale_generation in the config and build fresh.")
            completed = set(marker["completed_levels"])
    else:
        write_progress(az, job_name, snapshot_id, completed, marker_extra)

    # rewrite everything from the first gap: each level derives from the one
    # before it, so levels past a rewritten one are stale even if marked
    start = next((lv for lv in range(args.levels) if lv not in completed),
                 args.levels)
    todo = list(range(start, args.levels))
    if not todo:
        log.info("%s: all %d levels already complete -- nothing to do",
                 job_name, args.levels)
        return

    store = dest_store(config, args.dest_prefix)
    log.info("Writing %s levels %s -> %s (mode='a', topozarr %s)",
             job_vars, todo, args.dest_prefix, TOPOZARR_VERSION)
    for lv in todo:
        stats = pyramid.write(store, mode="a", levels=[lv],
                              max_workers=args.max_workers,
                              progress=True, stats=True)
        if stats:
            log.info("level %d stats: %s", lv, json.dumps(stats, default=str))
        completed.add(lv)
        write_progress(az, job_name, snapshot_id, completed, marker_extra)

    if provenance is not None:
        root = zarr.open_group(dest_store(config, args.dest_prefix), mode="a")
        root.attrs["provenance"] = provenance
    log.info("Job done: %s", job_name)


def main():
    args = parse_args()
    if (args.job is None) == (args.variables is None):
        raise SystemExit("pass exactly one of --job / --variables")

    config = load_config(args)
    if args.source_tag is None:
        args.source_tag = config.release_tag
    if args.dest_prefix is None:
        args.dest_prefix = config.global_runoff_multiscale_azure_prefix
    if args.source_tag is None or args.dest_prefix is None:
        raise SystemExit(
            "config lacks release_tag / global_runoff_multiscale_azure_prefix "
            "(added for v10 on 2026-08-12) and no CLI overrides were given")

    ds, snapshot_id = open_source(config, args.source_tag)
    log.info("Source: %s @ %s (snapshot %s) -> %s [%s]",
             config.global_runoff_icechunk_azure_prefix, args.source_tag,
             snapshot_id, args.dest_prefix, args.mode)

    if args.variables:
        job_vars = args.variables.split(",")
        groups = [("_".join(job_vars), job_vars)]
    elif args.job == "all":
        groups = list(JOBS.items())
    else:
        groups = [(args.job, JOBS[args.job])]

    for job_name, job_vars in groups:
        build_job(ds, job_name, job_vars, args, config, snapshot_id)


if __name__ == "__main__":
    main()
