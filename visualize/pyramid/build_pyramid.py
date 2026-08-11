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
for Zarr v3 shards that adlfs gets wrong) into a versioned prefix. The
icechunk repo stays the only source of truth; the pyramid is disposable and
regenerable -- bump the prefix suffix to bust caches on regeneration.

Jobs (one topozarr pass per variable group; a job owns whole arrays, so jobs
never write the same zarr object and can run concurrently once the store
exists):

    composites           runoff_onset_median, runoff_onset_mad, temporal_resolution_median
    runoff_onset         runoff_onset (water_year, latitude, longitude)
    temporal_resolution  temporal_resolution (water_year, latitude, longitude)

Run `--job composites` FIRST (alone -- it creates the root group, the level
groups, and the convention attrs), then the two yearly jobs in parallel.
Jobs are idempotent: rerun a failed job whole (the source is pinned to a tag,
so rewrites are value-stable).

Usage:
    python visualize/pyramid/build_pyramid.py --job composites
    python visualize/pyramid/build_pyramid.py --job runoff_onset
    python visualize/pyramid/build_pyramid.py --job all              # sequential local run
    python visualize/pyramid/build_pyramid.py --job composites --plan-only
    # shakedown: one small variable to a scratch prefix
    python visualize/pyramid/build_pyramid.py --variables runoff_onset_median \\
        --dest-prefix snowmelt/snowmelt_runoff_onset/scratch_pyramid_shakedown
"""

import argparse
import json
import logging
import sys
from pathlib import Path

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
}

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
    p.add_argument("--config-file", default="config/global_config_v10.txt",
                   help="Path to config file (resolved against the repo root)")
    p.add_argument("--source-tag", default="v10.0",
                   help="Icechunk tag to build from (never a branch -- the "
                        "pyramid must come from a pinned, released snapshot)")
    p.add_argument("--dest-prefix", default=None,
                   help="container/prefix for the pyramid store. Default: "
                        "snowmelt/snowmelt_runoff_onset/global_runoff_onset_<tag>_multiscale_1")
    p.add_argument("--job", choices=[*JOBS, "all"], default=None,
                   help="Variable group to build ('all' runs the groups sequentially)")
    p.add_argument("--variables", default=None,
                   help="Comma-separated variable override (instead of --job); "
                        "for shakedowns and partial rebuilds")
    p.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    p.add_argument("--max-workers", type=int, default=None,
                   help="topozarr region thread pool size (default: derived "
                        "from CPU count and available memory)")
    p.add_argument("--plan-only", action="store_true",
                   help="Print per-level shapes/chunks without writing")
    return p.parse_args()


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


def dest_store(config, dest_prefix, read_only=False):
    container, prefix = dest_prefix.split("/", 1)
    azure = AzureStore(account_name=config.azure_storage_account,
                       container_name=container, prefix=prefix,
                       sas_key=config.sas_token)
    return zarr.storage.ObjectStore(azure, read_only=read_only)


def build_job(ds, job_vars, args, config, snapshot_id):
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
        layer_hints={k: v for k, v in LAYER_HINTS.items() if k in job_vars},
    )

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

    store = dest_store(config, args.dest_prefix)
    log.info("Writing %s -> %s (mode='a', topozarr %s)",
             job_vars, args.dest_prefix, TOPOZARR_VERSION)
    stats = pyramid.write(store, mode="a", max_workers=args.max_workers,
                          progress=True, stats=True)
    if stats:
        log.info("write stats: %s", json.dumps(stats, default=str))

    # Deterministic provenance (identical across jobs of one build, so the
    # last-writer-wins race between parallel jobs is benign).
    root = zarr.open_group(dest_store(config, args.dest_prefix), mode="a")
    root.attrs["provenance"] = {
        "source_store": config.global_runoff_icechunk_azure_prefix,
        "source_tag": args.source_tag,
        "source_snapshot_id": snapshot_id,
        "levels": args.levels,
        "method": ("fill-aware integer mean-of-valid, cascaded level-from-level "
                   "on raw int16 (topozarr)"),
        "topozarr_version": TOPOZARR_VERSION,
        "builder": "visualize/pyramid/build_pyramid.py",
    }
    log.info("Job done: %s", job_vars)


def main():
    args = parse_args()
    if (args.job is None) == (args.variables is None):
        raise SystemExit("pass exactly one of --job / --variables")

    config = Config(args.config_file)
    if args.dest_prefix is None:
        args.dest_prefix = ("snowmelt/snowmelt_runoff_onset/"
                            f"global_runoff_onset_{args.source_tag}_multiscale_1")

    ds, snapshot_id = open_source(config, args.source_tag)
    log.info("Source: %s @ %s (snapshot %s)",
             config.global_runoff_icechunk_azure_prefix, args.source_tag, snapshot_id)

    if args.variables:
        groups = [args.variables.split(",")]
    elif args.job == "all":
        groups = list(JOBS.values())
    else:
        groups = [JOBS[args.job]]

    for job_vars in groups:
        build_job(ds, job_vars, args, config, snapshot_id)


if __name__ == "__main__":
    main()
