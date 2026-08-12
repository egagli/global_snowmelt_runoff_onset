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

Run `--job composites` FIRST (alone -- it creates the root group, the level
groups, and the convention attrs), then the two yearly jobs in parallel.

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
import sys
from pathlib import Path

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


def write_progress(az, job_name, snapshot_id, completed):
    payload = {"job": job_name, "source_snapshot_id": snapshot_id,
               "completed_levels": sorted(completed),
               "topozarr_version": TOPOZARR_VERSION}
    obstore.put(az, f"_build/{job_name}.json", json.dumps(payload).encode())


def build_job(ds, job_name, job_vars, args, config, snapshot_id):
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
        write_progress(az, job_name, snapshot_id, completed)

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
        write_progress(az, job_name, snapshot_id, completed)

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
