#!/usr/bin/env python3
"""
Process a single tile into the global icechunk store, one water year at a time.

Platform-agnostic entrypoint (GitHub Actions, CryoCloud, local): each water
year of the tile is processed sequentially and committed to the icechunk
repository individually, so memory stays bounded, a timeout mid-tile loses at
most one year, and any single bad year can be reprocessed without redoing the
other nine. After the annual layers, the cross-year composites
(runoff_onset_median/mad, temporal_resolution_median) are computed -- reading
back any years not processed in this run -- and committed last.

Commit semantics (see global_snowmelt_runoff_onset.status):
- success with data     -> commit writing the tile x water_year shard
- verified empty        -> empty marker commit (allow_empty=True) with an
                           empty_reason: no_seasonal_snow (snow phenology says
                           no seasonal snow), no_s1_data (successful STAC
                           search found no scenes), no_valid_pixels (scenes
                           exist but nothing survives quality filtering)
- failure               -> NO commit; the job exits nonzero and the
                           tile x water_year stays 'missing' for redispatch

A transient Planetary Computer failure is never recorded as empty: the STAC
search is retried with backoff and, if it keeps failing, the job fails.
"""

import argparse
import gc
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import xarray as xr
import dask
import icechunk
import odc.geo.xr
import odc.stac

# Add the repo root to the Python path so the package imports without install
sys.path.append(str(Path(__file__).parent.parent.parent))

from global_snowmelt_runoff_onset import processing, status
from global_snowmelt_runoff_onset.config import Config
from global_snowmelt_runoff_onset.provenance import collect_provenance
from global_snowmelt_runoff_onset.store import tile_region_slices

log = logging.getLogger("process_single_tile")

COMMIT_MAX_TRIES = 8
# Attempts per water year for load+compute; a retry re-searches so the
# Planetary Computer asset tokens (~45 min lifetime) are freshly signed.
YEAR_LOAD_MAX_TRIES = 2


def setup_logging(tile_row: int, tile_col: int) -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"tile_{tile_row}_{tile_col}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    for noisy in ("azure", "urllib3", "fsspec", "adlfs", "aiohttp", "botocore", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_memory(context: str) -> None:
    try:
        import psutil
        rss_gb = psutil.Process().memory_info().rss / 1e9
        log.info(f"[mem] {context}: rss={rss_gb:.2f}GB ({psutil.virtual_memory().percent:.0f}% system)")
    except Exception:
        pass


def open_output_repo(config: Config, local_store: str | None) -> icechunk.Repository:
    """Open the output repo on Azure, or a local filesystem repo for testing."""
    if local_store:
        storage = icechunk.local_filesystem_storage(local_store)
        return icechunk.Repository.open(storage, config=config.output_repo_config())
    return config.open_output_repo()


def commit_with_retry(repo, branch, write_fn, message, metadata, allow_empty=False) -> str:
    """
    Write and commit against a fresh session, retrying on conflicts/transient errors.

    ConflictDetector rebases automatically when concurrent commits touched
    disjoint chunks (always true across tiles, since shards align with tiles);
    the outer bounded retry handles expired sessions and transient storage
    errors by redoing the whole write. Bounded on purpose: a persistent error
    (e.g. expired SAS token) should fail the job, not loop forever.
    """
    last_error = None
    for attempt in range(COMMIT_MAX_TRIES):
        try:
            session = repo.writable_session(branch)
            write_fn(session)
            return session.commit(
                message,
                metadata=metadata,
                rebase_with=icechunk.ConflictDetector(),
                allow_empty=allow_empty,
            )
        except (ValueError, KeyError, TypeError):
            raise  # programming/schema errors: retrying won't help
        except Exception as e:
            last_error = e
            delay = min(60, 2 ** attempt) * random.uniform(0.5, 1.5)
            log.warning(
                f"commit attempt {attempt + 1}/{COMMIT_MAX_TRIES} failed "
                f"({type(e).__name__}: {e}); retrying in {delay:.1f}s"
            )
            time.sleep(delay)
    raise RuntimeError(f"commit failed after {COMMIT_MAX_TRIES} attempts") from last_error


def commit_empty_year(repo, branch, config, tile_row, tile_col, water_year,
                      reason, prov, duration_s) -> str:
    metadata = status.build_commit_metadata(
        status.KIND_TILE_YEAR, tile_row, tile_col, config.version,
        status.STATUS_EMPTY, water_year=water_year, empty_reason=reason,
        duration_s=duration_s, provenance=prov,
    )
    message = status.build_commit_message(
        status.KIND_TILE_YEAR, tile_row, tile_col, status.STATUS_EMPTY,
        water_year=water_year, empty_reason=reason,
    )
    snapshot_id = commit_with_retry(repo, branch, lambda s: None, message, metadata, allow_empty=True)
    log.info(f"WY{water_year}: committed empty marker ({reason}) -> {snapshot_id}")
    return snapshot_id


def check_store_grid_alignment(store_ds, tile, region_2d) -> None:
    """
    Tripwire: the tile geobox is by construction an exact slice of the global
    geobox the store was built from, so integer region slices are exact. Guard
    against a store initialized from a different grid/config.
    """
    tolerance = abs(tile.geobox.resolution.x) / 4
    store_lat = store_ds.latitude.isel(latitude=region_2d["latitude"]).values
    store_lon = store_ds.longitude.isel(longitude=region_2d["longitude"]).values
    tile_lat = tile.geobox.coordinates["latitude"].values
    tile_lon = tile.geobox.coordinates["longitude"].values
    if store_lat.shape != tile_lat.shape or store_lon.shape != tile_lon.shape:
        raise RuntimeError(
            f"Store grid mismatch for tile ({tile.row},{tile.col}): store region "
            f"{store_lat.shape}x{store_lon.shape} vs tile geobox {tile_lat.shape}x{tile_lon.shape}"
        )
    np.testing.assert_allclose(store_lat, tile_lat, atol=tolerance)
    np.testing.assert_allclose(store_lon, tile_lon, atol=tolerance)


def process_one_year(s1_wy_ds, mask_ds, water_year, config, gmba_clipped_gdf):
    """
    Run the single-water-year pipeline: mask -> denoise -> per-orbit quality
    filter -> temporal resolution + runoff onset, computed into memory.

    Returns:
        (onset_2d, tr_2d, stats) with float32 numpy arrays (NaN = nodata), or
        (None, None, stats) when no pixel survives filtering.
    """
    center_lat = (s1_wy_ds.rio.bounds()[1] + s1_wy_ds.rio.bounds()[3]) / 2
    if np.absolute(center_lat) < 3:
        s1_wy_ds = processing.remove_equator_crossing(s1_wy_ds)

    if gmba_clipped_gdf is not None:
        s1_wy_ds = s1_wy_ds.rio.clip_box(*gmba_clipped_gdf.total_bounds, crs=gmba_clipped_gdf.crs)
        s1_wy_ds = s1_wy_ds.rio.clip(gmba_clipped_gdf.geometry, drop=True)

    crs = s1_wy_ds.rio.crs
    s1_masked_ds = processing.apply_mask_for_year(s1_wy_ds, mask_ds)
    s1_masked_ds.rio.write_crs(crs, inplace=True)

    s1_masked_ds = processing.remove_bad_scenes_and_border_noise(
        s1_masked_ds, config.low_backscatter_threshold
    )
    s1_masked_ds.attrs.update(s1_wy_ds.attrs)

    s1_filtered_ds = processing.filter_insufficient_pixels_per_orbit(
        s1_rtc_masked_ds=s1_masked_ds,
        spatiotemporal_snow_cover_mask_ds=mask_ds,
        min_monthly_acquisitions=config.min_monthly_acquisitions,
        max_allowed_days_gap_per_orbit=config.max_allowed_days_gap_per_orbit,
    )
    s1_filtered_ds.rio.write_crs(crs, inplace=True)
    s1_filtered_ds.attrs.update(s1_wy_ds.attrs)

    temporal_resolution_da = processing.get_temporal_resolution(
        s1_filtered_ds, mask_ds
    ).astype(np.float32)

    runoff_onset_da = processing.calculate_runoff_onset(
        s1_filtered_ds,
        returned_dates_format="dowy",
        return_constituent_runoff_onsets=False,
    )

    onset_np, tr_np = dask.compute(runoff_onset_da, temporal_resolution_da)

    # xr_datetime_to_DOWY encodes NaT as -9999 int16; normalize to float32/NaN
    # so the store's CF encoding (int16, _FillValue=-9999, x10 scaling for
    # temporal resolution) is applied uniformly on write.
    onset_2d = onset_np.values.astype(np.float32)
    onset_2d[onset_2d < 1] = np.nan
    tr_2d = np.squeeze(tr_np.sel(water_year=water_year).values).astype(np.float32)
    tr_2d[~np.isfinite(tr_2d)] = np.nan

    valid_px = int(np.isfinite(onset_2d).sum())
    stats = {
        "valid_px": valid_px,
        "n_scenes": int(s1_wy_ds.time.size),
        "n_orbits": int(np.unique(s1_wy_ds["sat:relative_orbit"].values).size),
        "median_tr_days": (round(float(np.nanmedian(tr_2d)), 2) if np.isfinite(tr_2d).any() else None),
    }
    if valid_px == 0:
        return None, None, stats
    return onset_2d, tr_2d, stats


def process_tile(config, repo, tile_row, tile_col, water_years, branch,
                 skip_composites, read_chunks) -> dict:
    """
    Process the requested water years of one tile, then refresh composites.

    Args:
        read_chunks: dask chunking for the Sentinel-1 read. Time chunking
            only batches independent per-scene 2D reads and never affects
            values. SPATIAL read-chunk size does not change which COG
            overview odc reads (that follows the min-axis scale ratio, i.e.
            latitude), but it does shift scene-footprint-EDGE warping and
            ULP-level float noise; end-to-end vs the v9-equivalent 512:
            99.56% identical DOWY, 0.04% coverage change, ~0.4% of pixels
            flip between near-tied backscatter minima, tile statistics
            unchanged. The 2048 default trades that for ~27% fewer bytes
            and ~2x faster loading (accepted July 2026).

    Returns a per-year outcome dict for the step summary.
    """
    prov = collect_provenance()
    tile = config.get_tile(tile_row, tile_col)
    region_2d = tile_region_slices(config, tile_row, tile_col)
    all_water_years = [int(wy) for wy in config.water_years]
    wy_to_index = {wy: i for i, wy in enumerate(all_water_years)}
    outcomes = {}

    odc.stac.configure_rio(cloud_defaults=True)

    readonly_ds = xr.open_zarr(
        repo.readonly_session(branch).store, zarr_format=3, consolidated=False,
        mask_and_scale=True, decode_coords="all",
    )
    check_store_grid_alignment(readonly_ds, tile, region_2d)
    tile_lat = readonly_ds.latitude.isel(latitude=region_2d["latitude"]).values
    tile_lon = readonly_ds.longitude.isel(longitude=region_2d["longitude"]).values

    # --- snow phenology mask, computed once for all years (also the
    # early-exit check: no seasonal snow -> empty markers without any S1 work)
    log.info("Loading spatiotemporal snow cover mask...")
    tile_template = odc.geo.xr.xr_zeros(tile.geobox, dtype="float32")
    mask_ds = processing.get_spatiotemporal_snow_cover_mask(
        ds=tile_template,
        bbox_gdf=tile.bbox_gdf,
        snow_phenology_store=config.snow_phenology_store,
        extend_search_window_beyond_SDD_days=config.extend_search_window_beyond_SDD_days,
        min_consec_snow_days_for_seasonal_snow=config.min_consec_snow_days_for_seasonal_snow,
    ).compute()
    # Re-chunk the computed mask: comparing numpy mask fields against the
    # (time,) DOWY coordinate inside apply_mask_for_year would otherwise
    # broadcast EAGERLY to full (time, 2048, 2048) numpy booleans -- ~2.5 GB
    # per mask term on scene-dense tiles (e.g. 598 scenes/year at 70N).
    # Chunked, those comparisons stay lazy and memory tracks dask chunks.
    mask_ds = mask_ds.chunk({
        "water_year": 1,
        "latitude": config.spatial_chunk_dim_s1_process,
        "longitude": config.spatial_chunk_dim_s1_process,
    })
    log_memory("snow mask computed")

    mask_years = set(int(wy) for wy in mask_ds.water_year.values)
    missing_from_phenology = [wy for wy in water_years if wy not in mask_years]
    if missing_from_phenology:
        raise RuntimeError(
            f"Water years {missing_from_phenology} not present in the snow phenology "
            "store -- refusing to mark them empty; update the phenology input first."
        )

    years_to_process = []
    for wy in water_years:
        t0 = time.time()
        if not bool(mask_ds.binary_seasonal_snow_cover_presence.sel(water_year=wy).any()):
            commit_empty_year(repo, branch, config, tile_row, tile_col, wy,
                              status.EMPTY_NO_SEASONAL_SNOW, prov, time.time() - t0)
            outcomes[wy] = ("empty", status.EMPTY_NO_SEASONAL_SNOW)
        else:
            years_to_process.append(wy)

    # Hemisphere from the tile center (matches the UTM-zone-based rule the
    # algorithm applies to loaded data); determines each water year's window.
    center_lat = (tile.geobox.boundingbox.bottom + tile.geobox.boundingbox.top) / 2
    hemisphere = "northern" if center_lat >= 0 else "southern"

    def water_year_window(wy):
        if hemisphere == "northern":
            return f"{wy - 1}-10-01", f"{wy}-09-30"
        return f"{wy}-04-01", f"{wy + 1}-03-31"

    gmba_clipped_gdf = (
        processing.get_gmba_mountain_inventory(tile.bbox_gdf)
        if config.mountain_snow_only else None
    )

    # --- Sentinel-1, searched + signed PER WATER YEAR. Planetary Computer
    # asset tokens expire ~45 min after signing, so one up-front search for
    # the whole 2014-2025 period would leave later years of a 10-year job
    # computing against expired URLs. A per-year search keeps each year's
    # token fresh at the start of that year's load/compute, and the retry
    # below re-searches (fresh token) if a year's compute still outlives it.
    year_results = {}
    for wy in years_to_process:
        t0 = time.time()
        year_start, year_end = water_year_window(wy)

        result = None
        for load_attempt in range(YEAR_LOAD_MAX_TRIES):
            log.info(f"WY{wy}: searching Sentinel-1 items ({year_start} to {year_end})"
                     + (f" [attempt {load_attempt + 1}]" if load_attempt else "") + "...")
            items = processing.search_sentinel1_items(
                tile.geobox, start_date=year_start, end_date=year_end
            )
            if len(items) == 0:
                result = "no_items"
                break
            s1_year_ds = processing.load_sentinel1_rtc(
                items, tile.geobox, bands=config.bands,
                chunks_read=read_chunks,
                fail_on_error=True,
            )
            # belt-and-braces: keep only acquisitions the WY/DOWY logic assigns
            # to this water year (search window and WY assignment use the same
            # timestamps, so this is normally a no-op)
            s1_wy_ds = s1_year_ds.sel(time=s1_year_ds.water_year == wy)
            log.info(f"WY{wy}: {int(s1_wy_ds.time.size)} scenes ({len(items)} items)")
            if s1_wy_ds.time.size == 0:
                result = "no_scenes"
                break
            try:
                result = process_one_year(s1_wy_ds, mask_ds, wy, config, gmba_clipped_gdf)
                break
            except Exception as e:
                # Typically odc "Aborting load due to failure while reading":
                # a transient blob failure or the signed token expiring
                # mid-compute. Nothing was committed, so retrying the whole
                # year against a freshly signed search is safe.
                if load_attempt + 1 >= YEAR_LOAD_MAX_TRIES:
                    raise
                log.warning(f"WY{wy}: load/compute failed ({type(e).__name__}: {e}); "
                            "re-searching with fresh asset tokens and retrying")
                del s1_year_ds, s1_wy_ds
                gc.collect()

        if result in ("no_items", "no_scenes"):
            commit_empty_year(repo, branch, config, tile_row, tile_col, wy,
                              status.EMPTY_NO_S1_DATA, prov, time.time() - t0)
            outcomes[wy] = ("empty", status.EMPTY_NO_S1_DATA)
            continue

        onset_2d, tr_2d, stats = result
        log_memory(f"WY{wy} computed")

        if onset_2d is None:
            commit_empty_year(repo, branch, config, tile_row, tile_col, wy,
                              status.EMPTY_NO_VALID_PIXELS, prov, time.time() - t0)
            outcomes[wy] = ("empty", status.EMPTY_NO_VALID_PIXELS)
            gc.collect()
            continue

        year_index = wy_to_index[wy]
        ds_write = xr.Dataset(
            {
                "runoff_onset": (("water_year", "latitude", "longitude"), onset_2d[None]),
                "temporal_resolution": (("water_year", "latitude", "longitude"), tr_2d[None]),
            },
            coords={"water_year": [wy], "latitude": tile_lat, "longitude": tile_lon},
        )
        region = {"water_year": slice(year_index, year_index + 1), **region_2d}

        metadata = status.build_commit_metadata(
            status.KIND_TILE_YEAR, tile_row, tile_col, config.version,
            status.STATUS_DATA, water_year=wy, stats=stats,
            duration_s=time.time() - t0, provenance=prov,
        )
        message = status.build_commit_message(
            status.KIND_TILE_YEAR, tile_row, tile_col, status.STATUS_DATA,
            water_year=wy, valid_px=stats["valid_px"],
        )
        snapshot_id = commit_with_retry(
            repo, branch,
            lambda session: ds_write.to_zarr(
                session.store, region=region, zarr_format=3, consolidated=False, mode="r+"
            ),
            message, metadata,
        )
        log.info(f"WY{wy}: committed {stats['valid_px']:,} valid px -> {snapshot_id}")
        outcomes[wy] = ("data", stats["valid_px"])

        year_results[wy] = (onset_2d, tr_2d)
        del s1_wy_ds, s1_year_ds, ds_write
        gc.collect()
        log_memory(f"WY{wy} committed")

    if skip_composites:
        return outcomes

    # --- composites over ALL water years: in-memory results where available,
    # store readback for years committed in previous runs
    t0 = time.time()
    n_lat = region_2d["latitude"].stop - region_2d["latitude"].start
    n_lon = region_2d["longitude"].stop - region_2d["longitude"].start
    onset_stack = np.full((len(all_water_years), n_lat, n_lon), np.nan, np.float32)
    tr_stack = np.full_like(onset_stack, np.nan)

    readback_years = [
        wy for wy in all_water_years
        if wy not in year_results and outcomes.get(wy, ("",))[0] != "empty"
    ]
    if readback_years:
        log.info(f"Reading back WYs {readback_years} from the store for composites...")
        # re-open at current tip so years committed by this job are visible
        readback_ds = xr.open_zarr(
            repo.readonly_session(branch).store, zarr_format=3, consolidated=False,
            mask_and_scale=True,
        ).isel(region_2d)
        onset_readback = readback_ds.runoff_onset.sel(water_year=readback_years).values
        tr_readback = readback_ds.temporal_resolution.sel(water_year=readback_years).values
        for i, wy in enumerate(readback_years):
            onset_stack[wy_to_index[wy]] = onset_readback[i]
            tr_stack[wy_to_index[wy]] = tr_readback[i]
    for wy, (onset_2d, tr_2d) in year_results.items():
        onset_stack[wy_to_index[wy]] = onset_2d
        tr_stack[wy_to_index[wy]] = tr_2d

    coords = {"water_year": all_water_years, "latitude": tile_lat, "longitude": tile_lon}
    onset_da = xr.DataArray(onset_stack, dims=("water_year", "latitude", "longitude"), coords=coords)
    tr_da = xr.DataArray(tr_stack, dims=("water_year", "latitude", "longitude"), coords=coords)

    median_da, mad_da = processing.median_and_mad_with_min_obs(
        da=onset_da, dim="water_year", min_count=config.min_years_for_median_std
    )
    tr_median_da = processing.median_with_min_obs(
        da=tr_da, dim="water_year", min_count=config.min_years_for_median_std
    )

    composite_valid_px = int(np.isfinite(median_da.values).sum())
    years_with_data = sorted(int(wy) for wy in all_water_years
                             if np.isfinite(onset_stack[wy_to_index[wy]]).any())
    composite_stats = {
        "valid_px": composite_valid_px,
        "years_with_data": years_with_data,
        "n_years_with_data": len(years_with_data),
    }

    if composite_valid_px == 0:
        metadata = status.build_commit_metadata(
            status.KIND_TILE_COMPOSITE, tile_row, tile_col, config.version,
            status.STATUS_EMPTY, empty_reason=status.EMPTY_NO_VALID_PIXELS,
            stats=composite_stats, duration_s=time.time() - t0, provenance=prov,
        )
        message = status.build_commit_message(
            status.KIND_TILE_COMPOSITE, tile_row, tile_col, status.STATUS_EMPTY,
            empty_reason=status.EMPTY_NO_VALID_PIXELS,
        )
        snapshot_id = commit_with_retry(repo, branch, lambda s: None, message, metadata, allow_empty=True)
        log.info(f"composites: committed empty marker -> {snapshot_id}")
        outcomes["composites"] = ("empty", status.EMPTY_NO_VALID_PIXELS)
        return outcomes

    composites_write_ds = xr.Dataset(
        {
            "runoff_onset_median": (("latitude", "longitude"), median_da.values.astype(np.float32)),
            "runoff_onset_mad": (("latitude", "longitude"), mad_da.values.astype(np.float32)),
            "temporal_resolution_median": (("latitude", "longitude"), tr_median_da.values.astype(np.float32)),
        },
        coords={"latitude": tile_lat, "longitude": tile_lon},
    )
    metadata = status.build_commit_metadata(
        status.KIND_TILE_COMPOSITE, tile_row, tile_col, config.version,
        status.STATUS_DATA, stats=composite_stats,
        duration_s=time.time() - t0, provenance=prov,
    )
    message = status.build_commit_message(
        status.KIND_TILE_COMPOSITE, tile_row, tile_col, status.STATUS_DATA,
        valid_px=composite_valid_px,
    )
    snapshot_id = commit_with_retry(
        repo, branch,
        lambda session: composites_write_ds.to_zarr(
            session.store, region=region_2d, zarr_format=3, consolidated=False, mode="r+"
        ),
        message, metadata,
    )
    log.info(f"composites: committed {composite_valid_px:,} valid px -> {snapshot_id}")
    outcomes["composites"] = ("data", composite_valid_px)
    return outcomes


def write_step_summary(tile_row, tile_col, outcomes) -> None:
    """Per-year outcome table in the GitHub Actions step summary, if running there."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [f"### Tile ({tile_row}, {tile_col})", "", "| water year | outcome | detail |", "|---|---|---|"]
    for key, (outcome, detail) in outcomes.items():
        detail_str = f"{detail:,} valid px" if outcome == "data" else str(detail)
        lines.append(f"| {key} | {outcome} | {detail_str} |")
    with open(summary_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process a single tile (per water year) into the icechunk store")
    parser.add_argument("--tile-row", type=int, required=True)
    parser.add_argument("--tile-col", type=int, required=True)
    parser.add_argument("--config-file", type=str, default="global_config_v10.txt",
                        help="Config file name in config/ (e.g. global_config_v10.txt)")
    parser.add_argument("--water-years", type=str, default="all",
                        help="Comma-separated water years to process, 'all' (default), "
                             "or 'none' to only (re)compute composites from committed years")
    parser.add_argument("--skip-composites", action="store_true",
                        help="Skip the cross-year composite commit")
    parser.add_argument("--branch", type=str, default="main")
    parser.add_argument("--local-store", type=str, default=None,
                        help="Path to a local icechunk repo (testing; overrides Azure)")
    parser.add_argument("--dask-workers", type=int, default=None,
                        help="Threaded-scheduler worker count (default: all cores). "
                             "Peak memory scales with workers; cap this on "
                             "many-core machines with limited RAM.")
    parser.add_argument("--read-chunk-dim", type=int, default=2048,
                        help="Spatial chunk size for the Sentinel-1 read. Default 2048 "
                             "(whole tile): ~27%% fewer bytes and ~2x faster loading "
                             "than 512 (fewer halo re-reads/round trips). Not "
                             "bit-reproducible against the v9-equivalent 512: "
                             "measured 99.56%% identical DOWY, 0.04%% coverage "
                             "change at scene-footprint edges, ~0.4%% of pixels "
                             "flip to a different backscatter minimum (idxmin "
                             "near-ties), tile statistics unchanged. Use 512 for "
                             "exact v9 comparisons.")
    parser.add_argument("--read-chunk-time", type=int, default=1,
                        help="Time chunk size for the Sentinel-1 read (default 1: "
                             "scenes are independent 2D reads, so this is value-"
                             "identical to any other batching and maximizes download "
                             "parallelism).")
    args = parser.parse_args()

    setup_logging(args.tile_row, args.tile_col)
    if args.dask_workers:
        dask.config.set(scheduler="threads", num_workers=args.dask_workers)
    else:
        dask.config.set(scheduler="threads")

    config_name = args.config_file if args.config_file.endswith(".txt") else f"global_config_{args.config_file}.txt"
    config = Config(str(Path(__file__).parent.parent.parent / "config" / config_name))
    if not config.output_store_is_icechunk:
        log.error(f"{config_name} is a legacy (pre-icechunk) config; use configs >= v10 here.")
        sys.exit(2)

    water_years_arg = args.water_years.strip().lower()
    if water_years_arg in ("all", ""):
        water_years = [int(wy) for wy in config.water_years]
    elif water_years_arg in ("none", "composites_only"):
        water_years = []  # composites-only: refresh from already-committed years
    else:
        water_years = sorted(int(wy) for wy in args.water_years.split(","))

    repo = open_output_repo(config, args.local_store)

    read_chunks = {"x": args.read_chunk_dim, "y": args.read_chunk_dim,
                   "time": args.read_chunk_time}

    start = time.time()
    try:
        outcomes = process_tile(
            config, repo, args.tile_row, args.tile_col, water_years,
            args.branch, args.skip_composites, read_chunks,
        )
    except Exception:
        log.error(f"Tile ({args.tile_row}, {args.tile_col}) FAILED after "
                  f"{time.time() - start:.0f}s:\n{traceback.format_exc()}")
        sys.exit(1)

    write_step_summary(args.tile_row, args.tile_col, outcomes)
    log.info(f"Tile ({args.tile_row}, {args.tile_col}) done in {time.time() - start:.0f}s: "
             + ", ".join(f"{key}={outcome}" for key, (outcome, _) in outcomes.items()))
    sys.exit(0)


if __name__ == "__main__":
    main()
