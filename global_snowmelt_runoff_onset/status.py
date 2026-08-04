"""
Processing status derived from Icechunk commit history.

The v10+ pipeline records everything it does as structured metadata on
Icechunk commits (one commit per tile x water year, plus one composite
commit per tile). This module is the single reader of that record: it walks
``repo.ancestry()`` once and derives which tile x water years hold data,
which are verified-empty, and which are still missing (never attempted, or
attempted and failed -- failures never commit, so absence == not done).

This replaces the tile_results_*.csv + GitHub-artifact-consolidation status
tracking used through config v9. The commit history is the single source of
truth; nothing here writes anything.

Commit metadata schema (see build_commit_metadata):

    {
      "schema": 1,
      "kind": "tile_year" | "tile_composite",
      "tile": [row, col],
      "water_year": 2019,              # tile_year only
      "status": "data" | "empty",
      "empty_reason": "no_seasonal_snow" | "no_s1_data" | "no_valid_pixels",
      "stats": {...},                  # per-kind statistics, see the runner
      "config_version": "v10",
      "duration_s": 123.4,
      "provenance": {...},             # see provenance.collect_provenance()
    }
"""

import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Any, Dict, List, Optional

COMMIT_SCHEMA_VERSION = 1

KIND_TILE_YEAR = "tile_year"
KIND_TILE_COMPOSITE = "tile_composite"

STATUS_DATA = "data"
STATUS_EMPTY = "empty"

EMPTY_NO_SEASONAL_SNOW = "no_seasonal_snow"
EMPTY_NO_S1_DATA = "no_s1_data"
EMPTY_NO_VALID_PIXELS = "no_valid_pixels"

DEFAULT_TRAILING_BUFFER_DAYS = 120


# ---------------------------------------------------------------------------
# Hemisphere-aware water-year eligibility
# ---------------------------------------------------------------------------
# A water year closes at different times in each hemisphere (northern WY N:
# Oct 1 N-1 .. Sep 30 N; southern WY N: Apr 1 N .. Mar 31 N+1), so one
# hemisphere is ready to process ~6 months before the other. Both the
# dispatcher (get_remaining_work) and the processor (process_single_tile)
# gate on this rule; an ineligible year is simply left uncommitted
# ('missing') and flows into the work list automatically once its
# eligibility date passes. The gate is what makes it safe for the upstream
# snow phenology store to hold a water-year slot that one hemisphere hasn't
# computed yet: an all-fill phenology slab is indistinguishable from
# verified no-snow, and without the gate the tile would be durably (and
# wrongly) committed as empty(no_seasonal_snow).

def season_end(wy: int, hemisphere: str) -> datetime.date:
    """Last calendar day of water year ``wy`` for a hemisphere."""
    if hemisphere == "northern":
        return datetime.date(wy, 9, 30)
    if hemisphere == "southern":
        return datetime.date(wy + 1, 3, 31)
    raise ValueError(f"unknown hemisphere: {hemisphere!r}")


def wy_eligible(
    wy: int,
    hemisphere: str,
    today: Optional[datetime.date] = None,
    trailing_buffer_days: int = DEFAULT_TRAILING_BUFFER_DAYS,
) -> bool:
    """True once ``wy`` has fully elapsed for ``hemisphere`` plus the buffer.

    The buffer (config key ``trailing_buffer_days``) covers the phenology
    pipeline's own 90-day trailing buffer plus grace for its fleet to run —
    S1 itself is in the catalog within days of acquisition.
    """
    today = today or datetime.date.today()
    return today >= season_end(wy, hemisphere) + datetime.timedelta(
        days=trailing_buffer_days
    )


def tile_hemisphere(geometry) -> str:
    """Hemisphere of a registry tile from its centroid latitude.

    Matches the UTM-EPSG-based rule the processing algorithm applies to
    loaded data (epsg < 32700 -> northern), since estimate_utm_crs also
    keys off the geometry's center.
    """
    return "northern" if geometry.centroid.y >= 0 else "southern"


def build_commit_metadata(
    kind: str,
    tile_row: int,
    tile_col: int,
    config_version: str,
    status: str,
    water_year: Optional[int] = None,
    empty_reason: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    duration_s: Optional[float] = None,
    provenance: Optional[Dict[str, Any]] = None,
    missing_assets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build the structured metadata dictionary attached to every pipeline commit.

    Machine-readable counterpart to the human-readable commit message; all
    status derivation in this module parses these dictionaries (never the
    message strings).
    """
    metadata: Dict[str, Any] = {
        "schema": COMMIT_SCHEMA_VERSION,
        "kind": kind,
        "tile": [int(tile_row), int(tile_col)],
        "status": status,
        "config_version": config_version,
    }
    if water_year is not None:
        metadata["water_year"] = int(water_year)
    if empty_reason is not None:
        metadata["empty_reason"] = empty_reason
    if stats is not None:
        metadata["stats"] = stats
    if duration_s is not None:
        metadata["duration_s"] = round(float(duration_s), 1)
    if provenance is not None:
        metadata["provenance"] = provenance
    if missing_assets:
        # Scenes excluded because their blobs are gone from object storage
        # (404 BlobNotFound) even though the STAC catalog still lists them. The
        # year is otherwise complete; recording the ids here is what keeps a
        # thinned year distinguishable from a whole one, and lets these years be
        # found again if the upstream archive is ever repaired.
        metadata["missing_assets"] = sorted(missing_assets)
    return metadata


def build_commit_message(
    kind: str,
    tile_row: int,
    tile_col: int,
    status: str,
    water_year: Optional[int] = None,
    empty_reason: Optional[str] = None,
    valid_px: Optional[int] = None,
) -> str:
    """Human-readable commit message (for `icechunk log`-style inspection only)."""
    prefix = f"tile({tile_row},{tile_col})"
    if kind == KIND_TILE_COMPOSITE:
        what = "composites"
    else:
        what = f"WY{water_year}"
    if status == STATUS_EMPTY:
        return f"{prefix} {what}: empty ({empty_reason})"
    if valid_px is not None:
        return f"{prefix} {what}: {valid_px:,} valid px"
    return f"{prefix} {what}: processed"


def get_commit_records(repo, branch: str = "main", as_of_snapshot: Optional[str] = None) -> pd.DataFrame:
    """
    Walk the branch ancestry once and return one row per pipeline commit.

    Non-pipeline commits (store init, manifest rewrites, ...) are skipped by
    requiring the metadata schema key. Ancestry iterates newest -> oldest;
    the returned frame preserves that order in the 'ancestry_index' column
    (0 = newest), which is what staleness comparisons use.

    Args:
        as_of_snapshot: derive status as of this snapshot instead of the
            branch tip. Used to pin one consistent work list across all
            batches of a fleet run (batches start at different times, and
            re-deriving from a moving tip would shift batch boundaries).

    Returns:
        DataFrame with columns: ancestry_index, snapshot_id, written_at, kind,
        row, col, water_year (NaN for composites), status, empty_reason,
        config_version, stats (dict), duration_s, missing_assets,
        provenance (dict)
    """
    if as_of_snapshot:
        ancestry = repo.ancestry(snapshot_id=as_of_snapshot)
    else:
        ancestry = repo.ancestry(branch=branch)
    records = []
    for index, snap in enumerate(ancestry):
        meta = snap.metadata or {}
        if "schema" not in meta or "kind" not in meta or "tile" not in meta:
            continue
        row, col = meta["tile"]
        records.append({
            "ancestry_index": index,
            "snapshot_id": snap.id,
            "written_at": snap.written_at,
            "kind": meta["kind"],
            "row": int(row),
            "col": int(col),
            "water_year": meta.get("water_year"),
            "status": meta.get("status"),
            "empty_reason": meta.get("empty_reason"),
            "config_version": meta.get("config_version"),
            "stats": meta.get("stats"),
            "duration_s": meta.get("duration_s"),
            "missing_assets": meta.get("missing_assets"),
            "provenance": meta.get("provenance"),
        })
    columns = ["ancestry_index", "snapshot_id", "written_at", "kind", "row", "col",
               "water_year", "status", "empty_reason", "config_version", "stats", "duration_s",
               "missing_assets", "provenance"]
    return pd.DataFrame.from_records(records, columns=columns)


def get_tile_status_gdf(config, repo=None, branch: str = "main",
                        as_of_snapshot: Optional[str] = None,
                        today: Optional[datetime.date] = None) -> gpd.GeoDataFrame:
    """
    Per-tile processing status: the tile registry joined with commit history.

    For each registry tile, derives:
    - hemisphere: 'northern'/'southern' from the tile centroid
    - one column per water year ('wy_2015'...): 'data', 'empty', 'missing'
      (eligible but no commit yet), or 'ineligible' (the tile hemisphere's
      season has not fully elapsed + trailing_buffer_days)
    - n_years_done / n_years_eligible: committed vs currently-eligible counts
    - composites: 'data', 'empty', 'missing', or 'stale' (a year commit is
      newer than the composite commit, e.g. after a single-year reprocess)
    - tile_status: 'complete' (all ELIGIBLE years + fresh composites --
      ineligible years never hold a tile at 'partial'), 'partial', or
      'unprocessed' (no commits at all)

    Args:
        config: Config object (supplies the registry, water_years, and
            trailing_buffer_days; used to open the output repo when 'repo'
            isn't passed)
        repo: optionally, an already-open icechunk Repository
        branch: branch whose history to read
        today: eligibility reference date (defaults to today; for tests)

    Returns:
        GeoDataFrame, one row per registry tile, sorted like the registry
        (percent_valid_snow_pixels descending).
    """
    if repo is None:
        repo = config.open_output_repo()
    water_years = [int(wy) for wy in config.water_years]
    commits_df = get_commit_records(repo, branch=branch, as_of_snapshot=as_of_snapshot)

    # Newest commit wins for each (tile, water_year) and for each tile's composite.
    year_cols = {}
    composite_col = {}
    composite_index = {}
    newest_year_index = {}
    if not commits_df.empty:
        year_df = commits_df[commits_df.kind == KIND_TILE_YEAR]
        for record in year_df.itertuples():
            key = (record.row, record.col, int(record.water_year))
            if key not in year_cols:  # first seen == newest
                year_cols[key] = record.status
                tile_key = (record.row, record.col)
                newest_year_index[tile_key] = min(
                    newest_year_index.get(tile_key, record.ancestry_index), record.ancestry_index
                )
        comp_df = commits_df[commits_df.kind == KIND_TILE_COMPOSITE]
        for record in comp_df.itertuples():
            tile_key = (record.row, record.col)
            if tile_key not in composite_col:
                composite_col[tile_key] = record.status
                composite_index[tile_key] = record.ancestry_index

    tiles_gdf = gpd.read_file(config.valid_tiles_geojson_path)
    if "tile" in tiles_gdf.columns:
        tiles_gdf = tiles_gdf.drop(columns=["tile"])
    if "to_process" in tiles_gdf.columns:
        # v10+ registries keep excluded tiles (e.g. no VV items at probe date) for
        # documentation; only to_process == True rows define the work universe.
        tiles_gdf = tiles_gdf[tiles_gdf["to_process"]].copy()
    tiles_gdf = tiles_gdf.sort_values(by="percent_valid_snow_pixels", ascending=False)

    tiles_gdf["hemisphere"] = [tile_hemisphere(geom) for geom in tiles_gdf.geometry]
    buffer_days = getattr(config, "trailing_buffer_days", DEFAULT_TRAILING_BUFFER_DAYS)
    eligible_by_hemi = {
        hemi: {wy for wy in water_years
               if wy_eligible(wy, hemi, today=today, trailing_buffer_days=buffer_days)}
        for hemi in ("northern", "southern")
    }

    for wy in water_years:
        tiles_gdf[f"wy_{wy}"] = [
            year_cols.get(
                (row, col, wy),
                "missing" if wy in eligible_by_hemi[hemi] else "ineligible",
            )
            for row, col, hemi in zip(tiles_gdf.row, tiles_gdf.col, tiles_gdf.hemisphere)
        ]

    def _composites_status(row, col):
        tile_key = (row, col)
        if tile_key not in composite_col:
            return "missing"
        # Lower ancestry_index == newer commit. A year commit newer than the
        # composite commit means the composites no longer reflect all years.
        if newest_year_index.get(tile_key, np.inf) < composite_index[tile_key]:
            return "stale"
        return composite_col[tile_key]

    tiles_gdf["composites"] = [
        _composites_status(row, col) for row, col in zip(tiles_gdf.row, tiles_gdf.col)
    ]
    wy_columns = [f"wy_{wy}" for wy in water_years]
    tiles_gdf["n_years_done"] = (
        tiles_gdf[wy_columns].isin([STATUS_DATA, STATUS_EMPTY]).sum(axis=1)
    )
    tiles_gdf["n_years_eligible"] = [
        len(eligible_by_hemi[hemi]) for hemi in tiles_gdf.hemisphere
    ]

    def _tile_status(record):
        if record.n_years_done == 0 and record.composites == "missing":
            return "unprocessed"
        # Complete == all currently-ELIGIBLE years committed. Ineligible years
        # must not hold tiles at 'partial': when a new water year enters the
        # config, every tile would otherwise flip to 'partial' months before
        # its hemisphere's season has even closed.
        if record.n_years_done >= record.n_years_eligible and record.composites in (STATUS_DATA, STATUS_EMPTY):
            return "complete"
        return "partial"

    tiles_gdf["tile_status"] = tiles_gdf.apply(_tile_status, axis=1)
    return tiles_gdf


def get_remaining_work(
    config,
    repo=None,
    which: str = "incomplete",
    include_empty_years: bool = False,
    branch: str = "main",
    as_of_snapshot: Optional[str] = None,
    today: Optional[datetime.date] = None,
) -> List[Dict[str, Any]]:
    """
    List of work items for dispatch (GitHub Actions matrix / local runner).

    Args:
        which: which tiles to include:
            - 'incomplete': anything not complete -- unprocessed tiles, tiles
              with missing water years (incl. previously failed: they never
              commit), and tiles with missing/stale composites (default)
            - 'unprocessed': only tiles with no commits at all
            - 'all': every registry tile, all water years (full reprocess)
        include_empty_years: also redo years previously committed as empty
            (e.g. to pick up catalog backfills). Empty markers are only
            written after a *successful* STAC search, so this is normally off.

    Returns:
        List of {"row", "col", "water_years": [...]} dicts, ordered like the
        registry (snowiest tiles first). An empty "water_years" list means
        all years are committed and only the composites need (re)computing.
        Hemisphere-ineligible water years (season not elapsed +
        trailing_buffer_days for the tile's hemisphere) are absent from every
        mode — they enter the work list automatically once eligible.
    """
    water_years = [int(wy) for wy in config.water_years]
    tiles_gdf = get_tile_status_gdf(config, repo=repo, branch=branch,
                                    as_of_snapshot=as_of_snapshot, today=today)

    redo_statuses = {"missing", STATUS_EMPTY} if include_empty_years else {"missing"}

    work = []
    for record in tiles_gdf.itertuples():
        eligible_years = [
            wy for wy in water_years if getattr(record, f"wy_{wy}") != "ineligible"
        ]
        if which == "all":
            work.append({"row": int(record.row), "col": int(record.col), "water_years": eligible_years})
            continue
        if which == "unprocessed":
            if record.tile_status == "unprocessed":
                work.append({"row": int(record.row), "col": int(record.col), "water_years": eligible_years})
            continue
        if which == "incomplete":
            if record.tile_status == "complete" and not include_empty_years:
                continue
            missing_years = [
                wy for wy in water_years if getattr(record, f"wy_{wy}") in redo_statuses
            ]
            if missing_years or record.composites in ("missing", "stale"):
                work.append({"row": int(record.row), "col": int(record.col), "water_years": missing_years})
            continue
        raise ValueError("which must be one of ['incomplete', 'unprocessed', 'all']")
    return work
