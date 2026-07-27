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
        config_version, stats (dict), duration_s
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
        })
    columns = ["ancestry_index", "snapshot_id", "written_at", "kind", "row", "col",
               "water_year", "status", "empty_reason", "config_version", "stats", "duration_s"]
    return pd.DataFrame.from_records(records, columns=columns)


def get_tile_status_gdf(config, repo=None, branch: str = "main",
                        as_of_snapshot: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Per-tile processing status: the tile registry joined with commit history.

    For each registry tile, derives:
    - one column per water year ('wy_2015'...): 'data', 'empty', or 'missing'
    - n_years_done: count of water years with a commit (data or empty)
    - composites: 'data', 'empty', 'missing', or 'stale' (a year commit is
      newer than the composite commit, e.g. after a single-year reprocess)
    - tile_status: 'complete' (all years + fresh composites), 'partial',
      or 'unprocessed' (no commits at all)

    Args:
        config: Config object (supplies the registry and water_years; used to
            open the output repo when 'repo' isn't passed)
        repo: optionally, an already-open icechunk Repository
        branch: branch whose history to read

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
    tiles_gdf = tiles_gdf.sort_values(by="percent_valid_snow_pixels", ascending=False)

    for wy in water_years:
        tiles_gdf[f"wy_{wy}"] = [
            year_cols.get((row, col, wy), "missing")
            for row, col in zip(tiles_gdf.row, tiles_gdf.col)
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
    tiles_gdf["n_years_done"] = (tiles_gdf[wy_columns] != "missing").sum(axis=1)

    def _tile_status(record):
        if record.n_years_done == 0 and record.composites == "missing":
            return "unprocessed"
        if record.n_years_done == len(water_years) and record.composites in (STATUS_DATA, STATUS_EMPTY):
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
    """
    water_years = [int(wy) for wy in config.water_years]
    tiles_gdf = get_tile_status_gdf(config, repo=repo, branch=branch,
                                    as_of_snapshot=as_of_snapshot)

    redo_statuses = {"missing", STATUS_EMPTY} if include_empty_years else {"missing"}

    work = []
    for record in tiles_gdf.itertuples():
        if which == "all":
            work.append({"row": int(record.row), "col": int(record.col), "water_years": water_years})
            continue
        if which == "unprocessed":
            if record.tile_status == "unprocessed":
                work.append({"row": int(record.row), "col": int(record.col), "water_years": water_years})
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
