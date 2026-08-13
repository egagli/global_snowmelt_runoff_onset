"""
Sampling-based quality gates on the v10+ output store (`processing/3_quality_check_tiles.ipynb`).

`status.py` gates the **ledger** -- did every tile x water year commit? This module gates the
**data** those commits wrote: store schema, ledger/data agreement, per-pixel invariants,
composite arithmetic, tile seams, and a per-release fingerprint of the test-tile battery.

Everything here **samples**. The gates run on the eight-tile battery
(`processing/tile_data/test_tiles_<version>.txt`, each tile chosen for a failure mode the
others don't reach) plus a seeded random draw from the registry -- never on all ~4,400
processed tiles, which would cost more than the processing run it is checking. A passing
gate means "no defect of this class in the sample"; keeping the battery fixed is what makes
that statement comparable across releases.

Every check function returns a tidy DataFrame with an ``ok`` column, so `summarize()` can
reduce the lot to one PASS/FAIL row per gate.
"""

import hashlib
import pathlib
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from . import processing, status
from .store import NODATA_INT16, SCALED_VARIABLES, tile_region_slices

ANNUAL_VARIABLES = ("runoff_onset", "temporal_resolution")
COMPOSITE_VARIABLES = ("runoff_onset_median", "runoff_onset_mad", "temporal_resolution_median")
ALL_VARIABLES = ANNUAL_VARIABLES + COMPOSITE_VARIABLES

# Physical bounds. Onset is a day of water year; temporal resolution is
# search-window-length / valid-acquisition-count, so a single-acquisition pixel in a
# year-long window is the realistic ceiling.
DOWY_RANGE = (1.0, 366.0)
MAX_TEMPORAL_RESOLUTION_DAYS = 400.0

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _scale(var: str) -> float:
    """On-disk scale factor for a variable (0.1 for the scaled trio, else 1.0)."""
    return 0.1 if var in SCALED_VARIABLES else 1.0


def decode(raw: np.ndarray, var: str) -> np.ndarray:
    """Raw int16 -> float32 physical units with NaN at `_FillValue`."""
    out = raw.astype(np.float32) * np.float32(_scale(var))
    out[raw == NODATA_INT16] = np.nan
    return out


# ─── tile selection ─────────────────────────────────────────────────────────

def read_tile_list(path) -> List[Tuple[int, int]]:
    """
    Parse a `row,col`-per-line tile list (`#` comments, inline or whole-line).

    The format shared by `test_tiles_<version>.txt`, `station_tiles_<version>.txt`, and
    `manual_tiles_<version>.txt`.
    """
    tiles = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        row, col = line.split(",")
        tiles.append((int(row), int(col)))
    return tiles


def test_tiles(config) -> List[Tuple[int, int]]:
    """
    The validation battery for this config version, from `tile_data/test_tiles_<version>.txt`.

    That file -- not a list retyped into a notebook -- is the single source of truth: it
    carries the per-tile rationale and the v9 index for each, and the same list is what the
    fleet workflows dispatch via their `tiles_file` input.
    """
    path = _REPO_ROOT / "processing" / "tile_data" / f"test_tiles_{config.version}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"no test-tile battery for {config.version} at {path}; "
            "batteries are versioned alongside the config")
    return read_tile_list(path)


def sample_tiles(config, n: int, seed: int = 0,
                 exclude: Iterable[Tuple[int, int]] = ()) -> List[Tuple[int, int]]:
    """
    A seeded random draw of `n` `to_process` registry tiles.

    The battery covers the *known* failure modes; this covers the ordinary interior of the
    fleet, where a defect would otherwise have to be stumbled upon. Seeded so a rerun checks
    the same tiles -- change `seed` deliberately to widen coverage across runs.
    """
    import geopandas as gpd

    registry = gpd.read_file(config.valid_tiles_geojson_path)
    if "to_process" in registry.columns:
        registry = registry[registry["to_process"]]
    candidates = sorted({(int(r), int(c)) for r, c in zip(registry.row, registry.col)}
                        - set(exclude))
    return sorted(random.Random(seed).sample(candidates, min(n, len(candidates))))


# ─── store access ───────────────────────────────────────────────────────────

def open_release(config, tag: Optional[str] = None, branch: str = "main",
                 mask_and_scale: bool = False, chunks=None):
    """
    Open the output store at its release tag (falling back to `branch`).

    QC must describe a *fixed* snapshot: reading the branch tip while a fleet is committing
    would make the gates non-reproducible, and after a future water-year append the tag is
    the only way back to what was released.

    Returns:
        (ds, session, source) -- `source` is a human-readable provenance string to print.
    """
    repo = config.open_output_repo()
    tag = tag if tag is not None else getattr(config, "release_tag", None)
    if tag and tag in repo.list_tags():
        session = repo.readonly_session(tag=tag)
        source = f"tag {tag}"
    else:
        session = repo.readonly_session(branch)
        source = f"branch {branch}" + (f" ({tag!r} not found)" if tag else " (no release_tag)")
    ds = xr.open_zarr(session.store, zarr_format=3, consolidated=False, decode_coords="all",
                      chunks=chunks, mask_and_scale=mask_and_scale)
    return ds, session, f"{source}, snapshot {session.snapshot_id}"


@dataclass
class TileData:
    """One tile's five variables held as raw int16, loaded once and checked many times."""
    row: int
    col: int
    water_years: np.ndarray
    raw: Dict[str, np.ndarray]
    _decoded: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return f"({self.row},{self.col})"

    def dec(self, var: str) -> np.ndarray:
        """Decoded (float32, NaN-filled) view of `var`, computed once."""
        if var not in self._decoded:
            self._decoded[var] = decode(self.raw[var], var)
        return self._decoded[var]

    def valid_px(self, var: str, year_index: Optional[int] = None) -> int:
        arr = self.raw[var] if year_index is None else self.raw[var][year_index]
        return int(np.count_nonzero(arr != NODATA_INT16))


def load_tile(config, ds_raw: xr.Dataset, row: int, col: int) -> TileData:
    """
    Read one tile's five variables into memory as raw int16 (~210 MB for a full tile).

    Raw rather than decoded on purpose: it is the bytes actually stored, it is 4x smaller
    than the CF-decoded float64 the scaled variables would become, and `_FillValue` stays
    checkable as a value.
    """
    region = tile_region_slices(config, row, col)
    raw = {var: np.asarray(ds_raw[var].isel(region).values) for var in ALL_VARIABLES}
    return TileData(row=row, col=col,
                    water_years=np.asarray(ds_raw.water_year.values, dtype=int), raw=raw)


# ─── gate 1: schema ─────────────────────────────────────────────────────────

def check_schema(config, ds: xr.Dataset, session) -> pd.DataFrame:
    """
    Metadata-only lint of the store: dtype, fill, scaling, shard/chunk geometry, grid, coords.

    Cheap (no chunk reads) and the class of damage it catches is total: a wrong zarr
    `fill_value` makes every never-written region decode as a valid 0.0 rather than nodata,
    and a shard/chunk change silently destroys the one-shard-per-tile-year write isolation
    the whole fleet depends on.
    """
    tile_dim = config.spatial_chunk_dim_zarr_output
    inner = config.inner_chunk_dim
    n_y, n_x = (int(v) for v in config.global_geobox.shape.yx)
    n_wy = len(config.water_years)
    group = zarr.open_group(session.store, mode="r")
    rows = []

    def record(check, expected, actual, ok=None):
        rows.append({"check": check, "expected": str(expected), "actual": str(actual),
                     "ok": bool(expected == actual) if ok is None else bool(ok)})

    for var in ALL_VARIABLES:
        arr = group[var]
        is_3d = var in ANNUAL_VARIABLES
        record(f"{var}: shape", (n_wy, n_y, n_x) if is_3d else (n_y, n_x), tuple(arr.shape))
        record(f"{var}: dtype", "int16", str(arr.dtype))
        record(f"{var}: zarr fill_value", NODATA_INT16, arr.fill_value)
        record(f"{var}: _FillValue attr", NODATA_INT16, arr.attrs.get("_FillValue"))
        record(f"{var}: shards", (1, tile_dim, tile_dim) if is_3d else (tile_dim, tile_dim),
               tuple(arr.shards) if arr.shards else None)
        record(f"{var}: chunks", (1, inner, inner) if is_3d else (inner, inner),
               tuple(arr.chunks))
        # the store writes np.float32(0.1), which serializes as 0.10000000149011612 --
        # compare with a tolerance rather than exactly
        stored_scale = float(arr.attrs.get("scale_factor", 1.0))
        record(f"{var}: scale_factor", _scale(var), stored_scale,
               ok=np.isclose(stored_scale, _scale(var), rtol=1e-6))

    record("grid shape matches config geobox", (n_y, n_x),
           (ds.sizes["latitude"], ds.sizes["longitude"]))
    record("water_year coord matches config", list(int(wy) for wy in config.water_years),
           [int(wy) for wy in ds.water_year.values])
    record("latitude strictly decreasing", True,
           bool(np.all(np.diff(ds.latitude.values) < 0)))
    record("longitude strictly increasing", True,
           bool(np.all(np.diff(ds.longitude.values) > 0)))
    record("spatial_ref coordinate present", True, "spatial_ref" in ds.coords)
    return pd.DataFrame(rows)


# ─── gate 2: ledger vs data ─────────────────────────────────────────────────

def ledger_lookup(commits_df: pd.DataFrame) -> Dict[Tuple[int, int, Optional[int]], tuple]:
    """
    Newest commit per (row, col, water_year), with `water_year=None` for the composites.

    `get_commit_records` returns newest-first, so first-seen wins.
    """
    lookup = {}
    for rec in commits_df.sort_values("ancestry_index").itertuples():
        wy = None if rec.kind == status.KIND_TILE_COMPOSITE else int(rec.water_year)
        key = (int(rec.row), int(rec.col), wy)
        if key in lookup:
            continue
        stats = rec.stats if isinstance(rec.stats, dict) else {}
        lookup[key] = (rec.status, rec.empty_reason, stats.get("valid_px"))
    return lookup


def check_ledger(tile: TileData, lookup: dict) -> pd.DataFrame:
    """
    Does the store agree, pixel for pixel, with what the commit history claims?

    The commit stats already record `valid_px` per tile-year, so this is an **equality**
    test, not a tolerance: a `data` commit must find exactly that many valid pixels in its
    slab, and an `empty` marker (or no commit at all) must find a slab that was never
    written. The only check that can catch a durably-wrong empty marker -- the hemisphere
    trap in `processing/README.md`, where an all-fill phenology slab is indistinguishable
    from verified no-snow -- or a region write that landed on the wrong shard.
    """
    rows = []
    for index, wy in enumerate(tile.water_years):
        rows.append(_ledger_row(tile, int(wy), lookup.get((tile.row, tile.col, int(wy))),
                                tile.valid_px("runoff_onset", index)))
    rows.append(_ledger_row(tile, "composites", lookup.get((tile.row, tile.col, None)),
                            tile.valid_px("runoff_onset_median")))
    return pd.DataFrame(rows)


def _ledger_row(tile, water_year, entry, store_px):
    ledger_status, reason, ledger_px = entry if entry else ("missing", None, None)
    if ledger_status == status.STATUS_DATA:
        # valid_px is absent from a handful of very early commits; fall back to the sign test
        ok = store_px > 0 and (ledger_px is None or store_px == int(ledger_px))
        note = "" if ok else ("data commit but empty slab" if store_px == 0
                              else f"count mismatch ({store_px} vs {ledger_px})")
    else:
        ok = store_px == 0
        note = "" if ok else f"{ledger_status} but {store_px:,} px written"
    return {"tile": tile.name, "row": tile.row, "col": tile.col, "water_year": water_year,
            "ledger_status": ledger_status, "empty_reason": reason,
            "ledger_valid_px": ledger_px, "store_valid_px": store_px, "ok": ok, "note": note}


# ─── gate 3: invariants and composite arithmetic ────────────────────────────

def _n_outside(arr: np.ndarray, low: float, high: float) -> int:
    finite = np.isfinite(arr)
    return int(np.count_nonzero(finite & ((arr < low) | (arr > high))))


def check_invariants(tile: TileData) -> pd.DataFrame:
    """
    Per-pixel properties that must hold for every valid pixel, whatever the science says.

    The one-directional implication is deliberate: temporal resolution is
    search-window-length / valid-acquisition-count, so any pixel with an onset estimate must
    have one, but a pixel can have acquisitions and still yield no qualifying backscatter
    minimum -- valid `temporal_resolution` over nodata `runoff_onset` is expected, not a bug.
    """
    onset, tr = tile.dec("runoff_onset"), tile.dec("temporal_resolution")
    median, mad = tile.dec("runoff_onset_median"), tile.dec("runoff_onset_mad")
    tr_median = tile.dec("temporal_resolution_median")
    lo, hi = DOWY_RANGE
    tr_hi = MAX_TEMPORAL_RESOLUTION_DAYS
    checks = {
        f"runoff_onset in [{lo:.0f},{hi:.0f}]": _n_outside(onset, lo, hi),
        f"runoff_onset_median in [{lo:.0f},{hi:.0f}]": _n_outside(median, lo, hi),
        "runoff_onset_mad >= 0": int(np.count_nonzero(mad < 0)),
        f"temporal_resolution in (0,{tr_hi:.0f}]": _n_outside(tr, 1e-6, tr_hi),
        f"temporal_resolution_median in (0,{tr_hi:.0f}]": _n_outside(tr_median, 1e-6, tr_hi),
        "runoff_onset valid => temporal_resolution valid":
            int(np.count_nonzero(np.isfinite(onset) & ~np.isfinite(tr))),
    }
    return pd.DataFrame([
        {"tile": tile.name, "row": tile.row, "col": tile.col, "check": name,
         "n_violations": n, "ok": n == 0}
        for name, n in checks.items()
    ])


def check_composites(tile: TileData, config, slack: float = 1e-3) -> pd.DataFrame:
    """
    Recompute the composites from the tile's own annual layers and demand agreement.

    Calls the *same* `processing.median_*_with_min_obs` the pipeline used, so this tests the
    stored bytes rather than re-deriving the maths, including the `min_years_for_median_std`
    rule (a pixel with too few years must be nodata in the composite). Tolerance is exactly
    the int16 quantization step (0.5 day unscaled, 0.05 for the 0.1-scaled variables).

    It also catches a **stale composite** directly -- one written before a later single-year
    reprocess -- which `status.py` can only infer from commit ordering.

    **Tolerance carries two quantization terms, not one.** The pipeline reduced the annual
    layers while they were still float32 in memory and rounded only the result; this
    recompute starts from what the store holds, which is already on the int16 grid. So the
    source contributes up to half its own step before the output's rounding adds half of
    hers. That matters for `temporal_resolution_median` alone (0.05 + 0.05 = 0.1): it is the
    only composite whose source variable is scaled, since `runoff_onset` stores integer DOWY
    values exactly. Measured against the v10.0 release, ~0.01% of pixels on some tiles land
    at exactly one 0.1 step -- a real stale composite is off by days, so widening here costs
    the gate nothing.
    """
    dims = ("water_year", "latitude", "longitude")
    onset = xr.DataArray(tile.dec("runoff_onset"), dims=dims)
    tr = xr.DataArray(tile.dec("temporal_resolution"), dims=dims)
    median, mad = processing.median_and_mad_with_min_obs(
        da=onset, dim="water_year", min_count=config.min_years_for_median_std)
    tr_median = processing.median_with_min_obs(
        da=tr, dim="water_year", min_count=config.min_years_for_median_std)

    rows = []
    for var, source, recomputed in (
            ("runoff_onset_median", "runoff_onset", median),
            ("runoff_onset_mad", "runoff_onset", mad),
            ("temporal_resolution_median", "temporal_resolution", tr_median)):
        stored = tile.dec(var)
        expected = np.asarray(recomputed.values, dtype=np.float32)
        stored_valid, expected_valid = np.isfinite(stored), np.isfinite(expected)
        both = stored_valid & expected_valid
        # half a step for the output's own rounding, plus half a step for the source's if
        # the source is stored scaled (see the note above)
        source_step = _scale(source) if source in SCALED_VARIABLES else 0.0
        tolerance = 0.5 * _scale(var) + 0.5 * source_step + slack
        diff = np.abs(stored[both] - expected[both])
        n_mask = int(np.count_nonzero(stored_valid ^ expected_valid))
        n_over = int(np.count_nonzero(diff > tolerance))
        rows.append({
            "tile": tile.name, "row": tile.row, "col": tile.col, "variable": var,
            "n_compared": int(both.sum()), "n_nodata_mismatch": n_mask,
            "n_over_tolerance": n_over,
            "max_absdiff": float(diff.max()) if diff.size else 0.0,
            "tolerance": tolerance, "ok": n_mask == 0 and n_over == 0,
        })
    return pd.DataFrame(rows)


# ─── gate 4: tile seams ─────────────────────────────────────────────────────

def check_seams(config, ds_raw: xr.Dataset, tiles: Sequence[Tuple[int, int]],
                var: str = "runoff_onset_median", max_ratio: float = 1.5,
                min_pairs: int = 1000) -> pd.DataFrame:
    """
    Is the step across a tile boundary bigger than the steps just inside it?

    Tiles are processed as independent jobs, so a visible seam is the signature failure of
    the whole architecture -- and nothing else in the pipeline looks for one. For each
    internal boundary this compares the median |difference| of the pixel pairs straddling it
    against the median |difference| of the pairs one pixel in on either side. A ratio near 1
    means the boundary is invisible in the data; a large ratio localizes the seam.

    Reads four pixel-wide strips per edge (~2 MB), not whole tiles. Also the evidence the
    deferred "weighted mean cascade in the pyramid, only if QC finds edge artifacts"
    decision currently lacks.
    """
    tile_dim = config.spatial_chunk_dim_zarr_output
    n_y, n_x = (int(v) for v in config.global_geobox.shape.yx)
    rows = []
    for row, col in tiles:
        region = tile_region_slices(config, row, col)
        for edge, boundary, limit in (("right", (col + 1) * tile_dim, n_x),
                                      ("bottom", (row + 1) * tile_dim, n_y)):
            if boundary + 2 > limit:
                continue  # grid edge, no neighbour
            if edge == "right":
                strip = ds_raw[var].isel(latitude=region["latitude"],
                                         longitude=slice(boundary - 2, boundary + 2)).values
                lines = [strip[:, i] for i in range(4)]
            else:
                strip = ds_raw[var].isel(latitude=slice(boundary - 2, boundary + 2),
                                         longitude=region["longitude"]).values
                lines = [strip[i, :] for i in range(4)]
            a, b, c, d = (decode(line, var) for line in lines)
            across = _paired_absdiff(b, c)
            inside = np.concatenate([_paired_absdiff(a, b), _paired_absdiff(c, d)])
            if across.size < min_pairs or inside.size < min_pairs:
                rows.append({"tile": f"({row},{col})", "edge": edge, "n_pairs": int(across.size),
                             "across_median": np.nan, "inside_median": np.nan, "ratio": np.nan,
                             "ok": True, "note": "too few overlapping pixels to judge"})
                continue
            # floor the denominator at half the quantization step so a perfectly smooth
            # interior (median difference 0) doesn't send the ratio to infinity
            across_med, inside_med = float(np.median(across)), float(np.median(inside))
            ratio = across_med / max(inside_med, 0.5 * _scale(var))
            rows.append({"tile": f"({row},{col})", "edge": edge, "n_pairs": int(across.size),
                         "across_median": across_med, "inside_median": inside_med,
                         "ratio": ratio, "ok": ratio <= max_ratio, "note": ""})
    return pd.DataFrame(rows)


def _paired_absdiff(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    both = np.isfinite(first) & np.isfinite(second)
    return np.abs(first[both] - second[both])


# ─── regression fingerprint ─────────────────────────────────────────────────

def fingerprint(tile: TileData) -> pd.DataFrame:
    """
    A compact, comparable summary of one tile: per water year plus composites.

    This is what replaces "diff against v9" as the regression mechanism. v9 is a fixed
    historical store that v10 is *expected* to differ from for documented reasons, so the
    diff has no failing value; a fingerprint of the current release does. `hash16` is over
    the raw stored bytes, so it changes if any pixel changes, while the summary statistics
    say *how* it changed.
    """
    rows = []
    for index, wy in enumerate(tile.water_years):
        onset = tile.dec("runoff_onset")[index]
        tr = tile.dec("temporal_resolution")[index]
        rows.append({
            "row": tile.row, "col": tile.col, "water_year": int(wy),
            "valid_px": tile.valid_px("runoff_onset", index),
            **_percentiles(onset), "median_tr_days": _nanmedian(tr),
            "hash16": _hash16(tile.raw["runoff_onset"][index]),
        })
    rows.append({
        "row": tile.row, "col": tile.col, "water_year": "composites",
        "valid_px": tile.valid_px("runoff_onset_median"),
        **_percentiles(tile.dec("runoff_onset_median")),
        "median_tr_days": _nanmedian(tile.dec("temporal_resolution_median")),
        "hash16": _hash16(np.concatenate([tile.raw[var].ravel() for var in COMPOSITE_VARIABLES])),
    })
    return pd.DataFrame(rows)


def _percentiles(arr: np.ndarray) -> dict:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"p10_dowy": np.nan, "p50_dowy": np.nan, "p90_dowy": np.nan}
    p10, p50, p90 = np.percentile(finite, [10, 50, 90])
    return {"p10_dowy": round(float(p10), 2), "p50_dowy": round(float(p50), 2),
            "p90_dowy": round(float(p90), 2)}


def _nanmedian(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    return round(float(np.median(finite)), 2) if finite.size else np.nan


def _hash16(arr: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(arr).tobytes(), digest_size=8).hexdigest()


def compare_fingerprints(new: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """
    Diff a fresh fingerprint against a stored baseline; `ok` means byte-identical.

    Rows present on only one side are reported too (a battery tile gaining or losing a water
    year is exactly the kind of change worth stopping on).
    """
    keys = ["row", "col", "water_year"]
    stats = ["valid_px", "p10_dowy", "p50_dowy", "p90_dowy", "median_tr_days"]
    new = new.copy()
    baseline = baseline.copy()
    for frame in (new, baseline):
        frame["water_year"] = frame["water_year"].astype(str)
    merged = new.merge(baseline[keys + stats + ["hash16"]], on=keys, how="outer",
                       suffixes=("", "_baseline"), indicator=True)
    merged["ok"] = (merged["_merge"] == "both") & (merged["hash16"] == merged["hash16_baseline"])
    for stat in stats:
        merged[f"d_{stat}"] = merged[stat] - merged[f"{stat}_baseline"]
    merged["note"] = np.where(merged["_merge"] == "left_only", "new (not in baseline)",
                     np.where(merged["_merge"] == "right_only", "missing (in baseline only)",
                     np.where(merged["ok"], "", "bytes changed")))
    return merged[keys + ["ok", "note", "valid_px", "d_valid_px", "d_p50_dowy",
                          "d_median_tr_days", "hash16", "hash16_baseline"]]


# ─── verdict ────────────────────────────────────────────────────────────────

def summarize(gates: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per gate: how many checks ran, how many failed, PASS/FAIL."""
    rows = []
    for name, frame in gates.items():
        failed = 0 if frame.empty else int((~frame["ok"].astype(bool)).sum())
        rows.append({"gate": name, "checks": len(frame), "failed": failed,
                     "verdict": "PASS" if failed == 0 else "FAIL"})
    return pd.DataFrame(rows)


# ─── visual inspection ──────────────────────────────────────────────────────

def view_tile(config, ds: xr.Dataset, row: int, col: int, stride: int = 4,
              basemap: bool = True):
    """
    Look at one tile the way a person judges it: composites over a basemap, then per-year
    onset, per-year anomaly, and per-year temporal resolution.

    The gates above prove the store is internally consistent; they cannot tell you the
    pattern is wrong. Geographic context is what makes that judgeable, and the anomaly
    facet (year minus composite median) is where a single bad year shows up -- it is
    invisible in the absolute field. Styling comes from `plot_utils.variable_kw` so a QC
    render is directly comparable to the manuscript figures.

    Args:
        ds: dataset opened DECODED (`mask_and_scale=True`), e.g. via `open_release(...,
            mask_and_scale=True)`.
        stride: display decimation (4 -> 512 px panels from a 2048 px tile).
        basemap: reproject to UTM and draw a contextily basemap under the composites.
    """
    import matplotlib.pyplot as plt
    import rioxarray  # noqa: F401 -- registers the .rio accessor

    from . import plot_utils

    region = tile_region_slices(config, row, col)
    tile_ds = ds.isel(region).isel(latitude=slice(None, None, stride),
                                   longitude=slice(None, None, stride)).compute()

    composites = tile_ds[list(COMPOSITE_VARIABLES)]
    if basemap:
        composites = composites.rio.reproject(composites.rio.estimate_utm_crs())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="compressed")
    for ax, var in zip(axes, COMPOSITE_VARIABLES):
        composites[var].plot(ax=ax, **plot_utils.variable_kw(var))
        if basemap:
            try:
                import contextily as ctx
                ctx.add_basemap(ax=ax, crs=composites.rio.crs.to_string())
            except Exception as error:  # offline, or tile server unhappy -- not fatal
                print(f"basemap skipped: {error}")
        ax.set_title(var, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(f"tile ({row},{col}) composites — {config.version}")

    onset = tile_ds["runoff_onset"]
    years = [int(wy) for wy in onset.water_year.values
             if bool(np.isfinite(onset.sel(water_year=wy)).any())]
    if not years:
        print(f"tile ({row},{col}): no annual data in any water year")
        return
    facets = {
        "runoff_onset": onset.sel(water_year=years),
        "runoff_onset_anomaly": (onset - tile_ds["runoff_onset_median"]).sel(water_year=years),
        "temporal_resolution": tile_ds["temporal_resolution"].sel(water_year=years),
    }
    for var, data in facets.items():
        facet = data.plot.imshow(col="water_year", col_wrap=5,
                                 **plot_utils.variable_kw(var))
        facet.fig.suptitle(f"tile ({row},{col}) — {var}", y=1.02)
