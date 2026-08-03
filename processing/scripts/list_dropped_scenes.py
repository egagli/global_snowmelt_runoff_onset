"""Aggregate every scene the pipeline dropped as missing-from-storage.

The processor records confirmed-dead scenes (STAC-catalogued but blob 404) in
each tile-year commit's metadata under 'missing_assets' (both data commits and
the empty markers a fully-thinned year falls back to). This walks the commit
history -- the single durable record -- newest-wins per (tile, water_year), and
writes one row per (tile, water_year, scene id) plus a summary.

The same physical scene can appear under several tiles (footprints span tile
boundaries), so the summary also reports unique scene ids. Cross-reference with
find_missing_rtc_scenes.py output if you want to know whether a dropped scene
is also absent from the RTC catalog entirely (different failure: that script
finds scenes never RTC-processed; this one finds RTC items whose blobs died).

Usage:
    pixi run python processing/scripts/list_dropped_scenes.py \
        [--config-file global_config_v10.txt] [--out dropped_scenes.csv]
"""
import argparse
import contextlib
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="global_config_v10.txt")
    parser.add_argument("--out", type=str, default="dropped_scenes.csv")
    parser.add_argument("--branch", type=str, default="main")
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        from global_snowmelt_runoff_onset.config import Config
        from global_snowmelt_runoff_onset import status

        repo_root = Path(__file__).parent.parent.parent
        config_name = (args.config_file if args.config_file.endswith(".txt")
                       else f"global_config_{args.config_file}.txt")
        config = Config(str(repo_root / "config" / config_name))
        repo = config.open_output_repo()
        records = status.get_commit_records(repo, branch=args.branch)

    year_records = records[records.kind == status.KIND_TILE_YEAR]
    # Newest commit wins per (tile, water_year) -- a reprocessed year's drop
    # list supersedes older attempts (ancestry_index 0 = newest, and
    # get_commit_records preserves that order).
    newest = year_records.drop_duplicates(subset=["row", "col", "water_year"],
                                          keep="first")
    dropped = newest[newest.missing_assets.notna()]

    rows = []
    for record in dropped.itertuples():
        for scene_id in record.missing_assets:
            rows.append([int(record.row), int(record.col), int(record.water_year),
                         scene_id, record.status, record.empty_reason or "",
                         record.written_at, record.snapshot_id])
    rows.sort()
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "water_year", "scene_id", "status",
                         "empty_reason", "written_at", "snapshot_id"])
        writer.writerows(rows)

    unique_scenes = sorted({r[3] for r in rows})
    print(f"{len(rows)} dropped-scene records across {len(dropped)} tile-years "
          f"({len(unique_scenes)} unique scenes) -> {args.out}")
    if unique_scenes:
        per_wy = dropped.groupby("water_year").missing_assets.apply(
            lambda s: sum(len(x) for x in s))
        print("per water year:")
        print(per_wy.to_string())
        print("unique scenes:")
        for scene_id in unique_scenes:
            print(f"  {scene_id}")


if __name__ == "__main__":
    main()
