#!/usr/bin/env python3
"""
Generate tile work lists for batch processing from the icechunk commit history.

Replaces the tile_results_*.csv-driven matrix generation (configs <= v9):
remaining work is derived from the output repository's commit metadata via
global_snowmelt_runoff_onset.status, so failed tiles (which never commit) are
automatically re-dispatched and completed tile x water years are never redone.

Each work item is {"row": R, "col": C, "wys": "2015,2019"} where "wys" is the
comma-separated list of water years still missing for that tile ("all" when
none are committed yet, "none" when only the composites need refreshing).

Output modes (for GitHub Actions):
    --list-batches      {"batch_index": [0, 1, ...]} for the batch fan-out
    (default)           {"tile": [work items]} for one batch's job matrix
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

# Add the repo root to the Python path so the package imports without install
sys.path.append(str(Path(__file__).parent.parent.parent))

BATCH_SIZE = 256  # GitHub Actions matrix job limit


def get_work_items(config_file: str, which_tiles: str, include_empty_years: bool,
                   how_many: int, branch: str, as_of_snapshot: str | None) -> tuple:
    """
    Full ordered work list (snowiest tiles first), before batching.

    Returns (items, snapshot_id): status is derived as of 'as_of_snapshot' when
    given, otherwise as of the branch tip -- whose snapshot id is returned so a
    fleet run can pin every batch to the same consistent work list.
    """
    # Config loading and the ancestry walk print progress/config chatter;
    # keep stdout clean for the JSON output.
    with contextlib.redirect_stdout(io.StringIO()):
        from global_snowmelt_runoff_onset.config import Config
        from global_snowmelt_runoff_onset import status

        repo_root = Path(__file__).parent.parent.parent
        config_name = config_file if config_file.endswith(".txt") else f"global_config_{config_file}.txt"
        config = Config(str(repo_root / "config" / config_name))
        repo = config.open_output_repo()
        snapshot_id = as_of_snapshot or repo.lookup_branch(branch)
        work = status.get_remaining_work(
            config, repo=repo, which=which_tiles,
            include_empty_years=include_empty_years,
            branch=branch, as_of_snapshot=snapshot_id,
        )
        all_years = [int(wy) for wy in config.water_years]

    if how_many > 0:
        work = work[:how_many]

    items = []
    for entry in work:
        if entry["water_years"] == all_years:
            wys = "all"
        elif not entry["water_years"]:
            wys = "none"  # composites-only refresh
        else:
            wys = ",".join(str(wy) for wy in entry["water_years"])
        items.append({"row": entry["row"], "col": entry["col"], "wys": wys})
    return items, snapshot_id


def main():
    parser = argparse.ArgumentParser(description="Generate tile work lists from icechunk commit history")
    parser.add_argument("--config-file", type=str, default="global_config_v10.txt")
    parser.add_argument("--which-tiles", type=str, default="incomplete",
                        choices=["incomplete", "unprocessed", "all"],
                        help="incomplete: missing tile x water years + missing/stale composites "
                             "(includes failed -- they never commit); unprocessed: untouched tiles "
                             "only; all: full reprocess")
    parser.add_argument("--include-empty-years", action="store_true",
                        help="Also redo water years previously committed as verified-empty")
    parser.add_argument("--how-many", type=int, default=0,
                        help="Limit the number of tiles (0 = no limit)")
    parser.add_argument("--branch", type=str, default="main")
    parser.add_argument("--batch-index", type=int, default=0,
                        help=f"Which batch of {BATCH_SIZE}-tile batches to emit")
    parser.add_argument("--as-of-snapshot", type=str, default=None,
                        help="Derive status as of this snapshot id (pins a consistent "
                             "work list across the batches of one fleet run)")
    parser.add_argument("--list-batches", action="store_true",
                        help='Emit {"batch_index": [...], "snapshot_id": ...} instead of one batch\'s tiles')

    args = parser.parse_args()

    items, snapshot_id = get_work_items(args.config_file, args.which_tiles,
                                        args.include_empty_years, args.how_many,
                                        args.branch, args.as_of_snapshot)

    if args.list_batches:
        num_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE
        print(json.dumps({"batch_index": list(range(num_batches)),
                          "total_tiles": len(items),
                          "snapshot_id": snapshot_id},
                         separators=(",", ":")))
    else:
        start = args.batch_index * BATCH_SIZE
        print(json.dumps({"tile": items[start:start + BATCH_SIZE]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
