#!/usr/bin/env python3
"""
Run remaining tile work on this machine (laptop, CryoCloud, ...).

Platform-agnostic batch driver around process_single_tile.py: derives the
remaining work from the icechunk commit history (exactly like the GitHub
Actions matrix generation), then runs each tile as a subprocess -- so a run
started on GitHub Actions can be finished here, or vice versa, with no
coordination beyond the icechunk repository itself.

Each tile runs in its own subprocess for memory isolation; per-tile logs go to
logs/tile_<row>_<col>.log as usual.

Examples:
    # see what's left
    python processing/scripts/run_tiles.py --dry-run

    # process the next 10 incomplete tiles, two at a time
    python processing/scripts/run_tiles.py --how-many 10 --max-workers 2
"""

import argparse
import contextlib
import io
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add the repo root to the Python path so the package imports without install
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

PROCESS_SCRIPT = Path(__file__).parent / "process_single_tile.py"


def run_one(item, config_name, branch, local_store, dask_workers):
    wys = item["water_years"]
    wys_arg = "none" if not wys else ",".join(str(wy) for wy in wys)
    cmd = [
        sys.executable, str(PROCESS_SCRIPT),
        "--tile-row", str(item["row"]),
        "--tile-col", str(item["col"]),
        "--water-years", wys_arg,
        "--config-file", config_name,
        "--branch", branch,
    ]
    if local_store:
        cmd += ["--local-store", local_store]
    if dask_workers:
        cmd += ["--dask-workers", str(dask_workers)]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return item, result.returncode, time.time() - t0, result.stdout[-2000:] + result.stderr[-2000:]


def main():
    parser = argparse.ArgumentParser(description="Run remaining tile work locally")
    parser.add_argument("--config-file", type=str, default="global_config_v10.txt")
    parser.add_argument("--which-tiles", type=str, default="incomplete",
                        choices=["incomplete", "unprocessed", "all"])
    parser.add_argument("--include-empty-years", action="store_true")
    parser.add_argument("--how-many", type=int, default=0, help="0 = no limit")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Tiles processed concurrently (each is a subprocess)")
    parser.add_argument("--dask-workers", type=int, default=None,
                        help="Per-tile dask worker count (passed through to "
                             "process_single_tile.py; higher = faster S1 downloads, "
                             "more memory)")
    parser.add_argument("--branch", type=str, default="main")
    parser.add_argument("--local-store", type=str, default=None,
                        help="Path to a local icechunk repo (testing; overrides Azure)")
    parser.add_argument("--dry-run", action="store_true", help="List the work and exit")
    args = parser.parse_args()

    config_name = (args.config_file if args.config_file.endswith(".txt")
                   else f"global_config_{args.config_file}.txt")

    with contextlib.redirect_stdout(io.StringIO()):
        from global_snowmelt_runoff_onset.config import Config
        from global_snowmelt_runoff_onset import status
        import icechunk

        config = Config(str(REPO_ROOT / "config" / config_name))
        if args.local_store:
            repo = icechunk.Repository.open(
                icechunk.local_filesystem_storage(args.local_store),
                config=config.output_repo_config())
        else:
            repo = config.open_output_repo()
        work = status.get_remaining_work(
            config, repo=repo, which=args.which_tiles,
            include_empty_years=args.include_empty_years, branch=args.branch)

    if args.how_many > 0:
        work = work[:args.how_many]

    print(f"{len(work)} tile(s) to process ({args.which_tiles})")
    for item in work[:20]:
        wys = item["water_years"]
        print(f"  tile({item['row']},{item['col']}): "
              f"{'composites only' if not wys else wys}")
    if len(work) > 20:
        print(f"  ... and {len(work) - 20} more")
    if args.dry_run or not work:
        return

    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, item, config_name, args.branch,
                               args.local_store, args.dask_workers)
                   for item in work]
        for future in as_completed(futures):
            item, returncode, duration, tail = future.result()
            done += 1
            label = f"tile({item['row']},{item['col']})"
            if returncode == 0:
                print(f"[{done}/{len(work)}] {label} OK in {duration:.0f}s")
            else:
                failed.append(item)
                print(f"[{done}/{len(work)}] {label} FAILED (exit {returncode}) in {duration:.0f}s")
                print(f"    log: logs/tile_{item['row']}_{item['col']}.log; output tail:")
                for line in tail.strip().splitlines()[-8:]:
                    print(f"    {line}")

    print(f"\ndone: {len(work) - len(failed)} succeeded, {len(failed)} failed")
    if failed:
        print("failed tiles:", ", ".join(f"({i['row']},{i['col']})" for i in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
