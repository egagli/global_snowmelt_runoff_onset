"""List Sentinel-1 IW+VV GRD scenes that have no RTC item on Planetary Computer.

Sweeps the full MPC STAC GeoParquet indexes (sentinel-1-grd vs sentinel-1-rtc,
~154 monthly partitions each) and diffs granule ids: an RTC item id is its GRD
id + '_rtc', so `GRD(IW, VV-containing) - RTC` is exactly the set of scenes the
vv pipeline can never see even though the acquisition exists. This separates
"MPC never RTC-processed it" from true acquisition gaps (which show up in
NEITHER catalog -- spot-verified against ASF for (25,39) WY2016: the 48/72-day
holes there are true gaps, and the RTC deficit is ~0.1%).

Output: a CSV with one row per missing scene (id, datetime, bbox center) plus a
per-calendar-year summary on stdout. ~30-45 min for the full sweep (download
dominated). Requires only a Planetary Computer token (fetched automatically).

Usage:
    pixi run python processing/scripts/find_missing_rtc_scenes.py \
        [--out missing_rtc_scenes.csv] [--start 2014-10] [--end 2026-12]
"""
import argparse
import collections
import csv
import re

import adlfs
import planetary_computer
import pyarrow.parquet as pq

MONTH_RE = re.compile(r"part-\d+_(\d{4}-\d{2})")


def month_of(path: str) -> str:
    m = MONTH_RE.search(path.rsplit("/", 1)[-1])
    return m.group(1) if m else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="missing_rtc_scenes.csv")
    parser.add_argument("--start", default="2014-10", help="First month (YYYY-MM)")
    parser.add_argument("--end", default="2099-12", help="Last month (YYYY-MM)")
    args = parser.parse_args()

    token = planetary_computer.sas.get_token("pcstacitems", "items").token
    fs = adlfs.AzureBlobFileSystem("pcstacitems", sas_token=token)
    grd_parts = {month_of(f): f for f in fs.ls("items/sentinel-1-grd.parquet")
                 if f.endswith(".parquet")}
    rtc_parts = {month_of(f): f for f in fs.ls("items/sentinel-1-rtc.parquet")
                 if f.endswith(".parquet")}
    months = sorted(m for m in grd_parts if args.start <= m <= args.end)

    per_year = collections.Counter()
    per_year_total = collections.Counter()
    n_missing = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "datetime", "lon_center", "lat_center", "month"])
        for month in months:
            g = pq.ParquetFile(fs.open(grd_parts[month])).read(
                columns=["id", "sar:instrument_mode", "sar:polarizations",
                         "datetime", "bbox"])
            rows = {}
            for i, mode, pols, dt, bb in zip(
                    g["id"].to_pylist(), g["sar:instrument_mode"].to_pylist(),
                    g["sar:polarizations"].to_pylist(), g["datetime"].to_pylist(),
                    g["bbox"].to_pylist()):
                mode = mode[0] if isinstance(mode, list) and mode else mode
                if mode == "IW" and pols and "VV" in pols:
                    rows[i] = (dt, bb)
            year = month[:4]
            per_year_total[year] += len(rows)

            rtc_stems = set()
            if month in rtc_parts:
                r = pq.ParquetFile(fs.open(rtc_parts[month])).read(columns=["id"])
                rtc_stems = {i[:-4] if i.endswith("_rtc") else i
                             for i in r["id"].to_pylist()}

            missing = sorted(set(rows) - rtc_stems)
            for mid in missing:
                dt, bb = rows[mid]
                writer.writerow([mid, dt.isoformat(),
                                 round((bb["xmin"] + bb["xmax"]) / 2, 3),
                                 round((bb["ymin"] + bb["ymax"]) / 2, 3), month])
            per_year[year] += len(missing)
            n_missing += len(missing)
            print(f"{month}: {len(rows):>6} IW+VV GRD, {len(missing):>4} missing RTC"
                  + ("  (NO RTC PARTITION)" if month not in rtc_parts else ""),
                  flush=True)

    print(f"\nTOTAL missing: {n_missing}  ->  {args.out}")
    print(f"{'year':>6} {'IW+VV GRD':>10} {'missing':>8} {'pct':>6}")
    for year in sorted(per_year_total):
        t, m = per_year_total[year], per_year[year]
        print(f"{year:>6} {t:>10,} {m:>8,} {100*m/max(t,1):>5.2f}%")


if __name__ == "__main__":
    main()
