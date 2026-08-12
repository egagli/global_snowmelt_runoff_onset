"""
Open a GitHub issue when a new (water year, hemisphere) becomes eligible.

Run monthly by .github/workflows/water_year_watch.yml. Eligibility is purely
date-based: a hemisphere's water year is ready once its season has fully
elapsed plus a buffer. This repo uses a 120-day buffer — the upstream
MODIS_snow_phenology repo reminds at 90 days (its bidirectional cloud
filling needs post-season MOD10A2 context), and this dataset cannot extend
until that phenology fleet has run, so the extra month is grace for it.

Issues are deduplicated by exact title across open AND closed states, so
each (water year, hemisphere) reminds exactly once, ever — including the
southern-hemisphere season that closes ~6 months after the northern one of
the same water year (northern WY N ends Sep 30 N; southern ends Mar 31 N+1).

Stdlib only; needs gh CLI + GH_TOKEN (provided by the workflow).
"""

import datetime
import json
import subprocess
import sys

# Days past a hemisphere's season end before reminding. 90 (phenology
# trailing buffer) + ~30 grace for the phenology fleet to run.
BUFFER_DAYS = 120

# Years before this are handled by the initial v10 processing era; the
# watcher only reminds about seasons that close after it was put in place.
FIRST_WATCHED_WY = 2026

ISSUE_TITLE = "Ready to process: WY{wy} ({hemi} hemisphere)"
ISSUE_BODY = """\
Water year {wy} for the **{hemi} hemisphere** (ended {end}) is ready to add
to the snowmelt runoff onset dataset. S1 RTC coverage is complete within
days of acquisition; the gating input is the MODIS snow phenology dataset.

Prerequisite:
- [ ] MODIS_snow_phenology has committed WY{wy} for the {hemi} hemisphere
      (that repo opens its own "Ready to process" issue ~1 month before this
      one — check it's closed, or spot-check the phenology store).

Checklist:
- [ ] Bump `WY_end` in `config/global_config_v10.txt` (end_date derives
      automatically, covering the southern S1 window).
- [ ] Run `processing/5_add_water_year.ipynb` — checks eligibility and the
      phenology store (hemisphere trap), extends the store's `water_year`
      dimension through {wy}, and verifies the new work items appear.
- [ ] Dispatch the fleet — `get_remaining_work` emits (tile, {wy}) only for
      {hemi}-hemisphere tiles; the other hemisphere stays untouched until
      its own season closes. Composites refresh via the staleness rule.
- [ ] After the fleet: re-run `processing/4_finalize_icechunk_store.ipynb`
      (GC + backup sweep) with a bumped tag.

*Opened automatically by water_year_watch.yml.*
"""


def season_end(wy: int, hemisphere: str) -> datetime.date:
    return (datetime.date(wy, 9, 30) if hemisphere == "northern"
            else datetime.date(wy + 1, 3, 31))


def existing_issue_titles() -> set:
    out = subprocess.check_output(
        ["gh", "issue", "list", "--state", "all", "--limit", "500",
         "--json", "title"],
        text=True,
    )
    return {item["title"] for item in json.loads(out)}


def main():
    today = datetime.date.today()
    titles = existing_issue_titles()
    created = 0

    for hemi in ("northern", "southern"):
        for wy in range(FIRST_WATCHED_WY, today.year + 2):
            end = season_end(wy, hemi)
            if today < end + datetime.timedelta(days=BUFFER_DAYS):
                continue
            title = ISSUE_TITLE.format(wy=wy, hemi=hemi)
            if title in titles:
                continue
            body = ISSUE_BODY.format(wy=wy, hemi=hemi, end=end)
            subprocess.run(
                ["gh", "issue", "create", "--title", title, "--body", body],
                check=True,
            )
            print(f"opened: {title}")
            created += 1

    print(f"done: {created} issue(s) opened")
    return 0


if __name__ == "__main__":
    sys.exit(main())
