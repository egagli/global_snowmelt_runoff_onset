# dataset_evaluation

Evaluates the global snowmelt runoff onset dataset against independent reference measurements. This is the source of manuscript Sect. 2.3 (methodology), Sect. 4 (dataset evaluation, Fig. 4, Fig. 5), Fig. 6, Fig. A1, Fig. A5, and Table 1.

## Subfolders

| Folder | Manuscript tie-in | Status |
| --- | --- | --- |
| [`compare_to_all_public_snow_pillows/`](compare_to_all_public_snow_pillows/README.md) | Fig. 4, Fig. 5, Fig. A5 — the primary in-situ evaluation network (~1,500 stations: SNOTEL, CCSS, BC Snow Survey, NVE, SCAN, COOP) | **Primary / active** |
| [`compare_to_snotel/`](compare_to_snotel/README.md) | Fig. A1 (single-station case study) and earlier SNOTEL/CCSS-only evaluation work that `compare_to_all_public_snow_pillows/` has since superseded for the full network comparison | Active (case-study figure); superseded for network-wide stats |
| [`compare_to_passive/`](compare_to_passive/README.md) | Fig. 6 (Pan et al. 2021 passive microwave comparison, Alaska Range WY2020) | Active |
| [`calculate_spatial_coverage_and_temporal_resolution/`](calculate_spatial_coverage_and_temporal_resolution/README.md) | Table 1, and the "how much seasonal snow do we miss" text stat (Sect. 3.3) | Active |
| [`compare_to_NorSWE/`](compare_to_NorSWE/README.md) | Not used in the manuscript | Superseded by `compare_to_all_public_snow_pillows/` (per its own README — NorSWE ends in 2021 and many sites aren't daily) |
| [`compare_to_NH-SWE/`](compare_to_NH-SWE/README.md) | Not used in the manuscript | Parked/unfinished — stub README, no processing notebook yet |
| [`compare_to_ucla_reanalysis/`](compare_to_ucla_reanalysis/README.md) | Not used in the manuscript | Parked/unfinished — exploratory data search only |

## Notes

- `compare_to_NorSWE/` and `compare_to_NH-SWE/` are candidates for archiving/deletion if they won't be picked back up — see the root README's cleanup notes.
- `compare_to_ucla_reanalysis/` is similarly parked; see its own README for what would be needed to pick it up.
