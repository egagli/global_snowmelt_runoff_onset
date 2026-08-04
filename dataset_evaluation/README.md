# dataset_evaluation

Evaluates the global snowmelt runoff onset dataset against independent reference measurements. This is the source of manuscript Sect. 2.3 (methodology), Sect. 4 (dataset evaluation, Fig. 4, Fig. 5), Fig. 6, Fig. A1, Fig. A5, and Table 1.

## Subfolders

| Folder | Manuscript tie-in | Status |
| --- | --- | --- |
| [`compare_to_all_public_snow_pillows/`](compare_to_all_public_snow_pillows/README.md) | Fig. 4, Fig. 5, Fig. A5 — the primary in-situ evaluation network (~1,500 stations: SNOTEL, CCSS, BC Snow Survey, NVE, SCAN, COOP, Yukon) | **Primary / active** |
| [`compare_to_snotel/`](compare_to_snotel/README.md) | Fig. A1 (single-station case study) and earlier SNOTEL/CCSS-only evaluation work that `compare_to_all_public_snow_pillows/` has since superseded for the full network comparison | Active (case-study figure); superseded for network-wide stats. **Directory is still untracked in git** |
| [`compare_to_passive/`](compare_to_passive/README.md) | Fig. 6 (Pan et al. 2021 passive microwave comparison, Alaska Range WY2020) | Active (still v9-pinned) |
| [`calculate_spatial_coverage_and_temporal_resolution/`](calculate_spatial_coverage_and_temporal_resolution/README.md) | Table 1, and the "how much seasonal snow do we miss" text stat (Sect. 3.3) | Active (still v9-pinned) |
| [`compare_to_NorSWE/`](compare_to_NorSWE/README.md) | Not used in the manuscript | Superseded by `compare_to_all_public_snow_pillows/` (NorSWE ends in 2021 and many sites aren't daily) |
| [`compare_to_NH-SWE/`](compare_to_NH-SWE/README.md) | Not used in the manuscript | Superseded/abandoned — a download + Paradise-WA cross-check notebook exists and ran (3 figures in `figures/`); dropped because NH-SWE is ΔSNOW-**modeled** SWE, not observations |
| [`compare_to_ucla_reanalysis/`](compare_to_ucla_reanalysis/README.md) | Not used in the manuscript | Parked/unfinished — exploratory data search only |

## Output versioning

Every output in the four active subfolders is scoped by the dataset version it was
derived from, so evaluating a new version never overwrites the previous one:

| Output kind | Path |
| --- | --- |
| Figures | `figures/<version>/…` (e.g. `figures/v9/pixelwise_performance_analysis.png`) |
| Comparison datasets (Zarr) | `data/comparison_datasets/<version>/…` |
| Result tables | `results/<version>/…` in `calculate_spatial_coverage_and_temporal_resolution/`, `compare_to_all_public_snow_pillows/` (QC counts, binned stats, evaluation summary, station density), and `compare_to_passive/` (Fig. 6 stats) — written via `global_snowmelt_runoff_onset.results.save_result_table` |
| `compare_to_snotel/` NetCDFs | `comparison_datasets/…_<version>.nc` (kept flat — the filename already carries the version) |

Two known exceptions: `compare_to_snotel/figures/` still holds six unscoped
pre-versioning PNGs at its top level (documented in that README), and
`compare_to_passive/figures/v9/passive_comparison/` is empty pending a rerun of the
Kennicott/Denali notebook.

The version comes from `config.version`, so **switching versions means editing the one
`Config('config/global_config_vN.txt')` path** in a notebook that loads a config. The
downstream snow-pillow analysis notebooks (`2_`, `3_`, `4_`) don't need a config at all,
so they carry a single `VERSION = 'vN'` constant at the top instead — set it to match
the config used in `1_create_snow_pillow_comparison_dataset.ipynb`. (This drifted once
already: `4_snow_pillow_representativeness.ipynb` sat at `v9` while `1_`–`3_` moved to
v10, so Fig. A5 existed only under `figures/v9/`; fixed 2026-08-04, v10 rerun still
needed.)

Notebooks read the runoff onset dataset through `config.open_runoff_onset_dataset()`,
which hides the store difference between generations (icechunk for ≥ v10, legacy
consolidated Zarr v2 for ≤ v9) so a version bump doesn't need any other code change.

`compare_to_NorSWE/`, `compare_to_NH-SWE/` and `compare_to_ucla_reanalysis/` are not
version-scoped — they're parked/superseded (see below).

## Notes

- `compare_to_NorSWE/` and `compare_to_NH-SWE/` are candidates for archiving/deletion if they won't be picked back up; no decision recorded yet.
- `compare_to_ucla_reanalysis/` is similarly parked; see its own README for what would be needed to pick it up.
- Untracked clutter in this tree (all gitignored or ignorable, ~770 MB total): `analysis.log` files at the top level (7 MB), in `calculate_spatial_coverage_and_temporal_resolution/` (93 MB), `compare_to_NorSWE/` (667 MB), and `compare_to_snotel/` (0.5 MB), plus a `__pycache__/`. Safe to delete.
