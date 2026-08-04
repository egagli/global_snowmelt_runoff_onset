# Comparison to NH-SWE

Evaluates the global snowmelt runoff onset dataset against
[NH-SWE: Northern Hemisphere Snow Water Equivalent dataset based on in situ snow depth time
series](https://essd.copernicus.org/articles/15/2577/2023/essd-15-2577-2023.html)
([dataset on Zenodo, record 7565252](https://zenodo.org/records/7565252)).

## Why we stopped

**Superseded/abandoned:** NH-SWE is SWE *modeled* from snow depth via the regionalised ΔSNOW
model, not true SWE observations, so it can't serve as an independent in-situ reference for
runoff onset timing. The in-situ evaluation lives in
[`../compare_to_all_public_snow_pillows/`](../compare_to_all_public_snow_pillows/README.md).

## Notebooks

| Notebook | Description |
| --- | --- |
| [`compare_to_NH-SWE.ipynb`](compare_to_NH-SWE.ipynb) | Downloads Zenodo record 7565252, builds a working dataset from the matrix CSVs, and cross-checks NH-SWE against the co-located Paradise, WA SNOTEL SWE record. Ran to completion before the effort was dropped. |

## Outputs

`figures/` — three cross-check plots: `NH_SWE_vs_Paradise_SNOTEL_SWE.png`,
`NH_SWE_vs_Paradise_SNOTEL_SWE_2010_2024.png`, `NH_SWE_vs_Paradise_SNOTEL_SWE_2020_2022.png`.

## Data

`data/NH_SWE/` — `NH_SWE_METADATA.csv`, the Zenodo ZIP, and **~1.8 GB of extracted
`matrix_files/` CSVs** (SWE, QC flags, gap-fill flags, snow density, date/ID vectors).
Untracked and not gitignored; safe to delete — everything can be re-downloaded from Zenodo.
