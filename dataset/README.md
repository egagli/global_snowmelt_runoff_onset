# dataset

Pointers to the global snowmelt runoff onset dataset. The dataset itself is not stored in this repository.

## Dataset versions

- **v9 (published, frozen — the manuscript dataset).** Zarr v2, 81.10°N–60°S grid (195,970 × 499,998 px), WY2015–2024. Distributed as `.zarr.tar` archives on Zenodo — record [`19618062`](https://zenodo.org/records/19618062) holds the full dataset (`global_snowmelt_runoff_onset.zarr.tar`), the per-water-year splits (e.g. `global_snowmelt_runoff_onset_WY2015.zarr.tar`), the composites archive, and the kerchunk reference file; the manuscript cites the concept DOI [10.5281/zenodo.16953614](https://doi.org/10.5281/zenodo.16953614) (v1.1.0). (Record `19699063` is a leftover access-test record — "please ignore" — not part of the dataset.) Azure source of truth: `snowmelt/snowmelt_runoff_onset/global_v9.zarr`. This store stays frozen (it has a DOI).
- **v10 (July 2026 rebuild, in production).** [Icechunk](https://icechunk.io) repository (Zarr v3) on Azure — account `uwcryo`, container `snowmelt`, prefix `snowmelt_runoff_onset/global_runoff_onset_v10` ([`config/global_config_v10.txt`](../config/global_config_v10.txt)). Grid extended to 84.048°N / −63.4074°S (204,800 × 499,998 px, 100 × 245 tiles; v9 tile (r,c) == v10 tile (r+2,c)); water years **2015–2025**; five int16 variables with `_FillValue = -9999`, shards (1, 2048, 2048) with (1, 256, 256) inner chunks ([`global_snowmelt_runoff_onset/store.py`](../global_snowmelt_runoff_onset/store.py)). Open with `zarr_format=3, consolidated=False` via `Config.open_runoff_onset_dataset()`. **Not yet on Zenodo** — store initialized 2026-08-04, fleet runs in progress (see [`processing/README.md`](../processing/README.md)).

## Contents

- **`global_snowmelt_runoff_onset.zarr.tar.refs.json`** (git-tracked, ~13 MB) — a [Kerchunk](https://fsspec.github.io/kerchunk/) reference set (~102k refs) that lets `xarray`/`fsspec` lazily read the individual chunks inside the Zenodo-hosted **v9** `.zarr.tar` archive (record `19618062`) without downloading the whole file. See [`dataset_utils/test_open_zarr_lazy.ipynb`](../dataset_utils/test_open_zarr_lazy.ipynb) for a benchmark of this access pattern against direct Azure Zarr access, and [`dataset_utils/split_dataset.ipynb`](../dataset_utils/split_dataset.ipynb) for reference file creation.
- **`redistribution/`** — currently empty; reserved for redistributed/derived copies of the dataset (e.g. COGs from `dataset_utils/global_zarr_to_COG.ipynb`) that don't belong on Zenodo directly.
