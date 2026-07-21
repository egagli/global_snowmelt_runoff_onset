# dataset

Pointers to the published global snowmelt runoff onset dataset. The dataset itself is not stored in this repository — it's hosted on Zenodo (see the root [README](../README.md#data-access) for the DOI and a quick-start access snippet).

## Contents

- **`global_snowmelt_runoff_onset.zarr.tar.refs.json`** — a [Kerchunk](https://fsspec.github.io/kerchunk/) reference set that lets `xarray`/`fsspec` lazily read the individual chunks inside the Zenodo-hosted `.zarr.tar` archive without downloading the whole file. See [`dataset_utils/test_open_zarr_lazy.ipynb`](../dataset_utils/README.md) for a benchmark of this access pattern against direct Azure Zarr access.
- **`redistribution/`** — currently empty; reserved for redistributed/derived copies of the dataset (e.g. COGs from `dataset_utils/global_zarr_to_COG.ipynb`) that don't belong on Zenodo directly.
