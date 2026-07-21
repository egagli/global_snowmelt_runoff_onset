# dataset_utils

Utilities for packaging, distributing, and accessing the published global snowmelt runoff onset Zarr store. These notebooks operate on the finished dataset (after `processing/` has produced it) — they don't compute runoff onset themselves.

## Notebooks

| Notebook | Description |
| --- | --- |
| [`compress_and_download_zarr.ipynb`](compress_and_download_zarr.ipynb) | Reports total CPU-core-hours from `tile_results_*.csv`, then uses a Coiled worker to LZMA-tar-compress the published Zarr store for Zenodo upload, and benchmarks Zarr compression codecs/levels (zstd vs zlib, shuffle vs bitshuffle) to pick the final on-disk encoding. The first ~9 cells (Copernicus DEM sampling / vertical datum reprojection) are leftover scratch unrelated to this notebook's purpose and should be removed. |
| [`global_zarr_to_COG.ipynb`](global_zarr_to_COG.ipynb) | Converts the global Zarr dataset (or a variable/water-year slice of it) to Cloud-Optimized GeoTIFF for use in GIS software that doesn't read Zarr. |
| [`split_dataset.ipynb`](split_dataset.ipynb) | Splits the combined Zarr store into per-water-year and per-composite-product files, and demonstrates opening the dataset directly from cloud storage. |
| [`subset_global_dataset.ipynb`](subset_global_dataset.ipynb) | Example of clipping a small regional subset (Dischma basin, Swiss Alps) from the global store and exporting it to NetCDF for a collaborator comparison. |
| [`test_open_zarr_lazy.ipynb`](test_open_zarr_lazy.ipynb) | Benchmarks lazy-access download speed/bytes-transferred for two distribution mechanisms — Kerchunk references against the Zenodo-hosted tarball vs. direct Azure Zarr access — across four spatial extents and four mountain-range locations, to inform which access pattern to recommend to dataset users. |

## Related

See [`dataset/`](../dataset/README.md) for the Kerchunk reference file pointing at the published Zenodo archive, and the root [README](../README.md#data-access) for the published dataset DOI and quick-start access snippet.
