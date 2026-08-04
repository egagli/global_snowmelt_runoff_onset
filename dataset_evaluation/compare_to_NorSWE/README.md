# Comparison to NorSWE

Evaluates the global snowmelt runoff onset dataset against the
[Northern Hemisphere historical in-situ Snow Water Equivalent dataset
(NorSWE, 1979–2021)](https://zenodo.org/records/15263370), described in
[Pirazzini et al. (2025), ESSD](https://essd.copernicus.org/articles/17/3619/2025/).

NorSWE aggregates daily in-situ SWE and snow depth measurements from 10,153
stations across the Northern Hemisphere, spanning 1979–2021.

We decided not to go further down this route because outside of the SNOTEL network, there aren't many daily observations (see `1_create_NorSWE_comparison_dataset.ipynb`'s own conclusion: "doesn't look like enough daily data").

---

## Directory structure

```text
compare_to_NorSWE/
├── 0_download_and_preprocess_NorSWE.ipynb    # download, convert to Zarr, validate
├── 1_create_NorSWE_comparison_dataset.ipynb  # max-SWE timings + SAR chips; abandoned here
├── analysis.log                              # 667 MB, untracked/gitignored — safe to delete
├── figures/
│   └── NorSWE_station_counts_histogram.png
├── data/
│   ├── NorSWE/
│   │   └── NorSWE.zarr                       # converted dataset (see below)
│   └── comparison_datasets/
│       ├── max_NorSWE_swe_timing.zarr
│       └── runoff_onset_NorSWE_station_chips.zarr
└── README.md
```

## Notebooks

### `0_download_and_preprocess_NorSWE.ipynb`

End-to-end pipeline that:

1. **Downloads** the ZIP archive (~2.4 GB) from Zenodo record 15263370.
2. **Extracts** the NetCDF file (~21.65 GB on disk).
3. **Converts** it to a compact Zarr store (`NorSWE.zarr`, 0.1 GB), dropping
   the original NetCDF and ZIP afterwards to recover disk space.
4. **Cross-validates** against a co-located SNOTEL station (Paradise, WA —
   SNOTEL 679) to sanity-check time-series alignment.

### `1_create_NorSWE_comparison_dataset.ipynb`

Builds the comparison inputs: max-SWE (95%-of-max) timing per station/water year
(`data/comparison_datasets/max_NorSWE_swe_timing.zarr`), SAR runoff onset chips around each
station (`data/comparison_datasets/runoff_onset_NorSWE_station_chips.zarr`), and
`figures/NorSWE_station_counts_histogram.png`. Work stopped here — its final cell records the
abandonment reason ("doesn't look like enough daily data"), which is what the note at the top
of this README summarizes.

---

## Storage format note

NorSWE is distributed as a 21.65 GB NetCDF file. Four variables
(`qc_flag_snw`, `qc_flag_snd`, `data_flag_snw`, `data_flag_snd`) are stored
as HDF5 variable-length strings. xarray cannot back these lazily with dask —
opening them loads all ~637 million Python string objects into RAM (~127 GB),
crashing the kernel.

The Zarr copy encodes those four variables as `int8` using a small
per-variable vocabulary (e.g. `0=''`, `1='D'`, `2='M'`, `3='W'`). The
vocabulary and flag definitions are stored in each variable's `.attrs`.
Writing is done in chunks of 1000 time steps via `netCDF4` directly, so peak
memory during conversion stays well under 1 GB.

To decode a flag value:

```python
ds = xr.open_zarr("data/NorSWE/NorSWE.zarr")
vocab = ds["qc_flag_snw"].attrs["vocab"]   # list of strings
flag_str = vocab[int(ds["qc_flag_snw"][t, s])]
```
