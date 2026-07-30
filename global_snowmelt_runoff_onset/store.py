"""
Output store schema and initialization for the icechunk (v10+) pipeline.

The global runoff onset output store is an Icechunk repository holding a
Zarr v3 dataset with the same five variables as the published v9 store, but
using sharding: each shard spans exactly one tile x one water year, so the
per-tile-per-water-year processing jobs each write whole shards and
concurrent writers never touch the same object. Inner chunks keep point and
small-window reads (e.g. weather-station evaluation) cheap.

Store initialization is metadata-only: the template dataset is lazy (dask)
and written with compute=False / write_empty_chunks=False, so only Zarr
metadata and coordinate arrays are stored until tile jobs fill in regions.
"""

import numpy as np
import xarray as xr
import zarr
import dask.array
import odc.geo.xr

NODATA_INT16 = np.int16(-9999)

# Variables scaled by 0.1 on disk (0.1-day precision in int16).
SCALED_VARIABLES = ("runoff_onset_mad", "temporal_resolution", "temporal_resolution_median")

VARIABLE_DESCRIPTIONS = {
    "runoff_onset": "Estimated day of water year of snowmelt runoff onset for a given water year [unit=DOWY].",
    "runoff_onset_median": "Median estimated day of water year of snowmelt runoff onset for all water years [unit=DOWY].",
    "runoff_onset_mad": "Median absolute deviation of snowmelt runoff onset for all water years [unit=days].",
    "temporal_resolution": "Temporal resolution of runoff onset for a given water year [unit=days].",
    "temporal_resolution_median": "Median temporal resolution for all water years [unit=days].",
}

WATER_YEAR_DESCRIPTION = (
    "Water year. In northern hemisphere, water year starts on October 1st "
    "and ends on September 30th. For the southern hemisphere, water year "
    "starts on April 1st and ends on March 31st. e.g. in NH WY 2015 is "
    "[2014-10-01,2015-09-30] and in SH WY 2015 is [2015-04-01,2016-03-31]."
)


def build_template(config):
    """
    Build the lazy global template dataset and its Zarr v3 encoding.

    Args:
        config: Config object for a >= v10 (icechunk) configuration; supplies
            the global geobox, water years, tile size (shard spatial dims),
            and inner chunk size.

    Returns:
        Tuple of (template xr.Dataset backed by lazy dask arrays, encoding dict
        for xr.Dataset.to_zarr).
    """
    geobox = config.global_geobox
    water_years = config.water_years
    tile_dim = config.spatial_chunk_dim_zarr_output
    inner_dim = config.inner_chunk_dim
    shard_2d = (tile_dim, tile_dim)
    chunk_2d = (inner_dim, inner_dim)

    def empty_2d(name):
        return odc.geo.xr.wrap_xr(
            dask.array.full(geobox.shape.yx, NODATA_INT16, dtype=np.int16, chunks=shard_2d),
            geobox,
        ).rename(name)

    def empty_3d(name):
        return empty_2d(name).expand_dims({"water_year": water_years})

    template_ds = xr.combine_by_coords([
        empty_3d("runoff_onset"),
        empty_2d("runoff_onset_median"),
        empty_2d("runoff_onset_mad"),
        empty_3d("temporal_resolution"),
        empty_2d("temporal_resolution_median"),
    ])
    # odc names dims latitude/longitude for geographic geoboxes; rename defensively
    # in case of projected input.
    if "y" in template_ds.dims:
        template_ds = template_ds.rename({"y": "latitude", "x": "longitude"})

    template_ds.water_year.attrs["description"] = WATER_YEAR_DESCRIPTION
    for var, description in VARIABLE_DESCRIPTIONS.items():
        template_ds[var].attrs["description"] = description
        template_ds[var].attrs["grid_mapping"] = "spatial_ref"

    template_ds.attrs["title"] = "Global snowmelt runoff onset from Sentinel-1 SAR"
    template_ds.attrs["config_version"] = config.version

    compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=5)
    encoding = {}
    for var in VARIABLE_DESCRIPTIONS:
        is_3d = "water_year" in template_ds[var].dims
        encoding[var] = {
            "shards": (1, *shard_2d) if is_3d else shard_2d,
            "chunks": (1, *chunk_2d) if is_3d else chunk_2d,
            "compressors": [compressor],
            "dtype": "int16",
            # _FillValue is only the DECODE attribute; fill_value sets the
            # zarr-v3 array fill that ABSENT chunks materialize as. Without it
            # zarr defaults to 0, and every never-written region (empty years,
            # unprocessed tiles, ocean) reads back as valid 0.0 instead of NaN.
            "_FillValue": NODATA_INT16,
            "fill_value": NODATA_INT16,
        }
        if var in SCALED_VARIABLES:
            encoding[var]["scale_factor"] = np.float32(0.1)
            encoding[var]["add_offset"] = np.float32(0.0)

    return template_ds, encoding


def initialize_store(repo, config, extra_metadata=None) -> str:
    """
    Write the empty template into a freshly created icechunk repository.

    Metadata-only: no data chunks are written (compute=False,
    write_empty_chunks=False); coordinate arrays and Zarr metadata only.

    Args:
        repo: icechunk Repository (from config.create_output_repo())
        config: the >= v10 Config the template is built from
        extra_metadata: optional additional commit metadata (e.g. provenance)

    Returns:
        Snapshot ID of the initialization commit.
    """
    template_ds, encoding = build_template(config)
    session = repo.writable_session("main")
    template_ds.to_zarr(
        session.store,
        mode="w",
        zarr_format=3,
        compute=False,
        write_empty_chunks=False,
        consolidated=False,
        encoding=encoding,
    )
    metadata = {
        "schema": 1,
        "kind": "init",
        "config_version": config.version,
        **(extra_metadata or {}),
    }
    return session.commit(
        f"initialize store: empty template, WY{config.WY_start}-{config.WY_end}",
        metadata=metadata,
    )


def tile_region_slices(config, tile_row: int, tile_col: int):
    """
    Explicit integer index slices of a tile within the global store grid.

    The tile geoboxes are direct slices of the same global geobox the store
    template was built from, so integer arithmetic is exact -- no
    floating-point coordinate matching needed. Edge tiles (last row/column)
    are smaller than tile_dim and are clamped to the global shape.

    Returns:
        {"latitude": slice, "longitude": slice}
    """
    tile_dim = config.spatial_chunk_dim_zarr_output
    n_y, n_x = config.global_geobox.shape.yx
    y0 = tile_row * tile_dim
    x0 = tile_col * tile_dim
    return {
        "latitude": slice(y0, min(y0 + tile_dim, n_y)),
        "longitude": slice(x0, min(x0 + tile_dim, n_x)),
    }


def grid_pixel_offset(config, other_config):
    """
    Integer (row, col) pixel offset from `config`'s grid to `other_config`'s.

    The v10 grid was extended north and south of the <= v9 grid (2026-07-30) but
    kept the same resolution lattice, so the two grids differ only by a whole-pixel
    translation of their origins: v9 pixel (i, j) is v10 pixel (i + 4096, j).

    Args:
        config: Config whose grid the input indices are on.
        other_config: Config whose grid the output indices should be on.

    Returns:
        (row_offset, col_offset) to ADD to a `config` pixel index to get the
        `other_config` index of the same ground.

    Raises:
        ValueError: if the grids have different resolutions, or if their origins
            differ by a non-integer number of pixels (different lattices -- no
            exact pixel correspondence exists and data would need resampling).
    """
    if config.resolution != other_config.resolution:
        raise ValueError(
            f"Cannot map indices between grids at different resolutions: "
            f"{config.config_name} {config.resolution} vs "
            f"{other_config.config_name} {other_config.resolution}."
        )
    res = config.resolution
    src, dst = config.global_geobox.transform, other_config.global_geobox.transform
    # transform.f is the NORTH edge and the y step is -res, so a source row i sits at
    # src.f - i*res and lands on destination row i + (dst.f - src.f)/res. transform.c
    # is the WEST edge with a +res step, hence the opposite sign for columns.
    row_exact = (dst.f - src.f) / res
    col_exact = (src.c - dst.c) / res
    offsets = []
    for name, exact in (("row", row_exact), ("col", col_exact)):
        rounded = round(exact)
        if abs(exact - rounded) > 1e-6:
            raise ValueError(
                f"{config.config_name} and {other_config.config_name} are on different "
                f"pixel lattices: their origins differ by {exact} pixels in {name} "
                "(not a whole number), so no exact pixel mapping exists."
            )
        offsets.append(rounded)
    return offsets[0], offsets[1]


def tile_region_slices_on_grid(config, tile_row: int, tile_col: int, other_config):
    """
    Slices of `config`'s tile (tile_row, tile_col) expressed on `other_config`'s grid.

    Use this to read the same ground from a store written on a different (but
    lattice-compatible) grid -- e.g. comparing a v10 tile against the published v9
    store, whose grid starts 4096 rows further south:

        v10_region = tile_region_slices(config_v10, row, col)
        v9_region  = tile_region_slices_on_grid(config_v10, row, col, config_v9)

    Slices are clamped to `other_config`'s shape, so for a tile that only partially
    overlaps the other grid the two regions have DIFFERENT lengths; compare over the
    overlap (or check the returned lengths) rather than assuming they match.

    Raises:
        ValueError: if the tile lies entirely outside `other_config`'s grid, or the
            grids are not lattice-compatible (see grid_pixel_offset).
    """
    row_off, col_off = grid_pixel_offset(config, other_config)
    region = tile_region_slices(config, tile_row, tile_col)
    n_y, n_x = other_config.global_geobox.shape.yx
    out = {}
    for dim, off, size in (("latitude", row_off, n_y), ("longitude", col_off, n_x)):
        start = region[dim].start + off
        stop = region[dim].stop + off
        if stop <= 0 or start >= size:
            raise ValueError(
                f"Tile ({tile_row},{tile_col}) of {config.config_name} lies entirely "
                f"outside {other_config.config_name}'s grid in {dim} "
                f"(would be {start}:{stop} of {size}) -- it covers ground that grid "
                "does not include."
            )
        out[dim] = slice(max(start, 0), min(stop, size))
    return out
