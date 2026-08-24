"""
Consumer-side reader for the multiscale visualization pyramid.

The pyramid is a standalone plain-Zarr v3 multiscale store published next to
the icechunk repo (built by visualize/pyramid/build_pyramid.py), and the
container is anonymous-blob-readable — so READING it needs no SAS token.
This module is the installable home of that reader: the figure notebooks
(visualize/global/, visualize/regions/, visualize/methods/) previously
sys.path-hacked visualize/pyramid/ to import it from the builder script.

Reads are anonymous by default, which pairs with Config's lazy SAS token
loading: figure work against the pyramid runs with no credentials at all.
"""

import obstore.store
import xarray as xr
import zarr


def pyramid_store(config, dest_prefix=None, anonymous=True):
    """
    Read-only zarr ObjectStore onto the pyramid prefix (via obstore).

    obstore is used because its Rust object_store handles the Azure suffix
    byte-range requests that Zarr v3 shard-index reads need (adlfs gets them
    wrong). Anonymous by default — the container is public-read, so no SAS
    token is loaded or required; pass anonymous=False to sign with
    config.sas_token (only relevant for a store that isn't public yet).

    Args:
        config: Config (supplies the Azure account and the default pyramid
            prefix via global_runoff_multiscale_azure_prefix).
        dest_prefix: Override the container/prefix ('container/some/prefix').
        anonymous: Unsigned public reads (default) vs. SAS-signed.
    """
    if dest_prefix is None:
        dest_prefix = config.global_runoff_multiscale_azure_prefix
    container, prefix = dest_prefix.split("/", 1)
    credential = ({"skip_signature": True} if anonymous
                  else {"sas_key": config.sas_token})
    az = obstore.store.AzureStore(account_name=config.azure_storage_account,
                                  container_name=container, prefix=prefix,
                                  **credential)
    return zarr.storage.ObjectStore(az, read_only=True)


def open_pyramid_level(config, level, dest_prefix=None, decode=True,
                       chunks=None, anonymous=True):
    """
    Open one pyramid level as an xarray Dataset (read-only, via obstore).

    The consumer-side counterpart of the pyramid builder — used by the global
    and regional figure notebooks in place of the retired v9 coarsened store.
    Resolution at level n is ~80 m * 2**n (level 2 ~320 m, level 5 ~2.6 km,
    level 7 ~10 km). Open levels individually: xr.open_datatree on a
    multiscale hierarchy tries to align same-named dims with different
    coordinates across levels and fails.

    Args:
        config: Config (supplies the Azure account AND the pyramid prefix via
            global_runoff_multiscale_azure_prefix; no SAS token is touched
            unless anonymous=False).
        level: Pyramid level number (0 = native ~80 m).
        dest_prefix: Override the container/prefix.
        decode: Decode CF metadata (fill -> NaN, scale_factor applied).
        chunks: xarray chunking. None (default) = lazy backend arrays, no
            dask -- right for figure-scale levels (>= 4). Pass 'auto' for
            dask-backed reads of the big fine levels (0-3).
        anonymous: Unsigned public reads (default) vs. SAS-signed.
    """
    return xr.open_zarr(pyramid_store(config, dest_prefix, anonymous=anonymous),
                        group=str(level), zarr_format=3, consolidated=False,
                        mask_and_scale=decode, decode_coords="all",
                        chunks=chunks)
