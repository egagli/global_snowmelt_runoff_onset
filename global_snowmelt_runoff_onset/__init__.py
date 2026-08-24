"""
Global snowmelt runoff onset — processing pipeline and dataset access.

Submodules import lazily (PEP 562): ``import global_snowmelt_runoff_onset``
costs nothing, and ``global_snowmelt_runoff_onset.pyramid`` /
``from global_snowmelt_runoff_onset.pyramid import open_pyramid_level``
never pays for the heavy processing stack (odc.stac, easysnowdata →
earthengine-api, icechunk, …) that ``processing`` and ``config`` pull in.
Previously this module imported ``config`` and ``processing`` eagerly, so
any import of the package dragged the full stack in.
"""

import importlib

_SUBMODULES = (
    'config', 'plot_utils', 'processing', 'provenance', 'pyramid',
    'qc', 'results', 'status', 'store',
)


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))
