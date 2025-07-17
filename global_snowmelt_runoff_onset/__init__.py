import global_snowmelt_runoff_onset.global_snowmelt_runoff_onset
import global_snowmelt_runoff_onset.config
import global_snowmelt_runoff_onset.processing

# Optional import for analysis module (requires xdem and other dependencies)
try:
    import global_snowmelt_runoff_onset.analysis  # noqa: F401
except ImportError:
    pass
