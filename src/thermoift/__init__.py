import importlib as _importlib

from .ML_feeds import FeedsBuilder
from .ML_preprocessing import MLPreprocessing
from .ML_postprocessing import MLPostprocessing, plot_correlation_heatmap, print_model_metrics
from . import rng_utils

def __getattr__(name):
    _lazy = {
        "PLOT_SETTINGS":               f"{__name__}.PLOT_SETTINGS",
        "FeosPlugin":                  f"{__name__}.FeosPlugin",
        "semi_emperical_correlations":  f"{__name__}.semi_emperical_correlations",
        "Analyzer":                    f"{__name__}.Analyzer",
        "KIJ":                         f"{__name__}.BINARY_INTERACTION_PARAMETERS.KIJ",
    }
    if name in _lazy:
        mod = _importlib.import_module(_lazy[name])
        if name == "KIJ":
            obj = mod.BinaryInteractions
        elif name == "Analyzer":
            obj = mod.Analyzer
        else:
            obj = mod
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
