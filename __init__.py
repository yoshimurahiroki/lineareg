"""lineareg: Rigorous econometric analysis library.

This package provides publication-quality econometric estimators with
bootstrap-only inference, matching or exceeding R/Stata strictness.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "GLS",
    "GMM",
    "IV2SLS",
    "IVQR",
    "OLS",
    "QR",
    "RCT",
    "SAR2SLS",
    "SDID",
    "BaseEstimator",
    "BootConfig",
    "CallawaySantAnnaES",
    "DDDEventStudy",
    "DREventStudy",
    "EstimationResult",
    "EventStudyCS",
    "SpatialDID",
    "SyntheticControl",
    "diagnostics",
    "event_study_plot",
    "modelsummary",
    "moran_i",
    "moran_i_panel",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseEstimator": ("lineareg.estimators.base", "BaseEstimator"),
    "BootConfig": ("lineareg.estimators.base", "BootConfig"),
    "EstimationResult": ("lineareg.estimators.base", "EstimationResult"),
    "OLS": ("lineareg.estimators.ols", "OLS"),
    "GLS": ("lineareg.estimators.gls", "GLS"),
    "GMM": ("lineareg.estimators.gmm", "GMM"),
    "IV2SLS": ("lineareg.estimators.iv", "IV2SLS"),
    "IVQR": ("lineareg.estimators.qr", "IVQR"),
    "QR": ("lineareg.estimators.qr", "QR"),
    "RCT": ("lineareg.estimators.rct", "RCT"),
    "CallawaySantAnnaES": ("lineareg.estimators.eventstudy_cs", "CallawaySantAnnaES"),
    "EventStudyCS": ("lineareg.estimators.eventstudy_cs", "EventStudyCS"),
    "DREventStudy": ("lineareg.estimators.dr_eventstudy", "DREventStudy"),
    "DDDEventStudy": ("lineareg.estimators.eventstudy_ddd", "DDDEventStudy"),
    "SDID": ("lineareg.estimators.sdid", "SDID"),
    "SyntheticControl": ("lineareg.estimators.synthetic_control", "SyntheticControl"),
    "SpatialDID": ("lineareg.estimators.spatial_did", "SpatialDID"),
    "SAR2SLS": ("lineareg.spatial.spatial", "SAR2SLS"),
    "diagnostics": ("lineareg.output.summary", "diagnostics"),
    "modelsummary": ("lineareg.output.summary", "modelsummary"),
    "event_study_plot": ("lineareg.output.plots", "event_study_plot"),
    "moran_i": ("lineareg.spatial.spatial", "moran_i"),
    "moran_i_panel": ("lineareg.spatial.spatial", "moran_i_panel"),
}


def __getattr__(name: str) -> Any:
    """Lazily import public estimators and utilities on first access."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'lineareg' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Ensure dir() exposes lazily imported names."""
    return sorted(set(globals()) | set(__all__))
