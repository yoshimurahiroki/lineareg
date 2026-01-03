"""Estimator exports with lazy loading.

Public estimator classes and result containers. Uses lazy imports to avoid
circular dependencies.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GLS",
    "GMM",
    "IV2SLS",
    "OLS",
    "QR",
    "RCT",
    "SAR2SLS",
    "SDID",
    "BaseEstimator",
    "CallawaySantAnnaES",
    "DDDEventStudy",
    "DREventStudy",
    "EventStudyCS",
    "SpatialDID",
    "SyntheticControl",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseEstimator": ("lineareg.estimators.base", "BaseEstimator"),
    "BootConfig": ("lineareg.estimators.base", "BootConfig"),
    "EstimationResult": ("lineareg.estimators.base", "EstimationResult"),
    "OLS": ("lineareg.estimators.ols", "OLS"),
    "IV2SLS": ("lineareg.estimators.iv", "IV2SLS"),
    "GMM": ("lineareg.estimators.gmm", "GMM"),
    "GLS": ("lineareg.estimators.gls", "GLS"),
    "QR": ("lineareg.estimators.qr", "QR"),
    "IVQR": ("lineareg.estimators.qr", "IVQR"),
    "CallawaySantAnnaES": ("lineareg.estimators.eventstudy_cs", "CallawaySantAnnaES"),
    "EventStudyCS": ("lineareg.estimators.eventstudy_cs", "EventStudyCS"),
    "DREventStudy": ("lineareg.estimators.dr_eventstudy", "DREventStudy"),
    "DDDEventStudy": ("lineareg.estimators.eventstudy_ddd", "DDDEventStudy"),
    "SDID": ("lineareg.estimators.sdid", "SDID"),
    "SyntheticControl": ("lineareg.estimators.synthetic_control", "SyntheticControl"),
    "SpatialDID": ("lineareg.estimators.spatial_did", "SpatialDID"),
    "SAR2SLS": ("lineareg.spatial.spatial", "SAR2SLS"),
    "RCT": ("lineareg.estimators.rct", "RCT"),
}


def __getattr__(name: str) -> Any:
    """Lazily import estimator classes and shared containers."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        try:
            module = import_module(module_name)
        except ImportError:  # pragma: no cover - dev path fallback
            if name == "SAR2SLS":
                try:
                    module = import_module("lineareg.spatial.spatial")
                except ImportError:
                    root = __package__.split(".")[0] if __package__ else "lineareg"
                    module = import_module(f"{root}.spatial.spatial")
            else:
                raise
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'lineareg.estimators' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Expose lazily loaded attributes to ``dir()``."""
    return sorted(set(globals()) | set(__all__))
