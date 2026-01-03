# lineareg/output/__init__.py
"""Output and visualization module for econometric results."""
from .plots import (
    balance_plot,
    event_study_auto_plot,
    event_study_percentile_plot,
    event_study_plot,
    spatial_did_plot,
)
from .summary import diagnostics, modelsummary

__all__ = [
    # Alphabetical export list (grouped with comments)
    "balance_plot",
    "diagnostics",
    "event_study_auto_plot",
    "event_study_percentile_plot",
    "event_study_plot",
    "modelsummary",
    "spatial_did_plot",
]
