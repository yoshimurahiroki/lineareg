# lineareg/spatial/__init__.py
"""Spatial econometrics module."""
from .spatial import SAR2SLS, moran_i, moran_i_panel

__all__ = ["SAR2SLS", "moran_i", "moran_i_panel"]
