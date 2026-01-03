"""Optional GPU acceleration via CuPy.

This module provides a NumPy/CuPy abstraction layer for dense linear algebra operations.
GPU acceleration is enabled when CuPy is available and explicitly requested.
Results are returned as NumPy arrays to maintain API consistency.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

# NumPy is always present
import numpy as np

# Try CuPy lazily
try:  # pragma: no cover - optional dependency
    import cupy as _cp  # type: ignore
    import cupyx.scipy.linalg as _cpxla  # type: ignore
    _CUPY_OK = True
except ImportError:  # pragma: no cover - optional dependency
    _cp = None  # type: ignore
    _cpxla = None  # type: ignore
    _CUPY_OK = False


if TYPE_CHECKING:
    from collections.abc import Iterable


_LOGGER = logging.getLogger(__name__)


from functools import lru_cache


def _env_wants_gpu() -> bool:
    dev = str(os.environ.get("LINEAREG_DEVICE", "")).strip().lower()
    flag = str(os.environ.get("LINEAREG_USE_GPU", "")).strip().lower()
    if dev in {"gpu", "cuda"}:
        return True
    return flag in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Return True if CuPy is importable and at least one CUDA device is available."""
    if not _CUPY_OK:
        return False
    try:  # pragma: no cover - environment-specific
        n = _cp.cuda.runtime.getDeviceCount()  # type: ignore[attr-defined]
        return int(n) > 0
    except (AttributeError, RuntimeError) as exc:  # pragma: no cover - environment-specific
        _LOGGER.debug("GPU detection failed; assuming CPU. Error: %s", exc)
        return False


@lru_cache(maxsize=1)
def gpu_enabled() -> bool:
    """Return True when environment requests GPU and it's available."""
    return _env_wants_gpu() and gpu_available()


def xp_for(arrays: Iterable[Any] | None = None, prefer_gpu: bool | None = None):
    """Return the array module (numpy or cupy) to use.

    Heuristics:
    - If any input array is a CuPy array, return cupy.
    - Else, if prefer_gpu is True or environment enables GPU and it's available, return cupy.
    - Otherwise, return numpy.
    """
    if _CUPY_OK:
        # If any provided array is already CuPy, honor that
        if arrays is not None:
            for a in arrays:
                if isinstance(a, _cp.ndarray):  # type: ignore[attr-defined]
                    return _cp
    use_gpu = _env_wants_gpu() if prefer_gpu is None else bool(prefer_gpu)
    if use_gpu and gpu_available():
        return _cp if _CUPY_OK else np
    return np


def asarray(x: Any, xp=None, dtype=np.float64, copy: bool = False):
    """Convert to array on the selected backend (default float64)."""
    mod = np if xp is None else xp
    # NumPy's asarray historically does not accept copy= (pre-2.0). Avoid
    # relying on exceptions for control flow on the CPU path.
    if mod is np:
        if copy:
            return np.array(x, dtype=dtype, copy=True)
        return np.asarray(x, dtype=dtype)
    try:
        return mod.asarray(x, dtype=dtype, copy=copy)
    except (AttributeError, TypeError, ValueError) as exc:
        _LOGGER.debug("Falling back to NumPy.asarray; %s.asarray failed: %s", mod.__name__, exc)
        # Fallback to NumPy if backend coercion fails
        if copy:
            return np.array(x, dtype=dtype, copy=True)
        return np.asarray(x, dtype=dtype)


def to_device(x: Any, device: str | None = None, dtype=np.float64):
    """Move array to the requested device ("cpu"|"gpu"|None for heuristic)."""
    dev = (str(device).lower() if device is not None else None)
    if dev in {"gpu", "cuda"} or (dev is None and _env_wants_gpu() and gpu_available()):
        if _CUPY_OK and gpu_available():
            try:
                return _cp.asarray(x, dtype=dtype)  # type: ignore[attr-defined]
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                _LOGGER.debug("GPU transfer failed; retrying on CPU. Error: %s", exc)
    # default CPU
    return np.asarray(x, dtype=dtype)


def to_cpu(x: Any, dtype=np.float64):
    """Ensure array is on CPU (NumPy)."""
    if _CUPY_OK and isinstance(x, _cp.ndarray):  # type: ignore[attr-defined]
        return _cp.asnumpy(x).astype(dtype, copy=False)  # type: ignore[attr-defined]
    return np.asarray(x, dtype=dtype)


def dot(A: Any, B: Any, prefer_gpu: bool | None = None):
    """Backend-aware matrix multiply; returns CPU ndarray to preserve public API."""
    xp = xp_for([A, B], prefer_gpu=prefer_gpu)
    Ad = asarray(A, xp=xp, dtype=np.float64)
    Bd = asarray(B, xp=xp, dtype=np.float64)
    C = Ad @ Bd
    return to_cpu(C)  # public API expects NumPy


def cholesky(A: Any, lower: bool = True, prefer_gpu: bool | None = None):
    xp = xp_for([A], prefer_gpu=prefer_gpu)
    Ad = asarray(A, xp=xp, dtype=np.float64)
    try:
        if xp is np:
            L = np.linalg.cholesky(Ad)
            return L if lower else L.T
        # cupy path
        if _cpxla is not None:
            return to_cpu(_cpxla.cholesky(Ad, lower=lower))  # type: ignore[attr-defined]
        L_gpu = xp.linalg.cholesky(Ad)
        return to_cpu(L_gpu) if lower else to_cpu(L_gpu).T
    finally:
        # be proactive about freeing large temporaries
        free_gpu_cache()


def svd(A: Any, full_matrices: bool = False, prefer_gpu: bool | None = None):
    xp = xp_for([A], prefer_gpu=prefer_gpu)
    Ad = asarray(A, xp=xp, dtype=np.float64)
    U, s, Vt = xp.linalg.svd(Ad, full_matrices=full_matrices)
    return to_cpu(U), to_cpu(s), to_cpu(Vt)


def eigvalsh(A: Any, prefer_gpu: bool | None = None):
    xp = xp_for([A], prefer_gpu=prefer_gpu)
    Ad = asarray(A, xp=xp, dtype=np.float64)
    vals = xp.linalg.eigvalsh(Ad)
    return to_cpu(vals)


def qr(A: Any, mode: str = "reduced", prefer_gpu: bool | None = None):
    """QR without pivoting on GPU when possible; with pivoting remains CPU.

    Note: Pivoted QR (GEQP3) is not widely available on GPU. For pivoted QR
    the callers should use the CPU implementation from core.linalg.
    """
    xp = xp_for([A], prefer_gpu=prefer_gpu)
    Ad = asarray(A, xp=xp, dtype=np.float64)
    Q, R = xp.linalg.qr(Ad, mode=mode)
    return to_cpu(Q), to_cpu(R)


class DeviceGuard:
    """Context manager to temporarily force a device preference."""

    def __init__(self, device: str):
        self.device = str(device).lower()
        self._prev_env = None

    def __enter__(self):  # pragma: no cover - simple context shim
        self._prev_env = os.environ.get("LINEAREG_DEVICE")
        os.environ["LINEAREG_DEVICE"] = ("gpu" if self.device in {"gpu", "cuda"} else "cpu")
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - simple context shim
        if self._prev_env is None:
            os.environ.pop("LINEAREG_DEVICE", None)
        else:
            os.environ["LINEAREG_DEVICE"] = self._prev_env
        free_gpu_cache()
        return False


def free_gpu_cache() -> None:
    """Release cached GPU memory if CuPy is active."""
    if not _CUPY_OK:
        return
    try:  # pragma: no cover - environment-specific
        _cp.get_default_memory_pool().free_all_blocks()  # type: ignore[attr-defined]
        _cp.get_default_pinned_memory_pool().free_all_blocks()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError) as exc:  # pragma: no cover - environment-specific
        _LOGGER.debug("GPU cache cleanup failed: %s", exc)
