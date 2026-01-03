"""Linear algebra routines for regression analysis.

This module provides solvers (QR, SVD, Cholesky) with rank policies compatible
with R and Stata. It supports sparse matrices and batched operations for
efficient bootstrap computations. Explicit matrix inversion is avoided.
"""

from __future__ import annotations

# Standard library
import multiprocessing
import os
import warnings as _warnings
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import TYPE_CHECKING, Any, Callable

# NumPy
import numpy as np
from numpy.linalg import solve as _npsolve

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
else:
    Sequence = tuple  # type: ignore[assignment]
    NDArray = np.ndarray  # type: ignore[misc,assignment]

# Optional backend (NumPy/CuPy) for dense ops
try:
    from . import backend as _bk  # local GPU-aware helpers
except ImportError:  # pragma: no cover - fallback if backend import fails
    _bk = None  # type: ignore

# SciPy (optional)
try:
    import scipy.linalg as sla
    import scipy.sparse as sp

    spla = sla
except ImportError:
    sp = None
    sla = None
    spla = None

# Optional sparse linear algebra routines
try:  # pragma: no cover - optional SciPy sparse
    from scipy.sparse.linalg import eigs as _sparse_eigs  # type: ignore
except ImportError:  # pragma: no cover - optional SciPy sparse
    _sparse_eigs = None
try:  # pragma: no cover - optional SciPy sparse
    from scipy.sparse.linalg import spsolve as _spsolve  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional SciPy sparse
    _spsolve = None

# SuiteSparseQR (optional)
try:
    import sparseqr

    spqr = sparseqr
except ImportError:
    sparseqr = None
    spqr = None

# Matrix type alias
Matrix = Any


def finite_sample_quantile_bplus1(x: NDArray[np.float64], q: float) -> float:
    """Compute (B+1) rule quantile for bootstrap statistics."""
    xa = np.asarray(x, dtype=np.float64)
    xa = xa[np.isfinite(xa)]
    if xa.size == 0:
        return float("nan")
    xa.sort()
    B = xa.size
    k = int(np.ceil((B + 1) * float(q)))
    k = max(1, min(k, B))
    return float(xa[k - 1])


def quantile_type1_r(x: NDArray[np.float64], q: float) -> float:
    """Compute R type-1 quantile (inverse of empirical distribution function)."""
    xa = np.asarray(x, dtype=np.float64)
    xa = xa[np.isfinite(xa)]
    if xa.size == 0:
        return float("nan")
    xa.sort()
    n = xa.size
    k = int(np.ceil(float(n) * float(q)))
    k = max(1, min(k, n))
    return float(xa[k - 1])


def _assert_all_finite(*arrays: NDArray[np.float64]) -> None:
    """Raise ValueError if any input contains NaN or Inf."""
    for a in arrays:
        if a is None:
            continue
        ad = np.asarray(a)
        if not np.all(np.isfinite(ad)):
            raise ValueError(
                "Input contains NA/NaN/Inf; please drop/clean rows (R/Stata behavior).",
            )


def _check_array_finiteness(arr: NDArray[np.float64]) -> None:
    """Helper to validate array finiteness with clear error message."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "Input contains NA/NaN/Inf; please drop/clean rows (R/Stata behavior).",
        )


def _extract_sparse_data(M: Matrix) -> NDArray[np.float64]:
    """Safely extract data from sparse matrix, falling back to dense conversion if needed."""
    try:
        return M.data  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # Fallback: convert to dense for validation
        return np.asarray(to_dense(M))


def _assert_all_finite_matrix(*matrices: Matrix) -> None:
    """Check dense or sparse matrices for non-finite entries."""
    for M in matrices:
        if M is None:
            continue

        if _is_sparse(M):
            # Only inspect the stored data array to avoid densification.
            data = _extract_sparse_data(M)
            _check_array_finiteness(data)
        else:
            arr = np.asarray(M)
            _check_array_finiteness(arr)


def _solve_normal_eq_worker_top(args):
    """Top-level picklable worker for ProcessPoolExecutor.

    Expects args = (X, y, w, method, rank_policy)
    """
    X, y, w, method, rank_policy = args
    return solve_normal_eq(X, y, weights=w, method=method, rank_policy=rank_policy)


def _swap_columns(  # noqa: PLR0913
    W: NDArray[np.float64],
    R: NDArray[np.float64],
    vn1: NDArray[np.float64],
    vn2: NDArray[np.float64],
    P: NDArray[np.int64],
    k: int,
    j: int,
) -> None:
    """Swap columns k and j in QR decomposition workspace arrays."""
    W[:, [k, j]] = W[:, [j, k]]
    vn1[[k, j]] = vn1[[j, k]]
    vn2[[k, j]] = vn2[[j, k]]
    P[[k, j]] = P[[j, k]]
    R[:k, [k, j]] = R[:k, [j, k]]


def _compute_householder_vector(
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Compute Householder reflection vector and norm."""
    normx = np.linalg.norm(x)
    if normx == 0.0:
        return x, 0.0

    sign = -1.0 if x[0] < 0.0 else 1.0
    v = x.copy()
    v[0] += sign * normx
    v /= np.linalg.norm(v)
    return v, normx


def _update_column_norms(  # noqa: PLR0913
    vn1: NDArray[np.float64],
    vn2: NDArray[np.float64],
    R: NDArray[np.float64],
    W: NDArray[np.float64],
    k: int,
    n: int,
) -> None:
    """Update column norms using GEQP3 downdate strategy."""
    if k + 1 >= n:
        return

    colslice = slice(k + 1, n)
    akj = R[k, colslice]
    # GEQP3 / DLAQPS additive downdate: vn1 <- max(0, vn1 - akj^2)
    vn1[colslice] = np.maximum(0.0, vn1[colslice] - (akj * akj))

    # Re-evaluation trigger per LAPACK GEQP3 strategy.
    tol3z = float(np.sqrt(np.finfo(float).eps))
    mask_re = vn1[colslice] <= (tol3z * vn2[colslice])
    if np.any(mask_re):
        idx = np.nonzero(mask_re)[0] + (k + 1)
        blk = W[k + 1 :, idx]
        # recompute true squared norms
        vn1[idx] = np.einsum("ij,ij->j", blk, blk)
        vn2[idx] = vn1[idx].copy()


def _qr_cp_numpy(
    A: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Householder QR with greedy column pivoting (GEQP3 style)."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    m, n = A.shape
    W = A.copy()
    Q = np.eye(m, dtype=np.float64)
    R = np.zeros((min(m, n), n), dtype=np.float64)
    P = np.arange(n, dtype=np.int64)
    vn1 = np.sum(W * W, axis=0)
    vn2 = vn1.copy()
    rcols = min(m, n)

    for k in range(rcols):
        # Find pivot column (greedy maximum norm)
        j = k + int(np.argmax(vn1[k:]))
        if j != k:
            _swap_columns(W, R, vn1, vn2, P, k, j)

        # Compute Householder reflection
        x = W[k:, k]
        v, normx = _compute_householder_vector(x)
        if normx == 0.0:
            R[k, k] = 0.0
            continue

        # Apply reflection to W and Q
        tau = 2.0
        W_k = W[k:, k:]
        W[k:, k:] -= tau * np.outer(v, v @ W_k)
        R[k, k:] = W[k, k:]
        Q[:, k:] -= tau * np.outer(Q[:, k:] @ v, v)

        # Update column norms for remaining columns
        _update_column_norms(vn1, vn2, R, W, k, n)

    Q = Q[:, :rcols]
    R = R[:rcols, :]
    return Q, R, P


# --- Proper __all__ list ---
__all__ = [
    "Matrix",
    "_qr_ls_solve",
    "_rank_from_diag",
    "_validate_weights",
    "ar1_correlation_by_groups",
    "ar1_correlation_n",
    "ar1_covariance_from_weights_by_groups",
    "batched_crossprod_list",
    "batched_crossprod_pairs",
    "batched_solve_normal_eq_list",
    "batched_solve_normal_eq_parallel",
    "rank_from_diag",
    "sar_errors_covariance",
    "uniform_band_from_bootstrap",
    "xtwx_inv_via_qr",
    "xty",
]


def _is_sparse(A: Matrix) -> bool:
    return sp is not None and isinstance(A, sp.spmatrix)  # type: ignore[union-attr]


def to_dense(A: Matrix, *, allow_densify: bool = True) -> NDArray[np.float64]:
    """Convert a matrix-like object to a dense float64 numpy array.

    Parameters
    ----------
    A : Matrix
        Input matrix which may be a numpy array or a SciPy sparse matrix.
    allow_densify : bool, default True
        When False, refuse to convert sparse inputs to dense and raise a
        RuntimeError. This helps detect unintended densification of large
        sparse matrices in higher-level code. The default preserves the
        historical behaviour (i.e., densification allowed).

    Returns
    -------
    ndarray
        Dense float64 numpy array.

    """
    if _is_sparse(A):
        if not allow_densify:
            msg = "to_dense: densification of sparse matrix is disabled; pass allow_densify=True to intentionally densify."
            raise RuntimeError(msg)
        return np.asarray(A.todense(), dtype=np.float64)
    return np.asarray(A, dtype=np.float64)


def qr(A: Matrix, *, pivoting: bool = False, mode: str = "economic"):
    """Compute QR decomposition using SciPy (if available) or internal NumPy fallback."""
    # Sparse path: prefer SuiteSparseQR for exact QRCP when pivoting requested
    if _is_sparse(A) and pivoting and (spqr is not None):
        Ac = A.tocsc()  # type: ignore[attr-defined]
        Qs, Rs, E = spqr.qr(Ac)  # type: ignore[attr-defined]
        # Ensure Q has 'economic' shape (m x rcols) to match NumPy/SciPy behavior
        rcols = min(Ac.shape[0], Ac.shape[1])
        Qs = Qs[:, :rcols]
        # SuiteSparse returns a permutation E; convert to P array of indices
        try:
            P = np.asarray(E).ravel()
        except (AttributeError, TypeError, ValueError):
            # Fallback: try extracting permutation from sparse matrix structure
            P = np.array(E.tocsc().argmax(axis=0)).ravel()  # type: ignore[union-attr]
        return Qs, Rs, P
    Ad = to_dense(A).astype(np.float64)
    rcols = min(Ad.shape[0], Ad.shape[1])
    if sla is not None:
        if pivoting:
            Q, R, P = sla.qr(Ad, mode=mode, pivoting=True)
            Q = Q[:, :rcols]
            R = R[:rcols, :]
            return Q, R, P
        Q, R = sla.qr(Ad, mode=mode, pivoting=False)
        Q = Q[:, :rcols]
        R = R[:rcols, :]
        return Q, R
    # fallback (NumPy-only)
    if pivoting:
        Q, R, P = _qr_cp_numpy(Ad)
        Q = Q[:, :rcols]
        R = R[:rcols, :]
        return Q, R, P
    Q, R = np.linalg.qr(Ad, mode=("reduced" if mode == "economic" else mode))
    Q = Q[:, :rcols]
    R = R[:rcols, :]
    return Q, R


def eye(n: int, *, sparse: bool = False) -> Matrix:
    if sparse and sp is not None:
        return sp.eye(n, n, dtype=np.float64, format="csc")
    return np.eye(n, dtype=np.float64)


def hstack(arrays: Sequence[Matrix]) -> Matrix:
    """Column-wise stack that preserves sparsity if any chunk is sparse."""
    use_sparse = any(_is_sparse(A) for A in arrays)
    if use_sparse and sp is not None:
        return sp.hstack(arrays, format="csc")
    return np.hstack([to_dense(A) for A in arrays])


def vstack(arrays: Sequence[Matrix]) -> Matrix:
    """Row-wise stack that preserves sparsity if any chunk is sparse."""
    use_sparse = any(_is_sparse(A) for A in arrays)
    if use_sparse and sp is not None:
        return sp.vstack(arrays, format="csc")
    return np.vstack([to_dense(A) for A in arrays])


def column_stack(cols: Sequence[Matrix]) -> NDArray[np.float64]:
    """Column-wise stack that returns a dense ndarray.

    This is a small wrapper to centralize column stacking behavior so that
    external modules do not call numpy directly. It preserves dtype float64
    and converts sparse inputs to dense first.
    """
    return np.column_stack([to_dense(c) for c in cols]).astype(np.float64)


def hadamard(A: Matrix, B: Matrix) -> Matrix:
    """Elementwise product with sparsity awareness.
    Returns sparse if any operand is sparse (SciPy preserves sparsity).
    """
    if _is_sparse(A) and _is_sparse(B):
        return A.multiply(B)
    if _is_sparse(A):
        return A.multiply(np.asarray(B, dtype=np.float64))
    if _is_sparse(B):
        # use explicit sparse multiply to avoid SciPy-version-dependent
        # densification when performing elementwise product
        return B.multiply(np.asarray(A, dtype=np.float64))
    return np.asarray(A, dtype=np.float64) * np.asarray(B, dtype=np.float64)


def _rank_from_diag(diagR: NDArray[np.float64], ncols: int, mode: str = "stata") -> int:
    """Determine numerical rank from R diagonal entries using method-specific tolerance."""
    d = np.asarray(diagR, dtype=float).reshape(-1)
    if d.size == 0:
        return 0
    mode_lower = str(mode).lower()
    # Stata (Mata qrsolve) default tolerance: eta = 1e-13 * trace(|R|)/rows(R)
    if mode_lower == "stata":
        # use explicit trace(|R|)/rows(R) as in Stata Mata qrsolve documentation
        eta = 1e-13 * (float(np.sum(np.abs(d))) / float(d.size))
        tol = eta
    elif mode_lower in ("r", "r_strict"):
        # R's lm.fit documented default: tol = 1e-7 * max(|diag(R)|)
        # Use this relative threshold to mimic dqrdc2/dgeqp3 behavior in practice.
        tol = 1e-7 * float(np.max(np.abs(d)))
    else:  # numpy-like rcond style
        tol = np.finfo(float).eps * max(1, int(ncols)) * float(np.max(np.abs(d)))
    return int(np.sum(np.abs(d) > tol))


def rank_from_diag(
    diagR: NDArray[np.float64], ncols: int, *, mode: str = "stata",
) -> int:
    """Public wrapper for :func:`_rank_from_diag` (preserves Stata/R conventions)."""
    return _rank_from_diag(diagR, ncols, mode=mode)


def _qr_ls_solve(
    Ad: NDArray[np.float64], Bd: NDArray[np.float64], *, mode: str = "stata",
) -> NDArray[np.float64]:
    """Solve least squares via pivoted QR."""
    mode_norm = str(mode).lower().strip()
    # Ensure 2-D shapes
    Bd = Bd.reshape(-1, 1) if Bd.ndim == 1 else Bd
    # Use SciPy pivoted QR when available for deterministic pivoting
    if sla is not None:
        Q, R, P = sla.qr(Ad, mode="economic", pivoting=True)
    else:
        Q, R, P = _qr_cp_numpy(Ad)
    diagR = np.abs(np.diag(R))
    r = _rank_from_diag(diagR, Ad.shape[1], mode=mode)
    QtB = Q.T @ Bd
    if mode_norm.startswith("r"):
        out = np.full((Ad.shape[1], Bd.shape[1]), np.nan, dtype=np.float64)
    else:
        out = np.zeros((Ad.shape[1], Bd.shape[1]), dtype=np.float64)
    if r > 0:
        X_basic = np.linalg.solve(R[:r, :r], QtB[:r, :])
        out[P[:r], :] = X_basic
    return out.astype(np.float64)


def qr_solve_stata(
    A: Matrix, B: Matrix, *, tol: float | None = None, mode: str = "stata",
) -> NDArray[np.float64]:
    """Solve system using Stata-style QR tolerance/filling rules."""
    Ad = to_dense(A)
    Bd = to_dense(B)
    Bd = Bd.reshape(-1, 1) if Bd.ndim == 1 else Bd
    mode_norm = mode.lower().strip()
    if mode_norm not in {"stata", "r"}:
        raise ValueError("mode must be either 'stata' or 'r'")
    # R/Stata strict behavior: reject NA/Inf in inputs early (sparse-aware)
    _assert_all_finite_matrix(A, Bd)
    if sla is not None:
        Q, R, P = sla.qr(Ad, mode="economic", pivoting=True)
    else:
        Q, R, P = _qr_cp_numpy(Ad)
    diagR = np.abs(np.diag(R))
    # Stata/Mata default: when tol is None use the Mata recipe implemented by
    # _rank_from_diag(..., mode='stata'). If tol is explicitly provided, treat
    # it as an absolute threshold on |diag(R)|.
    if tol is None:
        r = _rank_from_diag(diagR, Ad.shape[1], mode=mode_norm)
    else:
        thresh = float(tol)
        r = int(np.sum(diagR > thresh))
    fill_value = np.nan if mode_norm == "r" else 0.0
    out = np.full((Ad.shape[1], Bd.shape[1]), fill_value, dtype=np.float64)
    if r > 0:
        QtB = Q.T @ Bd
        X_basic = np.linalg.solve(R[:r, :r], QtB[:r, :])
        out[P[:r], :] = X_basic
    return out


def qr_coef_r(A: Matrix, B: Matrix, *, tol: float = 1e-7) -> NDArray[np.float64]:
    """Solve system using R-style QR tolerance/filling rules (NA for dropped cols)."""
    Ad = to_dense(A)
    Bd = to_dense(B)
    Bd = Bd.reshape(-1, 1) if Bd.ndim == 1 else Bd
    if sla is not None:
        Q, R, P = sla.qr(Ad, mode="economic", pivoting=True)
    else:
        Q, R, P = _qr_cp_numpy(Ad)
    diagR = np.abs(np.diag(R))
    thresh = float(tol) * (float(np.max(diagR)) if diagR.size else 0.0)
    r = int(np.sum(diagR > thresh))
    coef = np.full((Ad.shape[1], Bd.shape[1]), np.nan, dtype=np.float64)
    if r > 0:
        QtB = Q.T @ Bd
        beta = np.linalg.solve(R[:r, :r], QtB[:r, :])
        coef[P[:r], :] = beta
    return coef


def ensure_unweighted(
    weights: Sequence[float] | None, *, routine: str = "this routine",
) -> None:
    """Raise error if weights are provided to an unweighted routine."""
    if weights is not None:
        raise ValueError(f"{routine} does not accept analytic weights under the strict policy.")


def assert_unweighted_model(weights: Sequence[float] | None, model_name: str) -> None:
    """Legacy wrapper for ensure_unweighted."""
    ensure_unweighted(weights, routine=model_name)


def dot(A: Matrix, B: Matrix) -> Matrix:
    """Matrix multiplication with sparsity awareness."""
    if _is_sparse(A) or _is_sparse(B):
        return A @ B  # type: ignore[operator]
    Ad = np.asarray(A, dtype=np.float64)
    Bd = np.asarray(B, dtype=np.float64)
    # Optional GPU fast-path for dense multiply; result converted back to CPU
    if _bk is not None and _bk.gpu_enabled():
        try:
            return _bk.dot(Ad, Bd)  # returns NumPy array
        finally:
            _bk.free_gpu_cache()
    return Ad @ Bd


def crossprod(X: Matrix, y: Matrix | None = None) -> NDArray[np.float64]:
    """Compute X'y (or X'X if y is None)."""
    if y is None:
        _warnings.warn(
            "Calling crossprod(X) with a single argument is deprecated and will be removed. "
            "Please use crossprod(X, X) or tdot(X) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        y = X
    # Ensure y is column-like before any sparse coercion to avoid shape issues
    if isinstance(y, np.ndarray) and y.ndim == 1:
        y = y.reshape(-1, 1)
    if _is_sparse(X) or _is_sparse(y):
        X_sparse = X if _is_sparse(X) else sp.csc_matrix(to_dense(X))
        y_sparse = y if _is_sparse(y) else sp.csc_matrix(to_dense(y))
        out = (X_sparse.T @ y_sparse).toarray().astype(np.float64)
        return out.reshape(-1, 1) if out.ndim == 1 else out
    out = (X.T @ y).astype(np.float64)
    return out.reshape(-1, 1) if out.ndim == 1 else out


def drop_rank_deficient_cols_r(
    A: Matrix,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Drop columns based on pivoted QR rank (R-style tolerance)."""
    Ad = to_dense(A).astype(np.float64)
    _assert_all_finite(Ad)
    if Ad.size == 0:
        return Ad, np.zeros((0,), dtype=bool)
    # Pivoted QR (SciPy preferred for determinism)
    if sla is not None:
        _Q, R, P = sla.qr(Ad, mode="economic", pivoting=True)
    else:
        _Q, R, P = _qr_cp_numpy(Ad)
    d = np.abs(np.diag(R))
    tol = 1e-7 * (d.max() if d.size else 0.0)
    r = int(np.sum(d > tol))
    keep_mask = np.zeros(Ad.shape[1], dtype=bool)
    if r > 0:
        keep_mask[P[:r]] = True
    return Ad[:, keep_mask], keep_mask


def drop_rank_deficient_cols_stata(
    A: Matrix,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Drop columns based on pivoted QR rank (Stata-style tolerance)."""
    Ad = to_dense(A).astype(np.float64)
    _assert_all_finite(Ad)
    if Ad.size == 0:
        return Ad, np.zeros((0,), dtype=bool)
    if sla is not None:
        _Q, R, P = sla.qr(Ad, mode="economic", pivoting=True)
    else:
        _Q, R, P = _qr_cp_numpy(Ad)
    d = np.abs(np.diag(R)).astype(float)
    eta = 1e-13 * (d.sum() / (d.size if d.size else 1.0))
    r = int(np.sum(d > eta))
    keep_mask = np.zeros(Ad.shape[1], dtype=bool)
    if r > 0:
        keep_mask[P[:r]] = True
    return Ad[:, keep_mask], keep_mask


def drop_rank_deficient_cols_stable(
    A: Matrix,
    *,
    mode: str = "stata",
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Order-preserving rank screening.

    Uses pivoted QR to determine numerical rank (Stata-style tolerance) but
    returns a boolean `keep_mask` aligned with the original column order
    so callers can transparently map names/indices. This helper mirrors the
    semantics of `drop_rank_deficient_cols_stata` but makes the keep mask
    explicit and stable for downstream re-indexing logic.
    """
    Ad = to_dense(A).astype(np.float64)
    _assert_all_finite(Ad)
    if Ad.size == 0:
        return Ad, np.zeros((0,), dtype=bool)
    if sla is not None:
        _Q, R, P = sla.qr(Ad, mode="economic", pivoting=True)
    else:
        _Q, R, P = _qr_cp_numpy(Ad)
    d = np.abs(np.diag(R)).astype(float)
    mode_norm = str(mode).lower()
    if mode_norm == "stata":
        eta = 1e-13 * (d.sum() / (d.size if d.size else 1.0))
        r = int(np.sum(d > eta))
    elif mode_norm in {"r", "r_strict"}:
        tol = 1e-7 * (float(np.max(d)) if d.size else 0.0)
        r = int(np.sum(d > tol))
    else:
        tol = np.finfo(float).eps * max(1, int(Ad.shape[1])) * (float(np.max(d)) if d.size else 0.0)
        r = int(np.sum(d > tol))
    keep_mask = np.zeros(Ad.shape[1], dtype=bool)
    if r > 0:
        # P[:r] are pivoted column indices (in original order space)
        keep_mask[P[:r]] = True
    return Ad[:, keep_mask], keep_mask


def diag_xmx(X: Matrix, M: Matrix) -> NDArray[np.float64]:
    """Compute row-wise diagonal of X M X' as vector of length n.

    Returns an (n,) float64 ndarray where element i equals x_i' M x_i.
    """
    Xd = to_dense(X)
    Md = to_dense(M)
    XM = Xd @ Md
    return np.sum(XM * Xd, axis=1).astype(np.float64)


def hat_diag_r(
    X: Matrix, weights: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Compute leverage values (diagonal of hat matrix) using pivoted QR."""
    Xd = to_dense(X)
    n, _ = Xd.shape
    # build weighted design if needed
    if weights is not None:
        w = _validate_weights(weights, n).reshape(-1, 1)
        Xw = Xd * np.sqrt(w)
    else:
        Xw = Xd
    # rank-revealing QR (QRCP).
    if sla is not None:
        Q, R, _P = sla.qr(Xw, mode="economic", pivoting=True)
    else:
        Q, R, _P = _qr_cp_numpy(Xw)
    diagR = np.abs(np.diag(R))
    tol = 1e-7 * (float(np.max(diagR)) if diagR.size else 0.0)
    r = int(np.sum(diagR > tol))
    if r == 0:
        return np.zeros(n, dtype=np.float64)
    return np.sum(Q[:, :r] * Q[:, :r], axis=1).astype(np.float64)


def hat_diag_stata(
    X: Matrix, weights: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Stata-compliant leverage values.

    Weighted: H = sqrt(W) X (X' W X)^+ X' sqrt(W)
      => diag(H)_i = w_i * x_i' (X' W X)^+ x_i
    Unweighted: H = X (X' X)^+ X'.

    Rank determination: Stata Mata qrsolve convention eta = 1e-13 * trace(|R|) / rows(R).

    Parameters
    ----------
    X : Matrix
        Design matrix (n x p).
    weights : Sequence[float] | None
        Observation weights (n,). If None, uses identity weights.

    Returns
    -------
    h : ndarray
        Leverage values (n,).

    Notes
    -----
        - This implementation follows Stata Mata's `qrsolve` convention for rank
            determination (eta = 1e-13 * trace(|R|) / rows(R)). The weighted form
            above is the practical WLS expression used in Stata regress.

    """
    Xd = to_dense(X).astype(np.float64)
    Ginv = xtwx_inv_via_qr(Xd, weights=weights)  # (X' W X)^+
    base = np.einsum("ij,jk,ik->i", Xd, Ginv, Xd)  # x_i' G^{-1} x_i
    if weights is None:
        return base.astype(np.float64)
    w = _validate_weights(weights, Xd.shape[0]).reshape(-1)
    return (w * base).astype(np.float64)


# Strict dispatcher API: force users to choose R or Stata explicitly
def hat_diag(
    X: Matrix, *, weights: Sequence[float] | None = None, convention: str = "stata",
) -> NDArray[np.float64]:
    """Strict leverage dispatcher (no ambiguity, no NotImplemented).

    Use hat_diag_r for R-compliant leverage (QR on sqrt(W)X, tol=1e-7*max|diag(R)|)
    or hat_diag_stata for Stata-compliant leverage (X(X'WX)^+X', eta=1e-13*trace|R|/rows).

    These differ in HC2/HC3 standard errors and must be explicitly chosen for correctness.

    Parameters
    ----------
    X : Matrix
        Design matrix (n x p).
    weights : Sequence[float] | None
        Observation weights (n,). If None, uses identity weights.
    convention : str, default="stata"
        "R" -> hat_diag_r, "stata" -> hat_diag_stata.

    Returns
    -------
    h : ndarray
        Leverage values (n,).

    """
    conv = convention.strip().lower()
    if conv == "r":
        return hat_diag_r(X, weights=weights)
    if conv == "stata":
        return hat_diag_stata(X, weights=weights)
    msg = "hat_diag: convention must be 'R' or 'stata'."
    raise ValueError(msg)


def effective_f_from_first_stage(
    pi: NDArray[np.float64], Sigma: NDArray[np.float64], Z: Matrix,
) -> float:
    """Effective F-statistic per Montiel-Olea & Pflueger (2013) only.

        This function implements the MOP2013 definition exactly:
            Qzz = Z'Z
            tilde{pi} = Qzz^{1/2} pi
            tilde{Sigma} = Qzz^{1/2} Sigma Qzz^{1/2}
            F_eff = (1/k) * \tilde{pi}' \tilde{Sigma}^{+} \tilde{pi}

    Only the MOP2013 mode is supported. Legacy trace-normalized modes were
    removed to enforce a single, well-specified diagnostic consistent with
    modern applied work and the MOP2013 paper.

    The symmetric part of matrices is used and SVD-based pseudo-inverse
    fallback uses rcond = sqrt(eps) (MASS::ginv compatibility).
    """
    Zd = to_dense(Z)
    Qzz = (Zd.T @ Zd).astype(np.float64)
    k = int(pi.shape[0])
    # Whiten by symmetric square-root of Qzz
    e, V = np.linalg.eigh(Qzz)
    e_clipped = np.maximum(e, 0.0)
    S = V @ np.diag(np.sqrt(e_clipped)) @ V.T
    pi_t = S @ pi
    Sig_t = S @ Sigma @ S.T
    # Solve Sig_t^{-1} * pi_t without forming the inverse when possible
    try:
        L = safe_cholesky(Sig_t, lower=True)
        z = np.linalg.solve(L, pi_t)
        q = np.linalg.solve(L.T, z)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        # Ensure symmetry, then use SVD pseudo-inverse with rcond = sqrt(eps)
        Sig_sym = 0.5 * (Sig_t + Sig_t.T)
        U, s, Vt = np.linalg.svd(Sig_sym, full_matrices=False)
        tol = float(np.sqrt(np.finfo(float).eps)) * (s.max() if s.size else 0.0)
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
        q = (Vt.T * s_inv) @ (U.T @ pi_t)
    val = float((pi_t.T @ q).ravel()[0])
    return val / float(k)


def drop_duplicate_cols(A: Matrix) -> NDArray[np.float64]:
    """Drop exactly duplicate columns from a (dense or sparse) matrix.

    Exact bitwise duplicate detection (leftmost preserved). This function
    performs a strict, R/Stata-like duplicate removal and does not attempt
    numeric tolerance-based deduplication. Inputs are densified and
    strictly checked for finite values to match R/Stata na.fail semantics.
    """
    # Convert to dense and validate inputs
    Ad = np.asarray(to_dense(A), dtype=np.float64, order="C")
    _assert_all_finite(Ad)
    if Ad.size == 0:
        return Ad
    ncols = Ad.shape[1]

    # Use bytes fingerprint for exact equality; keys are bytes of column data
    seen: dict[bytes, int] = {}
    keep_idx: list[int] = []
    for j in range(ncols):
        col_bytes = Ad[:, j].tobytes()
        if col_bytes in seen:
            # duplicate — skip (preserve leftmost)
            continue
        seen[col_bytes] = j
        keep_idx.append(j)

    if not keep_idx:
        return Ad[:, []]
    return Ad[:, keep_idx]


def tdot(X: Matrix) -> NDArray[np.float64]:
    """X' X (dense result)."""
    # Fail fast on NA/Inf in inputs (sparse-aware) to avoid silent densification later
    _assert_all_finite_matrix(X)
    Xd = to_dense(X)
    if _bk is not None and _bk.gpu_enabled():
        try:
            return _bk.dot(Xd.T, Xd).astype(np.float64)
        finally:
            _bk.free_gpu_cache()
    return (Xd.T @ Xd).astype(np.float64)


def gram(X: Matrix, weights: Sequence[float] | None) -> NDArray[np.float64]:
    """Compute A = X' W X with W = diag(w); if weights is None, W=I.
    Avoids forming W explicitly by pre-multiplying rows by sqrt(w).
    Preserves sparsity if possible.
    """
    # Validate inputs (sparse-aware) before any algebra to enforce strict NA/Inf rules
    _assert_all_finite_matrix(X)
    if weights is None:
        if _is_sparse(X):
            return (X.T @ X).toarray().astype(np.float64)
        Xd = to_dense(X)
        if _bk is not None and _bk.gpu_enabled():
            try:
                return _bk.dot(Xd.T, Xd).astype(np.float64)
            finally:
                _bk.free_gpu_cache()
        return (Xd.T @ Xd).astype(np.float64)
    w = _validate_weights(weights, X.shape[0]).reshape(-1, 1)
    _assert_all_finite_matrix(w)
    sqrt_w = np.sqrt(w)
    if _is_sparse(X):
        Xw = X.multiply(sqrt_w)
        return (Xw.T @ Xw).toarray().astype(np.float64)
    Xw = to_dense(X) * sqrt_w
    if _bk is not None and _bk.gpu_enabled():
        try:
            return _bk.dot(Xw.T, Xw).astype(np.float64)
        finally:
            _bk.free_gpu_cache()
    return (Xw.T @ Xw).astype(np.float64)


def xty(X: Matrix, y: Matrix, weights: Sequence[float] | None) -> NDArray[np.float64]:
    """Compute b = X' W y with W = diag(w); if weights is None, W=I. Preserves sparsity if possible."""
    yd = to_dense(y)
    if yd.ndim == 1:
        yd = yd.reshape(-1, 1)
    # Validate inputs (sparse-aware) before algebra to avoid densification surprises
    _assert_all_finite_matrix(X, yd)
    if weights is None:
        if _is_sparse(X) or _is_sparse(yd):
            X_sparse = X if _is_sparse(X) else sp.csc_matrix(to_dense(X))
            y_sparse = yd if _is_sparse(yd) else sp.csc_matrix(yd)
            out = (X_sparse.T @ y_sparse).toarray().astype(np.float64)
            return out.reshape(-1, 1) if out.ndim == 1 else out
        Xd = to_dense(X)
        if _bk is not None and _bk.gpu_enabled():
            try:
                return _bk.dot(Xd.T, yd).astype(np.float64)
            finally:
                _bk.free_gpu_cache()
        out = (Xd.T @ yd).astype(np.float64)
        return out.reshape(-1, 1) if out.ndim == 1 else out
    w = _validate_weights(weights, X.shape[0]).reshape(-1, 1)
    sqrt_w = np.sqrt(w)
    if _is_sparse(X) or _is_sparse(yd):
        X_sparse = X if _is_sparse(X) else sp.csc_matrix(to_dense(X))
        y_sparse = yd if _is_sparse(yd) else sp.csc_matrix(yd)
        Xw = X_sparse.multiply(sqrt_w)
        yw = y_sparse.multiply(sqrt_w)
        out = (Xw.T @ yw).toarray().astype(np.float64)
        return out.reshape(-1, 1) if out.ndim == 1 else out
    Xw = to_dense(X) * sqrt_w
    yw = yd * sqrt_w
    if _bk is not None and _bk.gpu_enabled():
        try:
            return _bk.dot(Xw.T, yw).astype(np.float64)
        finally:
            _bk.free_gpu_cache()
    return (Xw.T @ yw).astype(np.float64)


def gram_full(X: Matrix, W: Matrix) -> NDArray[np.float64]:
    """Compute A = X' W X for a general symmetric (possibly dense) weight matrix W.
    Preserves sparsity when X/W are sparse. Enforces finite/symmetric checks.
    """
    # Convert lazily but check finiteness/sparsity via helpers
    # to_dense will densify sparse matrices; ensure checks before heavy ops
    _assert_all_finite_matrix(X, W)
    Wd = to_dense(W)
    # Symmetry check (strict): mirror R-like diagnostics and suggest remedies
    if not np.allclose(Wd, Wd.T, atol=1e-10, rtol=1e-8):
        raise ValueError(
            "W must be symmetric (and PSD) for GLS/GMM; consider force_psd/nearPD if needed.",
        )
    # PSD check: allow tiny negative eigenvalues from roundoff but reject
    # definitively non-PSD matrices to match R/Stata strict behavior.
    evals = np.linalg.eigvalsh(0.5 * (Wd + Wd.T))
    if np.min(evals) < -1e-12 * max(1.0, np.max(np.abs(evals))):
        raise ValueError(
            "W must be symmetric (and PSD) for GLS/GMM; consider force_psd/nearPD if needed.",
        )
    # Use dense multiplication route; if inputs were sparse, to_dense previously
    # converted them safely. Preserve dtype float64.
    Xd = to_dense(X)
    if _bk is not None and _bk.gpu_enabled():
        try:
            # compute (Wd @ Xd) on CPU (Wd may be dense only), then GPU dot
            WX = (Wd @ Xd).astype(np.float64)
            return _bk.dot(Xd.T, WX).astype(np.float64)
        finally:
            _bk.free_gpu_cache()
    return (Xd.T @ (Wd @ Xd)).astype(np.float64)


def xty_full(X: Matrix, y: Matrix, W: Matrix) -> NDArray[np.float64]:
    """Compute b = X' W y for a general symmetric W. Preserves sparsity where
    possible and enforces finite/symmetry checks.
    """
    _assert_all_finite_matrix(X, y, W)
    Wd = to_dense(W)
    # Symmetry check with R-style guidance
    if not np.allclose(Wd, Wd.T, atol=1e-10, rtol=1e-8):
        raise ValueError(
            "W must be symmetric (and PSD) for GLS/GMM; consider force_psd/nearPD if needed.",
        )
    # PSD check: allow tiny negative due to numerical noise but enforce PSD
    evals = np.linalg.eigvalsh(0.5 * (Wd + Wd.T))
    if np.min(evals) < -1e-12 * max(1.0, np.max(np.abs(evals))):
        raise ValueError(
            "W must be symmetric (and PSD) for GLS/GMM; consider force_psd/nearPD if needed.",
        )
    Xd = to_dense(X)
    yd = to_dense(y)
    if _bk is not None and _bk.gpu_enabled():
        try:
            Wy = (Wd @ yd).astype(np.float64)
            return _bk.dot(Xd.T, Wy).astype(np.float64)
        finally:
            _bk.free_gpu_cache()
    return (Xd.T @ (Wd @ yd)).astype(np.float64)


def uniform_band_from_bootstrap(  # noqa: PLR0913
    beta_hat: NDArray[np.float64],
    boot_draws: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    studentize: bool = True,
    center: str = "multiplier",
    context: str = "did",  # allowed: "did","eventstudy","synthetic_control"
):
    """Construct a simultaneous (uniform) confidence band from bootstrap draws.

    This helper builds a (1-alpha) uniform band for a vector of estimates
    `beta_hat` given bootstrap draws `boot_draws`. It computes the studentized
    sup-norm critical value internally and returns coordinatewise lower/upper
    bounds only. Internal critical values are used to form the band but are
    not exposed through the public API.

    Parameters
    ----------
    beta_hat : (m,) point estimates
    boot_draws : (B, m) bootstrap draws aligned to beta_hat
    alpha : float, default 0.05
    studentize : bool, default True — must be True. The helper currently supports only
                studentized sup-norm multiplier bands to match the project's strict policy.
    center : str, allowed values are "multiplier" or "wild". Pairwise (pairs) bootstrap centering is
             explicitly disallowed by project policy; use multiplier/wild multiplier centering only.

    Returns
    -------
    lower, upper

    """
    # Enforce allowed contexts per project policy
    if str(context).lower() not in {"did", "eventstudy", "synthetic_control", "rct"}:
        raise ValueError(
            "uniform_band_from_bootstrap: allowed only for DID/event-study/synthetic control/RCT.",
        )
    b = np.asarray(beta_hat, dtype=np.float64).reshape(-1)
    Bm = np.asarray(boot_draws, dtype=np.float64)
    if Bm.ndim != 2 or Bm.shape[1] != b.size:
        raise ValueError("boot_draws must be shape (B, m) aligned with beta_hat")
    # Only multiplier/wild centering is implemented
    if str(center).lower() not in {"multiplier", "wild"}:
        raise ValueError("center must be one of {'multiplier', 'wild'}")
    if not bool(studentize):
        raise ValueError(
            "uniform_band_from_bootstrap: only studentized sup-norm bands are supported (studentize=True).",
        )

    # Center bootstrap draws by the point estimate `b` to match the
    # studentized sup-t definition used by boottest / fwildclusterboot and
    # Romano-Wolf style procedures (center at theta, not the bootstrap mean).
    # Bm: (B, m), b: (m,)
    Bc = Bm - b[None, :]

    # Studentized sup-norm: compute coordinatewise SD and max-|t| across draws
    # Use ddof=1 (unbiased sample SD) to match bootstrap_se / estimator conventions.
    s = Bc.std(axis=0, ddof=1)
    s_safe = s.copy()
    s_safe[s_safe <= 0.0] = np.nan
    # Studentized sup-norm draws per replication
    T = np.nanmax(np.abs(Bc / s_safe[None, :]), axis=1)
    # Finite-sample corrected bootstrap quantile (order-statistic, B+1 rule)
    valid = np.isfinite(T)
    if not np.any(valid):
        raise ValueError(
            "No valid studentized bootstrap draws available to form uniform band",
        )
    # Use the finite-sample B+1 rule quantile for bootstrap order-statistics
    try:
        c_alpha = float(finite_sample_quantile_bplus1(T[valid], 1.0 - float(alpha)))
    except (ValueError, FloatingPointError, TypeError):
        # Fallback: numeric-safe partition on the array
        Ta = np.asarray(T[valid], dtype=np.float64)
        Ta.sort()
        B = Ta.size
        if B == 0:
            raise ValueError("No valid bootstrap draws to compute quantile") from None
        k = int(np.ceil((B + 1) * (1.0 - float(alpha))))
        k = max(1, min(k, B))
        c_alpha = float(Ta[k - 1])
    half = c_alpha * s
    lower = b - half
    upper = b + half
    # Per policy: do not expose the internal critical value c_alpha; return only
    # the coordinatewise lower and upper bounds for constructing uniform bands.
    return lower, upper


def _uniform_halfwidth_from_bootstrap_t(
    observed_t: NDArray[np.float64],
    boot_t: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
    context: str = "did",
) -> float:
    """INTERNAL: Return the common studentized half-width for uniform CI construction only.

    This function computes the studentized sup-t critical value used to form
    simultaneous (uniform) confidence bands internally. It is intentionally
    private (leading underscore) and MUST NOT be used as a testing helper or
    to expose p-values/critical values in public APIs. Use the public
    helpers `uniform_band_from_bootstrap` / `bootstrap_uniform_supnorm_halfwidth`
    which consume this internal value to return coordinatewise bounds only.
    """
    conv = str(context).lower()
    if conv not in {"did", "eventstudy", "synthetic_control", "rct"}:
        raise ValueError(
            "_uniform_halfwidth_from_bootstrap_t is CI-only and allowed only for DID/EventStudy/SC/RCT.",
        )
    obs = np.asarray(observed_t, dtype=np.float64)
    Bt = np.asarray(boot_t, dtype=np.float64)
    if Bt.ndim != 2:
        raise ValueError("boot_t must be a 2-D array of shape (B, m)")
    if obs.size and obs.reshape(-1).shape[0] != Bt.shape[1]:
        raise ValueError("observed_t must align with the number of columns in boot_t")
    T = np.nanmax(np.abs(Bt), axis=1) if two_sided else np.nanmax(Bt, axis=1)
    valid = T[np.isfinite(T)]
    if valid.size == 0:
        raise ValueError("No finite bootstrap sup-statistics available to form CI")
    return float(finite_sample_quantile_bplus1(valid, 1.0 - float(alpha)))


def _validate_weights(
    weights: Sequence[float],
    n: int,
    *,
    allow_zero: bool = True,
) -> NDArray[np.float64]:
    """Validate nonnegative weights and return a dense float64 array of shape (n,).

    Parameters
    ----------
    weights : Sequence[float]
        Weight values to validate.
    n : int
        Expected length.
    allow_zero : bool, default True
        Whether to allow zero weights. If False, raises ValueError on any zero weight.

    Returns
    -------
    NDArray[np.float64]
        Validated weights as 1D array.

    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != n:
        msg = "weights length must match n."
        raise ValueError(msg)
    if np.any(~np.isfinite(w)):
        msg = "weights must be finite."
        raise ValueError(msg)
    if np.any(w < 0):
        msg = "weights must be nonnegative."
        raise ValueError(msg)
    if not allow_zero and np.any(w == 0):
        msg = "Zero weights not allowed (allow_zero=False)."
        raise ValueError(msg)
    # Strict econometric semantics: analytic/probability weights must retain at
    # least one effective observation; all-zero weights imply an empty sample.
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0.0:
        raise ValueError("weights must sum to a positive finite value")
    return w


def col_mean(
    X: Matrix, *, weights: Sequence[float] | None = None, ignore_nan: bool = False,
) -> NDArray[np.float64]:
    """Column means with optional observation weights and NA handling.

    Returns array of shape (p,) with column means.
    """
    Xd = to_dense(X)
    if weights is None:
        return (np.nanmean if ignore_nan else np.mean)(Xd, axis=0).astype(np.float64)
    w = _validate_weights(weights, Xd.shape[0]).reshape(-1, 1)
    if ignore_nan:
        # weighted mean with NaN handling (match R weighted.mean semantics)
        mask = np.isfinite(Xd)
        w_expanded = np.where(mask, w, 0.0)
        w_sum = np.sum(w_expanded, axis=0)
        num = np.nansum(Xd * w_expanded, axis=0)
        # If w_sum == 0 (e.g., all values missing), return NaN to match R's behavior
        out = np.full(Xd.shape[1], np.nan, dtype=np.float64)
        ok = w_sum > 0
        out[ok] = num[ok] / w_sum[ok]
        return out
    # No NaN-handling requested: simple weighted mean
    denom = float(np.sum(w))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("weights must sum to a positive finite value")
    return ((Xd * w).sum(axis=0) / denom).astype(np.float64)


def col_var_r(
    X: Matrix, weights: Sequence[float] | None = None, *, method: str = "unbiased",
) -> NDArray[np.float64]:
    """R-strict-compatible column variance (univariate form of stats::cov.wt).

    Definition:
        p_i = w_i / sum(w),  μ = sum(p_i * x_i)
        var_unbiased = ( sum p_i (x_i-μ)**2 ) / ( 1 - sum p_i**2 )
        var_ML       =   sum p_i (x_i-μ)**2
    If `weights` is None, method="unbiased" yields sample variance (ddof=1),
    while method="ML" yields population variance (ddof=0).
    """
    Xd = to_dense(X)
    if Xd.ndim == 1:
        Xd = Xd.reshape(-1, 1)
    n, k = Xd.shape
    if weights is None:
        ddof = 1 if method.lower().startswith("unb") else 0
        return np.var(Xd, axis=0, ddof=ddof).astype(np.float64)
    w = _validate_weights(weights, n).reshape(-1)
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0.0:
        raise ValueError("weights must sum to a positive finite value")
    p = w / wsum
    mu = np.sum(Xd * p[:, None], axis=0)
    num = np.sum(((Xd - mu) ** 2) * p[:, None], axis=0)
    if method.lower().startswith("unb"):
        denom = 1.0 - float(np.sum(p * p))
        if not np.isfinite(denom) or denom <= 0.0:
            return np.full((k,), np.nan, dtype=np.float64)
        out = num / denom
    else:
        out = num
    return out.astype(np.float64)


def col_var_stata_pw(X: Matrix, pweights: Sequence[float]) -> NDArray[np.float64]:
    """Column variance compatible with Stata probability weights (pweights).

    This follows the same definition as R's cov.wt(unbiased): p_i = w_i / sum(w),
    and the unbiased denominator is 1 - sum(p_i**2).
    """
    return col_var_r(X, pweights, method="unbiased")


def safe_cholesky(A: Matrix, *, lower: bool = True) -> NDArray[np.float64]:
    """Strict Cholesky factorization without implicit ridges.
    Raises np.linalg.LinAlgError if not positive definite.
    """
    Ad = to_dense(A)
    Ad = (Ad + Ad.T) * 0.5  # symmetrize
    if _bk is not None and _bk.gpu_enabled():
        try:
            return np.asarray(_bk.cholesky(Ad, lower=lower), dtype=np.float64)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            # Fall back to CPU factorization without modifying A.
            pass
        finally:
            _bk.free_gpu_cache()
    if sla is not None:
        try:
            return sla.cholesky(Ad, lower=lower, check_finite=False)  # type: ignore[union-attr]
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise np.linalg.LinAlgError(f"Cholesky factorization failed: {exc}") from exc
    else:
        try:
            return np.linalg.cholesky(Ad)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(f"Cholesky factorization failed: {exc}") from exc


def chol_psd(A: Matrix) -> NDArray[np.float64]:
    """Cholesky square root for positive semi-definite matrices.
    Enforces PSD by thresholding eigenvalues.
    """
    Ad = to_dense(A)
    Ad = (Ad + Ad.T) * 0.5  # symmetrize
    e, V = np.linalg.eigh(Ad)
    e = np.maximum(e, 0.0)
    return V @ np.diag(np.sqrt(e)) @ V.T


def force_psd(  # noqa: PLR0913
    A: Matrix,
    *,
    tol: float | None = None,
    symmetrize: bool = True,
    log: Callable[[str], None] | None = None,
    # R Matrix::nearPD compatible extra options (defaults chosen to match R)
    keepDiag: bool = False,
    corr: bool = False,
    doDykstra: bool = True,
    conv_tol: float = 1e-7,
    maxit: int = 100,
) -> NDArray[np.float64]:
    """Map A to the nearest positive semi-definite matrix using Higham's
    alternating projections with optional Dykstra correction. This function
    exposes options compatible with R's Matrix::nearPD: `keepDiag`, `corr`,
    `doDykstra`, `conv_tol`, and `maxit`.

    Parameters
    ----------
    A : Matrix
        Input square matrix.
    keepDiag : bool
        If True, restore the original diagonal of A at the end of the
        projection (Matrix::nearPD keepDiag behavior).
    corr : bool
        If True, treat the matrix as a correlation matrix and scale to unit
        diagonal during processing.
    doDykstra : bool
        If True, apply Dykstra's correction during alternating projections.
    conv_tol : float
        Convergence tolerance equivalent to Matrix::nearPD's conv.tol.
    maxit : int
        Maximum number of alternating-projection iterations.

    Returns
    -------
    ndarray
        A symmetric, positive semi-definite approximation of A (float64).

    """
    Ad = to_dense(A)
    if symmetrize:
        Ad = (Ad + Ad.T) * 0.5
    if tol is not None:
        conv_tol = float(tol)
    if corr:
        d = np.sqrt(np.maximum(np.diag(Ad), 0.0))
        Dinv = np.diag(np.where(d > 0, 1.0 / d, 0.0))
        Ad = Dinv @ Ad @ Dinv

    X = Ad.copy()
    Y = np.zeros_like(X)
    for _ in range(int(maxit)):
        # Projection 1: symmetrize
        X = 0.5 * (X + X.T)
        # Dykstra correction (R's nearPD doDykstra default True)
        Rm = X - Y if doDykstra else X
        # Projection 2: PSD projection via eigendecomposition
        eigval, eigvec = np.linalg.eigh(Rm)
        eigval[eigval < 0.0] = 0.0
        X_new = (eigvec * eigval) @ eigvec.T
        if doDykstra:
            Y = X_new - Rm
        # Convergence check using R-like conv.tol semantics
        if np.linalg.norm(X_new - X, ord=np.inf) <= conv_tol * max(
            1.0, np.linalg.norm(X, ord=np.inf),
        ):
            X = X_new
            break
        X = X_new

    if keepDiag:
        np.fill_diagonal(X, np.diag(Ad))
    if corr:
        d = np.sqrt(np.maximum(np.diag(X), 1e-16))
        D = np.diag(np.where(d > 0, 1.0 / d, 0.0))
        X = D @ X @ D
    if log is not None:
        log("nearPD completed (converged or reached maxit).")
    return X.astype(np.float64)


def solve(  # noqa: PLR0913
    A: Matrix,
    B: Matrix,
    *,
    sym_pos: bool = False,
    method: str = "qr",
    svd_rcond: float | None = None,
    rank_policy: str = "stata",
) -> NDArray[np.float64]:
    Bd = to_dense(B)
    Bd = Bd.reshape(-1, 1) if Bd.ndim == 1 else Bd

    # Validate method parameter: qr/svd are exact (default), lsqr/lsmr are opt-in approximate
    if method not in {"qr", "svd"}:
        raise ValueError(
            "solve: method must be 'qr' or 'svd' (approximate lsqr/lsmr are not permitted)",
        )

    # Sparse handling: exact QRCP first, then explicit opt-in iterative, never implicit approximation
    if _is_sparse(A):
        # Priority 1: Exact sparse QRCP via SuiteSparseQR (if available and method="qr")
        if spqr is not None and method == "qr":
            try:
                # Use exact column-pivoted QR for sparse systems (no approximation)
                Ac = A.tocsc()
                sols = []
                for j in range(Bd.shape[1]):
                    bj = Bd[:, j].reshape(-1, 1) if Bd.ndim == 1 else Bd[:, j : j + 1]
                    # Call vectorized_qr_solve with appropriate rank_policy
                    xj = vectorized_qr_solve(
                        Ac,
                        bj,
                        rank_policy=(
                            "R" if rank_policy.upper().startswith("R") else "stata"
                        ),
                    )
                    sols.append(xj.reshape(-1, 1))
                X = np.column_stack(sols).squeeze()
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                return X.astype(np.float64)
            except (
                AttributeError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                Ad = to_dense(A)
                Bd = to_dense(B)
                if Bd.ndim == 1:
                    Bd = Bd.reshape(-1, 1)

        Ad = to_dense(A)
        if Bd.ndim == 1:
            Bd = Bd.reshape(-1, 1)

    # Dense path (or densified sparse)
    Ad = to_dense(A)
    Bd = to_dense(B)
    if Bd.ndim == 1:
        Bd = Bd.reshape(-1, 1)

    # Allow Cholesky only for explicit SPD linear systems (not regression).
    if sym_pos and Ad.shape[0] == Ad.shape[1]:
        try:
            if sla is not None:
                c, low = sla.cho_factor(Ad, lower=True, check_finite=False)  # type: ignore[union-attr]
                X = sla.cho_solve((c, low), Bd, check_finite=False)  # type: ignore[union-attr]
            else:
                L = np.linalg.cholesky(Ad)
                X = np.linalg.solve(L.T, np.linalg.solve(L, Bd))
            return np.asarray(X, dtype=np.float64)
        except (np.linalg.LinAlgError, ValueError):
            sym_pos = False

    if method == "qr":
        return np.asarray(
            _qr_ls_solve(
                Ad, Bd, mode=("R" if rank_policy.upper().startswith("R") else "stata"),
            ),
            dtype=np.float64,
        )

    # SVD explicit path
    if method == "svd":
        U, s, Vt = np.linalg.svd(Ad, full_matrices=False)
        rcond = np.sqrt(np.finfo(float).eps) if svd_rcond is None else float(svd_rcond)
        tol = rcond * (s.max() if s.size else 0.0)
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
        X = (Vt.T * s_inv) @ (U.T @ Bd)
        return np.asarray(X, dtype=np.float64)

    raise RuntimeError("solve: unreachable branch encountered")


def solve_normal_eq(  # noqa: PLR0913
    X: Matrix,
    y: Matrix,
    *,
    weights: Sequence[float] | None = None,
    method: str = "qr",
    svd_rcond: float | None = None,
    rank_policy: str = "stata",
) -> NDArray[np.float64]:
    """Solve (X' W X) b = X' W y via numerically stable routes.

    - Default: QR on X_w = sqrt(W) X (rank-revealing; GEQP3-equivalent).
        - Default behavior: QR on X_w = sqrt(W) X (rank-revealing; GEQP3-equivalent).
            This uses a QRCP-based solver which reproduces R/Stata conventions
            for handling rank deficiency (R -> NA for unidentified coefficients,
            Stata -> 0 for dropped coefficients).
        - SVD (minimum-norm) solutions are NOT used automatically; request
            explicit method="svd" to obtain an SVD minimum-norm solution.
        - Sparse X: delegate to sparse_solve_normal_eq(..., method="qr") for strict parity.
    """
    if _is_sparse(X):
        # If the user explicitly requested SVD on a sparse X, densify and run
        # an exact dense SVD to return the minimum-norm solution. Iterative
        # approximations (LSQR/LSMR) are intentionally disallowed to preserve
        # strict parity with R/Stata behavior when method="svd" is requested.
        if method == "svd":
            Xd = to_dense(X)
            yd = to_dense(y)
            if yd.ndim == 1:
                yd = yd.reshape(-1, 1)
            if weights is not None:
                w = _validate_weights(weights, Xd.shape[0]).reshape(-1, 1)
                sw = np.sqrt(w)
                Xw = Xd * sw
                yw = yd * sw
            else:
                Xw, yw = Xd, yd
            U, s, Vt = np.linalg.svd(Xw, full_matrices=False)
            rcond = (
                np.sqrt(np.finfo(float).eps) if svd_rcond is None else float(svd_rcond)
            )
            tol = rcond * (s.max() if s.size else 0.0)
            s_inv = np.where(s > tol, 1.0 / s, 0.0)
            return (Vt.T * s_inv) @ (U.T @ yw)
        # Default sparse exact route: strict QRCP (SuiteSparseQR) or densify+QR
        return sparse_solve_normal_eq(
            X, y, weights=weights, method="qr", rank_policy=rank_policy,
        )

    Xd = to_dense(X)
    yd = to_dense(y)
    if yd.ndim == 1:
        yd = yd.reshape(-1, 1)
    # Strict finite-input check (R/Stata behavior)
    _assert_all_finite(Xd, yd)

    if weights is not None:
        w = _validate_weights(weights, Xd.shape[0]).reshape(-1, 1)
        sw = np.sqrt(w)
        Xw = Xd * sw
        yw = yd * sw
    else:
        Xw, yw = Xd, yd

    if method == "qr":
        # Use QRCP-based solver that reproduces R or Stata conventions exactly:
        # R-mode returns NA for unidentified coefficients; Stata-mode returns 0
        # for dropped coefficients. SVD minimum-norm is NOT used automatically;
        # request explicit method="svd" to obtain an SVD solution.
        return _qr_ls_solve(
            Xw, yw, mode=("R" if rank_policy.upper().startswith("R") else "stata"),
        )

    if method == "svd":
        U, s, Vt = np.linalg.svd(Xw, full_matrices=False)
        rcond = np.sqrt(np.finfo(float).eps) if svd_rcond is None else float(svd_rcond)
        tol = rcond * (s.max() if s.size else 0.0)
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
        return (Vt.T * s_inv) @ (U.T @ yw)

    raise ValueError("solve_normal_eq: method must be 'qr' or 'svd'")


def group_sum_multi(X: Matrix, codes_matrix: Matrix) -> NDArray[np.float64]:
    """Row-sum X within multi-way clusters defined by a codes matrix.

    Parameters
    ----------
    X : (n x p) matrix
    codes_matrix : (n x R) integer-like labels; each row is an R-tuple defining an intersection cluster.

    Returns
    -------
    (G x p) dense float64 array of sums over distinct rows of codes_matrix (lexicographically sorted).

    """
    Xd = to_dense(X)
    C = np.asarray(codes_matrix)
    if C.ndim != 2:
        msg = "codes_matrix must be 2-dimensional (n x R)"
        raise ValueError(msg)
    if C.shape[0] != Xd.shape[0]:
        msg = "codes_matrix number of rows must match X"
        raise ValueError(msg)
    # Use np.unique over rows to obtain inverse mapping to group indices.
    uniq, inv = np.unique(C, axis=0, return_inverse=True)
    G = uniq.shape[0]
    _n, p = Xd.shape
    out = np.zeros((G, p), dtype=np.float64)
    np.add.at(out, inv, Xd)
    return out


def group_sum(
    X: Matrix, codes: Matrix, *, order: str = "sorted",
) -> NDArray[np.float64]:
    """Sum rows of X within groups defined by a single integer-like codes vector.

    Parameters
    ----------
    X : (n x p) matrix
    codes : (n x 1) integer-like labels

    Returns
    -------
    (G x p) dense float64 array of sums over groups ordered lexicographically (unique sorted labels).

    """
    Xd = to_dense(X)
    codes_arr = np.asarray(codes).reshape(-1)
    if codes_arr.shape[0] != Xd.shape[0]:
        msg = "codes length must match number of rows in X"
        raise ValueError(msg)
    # Enforce single canonical ordering to match R/Stata semantics: 'sorted'
    if order != "sorted":
        msg = "order must be 'sorted'"
        raise ValueError(msg)
    uniq, inv = np.unique(codes_arr, return_inverse=True)
    G = uniq.shape[0]
    p = Xd.shape[1]
    out = np.zeros((G, p), dtype=np.float64)
    np.add.at(out, inv, Xd)
    return out


def pinv(A: Matrix, *, rcond: float | None = None) -> NDArray[np.float64]:
    """Compute Moore-Penrose pseudo-inverse with explicit rcond handling.

    A dense or sparse -> returns dense ndarray.
    """
    Ad = to_dense(A)
    U, s, Vt = np.linalg.svd(Ad, full_matrices=False)
    if rcond is None:
        # unify default with other SVD-based truncation defaults
        rcond = np.sqrt(np.finfo(float).eps)
    tol = float(rcond) * (s.max() if s.size else 0.0)
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    return (Vt.T * s_inv) @ U.T


# ------------------------------------------------------------------
# Additional high-performance / factor-reuse helpers
# ------------------------------------------------------------------


def is_spd(A: Matrix, *, tol: float | None = None) -> bool:
    """Strict symmetric positive definite check using an rcond-style threshold.

    If `tol` is None the threshold is computed as
    eps * max(A.shape) * ||A||_2 (rcond-style), otherwise `tol` is used
    as an absolute threshold on the minimum eigenvalue.
    """
    Ad = to_dense(A)
    Ad = 0.5 * (Ad + Ad.T)
    vals = np.linalg.eigvalsh(Ad)
    if tol is None:
        thr = np.finfo(float).eps * max(Ad.shape) * np.linalg.norm(Ad, 2)
    else:
        thr = float(tol)
    return bool(np.min(vals) > thr)


def chol_factor(A: Matrix, *, lower: bool = True) -> NDArray[np.float64]:
    """Return Cholesky factor (wrapper around `safe_cholesky`)."""
    return safe_cholesky(A, lower=lower)


def chol_solve(
    L: NDArray[np.float64], B: Matrix, *, lower: bool = True,
) -> NDArray[np.float64]:
    """Solve A X = B given Cholesky factor L of A (A = L L')."""
    Bd = to_dense(B)
    if Bd.ndim == 1:
        Bd = Bd.reshape(-1, 1)
    if lower:
        # solve L Y = B, then L' X = Y
        Y = np.linalg.solve(L, Bd)
        X = np.linalg.solve(L.T, Y)
    else:  # pragma: no cover - symmetric branch (rarely used)
        Y = np.linalg.solve(L.T, Bd)
        X = np.linalg.solve(L, Y)
    return np.asarray(X, dtype=np.float64)


def triangular_solve(
    L: NDArray[np.float64], B: Matrix, *, lower: bool = True,
) -> NDArray[np.float64]:
    """Solve L X = B for triangular L (single triangular solve, not full SPD solve).

    This is for whitening: given Omega = L L', compute L^{-1} B (not Omega^{-1} B).
    Use chol_solve for Omega^{-1} B when you need the full inverse.
    """
    Bd = to_dense(B)
    if Bd.ndim == 1:
        Bd = Bd.reshape(-1, 1)
    if sla is not None:
        X = sla.solve_triangular(L, Bd, lower=lower, check_finite=False)
    else:
        X = np.linalg.solve(L, Bd)
    return np.asarray(X, dtype=np.float64)


def block_solve_posdef(
    A: Matrix, B_list: Sequence[Matrix],
) -> list[NDArray[np.float64]]:
    """Solve A X_j = B_j for multiple RHS blocks with one factorization.

    Parameters
    ----------
    A : SPD matrix (n x n)
    B_list : iterable of matrices/vectors with n rows

    Returns
    -------
    List of dense float64 solution arrays.

    """
    L = chol_factor(A)
    return [chol_solve(L, B) for B in B_list]


def batched_tdot_list(X_list: Sequence[Matrix]) -> list[NDArray[np.float64]]:
    """Compute X'X for each X in a sequence with vectorized fast-paths.

    For dense inputs with identical column dimension this function stacks
    the inputs and uses a single einsum to compute all X_i' X_i in one
    fast BLAS-backed operation. When inputs are sparse or shapes differ
    it falls back to a memory-conscious loop that preserves sparsity.

    This version operates on a list of separate matrices (for non-bootstrap contexts).

    Returns
    -------
    list of (p x p) dense float64 arrays, one per input in X_list.

    """
    # Fast path: all dense numpy arrays and matching column dims -> stack + einsum
    if not X_list:
        return []
    try:
        dense_list = [to_dense(X) for X in X_list]
    except (RuntimeError, ValueError, TypeError):
        # to_dense may raise on sparse when allow_densify is False; fall back
        return [tdot(X) for X in X_list]

    shapes = [d.shape for d in dense_list]
    # Ensure all are 2-D and share same number of columns
    if all(len(s) == 2 for s in shapes) and len({s[1] for s in shapes}) == 1:
        # Stack into (K, n, p) where n may differ across inputs if they were ragged;
        # require same n for vectorized path.
        n_sizes = {s[0] for s in shapes}
        if len(n_sizes) == 1:
            stacked = np.stack(dense_list, axis=0)  # K x n x p
            # compute X_i' X_i for all i in one call: 'kni,knj->kij'
            res = np.einsum("kni,knj->kij", stacked, stacked)
            return [res[k].astype(np.float64) for k in range(res.shape[0])]

    # Fallback: preserve sparsity / differing shapes
    return [tdot(X) for X in X_list]


def batched_crossprod_list(
    X_list: Sequence[Matrix], Y: Matrix,
) -> list[NDArray[np.float64]]:
    """Compute X_i' Y for a common Y and sequence {X_i} with a fast-path.

    If all X_i are dense and share the same number of rows we vectorize the
    computation via einsum; otherwise fall back to the sparse-aware loop.

    This version operates on a list of separate matrices (for non-bootstrap contexts).
    """
    if not X_list:
        return []
    Yd = to_dense(Y)
    try:
        dense_list = [to_dense(X) for X in X_list]
    except (RuntimeError, ValueError, TypeError):
        return [crossprod(X, Yd) for X in X_list]

    shapes = [d.shape for d in dense_list]
    # All X_i must be 2-D and have same number of rows to vectorize
    if all(len(s) == 2 for s in shapes) and len({s[0] for s in shapes}) == 1:
        stacked = np.stack(dense_list, axis=0)  # K x n x p
        # Compute X_i' Y via einsum: 'kni,nj->kij' -> (K, p, q)
        res = np.einsum("kni,nj->kij", stacked, Yd)
        return [res[k].astype(np.float64) for k in range(res.shape[0])]

    # Fallback: preserve sparsity / differing shapes
    return [crossprod(X, Yd) for X in X_list]


def block_diag(mats: Sequence[Matrix]) -> Matrix:
    """Construct a (sparse-aware) block diagonal matrix.

    If any component is sparse and SciPy is available, builds a sparse
    block diagonal matrix; otherwise falls back to dense assembly.
    Note: Dense path may consume excessive memory for large matrices;
    prefer sparse inputs when possible.
    """
    if not mats:
        return np.zeros((0, 0), dtype=np.float64)
    use_sparse = any(_is_sparse(M) for M in mats)
    if use_sparse and sp is not None:  # type: ignore[union-attr]
        blocks = [M if _is_sparse(M) else sp.csc_matrix(to_dense(M)) for M in mats]  # type: ignore[attr-defined]
        return sp.block_diag(blocks, format="csc")  # type: ignore[attr-defined]
    # Dense path
    sizes = [to_dense(M).shape[0] for M in mats]
    total = sum(sizes)
    out = np.zeros((total, total), dtype=np.float64)
    offset = 0
    for M in mats:
        Md = to_dense(M)
        n = Md.shape[0]
        out[offset : offset + n, offset : offset + n] = Md
        offset += n
    return out


def symmetric_matrix_sqrt(A: Matrix) -> NDArray[np.float64]:
    """Compute symmetric square root of a symmetric positive semidefinite matrix using eigendecomposition.
    Returns A^{1/2} such that A^{1/2} @ A^{1/2} = A.
    For PSD matrices, eigenvalues are clipped to non-negative.
    """
    Ad = to_dense(A)
    Ad = (Ad + Ad.T) / 2.0  # symmetrize
    evals, evecs = np.linalg.eigh(Ad)
    evals = np.maximum(evals, 0.0)  # clip negative eigenvalues to 0
    sqrt_evals = np.sqrt(evals)
    return (evecs * sqrt_evals) @ evecs.T


def ar1_correlation_by_groups(
    times: NDArray[np.int64], seg_id: NDArray[np.int64], *, rho: float,
) -> NDArray[np.float64]:
    """Build a block-diagonal AR(1) correlation matrix R where observations
    belonging to the same segment (seg_id) form an AR(1) sub-block indexed by
    their `times`. For observations in different segments the correlation is
    zero. The (i,j) entry within a segment equals rho^{|t_i - t_j|}.

    This is used for DID/EventStudy/DDD estimators with heterogeneous treatment timing.

    Parameters
    ----------
    times : (n,) integer-like array of time indices (not necessarily contiguous)
    seg_id : (n,) integer-like array of segment identifiers
    rho : float in (-1,1)

    Returns
    -------
    R : (n x n) dense ndarray of float64 containing the AR(1) correlation structure.

    """
    # Strict parameter domain check (R/Stata conventions for AR(1))
    if not (-1.0 < float(rho) < 1.0):
        raise ValueError(f"ar1_correlation_by_groups: rho must be in (-1,1); got {rho}")
    times_a = np.asarray(times).reshape(-1)
    seg_a = np.asarray(seg_id).reshape(-1)
    if times_a.shape[0] != seg_a.shape[0]:
        msg = "times and seg_id must have the same length"
        raise ValueError(msg)
    n = times_a.shape[0]
    R = np.zeros((n, n), dtype=np.float64)
    # vectorized block assembly per unique segment to avoid Python loops per pair
    uniq_segs = np.unique(seg_a)
    for g in uniq_segs:
        idx = np.nonzero(seg_a == g)[0]
        if idx.size == 0:
            continue
        tt = times_a[idx]
        # pairwise absolute differences
        D = np.abs(tt[:, None] - tt[None, :])
        # rho^D yields 1 on diagonal and appropriate decay off-diagonal
        R_block = (rho**D).astype(np.float64, copy=False)
        R[np.ix_(idx, idx)] = R_block
    return R


def ar1_covariance_from_weights_by_groups(
    times: NDArray[np.int64],
    seg_id: NDArray[np.int64],
    *,
    rho: float,
    weights: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Assemble covariance matrix Sigma = Lambda^{-1/2} R Lambda^{-1/2} where
    R is the AR(1) correlation matrix constructed from `times` and `seg_id`,
    and Lambda = diag(w) for supplied inverse-variance weights `w`.

    If `weights` is None, this returns the pure AR(1) correlation R.

    This helper centralizes the creation of AR1+diag covariance matrices for
    DID/EventStudy/DDD estimators so that callers do not manipulate dense matrices directly.
    """
    R = ar1_correlation_by_groups(times, seg_id, rho=rho)
    if weights is None:
        return R
    # Precision weights must be strictly positive to avoid Sigma blow-up.
    w = _validate_weights(weights, R.shape[0], allow_zero=False).reshape(-1)
    inv_sqrt = 1.0 / np.sqrt(w)
    # Efficient outer product scaling: Sigma_ij = R_ij * inv_sqrt_i * inv_sqrt_j
    scale = inv_sqrt[:, None] * inv_sqrt[None, :]
    return R * scale


def svd(
    A: Matrix, full_matrices: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Singular value decomposition of a matrix with optional GPU fast-path.

    Returns U, s, Vt such that A = U @ np.diag(s) @ Vt. When the GPU backend
    is enabled, delegates to the backend SVD and converts results back to CPU
    float64 arrays to preserve the public API.
    """
    Ad = to_dense(A)
    if _bk is not None and _bk.gpu_enabled():
        try:
            U, s, Vt = _bk.svd(Ad, full_matrices=full_matrices)
            return np.asarray(U, dtype=np.float64), np.asarray(s, dtype=np.float64), np.asarray(Vt, dtype=np.float64)
        finally:
            _bk.free_gpu_cache()
    return np.linalg.svd(Ad, full_matrices=full_matrices)


def eigvalsh(A: Matrix) -> NDArray[np.float64]:
    """Eigenvalues of a symmetric matrix with optional GPU fast-path.
    """
    Ad = to_dense(A)
    if _bk is not None and _bk.gpu_enabled():
        try:
            return np.asarray(_bk.eigvalsh(Ad), dtype=np.float64)
        finally:
            _bk.free_gpu_cache()
    return np.linalg.eigvalsh(Ad)


def eigvals(A: Matrix) -> NDArray[np.complex128]:
    """Eigenvalues of a general matrix.

    Note: No GPU fast-path provided here because the backend focuses on
    symmetric (Hermitian) eigensolvers used in statistical routines.
    """
    Ad = to_dense(A)
    return np.linalg.eigvals(Ad)


def norm(
    A: Matrix,
    *,
    order: float | str | None = None,
    **kwargs,
) -> float:
    """Matrix/vector norm with optional GPU fast-path.

    Compatibility: accepts both ``ord=`` (NumPy-style) and ``order=``. If both
    are provided, ``order`` takes precedence.
    """
    Ad = to_dense(A)
    # Accept NumPy-style ord= alias via kwargs, with order taking precedence
    eff_order = order if order is not None else kwargs.get("ord")
    # For 2-norm on large dense matrices, GPU can help; for other norms,
    # rely on NumPy for exact parity and simplicity.
    if _bk is not None and _bk.gpu_enabled() and (eff_order in (None, 2)):  # type: ignore[comparison-overlap]
        try:
            # Use SVD-based 2-norm via backend if available
            _U, s, _Vt = _bk.svd(Ad, full_matrices=False)
            return float(np.max(np.asarray(s, dtype=np.float64)) if np.size(s) else 0.0)
        finally:
            _bk.free_gpu_cache()
    return float(np.linalg.norm(Ad, ord=eff_order))


def eigh(A: Matrix) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenvalues and eigenvectors of a symmetric matrix with GPU fast-path.
    """
    Ad = to_dense(A)
    gpu_result: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    if _bk is not None and _bk.gpu_enabled():
        try:
            xp = _bk.xp_for([Ad])  # type: ignore[attr-defined]
            Ad_dev = _bk.asarray(Ad, xp=xp, dtype=np.float64)  # type: ignore[attr-defined]
            w, v = xp.linalg.eigh(Ad_dev)  # type: ignore[attr-defined]
            gpu_result = (np.asarray(_bk.to_cpu(w), dtype=np.float64), np.asarray(_bk.to_cpu(v), dtype=np.float64))  # type: ignore[attr-defined]
        except (RuntimeError, ValueError, TypeError, AttributeError):  # pragma: no cover - fallback on backend error
            gpu_result = None
        finally:
            _bk.free_gpu_cache()
        if gpu_result is not None:
            return gpu_result
    return np.linalg.eigh(Ad)


def eig_tol(A: Matrix) -> float:
    """Compute a scale-aware tolerance for eigenvalue filtering.

    Returns eps * max(eigenvalue) * multiplier, where multiplier accounts
    for matrix size. This is used to determine which eigenvalues are
    numerically zero relative to the dominant eigenvalue.

    Parameters
    ----------
    A : Matrix
        Symmetric matrix for tolerance computation.

    Returns
    -------
    float
        Eigenvalue tolerance threshold.

    Notes
    -----
    This matches common numerical linear algebra practice for determining
    the numerical rank of a symmetric matrix from its eigendecomposition.
    Similar to MATLAB's eps(max(evals)) * max(size(A)).

    """
    Ad = to_dense(A)
    if Ad.size == 0:
        return np.finfo(float).eps
    # For symmetric matrices, use absolute maximum eigenvalue as scale
    try:
        evals = np.linalg.eigvalsh(Ad)
        max_eval = float(np.max(np.abs(evals)))
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: use matrix norm as scale
        max_eval = float(np.linalg.norm(Ad, ord=2))

    if max_eval == 0.0:
        return np.finfo(float).eps

    # Tolerance: eps * max_eigenvalue * size_multiplier
    # Use max dimension as multiplier (conservative, matches MATLAB convention)
    size_multiplier = max(Ad.shape[0], Ad.shape[1])
    return np.finfo(float).eps * max_eval * size_multiplier


def min_gen_eig(A: Matrix, B: Matrix, *, rcond: float | None = None) -> float:
    """Compute the minimum generalized eigenvalue of the matrix pair (A, B).

    This routine returns the smallest generalized eigenvalue lambda solving
        A v = lambda B v

    Behavior:
    - If B is (numerically) SPD, returns the minimum eigenvalue of
      B^{-1/2} A B^{-1/2}.
    - If B is positive semi-definite or singular, restricts the problem to
      the image (column) space of B using its SVD and returns the minimum
      eigenvalue of
        C = Sigma_r^{-1/2} U_r' A U_r Sigma_r^{-1/2}.

    The above projection yields the correct generalized eigenvalue on the
    image of B and agrees with the SPD formula when B is full-rank.

    Parameters
    ----------
    A : Matrix
        Square symmetric matrix (k x k).
    B : Matrix
        Square symmetric matrix (k x k), may be singular or PSD.
    rcond : float | None
        Relative cutoff for singular values of B. If None, a default
        rcond = sqrt(eps) is used.

    """
    Ad = to_dense(A).astype(np.float64)
    Bd = to_dense(B).astype(np.float64)
    # Symmetrize to absorb numerical noise
    Ad = 0.5 * (Ad + Ad.T)
    Bd = 0.5 * (Bd + Bd.T)

    # --- SPD fast path (exact & literature-consistent) ---
    try:
        L = safe_cholesky(Bd, lower=True)
        # Construct C = B^{-1/2} A B^{-1/2} without explicit inverse for stability
        # Use solve to compute L^{-1} A L^{-T} via two triangular solves
        # Solve L Y = A  => Y = L^{-1} A
        Y = _npsolve(L, Ad)
        # Solve L C^T = Y^T => C = (L^{-1} A L^{-T})
        C = _npsolve(L, Y.T).T
        C = 0.5 * (C + C.T)
        return float(np.min(np.linalg.eigvalsh(C)))
    except (RuntimeError, np.linalg.LinAlgError, ValueError):
        Bd = to_dense(B)
        if np.allclose(Bd, np.zeros_like(Bd), atol=1e-12):
            return float("nan")

    U, s, _Vt = np.linalg.svd(Bd, full_matrices=True)
    if rcond is None:
        rcond = np.sqrt(np.finfo(float).eps)
    tol = float(rcond) * (np.max(s) if s.size else 0.0)
    keep = s > tol
    r = int(np.sum(keep))
    if r == 0:
        # B is (numerically) zero -> generalized eigenvalues undefined
        return float("nan")

    Ur = U[:, keep]
    sr = s[keep]
    inv_sqrt = 1.0 / np.sqrt(sr)
    S_inv_sqrt = np.diag(inv_sqrt)

    # Project and normalize on the image of B: C = S^{-1/2} U' A U S^{-1/2}
    C = S_inv_sqrt @ (Ur.T @ Ad @ Ur) @ S_inv_sqrt
    # C should be symmetric by construction; compute its smallest eigenvalue
    return float(np.min(np.linalg.eigvalsh(C)))


def batched_solve_normal_eq_list(
    X_list: Sequence[Matrix],
    y_list: Sequence[Matrix],
    *,
    weights_list: Sequence[Sequence[float] | None] | None = None,
    method: str = "qr",
) -> list[NDArray[np.float64]]:
    """Batched solve_normal_eq for multiple (X, y) pairs (list version).

    Useful for non-bootstrap contexts with separate data matrices.
    Each pair is solved independently.

    This version operates on lists of separate (X, y) pairs.
    For bootstrap with shared X and 3D Y, use batched_solve_normal_eq instead.
    """
    if len(X_list) != len(y_list):
        msg = "X_list and y_list must have the same length."
        raise ValueError(msg)
    results = []
    for i, (X, y) in enumerate(zip(X_list, y_list)):
        w = weights_list[i] if weights_list else None
        beta = solve_normal_eq(X, y, weights=w, method=method)
        results.append(beta)
    return results


def batched_solve_normal_eq_parallel(
    X_list: Sequence[Matrix],
    y_list: Sequence[Matrix],
    *,
    weights_list: Sequence[Sequence[float] | None] | None = None,
    method: str = "qr",
    n_jobs: int = -1,
) -> list[NDArray[np.float64]]:
    """Parallel batched solve_normal_eq for multiple (X, y) pairs using multiprocessing.

    Useful for non-bootstrap contexts with separate data matrices.
    Each pair is solved independently in parallel.

    This version operates on lists of separate (X, y) pairs.
    For bootstrap with shared X and 3D Y, use batched_solve_normal_eq instead.
    """
    if len(X_list) != len(y_list):
        raise ValueError("X_list and y_list must have the same length.")
    if weights_list is not None and len(weights_list) != len(X_list):
        raise ValueError("weights_list length must match X_list.")
    tasks = [
        (X, y, (weights_list[i] if weights_list else None), method, "stata")
        for i, (X, y) in enumerate(zip(X_list, y_list))
    ]
    maxw = (os.cpu_count() or 1) if (n_jobs is None or n_jobs == -1) else int(n_jobs)
    maxw = max(1, min(maxw, len(tasks)))

    with ProcessPoolExecutor(max_workers=maxw) as ex:
        # executor.map preserves input order; use module-level picklable worker
        return list(ex.map(_solve_normal_eq_worker_top, tasks))


def xtwx_inv_via_qr(
    X: Matrix,
    weights: Sequence[float] | None = None,
    *,
    svd_rcond: float | None = None,
    rank_policy: str = "stata",
) -> NDArray[np.float64]:
    """Compute a stable inverse/generalized-inverse of $X'WX$ via pivoted QR on $X_w=\sqrt{W}X$.

        Returns a dense (p x p) ndarray.

        Rank-deficient behavior depends on ``rank_policy``:
        - ``rank_policy='stata'``: return a QR-based generalized inverse consistent with
            Stata/Mata `qrsolve` dropping unidentified directions (i.e., the dropped
            subspace contributes zeros in the generalized inverse).
        - ``rank_policy='R'``: return the Moore–Penrose pseudo-inverse implied by an
            SVD tolerance (MASS::ginv-style).

    This avoids forming Gram explicitly and prevents squaring condition numbers.

    Parameters
    ----------
    X : Matrix
        Design matrix (n x p).
    weights : Sequence[float] | None
        Observation weights (n,). If None, uses identity weights.
    svd_rcond : float | None
        SVD tolerance for pseudo-inverse (used only in rank-deficient fallback).
    rank_policy : str, default="stata"
        "R" (R lm.fit tol=1e-7*max|diag(R)|) or "stata" (Mata eta).

    Returns
    -------
    ndarray
        (X' W X)^{-1} as (p x p) matrix.

    """
    Xd = to_dense(X)
    _assert_all_finite(Xd)
    n, p = Xd.shape
    if weights is not None:
        w = _validate_weights(weights, n).reshape(-1, 1)
        Xw = Xd * np.sqrt(w)
    else:
        Xw = Xd

    # Pivoted QR provides a stable route to a full-rank inverse and also gives a
    # deterministic dropped-subspace convention for Stata-style generalized inverses.
    try:
        if sla is not None:
            _Q, R, P = sla.qr(Xw, mode="economic", pivoting=True)
        else:
            _Q, R, P = _qr_cp_numpy(Xw)
        diagR = np.abs(np.diag(R))
        r = _rank_from_diag(
            diagR,
            p,
            mode=("R" if str(rank_policy).lower().startswith("r") else "stata"),
        )
        if r == p:
            identity_p = np.eye(p, dtype=np.float64)
            if sla is not None:
                Rinv = sla.solve_triangular(
                    R, identity_p, lower=False, check_finite=False,
                )
            else:
                Rinv = np.linalg.solve(R, identity_p)
            A = Rinv @ Rinv.T
            invp = np.argsort(P[:p])
            A = A[invp][:, invp]
            return A.astype(np.float64)
        if r == 0:
            return np.zeros((p, p), dtype=np.float64)

        # Rank-deficient: Stata convention is a QR-based generalized inverse that
        # sets the dropped directions to zero. R convention uses MP pseudo-inverse.
        if not str(rank_policy).lower().startswith("r"):
            R11 = np.asarray(R[:r, :r], dtype=np.float64)
            identity_r = np.eye(r, dtype=np.float64)
            if sla is not None:
                R11_inv = sla.solve_triangular(
                    R11, identity_r, lower=False, check_finite=False,
                )
            else:
                R11_inv = np.linalg.solve(R11, identity_r)
            A_piv = np.zeros((p, p), dtype=np.float64)
            A_piv[:r, :r] = R11_inv @ R11_inv.T
            invp = np.argsort(P[:p])
            A = A_piv[invp][:, invp]
            return A.astype(np.float64)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        pass

    _U, s, Vt = np.linalg.svd(Xw, full_matrices=False)
    rcond = np.sqrt(np.finfo(float).eps) if svd_rcond is None else float(svd_rcond)
    svd_tol = rcond * (s.max() if s.size else 0.0)
    s_inv2 = np.where(s > svd_tol, 1.0 / (s * s), 0.0)
    A = (Vt.T * s_inv2) @ Vt
    return np.asarray(A, dtype=np.float64)


def batched_crossprod_pairs(
    X_list: Sequence[Matrix],
    y_list: Sequence[Matrix],
) -> list[NDArray[np.float64]]:
    """Batched crossprod for multiple (X_i, y_i) pairs (no name collision)."""
    return [crossprod(X, y) for X, y in zip(X_list, y_list)]


def sparse_solve_normal_eq(
    X: Matrix,
    y: Matrix,
    *,
    weights: Sequence[float] | None = None,
    method: str = "qr",
    rank_policy: str = "stata",
) -> NDArray[np.float64]:
    """Solve normal equations for sparse matrices using strict QR route only.
    Only QR is permitted for strict R/Stata compliance; iterative solvers are not allowed.
    """
    if not _is_sparse(X):
        return solve_normal_eq(X, y, weights=weights, method="qr")  # strict parity

    if method != "qr":
        raise ValueError(
            "sparse_solve_normal_eq: only method='qr' is permitted (lsqr/lsmr are not allowed)",
        )

    # If SuiteSparseQR is available, perform exact sparse QRCP without densifying
    if spqr is not None:
        Xc = X.tocsc()
        y_arr = to_dense(y)
        # Enforce strict finite-inputs check (R/Stata na.fail parity) without densifying Xc
        _assert_all_finite_matrix(Xc, y_arr)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        w = (
            None
            if weights is None
            else _validate_weights(weights, Xc.shape[0]).reshape(-1, 1)
        )
        if w is None:
            qr_res = spqr.qr(Xc)
            Qs, Rs, E = qr_res[0], qr_res[1], qr_res[2]
            c = Qs.T @ y_arr
        else:
            Xw = Xc.multiply(np.sqrt(w))
            qr_res = spqr.qr(Xw)
            Qs, Rs, E = qr_res[0], qr_res[1], qr_res[2]
            c = Qs.T @ (y_arr * np.sqrt(w))

        # Determine numerical rank from diagonal of R
        try:
            diagR = np.abs(np.asarray(Rs.diagonal()).ravel())
        except (AttributeError, TypeError, ValueError):
            diagR = np.abs(np.diag(np.asarray(Rs)))
        r = _rank_from_diag(
            diagR,
            Xc.shape[1],
            mode=("R" if rank_policy.upper().startswith("R") else "stata"),
        )
        n = Xc.shape[1]
        P = np.array(E).ravel().astype(np.int64)

        # Prepare output filled per policy: R -> NaN, Stata -> 0
        fill_val = np.nan if rank_policy.upper().startswith("R") else 0.0
        out = np.full((n, c.shape[1]), fill_val, dtype=np.float64)

        if r > 0:
            R11 = Rs[:r, :r]
            R11 = R11.toarray() if hasattr(R11, "toarray") else np.asarray(R11)
            rhs = c[:r, :]
            beta_r = (
                sla.solve_triangular(R11, rhs, lower=False, check_finite=False)
                if sla is not None
                else np.linalg.solve(R11, rhs)
            )
            out[P[:r], :] = beta_r

        return out

    # No SuiteSparseQR: densify and reuse dense QR path (strict parity, no iteration)
    return solve_normal_eq(
        to_dense(X), y, weights=weights, method="qr", rank_policy=rank_policy,
    )


# ============================================================================
# Additional AR(1) / SAR Functions (renamed to avoid collisions)
# ============================================================================


def ar1_correlation_n(rho: float, n: int) -> NDArray[np.float64]:
    """Construct the n x n AR(1) correlation matrix with parameter rho:
    [R]_{ij} = rho^{|i-j|},  rho ∈ (-1,1).
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"AR(1) parameter rho must be in (-1, 1), got {rho}")
    idx = np.arange(n, dtype=int)
    D = np.abs(idx[:, None] - idx[None, :])
    return (rho**D).astype(np.float64)


# --- FIX 1B: new function (present in __all__) ---
def sar_errors_covariance(W: Matrix, rho: float) -> NDArray[np.float64]:
    """Compute Omega = (I - rho W)^{-1} (I - rho W')^{-1} in a numerically stable way.

    Dense path avoids constructing A^{-1} explicitly by using the identity
        Omega = (A A')^{-1} where A = (I - rho W),
    and solving the SPD system via Cholesky on S = A A'. Sparse path uses
    multi-RHS exact solves without forming an explicit inverse.
    """
    Wd = to_dense(W)
    n, m = Wd.shape
    if n != m:
        raise ValueError("W must be square")
    # Admissible rho domain for A = I - rho W.
    # If W is (numerically) symmetric, use exact bounds implied by eigenvalues:
    #   rho ∈ (min(1/λ_min, 1/λ_max), max(1/λ_min, 1/λ_max)).
    # Otherwise, fall back to a safe sufficient condition |rho| < 1 / max|λ|.
    tol = np.sqrt(np.finfo(float).eps)
    is_sym = np.allclose(Wd, Wd.T, atol=1e-12, rtol=1e-12)
    if is_sym:
        lam = np.linalg.eigvalsh(Wd)
        lam_min, lam_max = float(lam.min()), float(lam.max())
        a = -np.inf if lam_max == 0.0 else 1.0 / lam_max
        b = np.inf if lam_min == 0.0 else 1.0 / lam_min
        lower, upper = (a, b) if a < b else (b, a)
        if not (lower + tol < rho < upper - tol):
            raise ValueError(
                f"sar_errors_covariance: rho={rho} outside admissible interval ({lower}, {upper}) implied by eigenvalues of symmetric W.",
            )
    else:
        # Non-symmetric: spectral-radius-based sufficient condition
        if _is_sparse(W) and sp is not None:
            if _sparse_eigs is None:
                raise RuntimeError("scipy.sparse.linalg.eigs unavailable")
            try:
                lam_max_abs = np.abs(
                    _sparse_eigs(W, k=1, which="LM", return_eigenvectors=False)[0],
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                lam_max_abs = np.max(np.abs(np.linalg.eigvals(Wd)))
        else:
            lam_max_abs = np.max(np.abs(np.linalg.eigvals(Wd)))
        if not (np.abs(rho) < (1.0 / (lam_max_abs + tol))):
            raise ValueError(
                "sar_errors_covariance: |rho| too large given spectral radius of W.",
            )
    identity = np.eye(n, dtype=np.float64)
    A = identity - rho * Wd
    # Sparse-aware exact solve without forming explicit inverse
    if _is_sparse(W):
        if _spsolve is None or sp is None:
            raise RuntimeError("scipy.sparse.linalg.spsolve unavailable")
        try:
            A_csc = (sp.eye(n, format="csc") - rho * W).astype(np.float64)  # type: ignore[union-attr]
            Ainv = _spsolve(
                A_csc, identity,
            )  # multi-RHS solve (computes explicit A^{-1} columns)
            return (Ainv @ Ainv.T).astype(np.float64)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            # Fall back to dense route below
            Wd_local = to_dense(W)
            A = identity - rho * Wd_local
    # Dense SPD route on S = A A'
    S = A @ A.T
    try:
        if sla is not None:
            c, low = sla.cho_factor(S, lower=True, check_finite=False)  # type: ignore[union-attr]
            Omega = sla.cho_solve((c, low), identity, check_finite=False)  # type: ignore[union-attr]
        else:
            L = np.linalg.cholesky(S)
            # Solve L L' X = I ⇒ first L Y = I, then L' X = Y
            Y = np.linalg.solve(L, identity)
            Omega = np.linalg.solve(L.T, Y)
        return np.asarray(Omega, dtype=np.float64)
    except (np.linalg.LinAlgError, ValueError):
        # As a last resort, fall back to two dense triangular solves constructing A^{-1} implicitly
        Ainv = np.linalg.solve(A, identity)
        return (Ainv @ Ainv.T).astype(np.float64)
    # Already handled above; unreachable code removed


# ============================================================================
# Parallel / Vectorized Matrix Operations for Bootstrap
# ============================================================================


def parallel_matrix_multiply(
    A: Matrix,
    B: Matrix,
    *,
    n_jobs: int | None = None,
    chunk_size: int = 1000,
) -> Matrix:
    """Parallel matrix multiplication for large matrices.

    Uses ThreadPoolExecutor to parallelize matrix multiplication
    by splitting along rows for better performance on large matrices.

    Parameters
    ----------
    A : Matrix
        Left matrix (m x k).
    B : Matrix
        Right matrix (k x n).
    n_jobs : int, optional
        Number of parallel jobs. Defaults to min(cpu_count, 4).
    chunk_size : int, optional
        Row chunk size for parallelization.

    Returns
    -------
    C : Matrix
        Product A @ B (m x n).

    Notes
    -----
    Only beneficial for very large matrices. For typical econometric applications
    (n ~ 10^6, p ~ 100), standard NumPy @ is sufficient.

    """
    if n_jobs is None:
        n_jobs = min(multiprocessing.cpu_count(), 4)

    A_dense = to_dense(A)
    B_dense = to_dense(B)

    m, k = A_dense.shape
    k2, _n = B_dense.shape
    if k != k2:
        msg = "Incompatible dimensions for matrix multiplication"
        raise ValueError(msg)

    # For small matrices, use standard multiplication
    if m <= chunk_size:
        return A_dense @ B_dense

    # Parallel chunked multiplication
    def _multiply_chunk(start_idx: int, end_idx: int) -> NDArray[np.float64]:
        return A_dense[start_idx:end_idx] @ B_dense

    chunks = [(i, min(i + chunk_size, m)) for i in range(0, m, chunk_size)]

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_multiply_chunk, s, e): s for (s, e) in chunks}
        out_ordered = [(futures[fut], fut.result()) for fut in as_completed(futures)]
    out_ordered.sort(key=lambda t: t[0])
    # Concatenate results in original row order
    return np.vstack([v for _, v in out_ordered])


def vectorized_qr_solve(
    A: Matrix,
    b: NDArray[np.float64],
    *,
    rcond: float | None = None,
    rank_policy: str = "stata",
) -> NDArray[np.float64]:
    """Vectorized QR-based linear system solver strictly matching R/Stata QRCP.

    This routine uses column-pivoted QR (GEQP3 via scipy.linalg.qr for dense,
    or SuiteSparseQR via sparseqr for sparse) and determines numerical rank
    using either R's lm.fit rule (tol = 1e-7 * max(|diag(R)|)) or Stata's
    Mata rule (eta = 1e-13 * trace(|R|)/rows(R)) depending on ``rank_policy``.
    When rank deficiency is detected, behavior is policy-specific: R-policy
    returns NA (np.nan) for unidentified coefficients; Stata-policy returns 0
    for dropped coefficients. Full-rank solves use triangular back-substitution
    and results are unpermuted to the original variable ordering.

        For sparse matrices:
            - If SuiteSparseQR (sparseqr) is available, use exact sparse QRCP.
            - Otherwise, densify and use the exact dense QRCP to preserve R/Stata parity.

    Parameters
    ----------
    A : Matrix
        Coefficient matrix (m x n), dense or sparse.
    b : ndarray
        Right-hand side vector or matrix (m,) or (m x k).
    rcond : float, optional
        SVD truncation cutoff used in the fallback; if None uses sqrt(eps).
    rank_policy : str, default="stata"
        "R" (R lm.fit tol=1e-7*max|diag(R)|) or "stata" (Mata eta).

    Returns
    -------
    x : ndarray
        Solution vector or matrix with shape (n, k).

    References
    ----------
    - R lm.fit: tol = 1e-7 * max(|diag(R)|)
    - Stata Mata qrsolve: eta = 1e-13 * trace(|R|) / rows(R)

    """
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    elif b.ndim != 2:
        msg = "b must be 1- or 2-dimensional"
        raise ValueError(msg)

    # Sparse-aware finite check without densifying first
    _assert_all_finite_matrix(A, b)
    m = A.shape[0] if _is_sparse(A) else to_dense(A).shape[0]
    if b.shape[0] != m:
        msg = "Incompatible dimensions"
        raise ValueError(msg)

    # Sparse QR (QRCP) path: SuiteSparseQR required for exact column-pivoted QR
    if _is_sparse(A):
        # Strict-parity fallback: if SuiteSparseQR is not available, densify *here* and continue
        if (sp is None) or (spqr is None):
            Ad = to_dense(A)
            _assert_all_finite(Ad, b)
            return vectorized_qr_solve(Ad, b, rcond=rcond, rank_policy=rank_policy)

        # CSC format recommended for SuiteSparse internal optimization
        Ac = A.tocsc()  # type: ignore[attr-defined]
        # sparseqr.qr returns (Q, R, E[, rank]). E is permutation (matrix or vector).
        qr_out = spqr.qr(Ac)  # type: ignore[attr-defined]
        if len(qr_out) == 4:
            Qs, Rs, E, r_ssqr = qr_out
        else:
            Qs, Rs, E = qr_out
            # Determine rank using R/Stata conventions
            if _is_sparse(Rs):
                diagR = np.abs(Rs.diagonal()).A.ravel()  # type: ignore[attr-defined]
            else:
                diagR = np.abs(np.diag(Rs))
            r_ssqr = _rank_from_diag(
                diagR,
                Ac.shape[1],
                mode=("R" if rank_policy.upper().startswith("R") else "stata"),
            )

        # Extract pivot array P (length n column indices) from E; coerce to int64
        if _is_sparse(E):
            # E is permutation matrix (n x n). Extract column indices via argmax.
            P = np.asarray(E.tocsc().argmax(axis=0)).ravel().astype(np.int64)  # type: ignore[attr-defined]
        else:
            P = np.asarray(E).ravel().astype(np.int64)

        # Q' b (sparse orthogonal matrix product). Ensure b is 2-D column-shaped.
        Qtb = Qs.T @ b  # type: ignore[operator]

        r = int(r_ssqr)
        n = Ac.shape[1]
        out = np.full(
            (n, b.shape[1]),
            np.nan if rank_policy.upper().startswith("R") else 0.0,
            dtype=np.float64,
        )

        if r > 0:
            # R11 is r x r upper triangular (densify for safe triangular solve)
            R11 = Rs[:r, :r]
            R11 = R11.toarray() if _is_sparse(R11) else np.asarray(R11)  # type: ignore[attr-defined]
            rhs = Qtb[:r, :]
            beta_r = (
                sla.solve_triangular(R11, rhs, lower=False, check_finite=False)
                if sla
                else np.linalg.solve(R11, rhs)
            )
            # Place solution in first r pivot columns (Stata: rest=0, R: rest=NaN)
            out[P[:r], :] = beta_r

        return out

    # Dense QR path
    A_dense = to_dense(A)
    if A_dense.ndim != 2:
        msg = "A must be 2-dimensional"
        raise ValueError(msg)

    m, n = A_dense.shape
    # Defensive: number of R columns produced by QR (economic) is at most min(m, n)
    rcols = min(m, n)

    if sla is not None:
        Q, R, P = sla.qr(A_dense, mode="economic", pivoting=True)
    else:
        Q, R, P = _qr_cp_numpy(A_dense)
    diagR = np.abs(np.diag(R))

    # Determine rank threshold per policy
    if rank_policy.upper().startswith("R"):
        tol = (1e-7 if rcond is None else float(rcond)) * (
            float(np.max(diagR)) if diagR.size else 0.0
        )
        r = int(np.sum(diagR > tol))
    else:
        r = _rank_from_diag(diagR, n, mode="stata")

    Qtb = Q.T @ b
    if r < n:
        # Rank-deficient case:
        # R-compatible: return NA (np.nan) for unidentified coefficients; only the first r identified columns are solved.
        if rank_policy.upper().startswith("R"):
            coef = np.full((n, b.shape[1]), np.nan, dtype=np.float64)
            if r > 0:
                beta_r = np.linalg.solve(R[:r, :r], Qtb[:r, :])
                coef[P[:r], :] = beta_r
            return coef
        # Stata-compatible: dropped columns receive 0; only the first r identified columns are solved.
        xb = np.linalg.solve(R[:r, :r], Qtb[:r, :])
        x_full = np.zeros((n, b.shape[1]), dtype=np.float64)
        x_full[P[:r], :] = xb
        return x_full
    # full rank: standard triangular solve, then undo column pivoting
    # full rank: solve using the leading square R block of size rcols
    xb = np.linalg.solve(R[:rcols, :rcols], Qtb[:rcols, :])  # xb is in pivot order
    x_full = np.zeros((n, b.shape[1]), dtype=np.float64)
    # SciPy/LAPACK convention: A[:, P] = Q R  ⇒  solution in pivot order maps as x[P] = xb
    x_full[P[:rcols], :] = xb

    return x_full


def solve_constrained_batch(  # noqa: PLR0913
    X: Matrix,
    Y_batch: Matrix,
    R: NDArray[np.float64],
    q_batch: NDArray[np.float64] | None,
    *,
    W: Matrix | None = None,
    symmetrize_W: bool = True,
    symmetry_atol: float = 1e-10,
    symmetry_rtol: float = 1e-8,
    prune_redundant: bool = False,
    prefer_WX_first: bool = False,
    ridge: float = 0.0,
    weight_policy: str = "forbid",
) -> NDArray[np.float64]:
    """Batched equality-constrained weighted least-squares solver.

    Solves, for each column j of Y_batch and corresponding q_batch column,
        minimize_b  (y_j - X b)' W (y_j - X b)  subject to  R b = q_j

    This implementation centralizes the KKT / transformed-KKT logic so all
    heavy linear algebra stays inside core.linalg. It prefers QR-based
    robust inversion via :func:`xtwx_inv_via_qr` and reuses factorizations
    across batch columns to minimize repeated work.

    Parameters mirror the higher-level helper in utils.constraints but the
    function accepts sparse-friendly Matrix types for X and W. The
    ``weight_policy`` flag enforces project-wide defaults whereby observation
    weights are forbidden unless the caller explicitly opts-in (GLS/GMM).
    """
    # Convert to dense-friendly representations where necessary through helpers
    Xd = to_dense(X)
    Yd = to_dense(Y_batch)
    R = np.asarray(R, dtype=np.float64, order="C")
    if Yd.ndim == 1 or (Yd.ndim == 2 and Yd.shape[1] == 1):
        Yd = Yd.reshape(-1, 1)

    n, k = Xd.shape
    m = R.shape[0]
    B = Yd.shape[1]
    if q_batch is None:
        qd = np.zeros((m, B), dtype=np.float64)
    else:
        qd = np.asarray(q_batch, dtype=np.float64, order="C")
        if qd.ndim == 1:
            qd = qd.reshape(-1, 1)
        if qd.shape[0] != m:
            msg = "q_batch must have same number of rows as R"
            raise ValueError(msg)
        if qd.shape[1] == 1 and B > 1:
            qd = np.repeat(qd, B, axis=1)
        elif qd.shape[1] != B:
            msg = "q_batch must have same number of columns as Y_batch (number of right-hand sides)."
            raise ValueError(msg)
        if not np.all(np.isfinite(qd)):
            msg = "q_batch contains non-finite values."
            raise ValueError(msg)

    if R.shape[1] != k:
        msg = "R must have same number of columns as X"
        raise ValueError(msg)
    if Yd.shape[0] != n:
        msg = "Y_batch must have same number of rows as X"
        raise ValueError(msg)

    # Validate/sanitize W and apply weighting/whitening so the objective becomes
    # unweighted least squares in transformed space.
    #
    # - If W is a vector (n,): interpret as diagonal observation weights w_i.
    # - If W is a matrix (n x n): interpret as a symmetric PSD weighting matrix.
    wp = str(weight_policy).lower()
    if wp not in {"allow", "forbid"}:
        msg = "weight_policy must be 'allow' or 'forbid'."
        raise ValueError(msg)
    if W is not None:
        if wp != "allow":
            msg = "Observation weights are forbidden unless weight_policy='allow'."
            raise ValueError(msg)
        # to_dense will handle sparse matrices safely
        W_arr = to_dense(W)
        if W_arr.ndim == 1:
            if W_arr.shape[0] != n:
                msg = "Length of weight vector W must equal number of rows of X."
                raise ValueError(msg)
            w_vec = _validate_weights(W_arr, n).reshape(-1, 1)
            sw = np.sqrt(w_vec)
            Xd = Xd * sw
            Yd = Yd * sw
        else:
            Wm = 0.5 * (W_arr + W_arr.T) if symmetrize_W else W_arr
            if not symmetrize_W and (not np.allclose(Wm, Wm.T, atol=symmetry_atol, rtol=symmetry_rtol)):
                msg = "Matrix W must be approximately symmetric (or set symmetrize_W=True)."
                raise ValueError(msg)
            _assert_all_finite(Wm)
            evals = np.linalg.eigvalsh(0.5 * (Wm + Wm.T))
            if np.min(evals) < -1e-12 * max(1.0, float(np.max(np.abs(evals)))):
                raise ValueError(
                    "W must be symmetric (and PSD) for constrained GLS/WLS; consider force_psd/nearPD if needed.",
                )
            # Whitening: find S such that (S'(y - Xb))'(S'(y - Xb)) = (y - Xb)' W (y - Xb).
            # Prefer strict Cholesky when W is PD; otherwise fall back to symmetric PSD square root.
            try:
                Lw = safe_cholesky(Wm, lower=True)
                S = Lw.T
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                S = symmetric_matrix_sqrt(Wm)
            Xd = S @ Xd
            Yd = S @ Yd

    # prefer_WX_first is kept for API compatibility; the implementation already
    # attempts an (X'WX)^{-1}-first route when feasible.

    # Try QR-first inversion for XtWX to support transformed-KKT
    A_inv: NDArray[np.float64] | None = None
    try:
        A_inv = xtwx_inv_via_qr(Xd, weights=None)
        if not np.all(np.isfinite(A_inv)):
            A_inv = None
    except (RuntimeError, np.linalg.LinAlgError, ValueError):
        A_inv = None

    zeros_m = np.zeros((m, m), dtype=np.float64)
    betas = np.empty((k, B), dtype=np.float64)

    # Optionally prune redundant constraint rows (caller may have already done so)
    if prune_redundant and m > 0:
        try:
            qr_res = qr(R.T, pivoting=True, mode="economic")
            if isinstance(qr_res, tuple) and len(qr_res) == 3:
                _, R_up, piv = qr_res
                R_upd = to_dense(R_up)
                diagR = np.abs(np.diag(R_upd))
                tol = float(diagR.max()) * 1e-10 if diagR.size else 0.0
                rank = int(np.sum(diagR > tol))
                if rank < R.shape[0]:
                    keep = np.sort(piv[:rank].astype(np.int64))
                    R = R[keep, :]
                    qd = qd[keep, :]
                    m = R.shape[0]
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            m = R.shape[0]

    # Precompute XtWX_sym for KKT branch
    # Compute XtWX (weighted Gram) once for KKT branches. Use gram() which
    # respects weights and sparse inputs.
    try:
        XtWX = gram(Xd, weights=None)
        XtWX_sym = 0.5 * (XtWX + XtWX.T)
    except (RuntimeError, ValueError):
        XtWX_sym = None

    # Helper to compute X'W y or X'y depending on W
    def _xty_for(y_col: NDArray[np.float64]) -> NDArray[np.float64]:
        # After whitening/weighting, the objective is unweighted LS in transformed space.
        return xty(Xd, y_col, weights=None)

    # Branch when A_inv (XtWX^{-1}) is not available: build full KKT and solve per RHS
    if A_inv is None:
        if XtWX_sym is None:
            # Fallback: compute XtWX via gram
            XtWX = gram(Xd, weights=None)
            XtWX_sym = 0.5 * (XtWX + XtWX.T)
        # Try SPD solve on XtWX_sym to use efficient KKT; only apply ridge if explicitly requested
        try:
            # Test for SPD by attempting a strict Cholesky factorization
            _ = safe_cholesky(XtWX_sym, lower=True)
            A11 = 2.0 * XtWX_sym
            A12 = R.T
            A21 = R
            A_kkt = np.block([[A11, A12], [A21, zeros_m]])
            for j in range(B):
                yj = Yd[:, j : j + 1]
                qj = qd[:, j : j + 1]
                Xty = _xty_for(yj)
                Bvec = np.vstack([2.0 * Xty, qj])
                sol = solve(A_kkt, Bvec, sym_pos=False)
                betas[:, j] = sol[:k, :].reshape(-1)
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            kkt_solved = False
            if ridge > 0.0:
                try:
                    XtWX_r = XtWX + ridge * eye(XtWX.shape[0])
                    A11 = 2.0 * (0.5 * (XtWX_r + XtWX_r.T))
                    A12 = R.T
                    A21 = R
                    A_kkt = np.block([[A11, A12], [A21, zeros_m]])
                    for j in range(B):
                        yj = Yd[:, j : j + 1]
                        qj = qd[:, j : j + 1]
                        Xty = _xty_for(yj)
                        Bvec = np.vstack([2.0 * Xty, qj])
                        sol = solve(A_kkt, Bvec, sym_pos=False)
                        betas[:, j] = sol[:k, :].reshape(-1)
                    kkt_solved = True
                except (RuntimeError, np.linalg.LinAlgError, ValueError):
                    kkt_solved = False
            if kkt_solved:
                return betas
        else:
            return betas
        # Last-resort per-RHS KKT without SPD assumption
        A11 = 2.0 * XtWX_sym
        A12 = R.T
        A21 = R
        A_kkt = np.block([[A11, A12], [A21, zeros_m]])
        for j in range(B):
            yj = Yd[:, j : j + 1]
            qj = qd[:, j : j + 1]
            Xty = _xty_for(yj)
            Bvec = np.vstack([2.0 * Xty, qj])
            sol = solve(A_kkt, Bvec, sym_pos=False)
            betas[:, j] = sol[:k, :].reshape(-1)
        return betas

    # If A_inv available: use transformed KKT that reuses constant blocks
    top_left = 2.0 * eye(k)
    top_right_const = dot(A_inv, R.T)
    bottom_left = R
    KKT_trans_const = np.block([[top_left, top_right_const], [bottom_left, zeros_m]])
    for j in range(B):
        yj = Yd[:, j : j + 1]
        qj = qd[:, j : j + 1]
        Xty = _xty_for(yj)
        Bvec = np.vstack([2.0 * dot(A_inv, Xty), qj])
        sol = solve(KKT_trans_const, Bvec, sym_pos=False)
        betas[:, j] = sol[:k, :].reshape(-1)
    return betas


def sparse_matrix_pseudoinverse(
    A: Matrix,
    *,
    rcond: float | None = None,
) -> NDArray[np.float64]:
    """Sparse-aware pseudoinverse computation.

    Computes Moore-Penrose pseudoinverse. For sparse matrices, densifies and uses SVD.

    Parameters
    ----------
    A : Matrix
        Input matrix (dense or sparse).
    rcond : float, optional
        Cutoff for small singular values.

    Returns
    -------
    A_pinv : ndarray
        Pseudoinverse of A (dense).

    Notes
    -----
    This intentionally densifies sparse inputs for numerical stability. Use with
    caution on very large sparse matrices.

    """
    # Force Moore-Penrose pseudoinverse via direct SVD on a dense matrix
    # to match R's MASS::ginv and avoid numerical degradation from normal
    # equations. Densify sparse inputs intentionally (documented policy).
    Ad = to_dense(A)
    rc = np.sqrt(np.finfo(float).eps) if rcond is None else float(rcond)
    return np.linalg.pinv(Ad, rcond=rc)


# ============================================================================
# Batched Operations for Bootstrap Efficiency
# ============================================================================


def batched_crossprod(
    X: Matrix,
    Y: NDArray[np.float64],
    *,
    n_jobs: int = -1,
    chunk_size: int = 50,
) -> NDArray[np.float64]:
    """Batched cross-product X'Y for bootstrap replications.

    Computes X' @ Y[:, :, b] for b = 0, 1, ..., B-1 in parallel across
    replications. This is the core operation for bootstrap coefficient
    computation when Y = y * multipliers (n x 1 x B).

    Parameters
    ----------
    X : Matrix
        Design matrix (n x p).
    Y : ndarray
        Response matrix with bootstrap dimension (n x k x B).
    n_jobs : int, default=-1
        Number of parallel workers (-1 uses all cores).
    chunk_size : int, default=50
        Number of replications per worker chunk.

    Returns
    -------
    XtY : ndarray
        Cross-product array (p x k x B).

    Notes
    -----
    This function parallelizes across the B bootstrap dimension using
    ThreadPoolExecutor for GIL-released NumPy operations. For large B
    (e.g., 2000 bootstrap replications), this provides significant speedup
    on multi-core systems.

    Examples
    --------
    >>> n, p, B = 1000, 10, 2000
    >>> X = np.random.randn(n, p)
    >>> y = np.random.randn(n, 1)
    >>> multipliers = np.random.randn(n, 1, B)
    >>> Y = y * multipliers  # (n, 1, B)
    >>> XtY = batched_crossprod(X, Y)  # (p, 1, B)

    """
    X_dense = to_dense(X)
    Y = np.asarray(Y, dtype=np.float64)

    if X_dense.ndim != 2:
        msg = "X must be 2-dimensional"
        raise ValueError(msg)
    if Y.ndim != 3:
        msg = "Y must be 3-dimensional (n x k x B)"
        raise ValueError(msg)

    n, p = X_dense.shape
    n2, k, B = Y.shape
    if n != n2:
        msg = f"Incompatible dimensions: X has {n} rows, Y has {n2} rows"
        raise ValueError(msg)

    # Sequential fallback for small B
    if B < 10:
        return np.stack([X_dense.T @ Y[:, :, b] for b in range(B)], axis=2)

    # Parallel computation across replications
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    def _crossprod_chunk(b_start: int, b_end: int) -> NDArray[np.float64]:
        result = np.empty((p, k, b_end - b_start), dtype=np.float64)
        for i, b in enumerate(range(b_start, b_end)):
            result[:, :, i] = X_dense.T @ Y[:, :, b]
        return result

    chunks = [(i, min(i + chunk_size, B)) for i in range(0, B, chunk_size)]
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_crossprod_chunk, s, e): s for (s, e) in chunks}
        out_ordered = [(futures[fut], fut.result()) for fut in as_completed(futures)]
    out_ordered.sort(key=lambda t: t[0])
    # Concatenate along bootstrap dimension in original order
    return np.concatenate([v for _, v in out_ordered], axis=2)


def batched_tdot(
    A: Matrix,
    B: NDArray[np.float64],
    *,
    n_jobs: int = -1,
    chunk_size: int = 50,
) -> NDArray[np.float64]:
    """Batched transposed dot product A' @ B[:, :, b] for bootstrap.

    Computes A.T @ B[:, :, b] for each bootstrap replication b in parallel.
    This is equivalent to batched_crossprod but with explicit naming for
    clarity in different contexts.

    Parameters
    ----------
    A : Matrix
        Left matrix (n x p).
    B : ndarray
        Right matrix with bootstrap dimension (n x k x B).
    n_jobs : int, default=-1
        Number of parallel workers.
    chunk_size : int, default=50
        Replications per worker chunk.

    Returns
    -------
    AtB : ndarray
        Transposed dot product array (p x k x B).

    """
    return batched_crossprod(A, B, n_jobs=n_jobs, chunk_size=chunk_size)


def batched_solve_normal_eq(  # noqa: PLR0913
    X: Matrix,
    Y: NDArray[np.float64],
    *,
    weights: NDArray[np.float64] | None = None,
    rcond: float | None = None,
    rank_policy: str = "stata",
    n_jobs: int = -1,
    chunk_size: int = 50,
) -> NDArray[np.float64]:
    """Batched normal equation solver for bootstrap coefficient estimates.

    Solves (X'WX) b = X'Wy for each bootstrap replication y[:, :, b] using
    QR factorization on sqrt(W)X. This is the primary bootstrap coefficient
    computation routine for OLS/IV/GMM estimators.

    Parameters
    ----------
    X : Matrix
        Design matrix (n x p).
    Y : ndarray
        Response with bootstrap dimension (n x k x B).
    weights : ndarray, optional
        Observation weights (n,) or (n, 1). Default is uniform weights.
    rcond : float, optional
        Rank tolerance for SVD fallback.
    rank_policy : str, default="stata"
        Rank determination policy: "stata" or "R".
    n_jobs : int, default=-1
        Number of parallel workers.
    chunk_size : int, default=50
        Replications per worker chunk.

    Returns
    -------
    beta : ndarray
        Coefficient estimates (p x k x B).

    Notes
    -----
    This function parallelizes across bootstrap replications and uses
    QR-based solving (via solve_normal_eq) to ensure numerical stability
    matching R/Stata QRCP conventions. SVD minimum-norm solutions are only
    produced when callers explicitly request method='svd'.

    References
    ----------
    - Stata Mata qrsolve: eta = 1e-13 * trace(|R|) / rows(R)
    - R lm.fit: tol = 1e-7 * max(|diag(R)|)

    """
    X_dense = to_dense(X)
    Y = np.asarray(Y, dtype=np.float64)

    if X_dense.ndim != 2:
        msg = "X must be 2-dimensional"
        raise ValueError(msg)
    if Y.ndim != 3:
        msg = "Y must be 3-dimensional (n x k x B)"
        raise ValueError(msg)

    n, p = X_dense.shape
    n2, k, B = Y.shape
    if n != n2:
        msg = f"Incompatible dimensions: X has {n} rows, Y has {n2} rows"
        raise ValueError(msg)

    # Validate weights
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if weights.shape[0] != n:
            msg = f"Weights must have length {n}, got {weights.shape[0]}"
            raise ValueError(msg)

    # Sequential fallback for small B
    if B < 10:
        result = np.empty((p, k, B), dtype=np.float64)
        for b in range(B):
            result[:, :, b] = solve_normal_eq(
                X_dense,
                Y[:, :, b],
                weights=weights,
                svd_rcond=rcond,
                rank_policy=rank_policy,
            )
        return result

    # Parallel computation across replications
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    def _solve_chunk(b_start: int, b_end: int) -> NDArray[np.float64]:
        result = np.empty((p, k, b_end - b_start), dtype=np.float64)
        for i, b in enumerate(range(b_start, b_end)):
            result[:, :, i] = solve_normal_eq(
                X_dense,
                Y[:, :, b],
                weights=weights,
                svd_rcond=rcond,
                rank_policy=rank_policy,
            )
        return result

    chunks = [(i, min(i + chunk_size, B)) for i in range(0, B, chunk_size)]
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_solve_chunk, s, e): s for (s, e) in chunks}
        out_ordered = [(futures[fut], fut.result()) for fut in as_completed(futures)]
    out_ordered.sort(key=lambda t: t[0])
    # Concatenate along bootstrap dimension in original order
    return np.concatenate([v for _, v in out_ordered], axis=2)


def vectorized_gram(
    X: Matrix,
    *,
    weights: NDArray[np.float64] | None = None,
    n_jobs: int = -1,
) -> NDArray[np.float64]:
    """Vectorized Gram matrix computation with optional parallelization.

    Computes X'WX efficiently with optional parallel chunking for very large
    matrices. For typical econometric applications (n ~ 10^6, p ~ 100), this
    provides minimal benefit, but for massive p it can help.

    Parameters
    ----------
    X : Matrix
        Design matrix (n x p).
    weights : ndarray, optional
        Observation weights (n,).
    n_jobs : int, default=-1
        Number of parallel workers (-1 uses all cores).

    Returns
    -------
    XtWX : ndarray
        Gram matrix (p x p).

    Notes
    -----
    This function is primarily a convenience wrapper around gram() with
    optional parallelization for very wide matrices. For standard use cases,
    gram() is sufficient.

    """
    return gram(X, weights=weights)


# ============================================================================
# Bootstrap Uniform Confidence Band Utilities (DID/EventStudy/SC only)
# ============================================================================


def bootstrap_uniform_supnorm_halfwidth(
    theta: NDArray[np.float64],
    theta_boot: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    center: str = "multiplier",
    context: str = "did",  # allowed: "did","eventstudy","synthetic_control"
) -> NDArray[np.float64]:
    """Compute uniform confidence band half-width for DID/EventStudy/SC using bootstrap.

    This low-level matrix operation computes the supremum norm of bootstrap
    deviations and returns the (1-alpha) quantile as the uniform half-width h.
    The upper-level estimator constructs confidence intervals as [theta - h, theta + h].

    NO statistical tests, p-values, or critical values are involved—this is
    pure linear algebra for bootstrap quantile computation. This function is
    ONLY for DID/event-study/synthetic-control estimators that are explicitly
    allowed to use bootstrap uniform confidence bands per project requirements.

    Parameters
    ----------
    theta : ndarray
        Point estimates, shape (T,) for scalar time series or (T, K) for
        multiple series (e.g., event-time ATT coefficients).
    theta_boot : ndarray
        Bootstrap samples, shape (T, B) for scalar or (T, K, B) for multiple
        series, where B is the number of bootstrap replications.
    alpha : float, default=0.05
        Significance level (e.g., 0.05 for 95% confidence).
    center : str, default="multiplier"
        Bootstrap centering method. Only multiplier (or wild) centering is
        supported for studentized sup-norm uniform bands (matching R's
        did::mboot and related implementations). Non-standard centering
        options such as "basic" or "percentile" are intentionally
        unsupported and will raise a ValueError to avoid ambiguous behavior.

    Returns
    -------
    h : ndarray
        Uniform half-width, same shape as theta. For "basic" method, all elements
        are the same scalar h; for "percentile", may vary by dimension.

    Notes
    -----
    Theory: Romano & Wolf (2005) "Exact and Approximate Stepdown Methods for
    Multiple Hypothesis Testing". For event-study designs, see Roth & Sant'Anna
    (2021) "When Is Parallel Trends Sensitive to Functional Form?".

    This function does NOT perform hypothesis tests or compute p-values. It only
    computes the bootstrap quantile of the supremum deviation, which is a pure
    matrix operation. Upper-level DID/event-study/SC estimators use this to
    construct [theta - h, theta + h] uniform confidence bands.

    References
    ----------
    - Romano & Wolf (2005): Bootstrap theory for supremum-based inference
    - Roth & Sant'Anna (2021): Pre-testing in event-study designs
    - Callaway & Sant'Anna (2021): Difference-in-Differences with multiple periods

    """
    # Enforce allowed contexts per project policy
    if str(context).lower() not in {"did", "eventstudy", "synthetic_control", "rct"}:
        msg = (
            "bootstrap_uniform_supnorm_halfwidth: allowed only for DID/event-study/"
            "synthetic control/RCT."
        )
        raise ValueError(msg)

    # Policy: only studentized sup-norm bands with multiplier/wild centering
    # are supported. Enforce allowed center values and delegate to
    # uniform_band_from_bootstrap which implements ddof=1 sample SD and the
    # finite-sample (B+1) order-statistic quantile selection.
    if str(center).lower() not in {"multiplier", "wild"}:
        msg = (
            "bootstrap_uniform_supnorm_halfwidth: only 'multiplier' (or 'wild') centering is "
            "supported; 'basic'/'percentile' are disallowed."
        )
        raise ValueError(msg)
    theta = np.asarray(theta)
    theta_boot = np.asarray(theta_boot)

    if theta.ndim == 1:
        T = theta.shape[0]
        if theta_boot.ndim != 2 or theta_boot.shape[0] != T:
            msg = "theta_boot must have shape (T, B) for scalar theta of shape (T,)"
            raise ValueError(msg)
        # Convert to uniform_band_from_bootstrap expected shapes:
        # beta_hat: (m,) where m = T, boot_draws: (B, m)
        B = theta_boot.shape[1]
        boot_draws = theta_boot.T.copy()  # shape (B, T)
        lower, upper = uniform_band_from_bootstrap(
            theta,
            boot_draws,
            alpha=alpha,
            studentize=True,
            center=str(center).lower(),
            context=context,
        )
        half = 0.5 * (upper - lower)
        return half.reshape((T,)).astype(np.float64)

    if theta.ndim == 2:
        T, K = theta.shape
        if theta_boot.ndim != 3 or theta_boot.shape[0] != T or theta_boot.shape[1] != K:
            msg = (
                "theta_boot must have shape (T, K, B) for theta of shape (T, K), got "
                f"{theta_boot.shape}"
            )
            raise ValueError(msg)
        # Flatten coordinates to a vector of length m = T*K and build boot_draws (B, m)
        B = theta_boot.shape[2]
        beta_hat = theta.reshape(-1)
        boot_draws = theta_boot.reshape(T * K, B).T.copy()  # (B, m)
        lower, upper = uniform_band_from_bootstrap(
            beta_hat,
            boot_draws,
            alpha=alpha,
            studentize=True,
            center=str(center).lower(),
            context=context,
        )
        half = 0.5 * (upper - lower)
        return half.reshape((T, K)).astype(np.float64)

    msg = f"theta must be 1D or 2D, got shape {theta.shape}"
    raise ValueError(msg)
