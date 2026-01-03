"""Fixed-effects absorption and recovery.

This module provides high-performance multi-way fixed effects absorption using
alternating projections (Frisch-Waugh-Lovell), along with singleton pruning and
coefficient recovery.
"""

# lineareg/core/fe.py
from __future__ import annotations

import numbers
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Union, cast, overload

import numpy as np
from scipy import sparse

from .linalg import (  # matrix ops (sparse-aware)
    _validate_weights,
    dot,
    group_sum,
    hstack,
    solve,
    to_dense,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = np.ndarray  # type: ignore[misc,assignment]

ArrayLike = Union[NDArray[Any], Sequence[int], Sequence[float]]

__all__ = [
    "FETransformResult",
    "absorb",
    "compute_fe_dof",
]


@dataclass(slots=True)
class FETransformResult:
    """Container for within-transformation results.

    Attributes
    ----------
    X : np.ndarray
        Demeaned regressor matrix.
    y : np.ndarray | None
        Demeaned outcome vector.
    Z : np.ndarray | None
        Demeaned additional matrix (e.g. instruments).
    mask : np.ndarray
        Boolean array indicating rows retained after NA/singleton checks.
    dropped : dict[str, int]
        Counts of dropped observations by reason.
    fe_codes : list[np.ndarray]
        Integer codes for each fixed-effect dimension after pruning.
    weights : np.ndarray | None
        Observation weights aligned with the demeaned sample.
    diagnostics : dict[str, Any] | None
        Convergence diagnostics from the alternating projection routine.
    """

    X: NDArray[np.float64]
    y: NDArray[np.float64] | None
    Z: NDArray[np.float64] | None
    mask: NDArray[np.bool_]
    dropped: dict[str, int] = field(default_factory=dict)
    fe_codes: list[NDArray[np.int64]] = field(default_factory=list)
    weights: NDArray[np.float64] | None = None
    diagnostics: dict[str, Any] | None = None

    @property
    def n_effective(self) -> int:
        """Number of observations retained after preprocessing."""
        return int(np.sum(self.mask))

    def copy(self, **updates: Any) -> FETransformResult:
        """Return a shallow copy updated with any supplied keyword overrides."""
        X = cast(NDArray[np.float64], updates.get("X", self.X))
        y = cast(NDArray[np.float64] | None, updates.get("y", self.y))
        Z = cast(NDArray[np.float64] | None, updates.get("Z", self.Z))
        mask = cast(NDArray[np.bool_], updates.get("mask", self.mask))
        dropped = cast(dict[str, int], updates.get("dropped", dict(self.dropped)))
        fe_codes = cast(
            list[NDArray[np.int64]],
            updates.get("fe_codes", [np.asarray(c) for c in self.fe_codes]),
        )
        weights = cast(NDArray[np.float64] | None, updates.get("weights", self.weights))
        diagnostics_raw = updates.get(
            "diagnostics",
            None if self.diagnostics is None else dict(self.diagnostics),
        )
        diagnostics: dict[str, Any] | None = (
            None
            if diagnostics_raw is None
            else dict(cast(dict[str, Any], diagnostics_raw))
        )
        return FETransformResult(
            X=X,
            y=y,
            Z=Z,
            mask=mask,
            dropped=dropped,
            fe_codes=fe_codes,
            weights=weights,
            diagnostics=diagnostics,
        )


# Helpers
# ---------------------------------------------------------------------


def _to_codes(z: ArrayLike) -> NDArray[np.int64]:
    """Map arbitrary labels to consecutive 0..G-1 integer codes."""
    arr = np.asarray(z).reshape(-1)
    _, inv = np.unique(arr, return_inverse=True)
    return cast(NDArray[np.int64], inv.astype(np.int64, copy=False))


def _isfinite_like(z: ArrayLike) -> NDArray[np.bool_]:
    """Check for finiteness/validity of FE IDs.

    Handles numeric arrays via `isfinite`, and object arrays by checking for
    None, NaN, pandas NA, or NaT.
    """
    # normalize to 1d array
    arr = np.asarray(z)
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    # Numeric kinds: rely on numpy's isfinite
    if arr.dtype.kind in {"i", "u", "f", "c"}:
        return cast(NDArray[np.bool_], np.isfinite(arr))

    # Object-like: try to leverage pandas.isna if available for broad coverage
    try:
        import pandas as pd
    except ImportError:
        isna: Callable[[object], bool] | None = None
    else:
        isna = pd.isna

    def _is_missing_via_pandas(value: object) -> bool:
        if isna is None:
            return False
        try:
            return bool(isna(value))
        except (TypeError, ValueError):
            return False

    def _datetime64_status(value: object) -> bool | None:
        method = getattr(value, "to_datetime64", None)
        if not callable(method):
            return None
        try:
            dt_value = method()
        except (TypeError, ValueError, AttributeError):
            return False
        if isinstance(dt_value, (np.datetime64, np.timedelta64)):
            return not np.isnat(dt_value)
        return True

    def _self_equal(value: object) -> bool:
        eq = getattr(value, "__eq__", None)
        if eq is None:
            return True
        try:
            result = eq(value)
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            return True
        if isinstance(result, np.ndarray):
            if result.shape == ():
                return bool(result)
            with np.errstate(all="ignore"):
                return bool(np.all(result))
        return bool(result)

    out = np.empty(arr.shape[0], dtype=bool)
    for idx, value in enumerate(arr):
        if value is None:
            out[idx] = False
            continue
        if isinstance(value, numbers.Number):
            out[idx] = bool(np.isfinite(float(cast(Any, value))))
            continue
        if _is_missing_via_pandas(value):
            out[idx] = False
            continue
        if isinstance(value, (np.datetime64, np.timedelta64)):
            out[idx] = not np.isnat(value)
            continue
        dt_status = _datetime64_status(value)
        if dt_status is not None:
            out[idx] = dt_status
            continue
        out[idx] = _self_equal(value)
    return out


def _group_means(
    X: NDArray[np.float64],
    codes: NDArray[np.int64],
    *,
    weights: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Compute group means for rows of X grouped by integer `codes`.

    If weights are provided, computes weighted means. Groups with zero total
    weight result in NaN.
    """
    n, p = X.shape
    codes = codes.astype(np.int64, copy=False)
    if codes.shape[0] != n:
        msg = "codes must have same length as rows of X"
        raise ValueError(msg)

    G = int(codes.max()) + 1 if codes.size else 0

    # Weighted branch
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != n:
            msg = "weights must have same length as rows of X"
            raise ValueError(msg)

        # Weighted sums per group using centralized group_sum helper
        Xw = X * w.reshape(-1, 1)
        sums = group_sum(Xw, codes, order="sorted")  # (G, p)

        # counts (total weight) per group via group_sum on weights column
        counts = group_sum(w.reshape(-1, 1), codes, order="sorted").astype(
            np.float64,
        )  # (G, 1)

        # Zero-total-weight groups: instead of raising here, produce NaN for
        # groups with zero total weight. Upstream callers (reghdfe path) may
        # prefer to drop such groups earlier; callers following fixest semantics
        # will receive NaNs and may keep rows until final post-processing.
        counts_flat = counts.reshape(-1)
        zero_mask = counts_flat == 0
        means = np.full((sums.shape[0], sums.shape[1]), np.nan, dtype=np.float64)
        pos = counts_flat > 0
        if np.any(pos):
            means[pos, :] = sums[pos, :] / counts[pos].reshape(-1, 1)
        return means[codes, :]

    # Unweighted branch
    # Unweighted counts and sums via centralized group_sum helper
    counts = (
        group_sum(np.ones((n, 1), dtype=np.float64), codes, order="sorted")
        .reshape(-1)
        .astype(np.int64)
    )
    sums = group_sum(X, codes, order="sorted")

    zero_mask = counts == 0
    means = np.empty((G, p), dtype=np.float64)
    pos = ~zero_mask
    if pos.any():
        means[pos, :] = sums[pos, :] / counts[pos].reshape(-1, 1)
    if zero_mask.any():
        means[zero_mask, :] = np.nan

    return means[codes, :]


def _prepare_fe(fe_ids: ArrayLike | Sequence[ArrayLike]) -> list[NDArray[np.int64]]:
    """Normalize FE specification into a list of integer code arrays."""
    if isinstance(fe_ids, (list, tuple)):
        return [_to_codes(z) for z in fe_ids]
    return [_to_codes(cast(ArrayLike, fe_ids))]


def _is_nested(target: NDArray[np.int64], others: list[NDArray[np.int64]]) -> bool:
    """Check if 'target' FE is perfectly determined by the concatenation of 'others'.
    Uses linalg.hstack for policy compliance and sparsity safety.
    """
    if len(others) == 0:
        return False
    keys = hstack([z.reshape(-1, 1) for z in others])  # (n x R-1)
    keys = np.asarray(keys)
    _, key_inv = np.unique(keys, axis=0, return_inverse=True)
    key_inv = key_inv.astype(np.int64, copy=False)
    kt = hstack([key_inv.reshape(-1, 1), target.reshape(-1, 1)])
    kt = np.asarray(kt)
    _, inv = np.unique(kt, axis=0, return_inverse=True)
    n_key = int(key_inv.max()) + 1
    n_kt = int(inv.max()) + 1
    return n_kt == n_key


def is_nested(target: NDArray[np.int64], others: list[NDArray[np.int64]]) -> bool:
    """Public wrapper for nested fixed-effect detection."""
    return _is_nested(target, others)


def _prepare_and_prune_fe(
    fe_ids: ArrayLike | Sequence[ArrayLike],
) -> list[NDArray[np.int64]]:
    """Convert FE ids to integer codes and prune any FE fully nested in others.
    """
    codes_list = _prepare_fe(fe_ids)
    pruned: list[NDArray[np.int64]] = []
    for j, z in enumerate(codes_list):
        others = [codes_list[k] for k in range(len(codes_list)) if k != j]
        if not _is_nested(z, others):
            pruned.append(z)
    return pruned


def _normalize_fixest_algo(
    fixest_algo: dict[str, Any] | None,
    backend: str,
) -> dict[str, int]:
    """Normalize and return a canonical fixest algorithm parameter dict.

    Ensures keys are present and integer-casts provided values. For non-fixest
    backends returns zeros to disable fixest-specific warmups/projections.
    """
    defaults = {
        "iter_warmup": 15,
        "iter_projAfterAcc": 40,
        "iter_grandAcc": 4,
        "extraProj": 0,
    }
    if backend != "fixest":
        return dict.fromkeys(defaults, 0)
    if not isinstance(fixest_algo, dict):
        return defaults
    out = defaults.copy()
    for key, default in defaults.items():
        if key not in fixest_algo:
            continue
        try:
            out[key] = int(fixest_algo[key])
        except (TypeError, ValueError):
            out[key] = default
    return out


def _defaults_for_backend(backend: str) -> dict[str, float | int | str]:
    """Return default demeaning parameters for the specified backend."""
    if backend == "fixest":
        # fixest public defaults (CRAN / setFixest_estimation):
        # tol = 1e-6, max_iter = 10000
        return {
            "tol": 1e-6,
            "max_iter": 10000,
            "transform": "kacz",
            "schedule": "cyclic",
        }

    # reghdfe-style defaults
    return {
        "tol": 1e-8,
        "max_iter": 16000,
        "transform": "symkacz",
        "schedule": "symmetric",
    }


def _should_fixest_extra_proj(
    iteration: int,
    algo: dict[str, int],
) -> tuple[bool, bool, bool]:
    """Determine projection steps for the current iteration (fixest algorithm)."""
    iw = int(algo.get("iter_warmup", 0))
    ipa = int(algo.get("iter_projAfterAcc", 0))
    iga = int(algo.get("iter_grandAcc", 0))
    do_warm = (iteration < iw) if iw > 0 else False
    do_after = ipa > 0 and iteration > 0 and (iteration % ipa == 0)
    do_grand = iga > 0 and iteration > 0 and (iteration % iga == 0)
    return do_warm, do_after, do_grand


# Note: _drop_to_2core removed — 2-core pre-reduction is deprecated and not
# part of the public reghdfe/fixest APIs. Use `_drop_singletons_iteratively`
# or external preprocessing for any k-core style reductions.


def _drop_singletons_iteratively(
    codes_list: list[NDArray[np.int64]],
    weights: NDArray[np.float64] | None = None,
    *,
    style: str = "reghdfe_iter",
) -> NDArray[np.bool_]:
    """Iteratively drop singleton groups.

    Drops observations belonging to groups with size 1, repeating until no
    singletons remain. Weights are ignored (count-based criterion).
    """
    if not codes_list:
        return np.array([], dtype=bool)
    n = codes_list[0].shape[0]
    if weights is not None:
        weights_arr = np.asarray(weights).reshape(-1)
        if weights_arr.shape[0] != n:
            raise ValueError("weights must have the same length as the FE codes")
    keep = np.ones(n, dtype=bool)

    # Only iterative recursive removal is supported for strict parity.
    # The older fixest one-pass semantics have been removed to maintain
    # consistent reghdfe-style recursive singleton dropping.
    if style is not None and style != "reghdfe_iter":
        raise ValueError("style must be 'reghdfe_iter' (iterative recursive removal)")

    def pass_once() -> bool:
        counts_any = np.zeros(n, dtype=bool)
        for codes in codes_list:
            # Count-based singleton detection only (weights are ignored by design)
            cnt = np.bincount(codes[keep], minlength=int(codes.max()) + 1)
            counts_any[keep] |= cnt[codes[keep]] == 1
        if np.any(counts_any):
            keep[counts_any] = False
            return True
        return False

    while pass_once():
        continue
    return keep


def _cluster_nested_dof_adjustment(
    codes_list: list[NDArray[np.int64]], clusters: Sequence[ArrayLike],
) -> int:
    """Conservative DoF adjustment for FE levels perfectly nested within clusters.

    For each FE dimension, compute for each provided cluster variable the number of
    FE levels that are perfectly nested (i.e., all observations with that FE
    level share the same cluster id). Use the maximum across cluster variables
    per FE dimension as a conservative adjustment and sum across dimensions.
    """
    # Deprecated: this function used to return a total adjustment by
    # subtracting entire L_k per nested FE which over-corrects degrees of
    # freedom. We now expose a boolean per-dimension helper and apply the
    # correct substitution Mk[k] = Lk in compute_fe_dof.
    # Keep a backward-compatible wrapper that computes the equivalent total
    # adjustment (sum Lk for nested dims) if callers still use it.
    if not clusters:
        return 0
    total_adj = 0
    for codes in codes_list:
        codes_arr = np.asarray(codes)
        Lk = int(codes_arr.max()) + 1 if codes_arr.size else 0
        if Lk == 0:
            continue
        nested_hit = False
        for clu in clusters:
            clu_arr = np.asarray(clu).reshape(-1)
            all_levels_nested = True
            for lvl in range(Lk):
                idx = np.nonzero(codes_arr == lvl)[0]
                if idx.size == 0:
                    all_levels_nested = False
                    break
                if np.unique(clu_arr[idx]).size != 1:
                    all_levels_nested = False
                    break
            if all_levels_nested:
                nested_hit = True
                break
        if nested_hit:
            total_adj += Lk
    return int(total_adj)


def _cluster_nested_dims(
    codes_list: list[NDArray[np.int64]], clusters: Sequence[ArrayLike],
) -> list[bool]:
    """Return boolean list 'nested[k]' telling whether FE-dimension k is fully nested within any cluster.

    This mirrors reghdfe's cluster nesting rule: if every level of FE k maps
    to a single cluster id for at least one provided cluster variable, then
    FE k is considered nested and its M_k should equal L_k (no DF loss).
    """
    if not clusters:
        return [False] * len(codes_list)
    nested = [False] * len(codes_list)
    for dim_idx, codes in enumerate(codes_list):
        codes_arr = np.asarray(codes)
        Lk = int(codes_arr.max()) + 1 if codes_arr.size else 0
        if Lk == 0:
            continue
        for clu in clusters:
            clu_arr = np.asarray(clu).reshape(-1)
            if clu_arr.shape[0] != codes_arr.shape[0]:
                raise ValueError("cluster array length must match FE codes length")
            all_levels_nested = True
            for lvl in range(Lk):
                level_mask = np.nonzero(codes_arr == lvl)[0]
                if level_mask.size == 0:
                    all_levels_nested = False
                    break
                if np.unique(clu_arr[level_mask]).size != 1:
                    all_levels_nested = False
                    break
            if all_levels_nested:
                nested[dim_idx] = True
                break
    return nested


def _continuous_slope_dof_adjustment(
    codes_list: list[NDArray[np.int64]],
    continuous: Sequence[tuple[ArrayLike, ArrayLike]],
) -> int:
    """Conservative DoF adjustment for continuous-slope terms.

    Each tuple is (fe_id_vector, continuous_vector). For each FE level, if the
    continuous variable is constant within that level, the slope for that level
    is not identified and reduces DoF. We count such occurrences conservatively
    and sum across provided continuous specifications.
    """
    if not continuous:
        return 0
    reference_n = codes_list[0].shape[0] if codes_list else None
    total_adj = 0
    for fe_vec, cont_vec in continuous:
        codes = np.asarray(fe_vec)
        cont = np.asarray(cont_vec).reshape(-1)
        if reference_n is not None and (
            codes.shape[0] != reference_n or cont.shape[0] != reference_n
        ):
            raise ValueError("continuous slope inputs must align with FE observations")
        Lk = int(codes.max()) + 1 if codes.size else 0
        if Lk == 0:
            continue
        adj = 0
        for lvl in range(Lk):
            idx = np.nonzero(codes == lvl)[0]
            if idx.size == 0:
                continue
            if np.nanstd(cont[idx]) == 0.0:
                adj += 1
        total_adj += adj
    return int(total_adj)


def compute_fe_dof(
    fe_ids: ArrayLike | Sequence[ArrayLike],
    *,
    include_intercept: bool = True,
    method: str = "pairwise",
    clusters: Sequence[ArrayLike] | None = None,
    continuous: Sequence[tuple[ArrayLike, ArrayLike]] | None = None,
) -> dict[str, Any]:
    """Compute degrees of freedom for fixed effects.

    Calculates the number of redundant levels (G - DoF). Exact for
    2 dimensions; uses the pairwise-maximum approximation for 3+ dimensions.
    Applies nested-cluster and continuous-slope adjustments if provided.

    Parameters
    ----------
    fe_ids : array-like or list of array-like
        Fixed effect identifiers.
    include_intercept : bool
        Whether the model includes a global intercept.
    method : str
        "pairwise" (default) or "firstpair".
    clusters : list of array-like, optional
        Cluster variables for nested-within-cluster adjustments.
    continuous : list of tuples, optional
        (fe_id, continuous_var) pairs for continuous slope adjustments.
    """
    # Normalize FE inputs to match absorb(): allow passing a DataFrame
    # (columns are treated as separate FE dimensions) or a 2D ndarray.
    raw_fe_list = fe_ids if isinstance(fe_ids, (list, tuple)) else [fe_ids]
    if raw_fe_list and hasattr(raw_fe_list[0], "columns"):
        df = raw_fe_list[0]
        raw_fe_list = [df[col].to_numpy() for col in df.columns]
    elif raw_fe_list and isinstance(raw_fe_list[0], np.ndarray) and raw_fe_list[0].ndim == 2:
        fe_arr = raw_fe_list[0]
        raw_fe_list = [fe_arr[:, j] for j in range(fe_arr.shape[1])]

    codes_list = _prepare_fe(raw_fe_list)
    J = len(codes_list)
    if J == 0:
        return {"total_levels": 0, "fe_dof": 0, "levels_per_fe": [], "Mk": []}

    levels = [int(c.max()) + 1 if c.size else 0 for c in codes_list]
    total_levels = int(sum(levels))

    def bipartite_components(ca: NDArray[np.int64], cb: NDArray[np.int64]) -> int:
        La = int(ca.max()) + 1 if ca.size else 0
        Lb = int(cb.max()) + 1 if cb.size else 0
        if La == 0 and Lb == 0:
            return 0
        parent = list(range(La + Lb))
        rank = [0] * (La + Lb)
        seen_a = np.zeros(La, dtype=bool)
        seen_b = np.zeros(Lb, dtype=bool)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for i in range(ca.shape[0]):
            ai, bi = int(ca[i]), int(cb[i])
            seen_a[ai] = True
            seen_b[bi] = True
            union(ai, La + bi)

        roots = set()
        for ai in np.nonzero(seen_a)[0]:
            roots.add(find(int(ai)))
        for bi in np.nonzero(seen_b)[0]:
            roots.add(find(La + int(bi)))
        return len(roots)

    # Accept a small set of well-defined methods for DF computation.
    method = method.lower()
    if method not in {"pairwise", "firstpair", "all"}:
        raise ValueError(
            "compute_fe_dof: method must be one of {'pairwise','firstpair','all'}",
        )

    Mk = [0] * J
    if J == 1:
        Mk[0] = 1 if include_intercept else 0
    else:
        Mk[0] = 1 if include_intercept else 0
        # exact for first two
        Mk[1] = bipartite_components(codes_list[0], codes_list[1])
        if method == "firstpair":
            # For 'firstpair' we compute exact Mk for the first two FE dims
            # and leave remaining Mk as zero (no additional DF loss beyond firstpair).
            for k in range(2, J):
                Mk[k] = 0
        else:
            # method == 'pairwise' or 'all' : conservative pairwise maxima for J>=3
            for k in range(2, J):
                Mk[k] = max(
                    bipartite_components(codes_list[i], codes_list[k]) for i in range(k)
                )

    # Cluster-based adjustments (reghdfe parity): if FE-k is nested in clusters,
    # then M_k should be set equal to L_k (no DoF loss from that FE dimension).
    if clusters:
        try:
            nested = _cluster_nested_dims(codes_list, clusters or [])
            for k, is_nested in enumerate(nested):
                if is_nested:
                    Mk[k] = levels[k]
        except (ValueError, TypeError, IndexError) as e:
            warnings.warn(
                f"compute_fe_dof: ignoring cluster nesting adjustment due to invalid clusters ({e}).",
                RuntimeWarning,
            )
    fe_dof = int(sum(L - M for L, M in zip(levels, Mk)))
    if continuous:
        try:
            cadj = _continuous_slope_dof_adjustment(codes_list, continuous)
            fe_dof -= cadj
        except (ValueError, TypeError) as e:
            warnings.warn(
                f"compute_fe_dof: ignoring continuous-slope adjustment due to invalid inputs ({e}).",
                RuntimeWarning,
            )
            fe_dof = max(0, fe_dof)

    # 'all' is functionally the same as 'pairwise' for our conservative
    # implementation; keep the method label for clarity in returned dict.
    return {
        "total_levels": total_levels,
        "fe_dof": fe_dof,
        "levels_per_fe": levels,
        "Mk": Mk,
        "method": method,
    }


# ---------------------------------------------------------------------
# Core within-transformation (alternating projections / SOR)
# ---------------------------------------------------------------------



@overload
def _demean_given_codes(  # noqa: PLR0913
    X: NDArray[np.float64],
    codes_list: list[NDArray[np.int64]],
    *,
    weights: Sequence[float] | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
    return_diagnostics: Literal[False] = False,
    verify_foc: bool = False,
    backend: str = "reghdfe",
    transform: str = "symkacz",
    accel: str | None = None,
    include_intercept: bool = True,
    schedule: str = "symmetric",
    fixest_algo: dict[str, Any] | None = None,
) -> NDArray[np.float64]: ...


@overload
def _demean_given_codes(  # noqa: PLR0913
    X: NDArray[np.float64],
    codes_list: list[NDArray[np.int64]],
    *,
    weights: Sequence[float] | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
    return_diagnostics: Literal[True],
    verify_foc: bool = False,
    backend: str = "reghdfe",
    transform: str = "symkacz",
    accel: str | None = None,
    include_intercept: bool = True,
    schedule: str = "symmetric",
    fixest_algo: dict[str, Any] | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]: ...


@overload
def _demean_given_codes(  # noqa: PLR0913
    X: NDArray[np.float64],
    codes_list: list[NDArray[np.int64]],
    *,
    weights: Sequence[float] | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
    return_diagnostics: bool = False,
    verify_foc: bool = False,
    backend: str = "reghdfe",
    transform: str = "symkacz",
    accel: str | None = None,
    include_intercept: bool = True,
    schedule: str = "symmetric",
    fixest_algo: dict[str, Any] | None = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], dict[str, Any]]: ...



def _demean_given_codes(  # noqa: PLR0913
    X: NDArray[np.float64],
    codes_list: list[NDArray[np.int64]],
    *,
    weights: Sequence[float] | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
    return_diagnostics: bool = False,
    verify_foc: bool = False,
    backend: str = "reghdfe",
    transform: str = "symkacz",
    accel: str | None = None,
    include_intercept: bool = True,
    schedule: str = "symmetric",
    fixest_algo: dict[str, Any] | None = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], dict[str, Any]]:
    """Execute alternating projections (MAP) to demean variables.

    Implements the specified algorithm (reghdfe/fixest style) with given
    tolerance and iteration limits.
    """
    # transform controls the projection scheduling: support reghdfe/fixest
    if backend not in {"reghdfe", "fixest"}:
        raise ValueError("backend must be 'reghdfe' or 'fixest'")
    transform = str(transform).lower() if transform is not None else None
    schedule = str(schedule).lower() if schedule is not None else None
    if transform not in {"kacz", "symkacz", "cimmino"}:
        raise ValueError("transform must be one of {'kacz','symkacz','cimmino'}")

    # Apply centralized defaults when caller did not provide explicit values
    defs = _defaults_for_backend(backend)
    if tol is None:
        tol = float(defs["tol"])
    if max_iter is None:
        max_iter = int(defs["max_iter"])
    transform = str(transform).lower()
    schedule = str(schedule).lower()

    # Allowed accelerators per backend
    if backend == "reghdfe":
        valid_accels = {"none", "aitken", "sd", "cg"}
        if accel is None:
            accel = "cg"
        if accel not in valid_accels:
            raise ValueError(
                f"accel must be one of {sorted(valid_accels)} for backend='reghdfe'",
            )
    else:
        # fixest: accel not accepted; control via fixest_algo
        if accel in ("it",):
            raise ValueError(
                "'it' accelerator is internal to fixest and not accepted publicly; use `fixest_algo`.",
            )
        if accel not in (None, "none"):
            raise ValueError(
                "backend='fixest' does not accept 'accel'; use `fixest_algo` to control fixest accelerators.",
            )
        accel = "none"
    if backend == "fixest":
        schedule = "cyclic"

    # Enforce symmetric transform when sd/cg accelerators are requested for reghdfe
    if backend == "reghdfe" and accel in {"sd", "cg"} and transform != "symkacz":
        transform = "symkacz"

    n = X.shape[0]
    # Keep a copy of original FE list (pre-intercept) for fixest warmup/top2 logic
    codes_all = list(codes_list)
    codes_list = list(codes_list)
    # fixest: reorder fixed effects by decreasing number of levels
    if backend == "fixest":
        levels = [int(c.max()) + 1 if c.size else 0 for c in codes_list]
        order = list(np.argsort(levels))[::-1]
        codes_all = [codes_all[i] for i in order]
        codes_list = [codes_list[i] for i in order]
    if include_intercept:
        codes_list = [*codes_list, np.zeros(n, dtype=np.int64)]

    # Prepare weight vector if provided
    w = _validate_weights(weights, n).reshape(-1, 1) if weights is not None else None

    Xw = X.copy()

    # fixest demeaning_algo parameters (normalized); for reghdfe they are zeros
    fixest_ALGO = _normalize_fixest_algo(fixest_algo, backend)
    extraProj = int(fixest_ALGO.get("extraProj", 0))
    iter_warmup = int(fixest_ALGO.get("iter_warmup", 0))
    iter_projAfterAcc = int(fixest_ALGO.get("iter_projAfterAcc", 0))
    iter_grandAcc = int(fixest_ALGO.get("iter_grandAcc", 0))
    if max_iter is None:
        max_iter = 16000 if backend == "reghdfe" else 10000

    # CG state carried across iterations for 1-step Fletcher-Reeves style update
    cg_state: dict[str, NDArray[np.float64] | None] = {"r_prev": None, "p_prev": None}

    # Irons & Tuck helper (vectorized extrapolation preserving fixed points)
    def irons_tuck_update(
        A_k: NDArray[np.float64],
        A_km1: NDArray[np.float64],
        A_km2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        d1 = A_km1 - A_km2
        d2 = A_k - A_km1
        delta = d2 - d1
        denom = float(np.sum(delta * delta))
        if not np.isfinite(denom) or denom <= 0.0:
            return A_k
        alpha = -float(np.sum(d2 * delta)) / (denom + 1e-300)
        return A_k + alpha * d2

    def subtract_means(
        A: NDArray[np.float64], codes: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        G = int(codes.max()) + 1 if codes.size else 0
        if G == 0:
            return A
        if w is None:
            ones = np.ones((A.shape[0], 1), dtype=np.float64)
            counts = group_sum(ones, codes, order="sorted").reshape(-1)
            sums = group_sum(A, codes, order="sorted")
            means = np.full((sums.shape[0], A.shape[1]), np.nan, dtype=np.float64)
            pos = counts > 0
            if np.any(pos):
                means[pos, :] = sums[pos, :] / counts[pos].reshape(-1, 1)
            return A - means[codes]
        # weighted
        Aw = A * w
        wsum = group_sum(w, codes, order="sorted").reshape(-1)
        sums = group_sum(Aw, codes, order="sorted")
        means = np.full((sums.shape[0], A.shape[1]), np.nan, dtype=np.float64)
        pos = wsum > 0
        # For groups with zero total weight, leave means as NaN (fixest semantics
        # may reinsert NaNs later; reghdfe path drops those rows upstream).
        if np.any(pos):
            with np.errstate(invalid="ignore", divide="ignore"):
                means[pos, :] = sums[pos, :] / wsum[pos].reshape(-1, 1)
        return A - means[codes]

    def sweep_once(A: NDArray[np.float64]) -> NDArray[np.float64]:
        return sweep_once_on(codes_list, A)

    def sweep_once_on(
        codes_sel: list[NDArray[np.int64]], A: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Run one projection sweep over the provided codes selection.

        This mirrors the original sweep_once logic but allows operating on a
        subset (used by fixest 2-FE warmup behaviour).
        """
        order = list(range(len(codes_sel)))
        # Sequential Kaczmarz or symmetric Kaczmarz (forward then backward)
        if transform in {"kacz", "symkacz"}:
            for j in order:
                A = subtract_means(A, codes_sel[j])
            if transform == "symkacz" or schedule == "symmetric":
                for j in reversed(order):
                    A = subtract_means(A, codes_sel[j])
            return A
        # Cimmino: simultaneous projections averaged across FE dims (reghdfe)
        if transform == "cimmino":
            corrections = None
            for codes in codes_sel:
                G = int(codes.max()) + 1 if codes.size else 0
                if G == 0:
                    continue
                Aj = subtract_means(A, codes)
                corrections = Aj if corrections is None else corrections + Aj
            if corrections is None:
                return A
            # Cimmino: simultaneous averaged projection — the average of each
            # dimension's projection becomes the next iterate (classic Cimmino).
            # Return the averaged projection itself (not A minus the average).
            return corrections / float(len(codes_sel))
        return A

    def sweep_top2(A: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sweep only the top-2 FE levels (used by fixest warmup/two-FE checks).

        If not applicable, returns A unchanged.
        """
        if len(codes_all) < 3:
            return A
        # pick top-2 by level counts (conservative heuristic)
        levels = [int(c.max()) + 1 if c.size else 0 for c in codes_all]
        # determine indices of FE dims sorted by descending size (fixest uses reorder)
        order_desc = list(np.argsort(levels))[::-1]
        top2_idx = order_desc[:2]
        # preserve forward-only cyclic order for the two FE dims
        codes_sel = [codes_all[i] for i in top2_idx]
        return sweep_once_on(codes_sel, A)

    # Two flavors of convergence metric: reghdfe uses max group-mean magnitude;
    # fixest uses max absolute change in group-means across iterations (coeff diff)
    def _group_means_list(
        A: NDArray[np.float64],
    ) -> list[NDArray[np.float64] | None]:
        out: list[NDArray[np.float64] | None] = []
        for codes in codes_list:
            G = int(codes.max()) + 1 if codes.size else 0
            if G == 0:
                out.append(None)
                continue
            if w is None:
                sums = group_sum(A, codes, order="sorted")
                cnts = group_sum(
                    np.ones((A.shape[0], 1), dtype=np.float64), codes, order="sorted",
                ).reshape(-1)
                means = np.divide(
                    sums,
                    cnts.reshape(-1, 1),
                    out=np.zeros_like(sums),
                    where=cnts.reshape(-1, 1) > 0,
                )
            else:
                sums = group_sum(A * w, codes, order="sorted")
                wsum = group_sum(w, codes, order="sorted").reshape(-1)
                means = np.divide(
                    sums,
                    wsum.reshape(-1, 1),
                    out=np.zeros_like(sums),
                    where=wsum.reshape(-1, 1) > 0,
                )
            out.append(means)
        return out

    def max_violation_reghdfe(A: NDArray[np.float64]) -> float:
        mv = 0.0
        gml = _group_means_list(A)
        for means in gml:
            if means is None:
                continue
            mv = max(mv, float(np.nanmax(np.abs(means))))
        return mv

    def max_violation_fixest(
        A_new: NDArray[np.float64], A_old: NDArray[np.float64],
    ) -> float:
        """fixest: stop when max absolute increase of FE coefficients (group means)
        is <= tol. This mirrors fixest's documented stopping rule which tracks
        changes in the FE "coefficients" (group means) rather than raw matrix
        element differences.
        """
        gm_new = _group_means_list(A_new)
        if A_old is None:
            vals = [np.nanmax(np.abs(m)) for m in gm_new if m is not None]
            return float(0.0 if len(vals) == 0 else np.nanmax(vals))
        gm_old = _group_means_list(A_old)
        diffs = []
        for mn, mo in zip(gm_new, gm_old):
            if (mn is not None) and (mo is not None):
                diffs.append(np.nanmax(np.abs(mn - mo)))
        return float(0.0 if len(diffs) == 0 else np.nanmax(diffs))

    def max_violation_for_codes(
        A: NDArray[np.float64], codes_sel: list[NDArray[np.int64]],
    ) -> float:
        """Compute max absolute group-mean magnitude for a subset of codes.

        Used to evaluate two-FE convergence during fixest warmup.
        """
        mv = 0.0
        for codes in codes_sel:
            G = int(codes.max()) + 1 if codes.size else 0
            if G == 0:
                continue
            if w is None:
                sums = group_sum(A, codes, order="sorted")
                cnts = group_sum(
                    np.ones((A.shape[0], 1), dtype=np.float64), codes, order="sorted",
                ).reshape(-1)
                means = np.divide(
                    sums,
                    cnts.reshape(-1, 1),
                    out=np.zeros_like(sums),
                    where=cnts.reshape(-1, 1) > 0,
                )
            else:
                sums = group_sum(A * w, codes, order="sorted")
                wsum = group_sum(w, codes, order="sorted").reshape(-1)
                means = np.divide(
                    sums,
                    wsum.reshape(-1, 1),
                    out=np.zeros_like(sums),
                    where=wsum.reshape(-1, 1) > 0,
                )
            mv = max(mv, float(np.nanmax(np.abs(means))))
        return mv

    def max_change_for_codes(
        A_new: NDArray[np.float64],
        A_old: NDArray[np.float64],
        codes_sel: list[NDArray[np.int64]],
    ) -> float:
        """Compute the maximum absolute change in group means between two iterates.

        This mirrors fixest's stopping rule which monitors coefficient changes
        (group means) rather than absolute magnitudes.
        """
        mv = 0.0
        for codes in codes_sel:
            G = int(codes.max()) + 1 if codes.size else 0
            if G == 0:
                continue
            if w is None:
                sums_new = group_sum(A_new, codes, order="sorted")
                sums_old = group_sum(A_old, codes, order="sorted")
                cnts = group_sum(
                    np.ones((A_new.shape[0], 1), dtype=np.float64),
                    codes,
                    order="sorted",
                ).reshape(-1)
                mn_new = np.divide(
                    sums_new,
                    cnts.reshape(-1, 1),
                    out=np.zeros_like(sums_new),
                    where=cnts.reshape(-1, 1) > 0,
                )
                mn_old = np.divide(
                    sums_old,
                    cnts.reshape(-1, 1),
                    out=np.zeros_like(sums_old),
                    where=cnts.reshape(-1, 1) > 0,
                )
            else:
                sums_new = group_sum(A_new * w, codes, order="sorted")
                sums_old = group_sum(A_old * w, codes, order="sorted")
                wsum = group_sum(w, codes, order="sorted").reshape(-1)
                mn_new = np.divide(
                    sums_new,
                    wsum.reshape(-1, 1),
                    out=np.zeros_like(sums_new),
                    where=wsum.reshape(-1, 1) > 0,
                )
                mn_old = np.divide(
                    sums_old,
                    wsum.reshape(-1, 1),
                    out=np.zeros_like(sums_old),
                    where=wsum.reshape(-1, 1) > 0,
                )
            mv = max(mv, float(np.nanmax(np.abs(mn_new - mn_old))))
        return mv

    converged = False
    it = 0
    final_diag: dict[str, object] = {"n_iter": 0, "max_violation": None, "converged": False}
    # keep last two iterates for accelerators (k-2, k-1)
    from collections import deque

    _hist: deque[NDArray[np.float64]] = deque(maxlen=2)  # stores [X_{k-2}, X_{k-1}]
    # For Aitken Δ² we also want the immediate previous two updates
    last_update: NDArray[np.float64] | None = None
    last_last_update: NDArray[np.float64] | None = None
    # fixest warmup bookkeeping
    if backend == "fixest" and len(codes_all) >= 3:
        # precompute top-2 FE indices by level size (descending)
        _levels_all = [int(c.max()) + 1 if c.size else 0 for c in codes_all]
        _order_desc = list(np.argsort(_levels_all))[::-1]
        _top2_codes = [codes_all[i] for i in _order_desc[:2]]

    # fixest demeaning_algo defaults (public docs) - normalized via helper
    fixest_ALGO = _normalize_fixest_algo(fixest_algo, backend)

    while it < max_iter:
        it += 1
        # snapshot before this iteration's update
        X_prev = Xw.copy()
        # keep up to last two iterates for accelerators
        _hist.append(X_prev)

        # fixest warmup: after iter_warmup iterations, run 2-FE sweeps to convergence
        if backend == "fixest" and len(codes_all) >= 3 and it == max(0, iter_warmup):
            it2 = 0
            prev = None
            mv_top2 = float("inf")
            while (it + it2) < max_iter:
                prev = Xw.copy()
                Xw = sweep_once_on(_top2_codes, Xw)
                mv_top2 = max_change_for_codes(Xw, prev, _top2_codes)
                it2 += 1
                if mv_top2 <= tol:
                    break
            # If we reached max_iter without meeting tolerance, raise an informative error
            if it2 == 0:
                raise RuntimeError(
                    "fixest warmup (top-2 FEs) could not run: max_iter is too small relative to iter_warmup.",
                )
            if (it + it2) >= max_iter and mv_top2 > tol:
                raise RuntimeError(
                    "fixest warmup (top-2 FEs) did not converge within max_iter during demeaning.",
                )

        # Acceleration / stepping
        # Note: CG accelerator requires a symmetric transform (e.g. 'symkacz').
        # The code enforces this by switching transform to 'symkacz' when
        # accel in {'sd','cg'} is requested for backend='reghdfe'.
        if backend == "reghdfe" and accel in {"aitken", "sd", "cg"}:
            # prepare operator f and B = I - f
            def apply_f(A: NDArray[np.float64]) -> NDArray[np.float64]:
                return sweep_once(A)

            def apply_b(V: NDArray[np.float64]) -> NDArray[np.float64]:
                return V - apply_f(V)

            Y1 = apply_f(X_prev)
            Y2 = apply_f(Y1)

            if accel == "aitken":
                # Aitken Δ² extrapolation applied to the projection updates
                # update = Y2 - Y1 (the step from k+1 to k+2)
                update = Y2 - Y1
                if last_update is None or last_last_update is None:
                    # fallback to plain Y2 if insufficient history
                    Xw = Y2
                else:
                    d1_u = update - last_update
                    d0_u = last_update - last_last_update
                    denom_arr = d1_u - d0_u
                    # elementwise safe Δ² extrapolation
                    mask = np.abs(denom_arr) > 1e-15
                    accel_u = np.zeros_like(update)
                    accel_u[mask] = (
                        update[mask] - (d1_u[mask] * d1_u[mask]) / denom_arr[mask]
                    )
                    accel_u[~mask] = update[~mask]
                    Xw = Y2 + accel_u
                # shift history
                last_last_update = last_update
                last_update = update
            elif accel == "sd":
                X0 = X_prev
                r = X0 - Y1
                Br = apply_b(r)
                denom = float(np.sum(r * Br))
                if denom <= 0:
                    Xw = Y1
                else:
                    alpha = float(np.sum(r * r) / denom)
                    Xw = X0 - alpha * r
            else:  # 'cg'
                X0 = X_prev
                r = X0 - Y1
                r_prev = cg_state["r_prev"]
                p_prev = cg_state["p_prev"]
                if (r_prev is None) or (p_prev is None):
                    p = r.copy()
                else:
                    denom_beta = max(1e-300, float(np.sum(r_prev * r_prev)))
                    beta = float(np.sum(r * r) / denom_beta)
                    p = r + beta * p_prev
                Bp = apply_b(p)
                denom = float(np.sum(p * Bp))
                if denom <= 0:
                    Xw = Y1
                    cg_state["r_prev"] = None
                    cg_state["p_prev"] = None
                else:
                    alpha = float(np.sum(r * p) / denom)
                    Xw = X0 - alpha * p
                    cg_state["r_prev"] = r
                    cg_state["p_prev"] = p
            # reghdfe accelerators handled above; fixest uses its own accel flow below
        # fixest: At each iteration apply Irons-Tuck acceleration to f applied twice
        elif backend == "fixest":

            def apply_f(A: NDArray[np.float64]) -> NDArray[np.float64]:  # one full sweep on all FEs (cyclic Kaczmarz)
                return sweep_once(A)

            Y1 = apply_f(X_prev)
            Y2 = apply_f(Y1)
            Xw = irons_tuck_update(Y2, Y1, X_prev)
            # extraProj: for parity with fixest, insert 3 * extraProj
            # additional plain (unaccelerated) projections per iteration.
            # This mirrors fixest's internal warmup/projection policy.
            if extraProj > 0:
                for _ in range(3 * extraProj):
                    Xw = apply_f(Xw)
            # iter_projAfterAcc: after a threshold, add one plain projection right after acceleration
            if iter_projAfterAcc > 0 and it >= iter_projAfterAcc:
                Xw = apply_f(Xw)
            # grand acceleration over f^k every 2*k iterations
            if iter_grandAcc > 0 and (it % (2 * iter_grandAcc) == 0):

                def apply_f_k(A: NDArray[np.float64]) -> NDArray[np.float64]:
                    Z = A
                    for _ in range(iter_grandAcc):
                        Z = apply_f(Z)
                    return Z

                H1 = apply_f_k(X_prev)
                H2 = apply_f_k(H1)
                Xw = irons_tuck_update(H2, H1, X_prev)
        else:
            # reghdfe: plain MAP step
            Xw = sweep_once(Xw)

        # store the new iterate for next round (history length <= 2)
        _hist.append(Xw.copy())

        # Choose convergence metric depending on backend
        if backend == "reghdfe":
            mv = max_violation_reghdfe(Xw)
        else:
            mv = max_violation_fixest(Xw, X_prev)
        final_diag["n_iter"] = it
        final_diag["max_violation"] = float(mv)
        if mv <= tol:
            converged = True
            break

    if not converged:
        msg = f"demean: no convergence under strict tolerance; max_violation={final_diag.get('max_violation')} after {it} iterations."
        raise RuntimeError(
            msg,
        )

    final_diag["converged"] = True
    # FOC verification (strict): ensure per-FE group means (first-order
    # conditions) are numerically near zero after convergence. When
    # verify_foc=True we compute group means per FE dimension (weighted if
    # weights were provided) and assert the maximum absolute group-mean
    # across all FE dims and columns is <= tol. Raise AssertionError on
    # violation to make this a strict, machine-enforceable contract.
    if verify_foc:
        gml = _group_means_list(Xw)
        vio = 0.0
        for gm in gml:
            if gm is None:
                continue
            vio = max(vio, float(np.nanmax(np.abs(gm))))
        if (not np.isfinite(vio)) or vio > float(tol):
            raise AssertionError(
                f"FOC violated: max |group mean| = {vio:.3e} > tol={float(tol):.3e}",
            )
    if return_diagnostics:
        return Xw, final_diag
    return Xw


# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------


def demean(  # noqa: PLR0913
    X: NDArray[np.float64],
    fe_ids: ArrayLike | Sequence[ArrayLike],
    *,
    weights: Sequence[float] | None = None,
    context: str = "ols",
    tol: float | None = None,
    max_iter: int | None = None,
    na_action: str = "drop",
    acceleration: bool = False,
    accel: str | None = None,
    schedule: str = "symmetric",
    drop_singletons: bool = True,
    include_intercept: bool = True,
    backend: str = "reghdfe",
    fixest_algo: dict[str, Any] | None = None,
    verify_foc: bool = False,
) -> NDArray[np.float64]:
    """Canonical helper: returns X with fixed effects absorbed.

    Thin wrapper over :func:`absorb` that returns only the transformed X.
    """
    if acceleration:
        raise ValueError(
            "`acceleration` is not supported. Use `accel` or `fixest_algo` to configure acceleration.",
        )
    if na_action != "drop":
        raise ValueError("Strict FE: na_action must remain 'drop'.")
    accel_value = (
        accel if accel is not None else ("cg" if backend == "reghdfe" else "none")
    )
    zero_y = np.zeros((X.shape[0], 1), dtype=np.float64)
    res = absorb(
        X,
        zero_y,
        fe_ids,
        Z=None,
        weights=weights,
        allow_weights=(context == "gls"),
        context=context,
        tol=tol,
        max_iter=max_iter,
        drop_na_fe_ids=True,
        drop_singletons=drop_singletons,
        drop_zero_weight_groups=True,
        backend=backend,
        schedule=schedule,
        diagnostics=False,
        accel=accel_value,
        include_intercept=include_intercept,
        fixest_algo=fixest_algo,
        verify_foc=verify_foc,
    )
    return res.X


def _absorb_impl(  # noqa: PLR0913
    X: NDArray[np.float64],
    Z: NDArray[np.float64],
    y: NDArray[np.float64],
    fe_ids: ArrayLike | Sequence[ArrayLike],
    *,
    weights: Sequence[float] | None = None,
    allow_weights: bool = False,
    context: str = "ols",
    tol: float | None = None,
    max_iter: int | None = None,
    drop_na_fe_ids: bool = True,
    drop_singletons: bool = True,
    drop_zero_weight_groups: bool = True,
    backend: str = "reghdfe",
    schedule: str = "symmetric",
    diagnostics: bool = False,
    accel: str = "none",
    include_intercept: bool = True,
    fixest_algo: dict[str, Any] | None = None,
    verify_foc: bool = False,
    na_action: str = "drop",
) -> FETransformResult:
    """Jointly demean (X, Z, y) with identical row filtering and FE absorption.

    Parameters mirror `demean_xy` with the addition of matrix Z. By default we
    drop FE singletons (observations whose group appears only once in *any* FE
    dimension), following best practice in high-dimensional FE literature
    (reghdfe / fixest) to avoid leverage pathologies. Set drop_singletons=False
    to retain them (a warning is issued in that case).
    Acceleration options: valid accel values are 'none', 'aitken', 'sd', and 'cg'.
    Note: 'sd' and 'cg' require a symmetric transform (e.g., backend='reghdfe'
    with transform='symkacz'). When ``diagnostics`` is True the returned
    :class:`FETransformResult` contains convergence metadata.
    """
    if (weights is not None) and (not allow_weights) and (context != "gls"):
        raise ValueError(
            "weights are forbidden outside GLS/GMM context (OLS/IV/QR/IV-QR must be unweighted).",
        )
    if na_action != "drop":
        msg = "Strict FE: na_action='drop' only permitted."
        raise ValueError(msg)
    # Default accel per backend
    if accel is None:
        accel = "cg" if backend == "reghdfe" else "none"
    if backend == "reghdfe":
        valid_accels = {"none", "aitken", "sd", "cg"}
        if accel not in valid_accels:
            raise ValueError(
                f"accel must be one of {sorted(valid_accels)} for backend='reghdfe'",
            )
    elif accel not in {None, "none"}:
        raise ValueError(
            "backend='fixest' does not accept 'accel'; use `fixest_algo` to control fixest accelerators.",
        )
    if schedule not in {"symmetric", "cyclic"}:
        msg = "schedule must be 'symmetric' or 'cyclic'"
        raise ValueError(msg)
    # Ensure backend is valid
    if backend not in {"reghdfe", "fixest"}:
        msg = "backend must be 'reghdfe' or 'fixest'"
        raise ValueError(msg)
    # Enforce project-wide policy: weights only allowed in GLS/GMM context
    if context != "gls" and weights is not None:
        raise ValueError(
            "weights are forbidden outside GLS/GMM context (OLS/IV/QR/IV-QR must be unweighted).",
        )
    if tol is None:
        tol = 1e-8 if backend == "reghdfe" else 1e-6
    if max_iter is None:
        max_iter = 16000 if backend == "reghdfe" else 10000
    raw_fe_list = fe_ids if isinstance(fe_ids, (list, tuple)) else [fe_ids]

    # Defensive validation: ensure FE IDs are properly structured
    # Handle pandas DataFrame: split each column into a separate 1D array
    if hasattr(raw_fe_list[0], "columns"):  # DataFrame-like
        df = raw_fe_list[0]
        raw_fe_list = [df[col].to_numpy() for col in df.columns]
    elif isinstance(raw_fe_list[0], np.ndarray) and raw_fe_list[0].ndim == 2:
        # If first element is 2D ndarray, split columns into separate arrays
        fe_arr = raw_fe_list[0]
        raw_fe_list = [fe_arr[:, j] for j in range(fe_arr.shape[1])]

    n = X.shape[0]
    mask = np.ones(n, dtype=bool)

    # Track dropped observations for statistics
    n_initial = n
    n_na_dropped = 0
    n_fe_id_na_dropped = 0
    n_singleton_dropped = 0
    n_zero_weight_dropped = 0

    # Optionally drop rows with missing FE ids; if False we keep FE-NA as a
    # separate category (encoded later by _to_codes).
    if drop_na_fe_ids:
        for z in raw_fe_list:
            fe_mask = _isfinite_like(z)
            n_fe_id_na_dropped += int(np.sum(~fe_mask & mask))
            mask &= fe_mask

    # Drop rows with NA in y
    y_mask = np.isfinite(y.reshape(-1))
    n_na_dropped += int(np.sum(~y_mask & mask))
    mask &= y_mask

    if weights is not None:
        w_all = _validate_weights(weights, n, allow_zero=True)
        if np.any(w_all < 0):
            raise ValueError("weights must be non-negative.")
        w_mask = np.isfinite(w_all) & (w_all >= 0)
        n_na_dropped += int(np.sum(~w_mask & mask))
        mask &= w_mask
    else:
        w_all = None

    # Projection mask: start from the basic mask (NA in FE IDs, y, weights).
    # We maintain a separate mask for the rows actually used in the alternating
    # projections because some rows (e.g. in zero-total-weight FE groups) must
    # be excluded from the projection operator to avoid undefined weighted means.
    mask_proj = mask.copy()

    # two_core option is not part of reghdfe/fixest public APIs
    # two_core removed: non-standard option not part of reghdfe/fixest public API

    # Strict handling of zero-total-weight FE groups according to mode
    zero_row_mask = np.zeros(n, dtype=bool)
    if drop_zero_weight_groups:
        # compute counts on the currently valid mask (NA/invalid weights removed)
        idx = np.nonzero(mask)[0]
        for j in range(len(raw_fe_list)):
            raw_masked = np.asarray(raw_fe_list[j])[idx].reshape(-1)
            _, inv = np.unique(raw_masked, return_inverse=True)
            inv = inv.astype(np.int64, copy=False)
            if weights is None:
                counts = group_sum(
                    np.ones((idx.size, 1), dtype=np.float64), inv, order="sorted",
                ).reshape(-1)
            else:
                if w_all is None:
                    raise RuntimeError("internal error: weights provided but w_all is None")
                w_masked = w_all[idx].reshape(-1, 1)
                counts = group_sum(w_masked, inv, order="sorted").reshape(-1)
            zero_groups = counts == 0
            if np.any(zero_groups):
                zero_row_mask[idx] |= zero_groups[inv]

        # Exclude zero-total-weight groups from the projection sample.
        # These rows carry zero weight and do not affect estimation. We keep the
        # demeaned arrays on the retained sample and expose `mask` to map back.
        n_zero_weight_dropped = int(np.sum(zero_row_mask & mask_proj))
        mask_proj &= ~zero_row_mask

    # Compute FE codes on the projection sample only.
    fe_ids_masked = [np.asarray(z)[mask_proj] for z in raw_fe_list]
    codes_list = _prepare_and_prune_fe(fe_ids_masked)

    n_before_singleton = int(np.sum(mask_proj))
    if drop_singletons:
        w_for_drop = (
            None
            if weights is None
            else (
                (w_all[mask_proj] if w_all is not None else None)
                if "w_all" in locals()
                else None
            )
        )
        # enforce recursive singleton removal for parity with reghdfe/fixest
        style = "reghdfe_iter"
        keep_mask = _drop_singletons_iteratively(
            codes_list, weights=w_for_drop, style=style,
        )
        full_mask = np.zeros(n, dtype=bool)
        full_mask[np.nonzero(mask_proj)[0][keep_mask]] = True
        n_singleton_dropped = n_before_singleton - int(np.sum(full_mask))
    else:
        full_mask = mask_proj
        n_singleton_dropped = 0

    X2 = X[full_mask]
    Z2 = Z[full_mask]
    y2 = y.reshape(-1, 1)[full_mask]
    # Recompute FE codes on the final kept rows to avoid length-mismatch bugs
    # (codes_list is built on the masked sample, not full n).
    fe_ids_final = [np.asarray(z)[full_mask] for z in raw_fe_list]
    codes2 = _prepare_and_prune_fe(fe_ids_final)
    # fixest: reorder by number of levels (unique categories), descending
    if backend == "fixest":
        levels2 = [int(c.max()) + 1 if c.size else 0 for c in codes2]
        order2 = np.argsort(levels2)[::-1]
        codes2 = [codes2[i] for i in order2]
    w2 = (
        None
        if weights is None
        else (
            (w_all[full_mask] if w_all is not None else None)
            if "w_all" in locals()
            else _validate_weights(weights, n)[full_mask]
        )
    )
    # Stack X,Z,y and project together
    XYZ = np.asarray(hstack([X2, Z2, y2]), dtype=np.float64, order="C")
    # Choose transform per backend and pass backend/transform through to internal routine
    transform = "kacz" if backend == "fixest" else "symkacz"
    XYZ_tilde = _demean_given_codes(
        XYZ,
        codes2,
        weights=cast(Sequence[float] | None, w2),
        tol=tol,
        max_iter=max_iter,
        backend=backend,
        transform=transform,
        accel=accel,
        schedule=schedule,
        include_intercept=include_intercept,
        fixest_algo=fixest_algo,
        return_diagnostics=diagnostics,
        verify_foc=verify_foc,
    )
    p = X.shape[1]
    q = Z.shape[1]
    if diagnostics:
        XYZ_tilde_arr, diag = cast(
            tuple[NDArray[np.float64], dict[str, Any]],
            XYZ_tilde,
        )
    else:
        diag = None
        XYZ_tilde_arr = cast(NDArray[np.float64], XYZ_tilde)
    X_tilde = XYZ_tilde_arr[:, :p]
    Z_tilde = XYZ_tilde_arr[:, p : p + q]
    # Preserve column vector shape for y (critical for bootstrap shape compatibility)
    y_tilde = XYZ_tilde_arr[:, p + q :].reshape(-1, 1)

    # Note: rows in zero-total-weight FE groups are excluded from the projection
    # sample (mask_proj) because weighted means are undefined there. Since such
    # observations carry zero weight, dropping them does not affect estimation.
    # We therefore keep the demeaned arrays in the filtered (kept) sample
    # throughout, and expose `mask` for mapping back to the original sample.

    dropped_stats = {
        "na": n_na_dropped,
        "fe_id_na": n_fe_id_na_dropped,
        "singleton": n_singleton_dropped,
        "zero_weight": n_zero_weight_dropped,
    }

    fe_codes_out = [np.asarray(c) for c in codes2]
    weights_out = None if weights is None else (None if w2 is None else np.asarray(w2))

    diag_payload: dict[str, Any] | None = None
    if diagnostics:
        diag_payload = dict(diag or {})
        diag_payload["fe_codes"] = [np.asarray(c) for c in codes2]
        diag_payload["mask"] = full_mask.copy()
        if weights_out is not None:
            diag_payload["weights_aligned"] = weights_out

    return FETransformResult(
        X=X_tilde,
        y=y_tilde,
        Z=Z_tilde,
        mask=full_mask.copy(),
        dropped=dropped_stats,
        fe_codes=fe_codes_out,
        weights=weights_out,
        diagnostics=diag_payload,
    )


def absorb(  # noqa: PLR0913
    X: ArrayLike,
    y: ArrayLike,
    fe_ids: ArrayLike | Sequence[ArrayLike],
    *,
    Z: ArrayLike | None = None,
    weights: Sequence[float] | None = None,
    allow_weights: bool = False,
    context: str = "ols",
    tol: float | None = None,
    max_iter: int | None = None,
    drop_na_fe_ids: bool = True,
    drop_singletons: bool = True,
    drop_zero_weight_groups: bool = True,
    backend: str = "reghdfe",
    schedule: str = "symmetric",
    diagnostics: bool = False,
    accel: str = "none",
    na_action: str = "drop",
    include_intercept: bool = True,
    fixest_algo: dict[str, Any] | None = None,
    verify_foc: bool = False,
) -> FETransformResult:
    """High-level FE absorption entry point returning :class:`FETransformResult`.

    Parameters mirror :func:`_absorb_impl` with the addition that ``Z`` may be
    omitted (defaults to a zero-column matrix). Inputs are converted to NumPy
    arrays with dtype float64 before processing to guarantee consistent numeric
    kernels across estimators.
    """
    X_arr = np.asarray(X, dtype=np.float64, order="C")
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    Z_arr: NDArray[np.float64]
    if Z is None:
        Z_arr = np.zeros((X_arr.shape[0], 0), dtype=np.float64)
    else:
        Z_arr = np.asarray(Z, dtype=np.float64, order="C")
        if Z_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("Z must have the same number of rows as X.")

    return _absorb_impl(
        X_arr,
        Z_arr,
        y_arr,
        fe_ids,
        weights=weights,
        allow_weights=allow_weights,
        context=context,
        tol=tol,
        max_iter=max_iter,
        drop_na_fe_ids=drop_na_fe_ids,
        drop_singletons=drop_singletons,
        drop_zero_weight_groups=drop_zero_weight_groups,
        backend=backend,
        schedule=schedule,
        diagnostics=diagnostics,
        accel=accel,
        na_action=na_action,
        include_intercept=include_intercept,
        fixest_algo=fixest_algo,
        verify_foc=verify_foc,
    )


def demean_xyz(  # noqa: PLR0913
    X: NDArray[np.float64],
    Z: NDArray[np.float64],
    y: NDArray[np.float64],
    fe_ids: ArrayLike | Sequence[ArrayLike],
    *,
    weights: Sequence[float] | None = None,
    allow_weights: bool = False,
    context: str = "ols",
    tol: float | None = None,
    max_iter: int | None = None,
    na_action: str = "drop",
    drop_na_fe_ids: bool = True,
    drop_singletons: bool = True,
    return_mask: bool = False,
    return_dropped_stats: bool = False,
    drop_zero_weight_groups: bool = True,
    backend: str = "reghdfe",
    schedule: str = "symmetric",
    diagnostics: bool = False,
    acceleration: bool = False,
    accel: str = "none",
    include_intercept: bool = True,
    fixest_algo: dict[str, Any] | None = None,
    verify_foc: bool = False,
) -> tuple[Any, ...]:
    """Canonical tuple-returning wrapper over :func:`absorb`.

    Returns tuples matching the (X, Z, y, ...) contract.
    """
    if acceleration:
        raise ValueError(
            "`acceleration` is not supported. Use `accel` or `fixest_algo` to control acceleration.",
        )
    if na_action != "drop":
        raise ValueError("Strict FE: na_action must remain 'drop'.")

    result = absorb(
        X,
        y,
        fe_ids,
        Z=Z,
        weights=weights,
        allow_weights=allow_weights,
        context=context,
        tol=tol,
        max_iter=max_iter,
        drop_na_fe_ids=drop_na_fe_ids,
        drop_singletons=drop_singletons,
        drop_zero_weight_groups=drop_zero_weight_groups,
        backend=backend,
        schedule=schedule,
        diagnostics=diagnostics,
        accel=accel,
        na_action=na_action,
        include_intercept=include_intercept,
        fixest_algo=fixest_algo,
        verify_foc=verify_foc,
    )

    payload: list[Any] = [
        result.X,
        result.Z if result.Z is not None else np.zeros((result.X.shape[0], 0)),
        result.y,
    ]
    if return_mask:
        payload.append(result.mask)
    if return_dropped_stats:
        payload.append(result.dropped)
    if diagnostics:
        payload.append(result.diagnostics or {})
    if len(payload) == 3:
        return tuple(payload[:3])
    return tuple(payload)


def demean_xy(  # noqa: PLR0913
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    fe_ids: ArrayLike | Sequence[ArrayLike],
    *,
    weights: Sequence[float] | None = None,
    allow_weights: bool = False,
    context: str = "ols",
    tol: float | None = None,
    max_iter: int | None = None,
    na_action: str = "drop",
    drop_na_fe_ids: bool = True,
    drop_singletons: bool = True,
    return_mask: bool = False,
    return_dropped_stats: bool = False,
    drop_zero_weight_groups: bool = True,
    backend: str = "reghdfe",
    schedule: str = "symmetric",
    diagnostics: bool = False,
    acceleration: bool = False,
    accel: str = "none",
    include_intercept: bool = True,
    fixest_algo: dict[str, Any] | None = None,
    verify_foc: bool = False,
) -> tuple[Any, ...]:
    """Canonical wrapper returning (X, y) with fixed effects absorbed."""
    if acceleration:
        raise ValueError(
            "`acceleration` is not supported. Use `accel` or `fixest_algo` to control acceleration.",
        )
    if na_action != "drop":
        raise ValueError("Strict FE: na_action must remain 'drop'.")
    result = absorb(
        X,
        y,
        fe_ids,
        Z=None,
        weights=weights,
        allow_weights=allow_weights,
        context=context,
        tol=tol,
        max_iter=max_iter,
        drop_na_fe_ids=drop_na_fe_ids,
        drop_singletons=drop_singletons,
        drop_zero_weight_groups=drop_zero_weight_groups,
        backend=backend,
        schedule=schedule,
        diagnostics=diagnostics,
        accel=accel,
        na_action=na_action,
        include_intercept=include_intercept,
        fixest_algo=fixest_algo,
        verify_foc=verify_foc,
    )
    payload: list[Any] = [result.X, result.y]
    if return_mask:
        payload.append(result.mask)
    if return_dropped_stats:
        payload.append(result.dropped)
    if diagnostics:
        payload.append(result.diagnostics or {})
    if len(payload) == 2:
        return tuple(payload[:2])
    return tuple(payload)


# NOTE: parallel simultaneous updates were removed. Simultaneous parallel
# projection is not mathematically equivalent to sequential MAP/SOR iterations
# used by reghdfe/fixest and therefore is not provided as a public API.


def vectorized_fe_absorption(
    X: NDArray[np.float64],
    fe_codes: Sequence[NDArray[np.int64]],
    *,
    weights: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Vectorized fixed-effect absorption using alternating projections only.

    This function enforces strict parity with reghdfe/fixest: only
    alternating (MAP / sequential Kaczmarz-like) projection is supported.
    Simultaneous pinv-based absorption is a non-standard approach and is
    deliberately not provided.

    Notes
    -----
    - Analytic observation weights are not allowed on the strict OLS/FE
      path; if weights are required use GLS/GMM-style APIs.
    - The global intercept should be included only on the first call when
      repeatedly applying `demean` across multiple FE dimensions.

    """
    if weights is not None:
        raise ValueError(
            "vectorized_fe_absorption: analytic per-observation weights are not allowed on the strict OLS/FE path; use GLS/GMM APIs.",
        )
    # Perform joint multi-way absorption by calling the public `demean`
    # implementation with all FE codes. Use documented reghdfe defaults for
    # joint absorption (symmetric schedule, cg acceleration allowed).
    return demean(
        X,
        fe_codes,
        weights=None,
        include_intercept=True,
        backend="reghdfe",
        accel="cg",
        schedule="symmetric",
    )


def sparse_fe_solver(
    X: NDArray[np.float64] | sparse.spmatrix,
    y: NDArray[np.float64],
    fe_codes: Sequence[NDArray[np.int64]],
    *,
    weights: Sequence[float] | None = None,
    lambda_reg: float = 0.0,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Sparse-aware fixed effects regression solver (exact sparse QR with R/Stata parity).

    Solves FE regression using exact column-pivoted QR (QRCP) for both sparse
    and dense inputs. Sparse matrices use SuiteSparseQR (sparseqr); dense use
    SciPy GEQP3. Rank determination and coefficient filling follow strict R/Stata
    conventions (see vectorized_qr_solve).

    Parameters
    ----------
    X : ndarray or sparse matrix
        Design matrix of shape (n, p).
    y : ndarray
        Response vector of shape (n,).
    fe_codes : sequence of ndarray
        List of FE codes, each of shape (n,).
    weights : sequence, optional
        Not allowed for strict FE OLS (use GLS/GMM APIs for weighted estimation).
    lambda_reg : float, optional
        Ridge regularization parameter.

    Returns
    -------
    beta : ndarray
        Coefficients of shape (p,).
    fe_effects : list of ndarray
        Fixed effects for each FE dimension (sum-to-zero normalization).

    """
    from .linalg import vectorized_qr_solve

    y = y.reshape(-1, 1)

    # Strict OLS parity: weights are not allowed in FE OLS absorption
    if weights is not None:
        msg = "sparse_fe_solver: analytic per-observation weights are not allowed for strict FE OLS; use GLS/GMM APIs."
        raise ValueError(msg)

    # Disallow ridge regularization for strict FE OLS sparse solver.
    if float(lambda_reg) != 0.0:
        raise ValueError(
            "sparse_fe_solver: lambda_reg (ridge) is not allowed for strict FE OLS; set lambda_reg=0.0 or use a GLS/GMM solver that supports regularization.",
        )

    # Absorb FEs from X and y (unweighted)
    X_dm = vectorized_fe_absorption(X, fe_codes, weights=None)
    y_dm = vectorized_fe_absorption(y, fe_codes, weights=None)

    # Note: ridge branch intentionally removed; regularization must be handled
    # by dedicated GLS/GMM APIs which perform proper weighting/whitening.

    # Exact solve: sparse→SuiteSparseQR, dense→GEQP3 (both QRCP + R/Stata rank)
    beta = vectorized_qr_solve(X_dm, y_dm, rank_policy="stata").reshape(-1)

    # Recover FE under sum-to-zero normalization (strict R/Stata parity)
    X_orig = to_dense(X) if hasattr(X, "toarray") or sparse.issparse(X) else X
    r_full = y.reshape(-1, 1) - dot(X_orig, beta.reshape(-1, 1))
    fe_effects = recover_fe(
        fe_codes, r_full, normalization="sum_to_zero", backend="reghdfe",
    )
    return beta, fe_effects


def recover_fe(  # noqa: PLR0913
    fe_codes: Sequence[NDArray[np.int64]],
    residuals: NDArray[np.float64],
    *,
    normalization: str = "auto",
    refs: dict[int, Any] | None = None,
    level_order: dict[int, Sequence[int]] | None = None,
    backend: str | None = None,
) -> list[NDArray[np.float64]]:
    """Recover fixed-effect coefficients under explicit normalization (strict R/Stata parity).

    New implementation (strict R/Stata parity):
    Solve  min_a 1/2 || D a - r ||^2  s.t.  R a = 0,
    where D=[D1 ... DJ] stacks one-hot incidence for each FE dimension
    and R imposes sum-to-zero within each FE dimension and connected component.
    We build connected components over FE levels by union-find across
    all dimensions using co-occurrence within observations.
    The KKT system [D'D  R'; R  0][a; λ] = [D'r; 0] is solved via QR/SVD.

    This routine recovers FE coefficients under either 'sum_to_zero' or
    'references' normalization. If normalization='auto' the backend determines
    the default: 'references' for backend=='fixest' and 'sum_to_zero' for
    backend=='reghdfe' (or 'sum_to_zero' if backend is None).

    - sum_to_zero: impose that the FE coefficients within each connected
      component sum to zero (KKT rows = one row per FE-dimension present).
    - references: for each FE-dimension and connected component pick a
      reference level (the smallest-level index observed) and fix its
      coefficient to zero. This matches fixest's returned fixef references.

    Parameters
    ----------
    fe_codes : sequence of ndarray
        List of FE codes for each dimension, each of shape (n,).
    residuals : ndarray
        Residuals from the regression, shape (n,) or (n, 1).
    normalization : str, optional
        'auto' (choose by backend), 'sum_to_zero', or 'references'.
    backend : str | None
        One of {'reghdfe','fixest'} or None. Used only when normalization='auto'.

    Returns
    -------
    fe_effects : list of ndarray
        Fixed effects for each dimension with the chosen normalization.

    """
    # determine normalization
    norm = normalization.lower().strip()
    if norm == "auto":
        norm = "references" if backend == "fixest" else "sum_to_zero"
    if norm not in {"sum_to_zero", "references"}:
        raise ValueError(
            "normalization must be one of {'auto','sum_to_zero','references'}",
        )

    r = residuals.reshape(-1, 1).astype(np.float64)
    n = r.shape[0]
    J = len(fe_codes)

    if J == 0:
        return []

    # 1) Encode global ids for all FE levels across dimensions
    levels = [int(c.max()) + 1 if c.size else 0 for c in fe_codes]
    offsets = np.cumsum([0, *levels[:-1]])
    total_L = int(sum(levels))

    if total_L == 0:
        return [np.zeros(0, dtype=np.float64) for _ in range(J)]

    # 2) Union-find over FE levels: connect all levels that co-occur in an observation
    parent = np.arange(total_L, dtype=np.int64)
    rank_arr = np.zeros(total_L, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank_arr[ra] < rank_arr[rb]:
            parent[ra] = rb
        elif rank_arr[ra] > rank_arr[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank_arr[ra] += 1

    # co-occurrence edges: connect FE levels that appear in same observation
    for i in range(n):
        gids = []
        for k in range(J):
            c = int(fe_codes[k][i])
            if c >= 0:
                gids.append(offsets[k] + c)
        # connect consecutive pairs (transitive closure via union-find)
        for a in range(len(gids) - 1):
            union(gids[a], gids[a + 1])

    comp_id = np.array([find(i) for i in range(total_L)], dtype=np.int64)

    # 3) For each component, build sparse D_c and constraints R_c, then solve KKT
    out: list[NDArray[np.float64]] = [
        np.zeros(level_size, dtype=np.float64) for level_size in levels
    ]

    for root in np.unique(comp_id):
        # columns (levels) in this component
        cols = np.nonzero(comp_id == root)[0]
        if cols.size == 0:
            continue

        # map global level index -> local column
        col_map = {int(g): j for j, g in enumerate(cols)}

        # collect rows (observations) that hit this component
        rows = []
        for i in range(n):
            hit = False
            for k in range(J):
                c = int(fe_codes[k][i])
                if c >= 0 and (offsets[k] + c) in col_map:
                    hit = True
                    break
            if hit:
                rows.append(i)

        if len(rows) == 0:
            continue

        # Build D_c (|rows| x |cols|) with ones at each (i, level) present
        data, rowi, coli = [], [], []
        for idx, i in enumerate(rows):
            for k in range(J):
                c = int(fe_codes[k][i])
                g = offsets[k] + c
                if c >= 0 and g in col_map:
                    rowi.append(idx)
                    coli.append(col_map[g])
                    data.append(1.0)

        D = sparse.csc_matrix((data, (rowi, coli)), shape=(len(rows), len(cols)))

        # KKT: [D'D R'; R 0][a;λ] = [D'r; 0]
        DtD = dot(D.T, D).astype(float)
        Dtr = dot(D.T, r[rows]).astype(float)

        # build R according to normalization policy
        R_blocks = []
        if norm == "sum_to_zero":
            # one sum-to-zero constraint per FE-dimension present in this component
            for k in range(J):
                mask = []
                for g in cols:
                    off = offsets[k]
                    Lk = levels[k]
                    mask.append(1.0 if (off <= g < off + Lk) else 0.0)
                if any(mask):
                    R_blocks.append(np.array(mask, dtype=float).reshape(1, -1))
        else:  # references
            # for each FE-dimension present, pick the reference according to priority:
            # 1) explicit `refs` mapping {k: {comp_id: level_code}}
            # 2) `level_order[k]`: list/array with preferred levels in order (first present is chosen)
            # 3) fallback: smallest numeric local level
            for k in range(J):
                # collect local cols that belong to this FE dimension
                local_map = []
                for j, g in enumerate(cols):
                    off = offsets[k]
                    Lk = levels[k]
                    if off <= g < off + Lk:
                        local_map.append((j, g - off))
                if not local_map:
                    continue
                comp_levels = np.array([lv for (_, lv) in local_map], dtype=int)
                # determine component id for this root to index into refs/level_order
                comp_root = int(root)
                # 1) explicit refs mapping
                ref_level = None
                if refs is not None and k in refs:
                    # refs[k] may be mapping of comp_id->level_code or single level
                    ref_map = refs[k]
                    if isinstance(ref_map, dict) and comp_root in ref_map:
                        candidate = int(ref_map[comp_root])
                        if candidate in comp_levels:
                            ref_level = candidate
                    elif not isinstance(ref_map, dict) and int(ref_map) in comp_levels:
                        ref_level = int(ref_map)
                # 2) level_order
                if ref_level is None and level_order is not None and k in level_order:
                    for lev in level_order[k]:
                        if int(lev) in comp_levels:
                            ref_level = int(lev)
                            break
                # 3) fallback: smallest numeric level
                if ref_level is None:
                    ref_level = int(comp_levels.min())
                # find local index for ref_level
                ref_loc = None
                for j, lv in local_map:
                    if lv == ref_level:
                        ref_loc = j
                        break
                if ref_loc is None:
                    # defensive fallback
                    ref_loc = local_map[0][0]
                row = np.zeros((len(cols),), dtype=np.float64)
                row[ref_loc] = 1.0
                R_blocks.append(row.reshape(1, -1))

        R = np.vstack(R_blocks) if R_blocks else np.zeros((0, DtD.shape[0]))

        # Assemble KKT
        K11 = DtD.toarray() if sparse.issparse(DtD) else DtD
        K = np.block([[K11, R.T], [R, np.zeros((R.shape[0], R.shape[0]))]])
        b = np.vstack([Dtr, np.zeros((R.shape[0], 1))])

        # Solve via centralized linear algebra (stable; no direct np.linalg here).
        # Use explicit SVD-based route to match np.linalg.lstsq minimum-norm behavior.
        sol = solve(K, b, method="svd").reshape(-1)
        a_loc = sol[: K11.shape[0]]

        # scatter back to global output
        for g, j in col_map.items():
            # global -> (dim, level)
            k = int(np.searchsorted(offsets, g, side="right") - 1)
            j0 = g - offsets[k]
            out[k][j0] = a_loc[j]

    return out
