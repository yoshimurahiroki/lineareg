# lineareg/core/bootstrap.py
"""Bootstrap helpers: wild/multiplier engines, clustering, and uniform bands.

Policy (theory-first and library-wide):

- Inference is bootstrap-only. No analytical SEs, p-values, or critical values
    are ever returned by any estimator.
- Default number of replications is ``B=2000`` via :data:`DEFAULT_BOOTSTRAP_ITERATIONS`.
- Only wild/multiplier bootstrap is implemented. Pairs bootstrap is forbidden.
- IID, cluster, and multiway clustering are supported. When the Webb
    distribution is applicable and enumeration is feasible, we enumerate first
    (mirroring boottest/fwildclusterboot behavior) before falling back to draws.
- All studentization, recentering, and B+1 quantile rules are centralized here
    to guarantee consistent inference across estimators.
- Uniform confidence bands (sup-t) are constructed here for DID/event-study
    estimators (pre, post, full). Linear models like OLS/IV/GLS/GMM/QR must not
    request bands.

All matrix operations delegate to :mod:`lineareg.core.linalg` to avoid explicit
matrix inverses and to preserve sparse structure when possible.
"""

from __future__ import annotations

import multiprocessing
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import pi
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# Import core linear algebra helpers (use both module and symbols for convenience)
from . import linalg as la
from .linalg import (
    Matrix,
    _validate_weights,
    crossprod,
    dot,
    gram,
    hadamard,
    hat_diag_r,  # Use R convention for HC2/HC3 (consistent with wild bootstrap literature)
    norm,
    to_dense,
)

# Project-wide bootstrap default (user requested): 2000 iterations
DEFAULT_BOOTSTRAP_ITERATIONS: int = 2000

# Store last Satterthwaite df selection metadata for reproducibility/logging.
# Populated when `cluster_robust_vcov` computes Satterthwaite dfs.
_LAST_DF_INFO: dict | None = None


def canonicalize_multipliers(
    multipliers: Any, *, n: int | None = None, B: int | None = None,
) -> np.ndarray:
    """Canonicalize bootstrap multipliers to a (n, B) numpy.ndarray.

    Accepts numpy arrays, pandas Series/DataFrame, or None. If ``multipliers``
    is ``None``, this function returns an array of shape ``(n, B)`` where
    ``B`` defaults to :data:`DEFAULT_BOOTSTRAP_ITERATIONS` and the values are
    filled with zeros; callers should replace them with draws from the chosen
    wild multiplier factory. If ``multipliers`` is a 1-D array/Series it is
    interpreted as a single draw (B=1) and reshaped to (n, 1). If a 2-D
    array or DataFrame is provided, it must have shape (n, B) or (B, n) — in
    the latter case we transpose to (n, B).

    The function validates finite entries and raises :class:`ValueError` on
    shape mismatches or non-finite entries. This centralization avoids
    ad-hoc shape handling across estimators.
    """
    # Quick path for None: caller must supply n and may supply B
    if multipliers is None:
        if n is None:
            msg = (
                "n must be provided when multipliers is None so "
                "DEFAULT_BOOTSTRAP_ITERATIONS can be applied"
            )
            raise ValueError(msg)
        Bout = DEFAULT_BOOTSTRAP_ITERATIONS if B is None else int(B)
        return np.zeros((n, Bout), dtype=np.float64)

    # Convert pandas objects or array-likes to ndarray
    arr: np.ndarray
    try:
        import pandas as pd  # local import to avoid forcing pandas on import
    except ImportError:
        pd = None  # type: ignore[assignment]

    if pd is not None and isinstance(multipliers, (pd.Series, pd.DataFrame)):
        arr = multipliers.to_numpy()
    else:
        arr = np.asarray(multipliers)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        # If shape matches (B, n) but not (n, B), try transposing when n given
        if n is not None and arr.shape[0] != n and arr.shape[1] == n:
            arr = arr.T

    if n is not None and arr.shape[0] != n:
        msg = f"Multipliers have incompatible row dimension {arr.shape[0]}; expected n={n}"
        raise ValueError(msg)

    if B is not None and arr.shape[1] != B:
        # allow B to be None (infer from array)
        msg = f"Multipliers have incompatible column dimension {arr.shape[1]}; expected B={B}"
        raise ValueError(msg)

    if not np.all(np.isfinite(arr)):
        msg = "Non-finite values found in multiplier array"
        raise ValueError(msg)

    return arr.astype(np.float64, copy=False)


# Enumeration thresholds used by WildDist policies. These are conservative
# literature-backed defaults: boottest prefers 11, strict policy uses 12.
ENUM_THRESH_BOOTTEST: int = 11
ENUM_THRESH_STRICT: int = 12

# Note: automatic Webb promotion and magic enumeration thresholds are disabled.
# Exact enumeration is decided solely by the caller via enumeration_mode='boottest'
# (which uses the strict condition B >= 2^G). Webb must be requested explicitly
# by passing dist='webb'. This avoids hidden heuristics and preserves theoretical
# transparency: enumeration is exact only when the user requests it or when
# the boottest condition (n_boot >= 2^G) holds.

# Valid VCV kinds exposed across the library (includes jackknife CRV3J/HC3J)
VALID_VCV = {
    "HC0",
    "HC1",
    "HC2",
    "HC3",
    "CRV1",
    "CRV2",
    "CRV3",
    "CRV3J",
}


# ---------------------------------------------------------------------
# HAC kernels, NW94 bandwidth, and dependent wild multipliers (time/space)
# ---------------------------------------------------------------------


def _hac_kernel(x: np.ndarray, kernel: str) -> np.ndarray:
    """Andrews (1991) HAC kernel family. Supported kernels: 'bartlett', 'parzen', 'qs'.

    Returns kernel weights k(x) evaluated elementwise for array x.
    """
    k = kernel.lower()
    z = np.asarray(x, dtype=np.float64)
    az = np.abs(z)
    if k in {"bartlett", "nw", "newey-west"}:
        return np.maximum(0.0, 1.0 - az)
    if k == "parzen":
        # Parzen window (Andrews 1991). Use absolute z for the |x| powers while
        # keeping the x^2 term (z*z) for exact algebraic agreement with standard formulas.
        return np.where(
            az <= 0.5,
            1.0 - 6.0 * z * z + 6.0 * (az**3),
            np.where(az <= 1.0, 2.0 * np.power(1.0 - az, 3.0), 0.0),
        )
    if k in {"qs", "quadratic-spectral", "andrews"}:
        return _qs_kernel(z)
    msg = f"unknown kernel: {kernel}"
    raise ValueError(msg)


def bandwidth_nw94(
    n_obs: int, *, kernel: str = "bartlett", as_integer: bool | None = None,
) -> float:
    """Newey & West (1994) bandwidth rule with a stricter policy:

    - Apply the NW94 formula only for Bartlett-family kernels ("bartlett", "nw",
      "newey-west").
    - For other commonly-used kernels ("qs" / Quadratic-Spectral and "parzen"),
      delegate bandwidth selection to the Andrews (1991) rule implemented in
      :func:`bandwidth_andrews91` which provides the recommended continuous
      bandwidths for those kernels.

    Rounding policy (as_integer): when ``as_integer`` is None the default is to
    return an integer lag only for Bartlett-family kernels (Stata legacy). QS
    and Parzen will by default return a continuous bandwidth consistent with
    Andrews/R semantics. Call with ``as_integer=False`` to force continuous
    (R-style) behavior, or ``as_integer=True`` to force integer truncation.
    """
    T = int(n_obs)
    k = kernel.lower()

    # Default integer policy: only Bartlett-family yields integer by default
    if as_integer is None:
        as_integer = k in {"bartlett", "nw", "newey-west"}
    else:
        as_integer = bool(as_integer)

    # Delegate QS/Parzen to Andrews (1991) rule-of-thumb constants
    if k in {"qs", "quadratic-spectral", "andrews", "parzen"}:
        b = bandwidth_andrews91(T, kernel=k)
        return float(int(np.floor(b))) if as_integer else float(b)

    # NW94 (Bartlett-family) default formula
    b = 4.0 * (T / 100.0) ** (2.0 / 9.0)
    return float(int(np.floor(b))) if as_integer else float(b)


def bandwidth_andrews91(n_obs: int, *, kernel: str = "qs") -> float:
    """Andrews (1991) optimal bandwidth rules (continuous where appropriate).

    Implements commonly used rule-of-thumb constants to match R conventions
    (sandwich / Andrews implementations):
      - QS / Andrews (continuous): b = 1.3221 * T^(1/5)
      - Bartlett / Newey-West (integer in many implementations): b = 1.1447 * T^(1/3)
      - Parzen: b = 2.6614 * T^(1/5)

    Returns a floating-point (continuous) bandwidth. Callers that require an
    integer lag should floor/round according to their compatibility policy.
    """
    T = float(int(n_obs))
    k = kernel.lower()
    if k in {"qs", "quadratic-spectral", "andrews"}:
        return 1.3221 * (T ** (1.0 / 5.0))
    if k in {"bartlett", "nw", "newey-west"}:
        return 1.1447 * (T ** (1.0 / 3.0))
    if k == "parzen":
        return 2.6614 * (T ** (1.0 / 5.0))
    raise ValueError(f"unknown kernel for Andrews91: {kernel}")


def _toeplitz_cov_by_index(
    index: np.ndarray, bandwidth: float, kernel: str,
) -> np.ndarray:
    """Build a Toeplitz covariance matrix K_ij = k(|rank(t_i) - rank(t_j)| / b).

    Uses rank order of provided (possibly irregular) time values to be consistent
    with common practice in Newey-West / sandwich implementations.
    """
    t = np.asarray(index, dtype=np.float64).reshape(-1)
    if t.size == 0:
        msg = "index must be a non-empty 1-D array"
        raise ValueError(msg)
    ord_idx = np.argsort(t)
    ranks = np.empty_like(ord_idx, dtype=np.int64)
    ranks[ord_idx] = np.arange(len(t))
    d = np.abs(ranks.reshape(-1, 1) - ranks.reshape(1, -1)).astype(np.float64)
    x = d / float(bandwidth)
    K = _hac_kernel(x, kernel)
    np.fill_diagonal(K, 1.0)
    return K


def _qs_kernel(z: np.ndarray) -> np.ndarray:
    """Quadratic-Spectral (QS) kernel per Andrews (1991) — exact formula.

    k_QS(x) = 25/(12*pi^2*x^2) * [ sin(6*pi*x/5)/(6*pi*x/5) - cos(6*pi*x/5) ]

    For x == 0 the analytic limit is 1.0 and is applied exactly. This
    implementation avoids interpolation and uses a small-taylor expansion
    for numerical stability only when x is extremely close to zero.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    out.fill(0.0)
    # handle zeros explicitly
    zero_mask = z == 0.0
    if np.any(zero_mask):
        out[zero_mask] = 1.0
    nz = ~zero_mask
    if not np.any(nz):
        return out
    zz = z[nz]
    # convert to t = 6*pi*z/5 per Andrews
    t = (6.0 * pi / 5.0) * zz
    const = 25.0 / (12.0 * pi * pi)
    # avoid cancellation: when |t| is extremely small use series expansion
    small = np.abs(t) < 1e-12
    num = np.empty_like(t, dtype=np.float64)
    if np.any(~small):
        tt = t[~small]
        num[~small] = (np.sin(tt) / tt) - np.cos(tt)
    if np.any(small):
        ts = t[small]
        # series: (sin t)/t - cos t = t^2/3 - t^4/30 + O(t^6)
        num[small] = (ts * ts) / 3.0 - (ts**4) / 30.0
    out[nz] = const * num / (zz * zz)
    return out


def finite_sample_quantile(t_stats: np.ndarray, alpha: float) -> float:
    """Return the finite-sample corrected bootstrap quantile using the (B+1)
    adjustment: k = ceil((B+1) * alpha). This matches common practice in
    boottest/fwidlclusterboot.

    Parameters
    ----------
    t_stats : array-like of shape (B,)
        Bootstrap replicates of the statistic.
    alpha : float
        Target quantile in (0,1].

    """
    arr = np.asarray(t_stats, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("t_stats must contain at least one bootstrap draw")
    B = int(arr.shape[0])
    k = int(np.ceil((B + 1) * float(alpha)))
    k = min(max(k, 1), B)
    # zero-indexed partition
    quantile_value = np.partition(arr, k - 1)[k - 1]
    return float(quantile_value)


def rademacher_enumeration(G: int) -> np.ndarray:
    """Enumerate the full Rademacher sign grid {+1, -1}^G in lexicographic order.
    Returns an array of shape (2**G, G). Double precision.
    """
    if not isinstance(G, int) or G < 0:
        raise ValueError("G must be a non-negative integer")
    if G == 0:
        return np.ones((1, 0), dtype=np.float64)
    # 2**G rows sign matrix using bit operations (columns correspond to cluster indices)
    rows = 1 << G  # 2**G
    out = np.empty((rows, G), dtype=np.float64)
    for j in range(G):
        block = 1 << j
        pattern = np.tile(
            np.concatenate([np.full(block, -1.0), np.full(block, +1.0)]),
            rows // (2 * block),
        )
        out[:, j] = pattern
    return out


def webb_enumeration(G: int, *, enum_max_g_webb: int = 4) -> np.ndarray:
    """Enumerate the full Webb 6-point grid in lexicographic order.
    Values are {±sqrt(3/2), ±1, ±1/sqrt(2)} each with prob 1/6.
    Returns array (6**G, G). For G > enum_max_g_webb raises ValueError.

    Note:
    ----
    Webb's 6-point design is point-symmetric and is generally not recommended
    for very small numbers of clusters; theoretical properties typically
    assume at least G >= 6. For G < 6 Webb enumeration is technically
    possible but practitioners commonly avoid it; callers should prefer
    Rademacher/Mammen for small G or use exact enumeration only when
    2**G is tractable.

    """
    if not isinstance(G, int) or G < 0:
        raise ValueError("G must be a non-negative integer")
    if int(enum_max_g_webb) < G:
        raise ValueError(
            f"Webb enumeration exploding at 6**G; set enum_max_g_webb>=G or sample instead (G={G}).",
        )
    if G == 0:
        return np.ones((1, 0), dtype=np.float64)
    vals = np.array(
        [
            -np.sqrt(1.5),
            -1.0,
            -1.0 / np.sqrt(2.0),
            1.0 / np.sqrt(2.0),
            1.0,
            np.sqrt(1.5),
        ],
        dtype=np.float64,
    )
    total = 6**G
    out = np.empty((total, G), dtype=np.float64)
    for j in range(G):
        block = 6**j
        pattern = np.tile(np.repeat(vals, block), total // (6 * block))
        out[:, j] = pattern
    return out


# NOTE: variant parser canonicalization moved later in the file. The
# duplicate/legacy definition above was removed to avoid contradictory
# mappings. See the canonical _parse_bootstrap_variant definition below
# which follows fwildclusterboot / boottest conventions.


def _kernel_cov_from_coords(
    coords: np.ndarray,
    bandwidth: float,
    kernel: str,
    *,
    metric: str = "euclidean",
    earth_radius_km: float = 6371.0,
) -> np.ndarray:
    """Spatial kernel covariance (Conley-style): K_ij = k(dist(i,j) / b).

    Constructs the full pairwise distance matrix using vectorized operations and
    applies the HAC kernel; does not form spatial clusters. Both 'euclidean'
    and 'haversine' (great-circle) metrics are implemented. When using
    'haversine', coordinates should be provided as (lon_deg, lat_deg) and the
    returned distances are in the same units as `earth_radius_km` (default km).
    """
    C = np.asarray(coords, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] == 0:
        msg = "coords must be a non-empty (n x d) array"
        raise ValueError(msg)
    metr = metric.lower()
    if metr == "euclidean":
        # squared distances via (c_i - c_j)^2 = c_i^2 + c_j^2 - 2 c_i c_j'
        s = np.sum(C * C, axis=1).reshape(-1, 1)
        # use la.dot for matrix product to respect core.linalg abstractions
        D2 = s + s.T - 2.0 * la.dot(C, C.T)
        np.maximum(D2, 0.0, out=D2)
        D = np.sqrt(D2, dtype=np.float64)
    elif metr in {"haversine", "greatcircle", "great-circle"}:
        # Expect coords columns as (lon_deg, lat_deg). Use numerically-stable
        # vectorized haversine helper to compute pairwise great-circle distances.
        if C.shape[1] < 2:
            msg = "haversine metric requires coords with longitude and latitude columns"
            raise ValueError(msg)
        D = _pairwise_haversine(C[:, :2], R=float(earth_radius_km))
    else:
        msg = "unsupported metric: choose 'euclidean' or 'haversine'"
        raise ValueError(msg)
    x = D / float(bandwidth)
    K = _hac_kernel(x, kernel)
    np.fill_diagonal(K, 1.0)
    return K


def _pairwise_haversine(
    latlon: np.ndarray, R: float = 6371.0, order: str = "lonlat",
) -> np.ndarray:
    """Pairwise great-circle distances (same units as R) with numeric safeguards.

    Parameters
    ----------
    latlon : ndarray
        Array with two columns containing geographic coordinates in degrees.
        By default the function expects columns in (lon_deg, lat_deg) order
        (``order="lonlat"``). If your input is (lat_deg, lon_deg) set
        ``order="latlon"`` and the columns will be swapped internally.
    R : float
        Earth radius (same units as desired output). Default is 6371.0 km.
    order : str
        Either ``"lonlat"`` (default) or ``"latlon"`` to indicate the
        ordering of the two columns in ``latlon``.

    Notes
    -----
    The function validates input shape and clips intermediate values for
    numerical stability. It always returns a symmetric distance matrix with
    zeros on the diagonal.

    """
    LL = np.asarray(latlon, dtype=np.float64)
    if LL.ndim != 2 or LL.shape[1] < 2:
        raise ValueError(
            "latlon must be an (n x 2) array with longitude and latitude in degrees",
        )

    order = (order or "lonlat").lower()
    if order not in {"lonlat", "latlon"}:
        raise ValueError("order must be either 'lonlat' or 'latlon'")

    # normalize to (lon, lat) ordering internally
    if order == "latlon":
        LL = LL[:, [1, 0]]

    lon = np.deg2rad(LL[:, 0].astype(np.float64)).reshape(-1, 1)
    lat = np.deg2rad(LL[:, 1].astype(np.float64)).reshape(-1, 1)
    dlat = lat - lat.T
    dlon = lon - lon.T
    # haversine 'a' term, vectorized
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat) * np.cos(lat.T) * (
        np.sin(dlon * 0.5) ** 2
    )
    # numerical guards: clip to [0,1]
    np.clip(a, 0.0, 1.0, out=a)
    # central angle
    c = 2.0 * np.arcsin(np.sqrt(a))
    D = (R * c).astype(np.float64)
    # ensure symmetry and zero diagonal
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _draw_multipliers_from_cov(
    K: np.ndarray,
    B: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw columns of W ~ N(0, K) using a strict PSD cholesky-like square-root.

    Symmetrize K, compute a PSD square-root via core.linalg.chol_psd and
    multiply by standard normals to obtain W = L @ Z where L @ L.T == PSD(K).
    """
    rng = rng or np.random.default_rng(seed)
    K = 0.5 * (np.asarray(K, dtype=np.float64) + np.asarray(K, dtype=np.float64).T)
    # Use robust PSD cholesky provided by core.linalg to ensure numerical
    # positive semidefiniteness (consistent across platforms).
    L = la.chol_psd(K)
    Z = rng.standard_normal(size=(K.shape[0], int(B)))
    W = la.dot(L, Z)
    return np.asarray(W, dtype=np.float64, order="C")


def time_dependent_multipliers(  # noqa: PLR0913
    time: Sequence[int | float],
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    *,
    kernel: str = "qs",
    bandwidth: float | None = None,
    bandwidth_method: str | None = None,
    as_integer_bandwidth: bool = False,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Dependent Wild Bootstrap multipliers for time series (DWB-like).

    Returns (W (T x B), logdict). W columns are independent draws from N(0, K)
    where K_ij = k(|rank(t_i)-rank(t_j)| / b).
    """
    t = np.asarray(time)
    if t.ndim != 1 or t.size == 0:
        msg = "time must be a non-empty 1-D array"
        raise ValueError(msg)
    T = int(t.size)
    if bandwidth is None:
        bm = (bandwidth_method or "auto").lower()
        # Auto inference: if bandwidth_method is 'auto' or None, choose Andrews91
        # for QS-like kernels and NW94 for Bartlett/Parzen-like kernels.
        if bm in {"auto", "none"}:
            k = kernel.lower()
            # Explicit policy: Bartlett-family uses Newey-West (NW94) integer
            # lag rule by default; QS/Parzen delegate to Andrews (1991).
            if k in {"bartlett", "nw", "newey-west"}:
                bandwidth = bandwidth_nw94(T, kernel=k, as_integer=as_integer_bandwidth)
            elif k in {"qs", "quadratic-spectral", "andrews"}:
                bandwidth = bandwidth_andrews91(T, kernel=kernel)
            else:
                # default fallback: use NW94 for unknown strings to preserve
                # conservative integer lag behavior
                bandwidth = bandwidth_nw94(
                    T, kernel=kernel, as_integer=as_integer_bandwidth,
                )
        elif bm == "nw94":
            bandwidth = bandwidth_nw94(
                T, kernel=kernel, as_integer=as_integer_bandwidth,
            )
        elif bm in {"andrews", "andrews91"}:
            bandwidth = bandwidth_andrews91(T, kernel=kernel)
        else:
            raise ValueError(
                "bandwidth_method must be 'nw94', 'andrews91', or 'auto' when bandwidth is None",
            )
    K = _toeplitz_cov_by_index(t, float(bandwidth), kernel)
    W = _draw_multipliers_from_cov(K, n_boot, seed=seed, rng=rng)
    log = {
        "kind": "time-dwb",
        "kernel": kernel,
        "bandwidth": float(bandwidth),
        "bandwidth_method": bandwidth_method,
        "n_obs": T,
        "n_boot": int(n_boot),
    }
    return W, log


def spatial_kernel_multipliers(  # noqa: PLR0913
    coords: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    *,
    kernel: str = "qs",
    bandwidth: float | None = None,
    metric: str = "euclidean",
    earth_radius_km: float = 6371.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Spatial kernel multipliers (distance-based, cluster-free): returns W ~ N(0, K)
    with K_ij = k(dist(i,j)/b). The user must supply a positive bandwidth.
    """
    if bandwidth is None or bandwidth <= 0:
        msg = "positive 'bandwidth' must be provided for spatial multipliers"
        raise ValueError(msg)
    K = _kernel_cov_from_coords(
        coords,
        float(bandwidth),
        kernel,
        metric=metric,
        earth_radius_km=float(earth_radius_km),
    )
    W = _draw_multipliers_from_cov(K, n_boot, seed=seed, rng=rng)
    log = {
        "kind": "spatial-dwb",
        "kernel": kernel,
        "bandwidth": float(bandwidth),
        "metric": metric,
        "n_obs": int(np.asarray(coords).shape[0]),
        "n_boot": int(n_boot),
    }
    return W, log


def _parse_bootstrap_variant(variant: str) -> tuple[str, str]:
    """Canonical MNW / boottest mapping.

    Mapping (fwildclusterboot / boottest conventions):
      11 -> (WCR, CRV1)
      13 -> (WCR, CRV3)
      31 -> (WCU_score, CRV1)
      33 -> (WCU_score, CRV3)
      33J -> (WCU_score, CRV3J)

    The parser accepts case-insensitive and separator-insensitive inputs
    (e.g., '33j', '33-J', 'mnw_33j'). Raises ValueError if unknown.
    """
    v = str(variant).upper().replace("-", "").replace("_", "")
    table = {
        "11": ("WCR", "CRV1"),
        "13": ("WCR", "CRV3"),
        "31": ("WCU_score", "CRV1"),
        "33": ("WCU_score", "CRV3"),
        "33J": ("WCU_score", "CRV3J"),
    }
    if v not in table:
        raise ValueError(
            f"variant must be one of {sorted(table.keys())}; got {variant!r}",
        )
    return table[v]


def _normalize_ssc(ssc: dict[str, object] | None) -> dict[str, object]:
    """Normalize ssc dictionaries to canonical keys.

    Accepts only {'adj', 'fixef.K', 'cluster.adj', 'cluster.df'}. Any other keys
    will raise an error. Defaults mirror fixest::ssc().
    """
    defaults = {
        "adj": True,
        "fixef.K": "none",
        "cluster.adj": "conventional",
        "cluster.df": "conventional",
    }
    if ssc is None:
        return dict(defaults)
    out: dict[str, object] = {}
    for k, v in ssc.items():
        if k not in defaults:
            raise KeyError(
                f"Invalid ssc key '{k}'. Allowed keys are {sorted(defaults.keys())}.",
            )
        out[k] = v
    # fill defaults
    for k, v in defaults.items():
        out.setdefault(k, v)
    return out


# ---------------------------------------------------------------------
# Wild multipliers
# ---------------------------------------------------------------------


class WildDist:
    """Wild multiplier distribution (mean 0, variance 1) used across all estimators.

    Supported distributions
    -----------------------
    rademacher / rad
    Two-point {-1,+1} with equal probability. Compatible with *exact enumeration*
    of all sign patterns for small number of clusters G (priority path for finite-sample
        validity when 2^G is tractable). Enumeration happens **before** any automatic
    Webb upgrade so that exact reference distributions are not pre-empted.
    webb
    Six-point distribution {±sqrt(3/2), ±1, ±1/sqrt(2)} each with probability 1/6.
        Recommended in practice for very small G when enumeration is not used
        (mirrors boottest / fwildclusterboot guidance; see Webb 2013 WP, 2023 CJE).
    mammen
    Two-point distribution with golden-ratio support ensuring mean 0 and variance 1.

    Policy notes
    ------------
    - Default wild distribution is Rademacher; for clustered designs we first attempt
      exact enumeration of all 2^G sign patterns when G ≤ enum_max_g (default 12).
      Only if enumeration is disabled / infeasible do we consider a Webb switch.
      Threshold 12 is conservative; literature (e.g., boottest) often uses ≤10-11.
      For boottest compatibility, set policy='boottest' to use threshold=11.
      See Webb (2013, Working Paper; 2023, Canadian Journal of Economics) for theoretical justification of the six-point distribution in small-G settings.
    - HC2 / HC3 leverage corrections are *off* by default for wild / clustered bootstrap
    because their finite-sample motivation (EHW style) does not directly transfer under
    cluster multipliers (see Imbens-Kolesar 2012). Users may still enable them explicitly.
      For small-sample clustered designs, consider CR2/CR3 corrections (Bell-McCaffrey / clubSandwich)
      as alternatives; these can be implemented via plugin_se in uniform_confidence_band.
    - All outputs are float64; no pairs bootstrap is implemented anywhere in the project.
    """

    def __init__(self, name: str, *, policy: str = "strict") -> None:
        self.name = name.lower().strip()
        self.policy = policy.lower().strip()
        # Use literature-backed defaults for enumeration thresholds. For boottest
        # compatibility prefer exact enumeration when G <= ENUM_THRESH_BOOTTEST.
        if self.policy == "boottest":
            self.small_g_threshold = ENUM_THRESH_BOOTTEST
        else:
            # strict policy disables automatic Webb promotion
            self.small_g_threshold = ENUM_THRESH_STRICT

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"WildDist({self.name!r})"

    _ALLOWED_DISTS = frozenset({"rademacher", "rad", "webb", "mammen", "normal", "gaussian", "standard_normal"})

    def draw(
        self,
        size: tuple[int, int],
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Draw an (R x C) array of multipliers using an external RNG if provided.
        All supported distributions are centered (mean 0) with unit variance.
        """
        if self.name not in self._ALLOWED_DISTS:
            msg = f"Unknown wild distribution: '{self.name}'. Allowed: {sorted(self._ALLOWED_DISTS)}"
            raise ValueError(msg)
        rng = rng or np.random.default_rng(seed)
        r, c = size
        if self.name in {"rademacher", "rad"}:
            return rng.choice(np.array([-1.0, 1.0]), size=(r, c)).astype(
                np.float64,
                copy=False,
            )
        if self.name in {"webb"}:
            # Webb six-point distribution (fwildclusterboot / boottest compatible):
            # values = {±√(3/2), ±1, ±1/√2} with equal probabilities 1/6 each.
            # These values are provided in full double precision below.
            vals = np.array(
                [
                    -1.22474487139158904909,  # -√(3/2)
                    -1.0,
                    -0.70710678118654752440,  # -1/√2
                    +0.70710678118654752440,  # +1/√2
                    +1.0,
                    +1.22474487139158904909,  # +√(3/2)
                ],
                dtype=np.float64,
            )
            probs = np.array([1.0 / 6.0] * 6, dtype=np.float64)
            return rng.choice(vals, size=(r, c), p=probs).astype(np.float64, copy=False)
        if self.name in {"normal", "gaussian", "standard_normal"}:
            # Standard normal draws (mean 0, variance 1)
            return rng.standard_normal(size=(r, c)).astype(np.float64, copy=False)
        if self.name == "exp":
            msg = "Exponential distribution is not supported for wild multipliers; use only for WGB-specific methods."
            raise ValueError(msg)
        # Mammen two-point (mean 0, var 1)
        a = (1.0 - np.sqrt(5.0)) / 2.0
        b = (1.0 + np.sqrt(5.0)) / 2.0
        pa = (np.sqrt(5.0) + 1.0) / (2.0 * np.sqrt(5.0))
        pb = 1.0 - pa
        return rng.choice(
            np.array([a, b], dtype=np.float64),
            size=(r, c),
            p=np.array([pa, pb], dtype=np.float64),
        ).astype(np.float64, copy=False)


def wild_multipliers(
    n_obs: int,
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    dist: WildDist | str = "rademacher",
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """IID wild multipliers: (n x B)."""
    D = dist if isinstance(dist, WildDist) else WildDist(str(dist))
    return D.draw((n_obs, n_boot), seed=seed, rng=rng)


def wgb_cluster_multipliers(
    G: int, B: int, *, seed: int | None = None,
) -> NDArray[np.float64]:
    """Return a (G x B) array of Mammen multipliers for Wild Gradient Bootstrap (WGB).

    This thin wrapper clarifies the contract used by QR WGB (Hagemann 2017): the
    caller expects G-by-B cluster multipliers which will later be expanded to
    observation length via cluster inverse indices. Keeping this as a dedicated
    helper avoids overloading the general-purpose `cluster_multipliers` API and
    prevents future regressions.
    """
    D = WildDist("mammen")
    return D.draw((G, B), seed=seed)


def _maybe_switch_to_webb(  # noqa: PLR0911
    dist: WildDist | str,
    G: int,
    *,
    threshold: int | None = None,
    enumeration_successful: bool = False,
) -> WildDist:
    """Decide whether to promote to Webb 6-point distribution for tiny G.

    Keep existing Webb selection if user already selected Webb. Return a
    WildDist instance in all cases.
    """
    D = dist if isinstance(dist, WildDist) else WildDist(str(dist))
    # keep Webb if user explicitly requested it
    # If the user explicitly requested Webb but the number of clusters is
    # too small to support the six-point design reliably (practical guidance
    # suggests at least G >= 6), fall back to Rademacher and let the caller
    # record/log a warning. This mirrors boottest practice of reverting Webb
    # when the Webb design is infeasible.
    if D.name == "webb":
        if G < 6:
            pol = getattr(D, "policy", "strict")
            # boottest policy: warn and continue with Webb; strict policy: downgrade
            if pol == "boottest":
                warnings.warn(
                    "G<6 under Webb: continuing with Webb per boottest policy.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return D
            return WildDist("rademacher", policy=pol)
        return D

    # Determine promotion rules strictly: do not auto-promote under 'strict'
    # policy. For compatibility with boottest/fwildclusterboot we allow an
    # automatic promotion to Webb only when the distribution policy is
    # 'boottest', enumeration was NOT used, and G is within the conservative
    # small-G threshold (ENUM_THRESH_BOOTTEST). This mirrors the literature
    # guidance: prefer exact enumeration when feasible, otherwise Webb may be
    # used for small G to improve finite-sample behaviour.
    if enumeration_successful:
        return D

    policy = getattr(D, "policy", "strict")
    # If user explicitly selected Webb keep it
    if D.name == "webb":
        return D

    # Only promote to Webb under boottest policy and when G is small enough
    if policy == "boottest":
        thr = threshold if threshold is not None else ENUM_THRESH_BOOTTEST
        if thr >= G:
            return WildDist("webb", policy=policy)
    # Otherwise preserve the caller's choice
    return D


def cluster_multipliers(  # noqa: PLR0913
    clusters: Sequence[int],
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    *,
    dist: WildDist | str = "rademacher",
    use_enumeration: bool = True,
    enumeration_mode: str = "boottest",
    enum_max_g: int | None = None,
    enum_max_g_webb: int | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    policy: str = "boottest",
    bootcluster: str | int | None = "max",
) -> tuple[np.ndarray, dict]:
    """Strict multiway policy: choose a single bootcluster dimension (R/Stata compatible).
    - If multiple ids provided, pick by rule:
        'intersection' -> use the intersection of provided cluster id columns (R/fwildclusterboot default)
        'largest' -> dim with max unique clusters
        'first'   -> first provided
        or integer index
    - Within chosen dim: if G <= enum_max_g -> full enumeration, else random draws.
    Returns: (Wobs (n, R), log)

    Default enumeration_mode='boottest' enumerates exactly when 2^G <= enum_max_g
    (default uses the boottest threshold) and otherwise falls back to Webb
    promotion only when enumeration is infeasible. This matches Stata's
    boottest and R's fwildclusterboot semantics: prefer exact enumeration under
    the boottest policy when the enumerated set is tractable; do not enumerate
    simply because enumeration is feasible unless the caller explicitly
    requested 'always_if_feasible'.
    """
    # rng and seed
    rng = rng or np.random.default_rng(seed)

    clusters_arr = np.asarray(clusters)
    if clusters_arr.ndim == 2:
        clusters_list = [clusters_arr[:, j].astype(int) for j in range(clusters_arr.shape[1])]
    elif (
        isinstance(clusters, (list, tuple))
        and len(clusters) > 0
        and hasattr(clusters[0], '__len__')
        and not isinstance(clusters[0], str)
    ):
        clusters_list = [np.asarray(c, dtype=int).reshape(-1) for c in clusters]
    else:
        clusters_list = None

    if clusters_list is not None:
        if len(clusters_list) == 0 or clusters_list[0] is None:
            msg = "No cluster ids provided"
            raise ValueError(msg)
        n_obs = len(clusters_list[0])
        for j, arr in enumerate(clusters_list):
            if len(arr) != n_obs:
                raise ValueError(f"Cluster dimension {j} has length {len(arr)} != n={n_obs}")
        sizes = [np.unique(ids_dim).size for ids_dim in clusters_list]
        if isinstance(bootcluster, str):
            bk = bootcluster.lower()
            if bk in {"max", "largest"}:
                dim_idx = int(np.argmax(sizes))
            elif bk in {"min", "smallest"}:
                dim_idx = int(np.argmin(sizes))
            elif bk in {"first"}:
                dim_idx = 0
            elif bk in {"intersection", "intersect", "all"}:
                keys = la.column_stack([c.reshape(-1, 1) for c in clusters_list])
                _, inv = np.unique(keys, axis=0, return_inverse=True)
                ids = inv.astype(int, copy=False)
                dim_idx = -1
            else:
                msg = "bootcluster must be 'intersection','max','min','first', or integer index"
                raise ValueError(msg)
        elif isinstance(bootcluster, int) and 0 <= int(bootcluster) < len(
            clusters_list,
        ):
            dim_idx = int(bootcluster)
        else:
            msg = "invalid bootcluster"
            raise ValueError(msg)
        if "ids" not in locals():
            ids = np.asarray(clusters_list[dim_idx])
    else:
        ids = clusters_arr.reshape(-1).astype(int)
        dim_idx = 0

    labels, inv = np.unique(ids, return_inverse=True)
    G = labels.size

    D = dist if isinstance(dist, WildDist) else WildDist(str(dist), policy=policy)

    # Resolve default enum_max_g from policy when caller did not supply one.
    # boottest policy uses the more permissive threshold (ENUM_THRESH_BOOTTEST)
    # while strict uses the more conservative threshold (ENUM_THRESH_STRICT).
    if enum_max_g is None:
        enum_max_g = (
            ENUM_THRESH_BOOTTEST
            if getattr(D, "policy", policy) == "boottest"
            else ENUM_THRESH_STRICT
        )

    # Enumeration logic: only 'boottest' mode is supported (R/Stata parity).
    # Under boottest semantics enumerate exactly when n_boot >= 2**G and the
    # provided enum_max_g (safety cap) allows enumeration. Nonstandard convenience
    # modes (e.g. 'always_if_feasible') are intentionally removed for strictness.
    if use_enumeration and D.name in {"rademacher", "rad", "webb"}:
        if enumeration_mode != "boottest":
            raise ValueError("invalid enumeration_mode")
        if D.name in {"rademacher", "rad"}:
            total = 1 << G
            should_enum = (int(n_boot) >= total) and (
                (enum_max_g is None) or (int(enum_max_g) >= G)
            )
        elif D.name == "webb":
            emw = (
                4 if enum_max_g_webb is None else int(enum_max_g_webb)
            )
            if G > emw:
                should_enum = False
                total = 0
            else:
                total = 6 ** G
                should_enum = (int(n_boot) >= total) and (int(emw) >= G)
        else:
            total = 0
            should_enum = False
        if should_enum:
            if D.name in {"rademacher", "rad"}:
                Wg = rademacher_enumeration(G)
                total = Wg.shape[0]
            elif D.name == "webb":
                Wg = webb_enumeration(G, enum_max_g_webb=emw)
                total = Wg.shape[0]
            elif D.name == "mammen":
                # Under 'boottest' enumeration semantics we do not enumerate
                # Mammen; fall back to RNG. If enumeration_mode != 'boottest'
                # and the caller explicitly requested enumeration, we allow
                # Mammen enumeration here.
                if enumeration_mode == "boottest":
                    Wg = None
                    total = 0
                else:
                    # Mammen two-point enumeration using exact algebraic roots
                    total = 1 << G
                    Wg = np.empty((G, total), dtype=np.float64)
                    a = (1.0 - np.sqrt(5.0)) / 2.0
                    b = (1.0 + np.sqrt(5.0)) / 2.0

                    def enum_vec_mammen(idx: int) -> NDArray[np.float64]:
                        bits = np.array(
                            [(idx >> g) & 1 for g in range(G)], dtype=np.int64,
                        )
                        return np.where(bits == 0, a, b).astype(np.float64)

                    for idx_bits in range(total):
                        Wg[:, idx_bits] = enum_vec_mammen(idx_bits)
                    Wg = Wg.T  # shape (total, G)
            else:
                # fallback: not a two-point family we can enumerate
                Wg = None
                total = 0

            if Wg is None:
                # cannot enumerate this distribution; fall back to random draws
                pass
            else:
                # When enumerating, always return the full enumerated grid.
                # Do not truncate or warn if the caller requested a larger n_boot.
                # Ensure rows correspond to clusters (G x total)
                if Wg.shape[0] == G and Wg.shape[1] == total:
                    Wg_proc = Wg
                elif Wg.shape[0] == total and Wg.shape[1] == G:
                    # transpose to (G x total)
                    Wg_proc = Wg.T
                else:
                    Wg_proc = Wg.T

                log = {
                    "selected_dim": int(dim_idx),
                    "G": int(G),
                    "enumerated": True,
                    "effective_dist": D.name,
                    "effective_B": int(total),
                    "variant": str(dist),
                    "bootcluster": bootcluster,
                    "enumeration_mode": enumeration_mode,
                    "policy": policy,
                    "enum_threshold": int(enum_max_g)
                    if enum_max_g is not None
                    else None,
                    "webb_enum": D.name == "webb",
                    "enum_max_g_webb": int(enum_max_g_webb)
                    if enum_max_g_webb is not None
                    else None,
                }
                W = Wg_proc[inv, :]
                return W, log
    # fallback: random draws per cluster from selected distribution. Multiway
    # clustering must never enumerate (fwildclusterboot/boottest parity).
    # Potential Webb promotion is kept only for small G under non-multiway
    # single-way setups and when the caller requested Webb explicitly.
    D = _maybe_switch_to_webb(D, G, threshold=enum_max_g, enumeration_successful=False)
    rng_eff = rng or np.random.default_rng(seed)
    Wg = D.draw((G, n_boot), rng=rng_eff)
    log = {
        "selected_dim": int(dim_idx),
        "G": int(G),
        "enumerated": False,
        "effective_dist": D.name,
        "effective_B": int(n_boot),
        "variant": str(dist),
        "bootcluster": bootcluster,
        "enumeration_mode": enumeration_mode,
        "policy": policy,
        "enum_threshold": int(enum_max_g) if enum_max_g is not None else None,
    }
    W = Wg[inv, :]
    return W, log

    # Note: legacy single-way branch removed. The canonical implementation
    # above handles both single- and multi-way inputs by selecting a single
    # bootcluster axis (or using the provided integer index) and applying the
    # exact enumeration / Webb promotion rules consistently. This avoids
    # duplicate/contradictory code paths resulting from merge artifacts.
    # End of cluster_multipliers


def multiway_multipliers(  # noqa: PLR0913
    cluster_list: Sequence[Sequence[int]],
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    *,
    dist: WildDist | str = "rademacher",
    dist_seed: int | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    enumeration_mode: str = "boottest",
    policy: str = "boottest",
) -> NDArray[np.float64]:
    """MNW-style multiway wild multipliers: generate independent cluster multipliers
    for each cluster dimension and combine them per-observation by elementwise
    product (Hadamard) to produce an (n x B) multiplier matrix.

    This mirrors the multiway composition used by MNW (2021, JBES) and keeps
    enumeration restricted to 1-way (boottest compatibility). The function
    returns an (n, B) array of multipliers where B == n_boot.
    """
    # Normalize RNG
    rng = rng or np.random.default_rng(seed or dist_seed)

    if not isinstance(cluster_list, (list, tuple)) or len(cluster_list) == 0:
        msg = "cluster_list must be a non-empty sequence of cluster id arrays"
        raise ValueError(msg)

    K = len(cluster_list)
    # Start with None and multiply in each dimension's multipliers
    W_prod: NDArray[np.float64] | None = None
    for k in range(K):
        cl = np.asarray(cluster_list[k], dtype=int).reshape(-1)
        Wk, _ = cluster_multipliers(
            cl,
            n_boot=n_boot,
            dist=dist,
            use_enumeration=(K == 1),  # only allow enumeration when single-way
            enumeration_mode=enumeration_mode,
            enum_max_g=None,
            seed=seed,
            rng=rng,
            policy=policy,
        )
        # Expand per-cluster multipliers to observation length
        # cluster_multipliers already returns Wobs (n, B) when passed full ids
        W_prod = Wk if W_prod is None else hadamard(W_prod, Wk)

    return np.asarray(W_prod, dtype=np.float64)


# ---------------------------------------------------------------------
# Bootstrap payload builder
# ---------------------------------------------------------------------


def _wls_fit(
    y: NDArray[np.float64],
    X: Matrix,
    weights: Sequence[float] | None = None,
    *,
    method: str = "qr",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Weighted least squares fitted values and residuals via QR/SVD solve.
    Avoids forming normal equations for numerical stability.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    # Use core.linalg.solve (QR-first, SVD fallback) on (possibly) sqrt-weighted design
    if weights is None:
        beta = la.solve(X, y, method=(method if method in {"qr", "svd"} else "qr"))
    else:
        w = _validate_weights(weights, y.shape[0]).reshape(-1, 1)
        sqrt_w = np.sqrt(w)
        Xw = hadamard(X, sqrt_w)
        yw = hadamard(y, sqrt_w)
        beta = la.solve(Xw, yw, method=(method if method in {"qr", "svd"} else "qr"))
    yhat = dot(X, beta)
    resid = y - yhat
    return np.asarray(yhat, dtype=np.float64), np.asarray(resid, dtype=np.float64)


def _wls_fit_restricted(  # noqa: PLR0913
    y: NDArray[np.float64],
    X: Matrix,
    R: NDArray[np.float64],
    r: NDArray[np.float64],
    weights: Sequence[float] | None = None,
    *,
    method: str = "qr",
    drop_rank_deficient: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute restricted WLS with linear constraints :math:`R \\beta = r` using the null-space method.

    Implementation details
    ----------------------
    - Avoids forming normal equations by projecting onto the null space of ``R``.
    - Uses SVD-based rank diagnostics for the constraint matrix.
    - Falls back to unrestricted WLS when no restrictions are present.
    - Optionally drops rank-deficient columns of ``X`` (Stata policy) before solving.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    r = np.asarray(r, dtype=np.float64).reshape(-1, 1)
    Xd_orig = to_dense(X)
    R_orig = np.asarray(R, dtype=np.float64)
    Xd = Xd_orig.copy()
    R_eff = R_orig.copy()
    keep_mask: NDArray[np.bool_] | None = None
    if drop_rank_deficient:
        Xd_reduced, keep_mask = la.drop_rank_deficient_cols_stata(Xd)
        if not np.any(keep_mask):
            msg = "All columns would be dropped due to rank deficiency."
            raise ValueError(msg)
        Xd = Xd_reduced
        R_eff = R_eff[:, keep_mask]
    if R_eff.shape[1] != Xd.shape[1]:
        msg = "R must have the same number of columns as X."
        raise ValueError(msg)
    if r.shape[0] != R_eff.shape[0]:
        msg = "r must have the same number of rows as R."
        raise ValueError(msg)
    if R_eff.shape[0] == 0:
        if weights is None:
            beta = la.solve(Xd, y, method=(method if method in {"qr", "svd"} else "qr"))
        else:
            w = _validate_weights(weights, y.shape[0]).reshape(-1, 1)
            sqrt_w = np.sqrt(w)
            Xw = hadamard(Xd, sqrt_w)
            yw = hadamard(y, sqrt_w)
            beta = la.solve(
                Xw, yw, method=(method if method in {"qr", "svd"} else "qr"),
            )
        if keep_mask is not None:
            beta_full = np.zeros((keep_mask.size, 1), dtype=np.float64)
            beta_full[keep_mask, :] = beta
            beta = beta_full
            yhat = dot(Xd_orig, beta)
        else:
            yhat = dot(Xd, beta)
        resid = y - yhat
        return np.asarray(yhat, dtype=np.float64), np.asarray(resid, dtype=np.float64)

    U, s, Vt = la.svd(R_eff, full_matrices=False)
    eps = np.finfo(float).eps
    rcond = max(R_eff.shape) * eps
    tol = rcond * (s.max() if s.size else 1.0)
    rank = np.sum(s > tol)
    null_space = Vt[rank:].T  # (p, p - rank)
    if rank == 0:
        particular = np.zeros((Xd.shape[1], 1), dtype=np.float64)
    else:
        Sr_inv = np.diag(1.0 / s[:rank])
        particular = dot(Vt[:rank, :].T, dot(Sr_inv, dot(U[:, :rank].T, r)))

    X_proj = dot(Xd, null_space)  # (n, p - rank)
    y_proj = y - dot(Xd, particular)  # (n, 1)

    if weights is None:
        gamma = la.solve(
            X_proj, y_proj, method=(method if method in {"qr", "svd"} else "qr"),
        )
    else:
        sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float64).reshape(-1, 1))
        X_proj_w = hadamard(X_proj, sqrt_w)
        y_proj_w = hadamard(y_proj, sqrt_w)
        gamma = la.solve(
            X_proj_w, y_proj_w, method=(method if method in {"qr", "svd"} else "qr"),
        )

    beta_R = particular + dot(null_space, gamma)

    res = norm(dot(R_eff, beta_R) - r)
    if not np.isfinite(res) or res > 1e-10 * (
        norm(R_eff) * norm(beta_R) + norm(r) + 1.0
    ):
        msg = "Infeasible linear restrictions: ||R beta - r|| too large."
        raise ValueError(msg)

    if keep_mask is not None:
        beta_full = np.zeros((keep_mask.size, 1), dtype=np.float64)
        beta_full[keep_mask, :] = beta_R
        beta_R = beta_full
        yhat = dot(Xd_orig, beta_R)
    else:
        yhat = dot(Xd, beta_R)
    resid = y - yhat
    return np.asarray(yhat, dtype=np.float64), np.asarray(resid, dtype=np.float64)


def wls_fit_restricted(  # noqa: PLR0913
    y: NDArray[np.float64],
    X: Matrix,
    R: np.ndarray,
    r: np.ndarray,
    *,
    weights: Sequence[float] | None = None,
    method: str = "qr",
    drop_rank_deficient: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Public wrapper for :func:`_wls_fit_restricted`."""
    return _wls_fit_restricted(
        y,
        X,
        R,
        r,
        weights=weights,
        method=method,
        drop_rank_deficient=drop_rank_deficient,
    )


def apply_wild_bootstrap(  # noqa: PLR0913
    yhat: np.ndarray | None = None,
    resid: np.ndarray | None = None,
    multipliers: np.ndarray | None = None,
    *,
    y: np.ndarray | None = None,
    X: Matrix | None = None,
    residual_type: str = "WCR",
    clusters: Sequence[int] | None = None,
    weights: Sequence[float] | None = None,
    allow_weights: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    na_action: str = "drop",
    null_R: NDArray[np.float64] | None = None,
    null_r: NDArray[np.float64] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Construct Y* = yhat + u_adj ⊙ W for all replications at once.

    Residuals
    ---------
    - If (yhat,resid) are provided, they are used as-is.
    - Otherwise, if (y,X) are provided:
        * residual_type="unrestricted" -> WLS residuals
        * residual_type="restricted"   -> restricted WLS with R beta = r

    For WCU (unrestricted), do NOT recenter residuals under the null (MNW/boottest
    semantics). For WCU_score, perform MNW-style score recentering which solves
    M_gg z = u_g for each cluster (see _score_recentering).

    NA handling
    -----------
    - na_action="propagate": keep NaNs as-is (they propagate through the product).
    - na_action="drop": drop rows with any NaN in (yhat,resid) or multipliers and return mask.
    """
    # Initialize RNG if provided for future extensions (kept for API compatibility)
    if rng is None and seed is not None:
        rng = np.random.default_rng(seed)
    _aliases = {"restricted": "WCR", "unrestricted": "WCU"}
    residual_type = _aliases.get(residual_type, residual_type)
    if residual_type not in {"WCR", "WCU", "WCU_score"}:
        msg = 'residual_type must be "WCR", "WCU", or "WCU_score".'
        raise ValueError(msg)
    if na_action != "drop":
        msg = "Strict bootstrap: na_action='drop' only."
        raise ValueError(msg)

    # Use provided multipliers (W) — caller must supply or use cluster_multipliers/wild_multipliers
    if multipliers is None:
        msg = "multipliers must be provided to apply_wild_bootstrap; use cluster_multipliers or wild_multipliers to build them."
        raise ValueError(msg)

    # Normalize inputs and optionally drop zero-weight observations
    W = np.asarray(multipliers, dtype=np.float64)

    # Enforce policy: weights are forbidden unless explicitly allowed (GLS/GMM only)
    if (weights is not None) and (not allow_weights):
        raise ValueError(
            "Weights are forbidden for OLS/IV/QR/IV-QR per policy. Use GLS/GMM with allow_weights=True.",
        )

    # Compute yhat/resid if not supplied
    if yhat is not None and resid is not None:
        yh = np.asarray(yhat, dtype=np.float64)
        u = np.asarray(resid, dtype=np.float64)
    else:
        if y is None or X is None:
            msg = "If (yhat,resid) are not provided, supply (y,X)."
            raise ValueError(msg)
        if residual_type == "WCR":
            if null_R is None or null_r is None:
                msg = "restricted residuals require (null_R, null_r)."
                raise ValueError(msg)
            yh, u = _wls_fit_restricted(y, X, null_R, null_r, weights=weights)
        else:
            yh, u = _wls_fit(y, X, weights=weights)

    # Recenter under null (per-cluster): implement MNW-consistent score recentering.
    # - WCU_score: apply M_gg^{-1} to u_g where M_gg = I - H_g (H_g uses global X (weighted if provided)).
    #   The operation is performed by solving M_gg z = u_g (avoid explicit inverse), stable via cholesky or pinv fallback.
    # - WCU: classic unrestricted bootstrap: do NOT recenter residuals (MNW/boottest semantics).
    if residual_type == "WCU_score" and clusters is not None:
        if X is None:
            msg = "WCU_score recentering requires X to be provided (cannot use yhat/resid alone)."
            raise ValueError(msg)
        # Centralized recentering helper which ensures MNW/boottest semantics and numeric stability.
        u = _score_recentering(u, X, clusters, weights=weights)
    elif residual_type == "WCU":
        # Classic WCU: no recentering under null (MNW/boottest semantics)
        pass

    # Build bootstrap outcomes: yh + u ⊙ W
    # Ensure shapes are compatible: yh (n,1), u (n,1), W (n,B) -> Ystar (n,B)
    # Broadcasting rules: (n,1) + (n,1) * (n,B) = (n,1) + (n,B) = (n,B)

    # Validate inputs before reshaping
    if yh.ndim > 2:
        msg = f"yhat must be 1D or 2D, got {yh.ndim}D with shape {yh.shape}"
        raise ValueError(msg)
    if u.ndim > 2:
        msg = f"resid must be 1D or 2D, got {u.ndim}D with shape {u.shape}"
        raise ValueError(msg)

    # Reshape to column vectors if needed
    if yh.ndim == 1:
        yh = yh.reshape(-1, 1)
    elif yh.shape[1] != 1:
        msg = f"yhat must be a column vector (n,1), got shape {yh.shape}. Multiple outcome columns are not supported."
        raise ValueError(msg)

    if u.ndim == 1:
        u = u.reshape(-1, 1)
    elif u.shape[1] != 1:
        msg = f"resid must be a column vector (n,1), got shape {u.shape}. Multiple outcome columns are not supported."
        raise ValueError(msg)

    if yh.shape[0] != W.shape[0]:
        msg = f"Shape mismatch: yhat has {yh.shape[0]} rows but multipliers have {W.shape[0]} rows"
        raise ValueError(msg)
    if u.shape[0] != W.shape[0]:
        msg = f"Shape mismatch: resid has {u.shape[0]} rows but multipliers have {W.shape[0]} rows"
        raise ValueError(msg)

    # Element-wise multiplication with broadcasting: u (n,1) * W (n,B) = (n,B)
    Ystar = yh + u * W

    mask_ret: np.ndarray | None = None
    if na_action == "drop":
        mask_ret = np.all(np.isfinite(Ystar), axis=1)
        Ystar = Ystar[mask_ret, :]
    return np.asarray(Ystar, dtype=np.float64), mask_ret


# ---------------------------------------------------------------------
# Uniform (sup-t) band
# ---------------------------------------------------------------------


def _sup_t_distribution_internal(  # noqa: PLR0913
    theta: NDArray[np.float64],
    theta_star: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    studentize: str = "bootstrap",
    zero_se_tol: float = 1e-12,
    zero_se_rel: float = 1e-12,
    scale: str = "sd",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Compute sup-t bootstrap studentization objects from bootstrap draws.

    This routine implements bootstrap-based studentization only (studentize='bootstrap'):
    - se: K-vector of bootstrap standard errors computed from theta_star (across B draws)
    - T: (K x B) studentized deviations (theta_star - theta) / se
    - Tmax: length-B vector of sup-t statistics (max abs across params per draw)
    - c: critical value (order-statistic with (B+1) finite-sample correction)

    The function is intentionally conservative and raises if bootstrap standard
    errors are non-finite or below the specified floors. This avoids any
    dependence on analytic VCVs when callers request bootstrap studentization.

    Parameters
    ----------
    scale : str, default "sd"
        Scale estimator for studentization: "sd" uses sample standard deviation
        (ddof=1), "iqr" uses IQR normalized to Gaussian scale (did/DRDID compatible).
    """
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    thb = np.asarray(theta_star, dtype=np.float64)
    if thb.ndim != 2:
        msg = "theta_star must be (K x B)."
        raise ValueError(msg)
    K, B = thb.shape
    if B < 2:
        raise ValueError("Uniform bands require at least 2 bootstrap draws (B>=2).")
    if th.shape[0] != K:
        msg = "theta and theta_star dimension mismatch."
        raise ValueError(msg)

    if studentize != "bootstrap":
        msg = "sup_t_distribution currently supports only studentize='bootstrap'."
        raise ValueError(msg)

    if scale == "iqr":
        from scipy import stats
        iqr_norm = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
        se = np.zeros((K,), dtype=np.float64)
        for k in range(K):
            iqr_val = float(np.percentile(thb[k, :], 75) - np.percentile(thb[k, :], 25))
            se[k] = iqr_val / iqr_norm if iqr_norm > 0 else 0.0
    else:
        se = np.std(thb, axis=1, ddof=1) if B > 1 else np.zeros((K,), dtype=np.float64)

    floor = max(
        zero_se_tol, zero_se_rel * float(np.max(np.abs(se)) if se.size else 0.0),
    )
    # Treat coordinates with non-finite or near-zero SE as non-contributing to
    # the sup statistic (i.e. their studentized T==0 and they receive band
    # half-width 0). This preserves theoretical behaviour when a coordinate
    # is perfectly identified across all bootstrap draws (enumeration/completely
    # identified parameter). We set these SEs to +inf so the studentized T
    # evaluates to zero and they do not affect the sup.
    bad = (~np.isfinite(se)) | (se <= floor)
    if np.any(bad):
        se[bad] = np.inf

    # Studentized T draws and sup-t per-draw
    T = (thb - th.reshape(-1, 1)) / se.reshape(-1, 1)
    Tmax = np.max(np.abs(T), axis=0)

    # R-compatible order statistic: k_idx = ceil((1 - alpha)(B + 1)) - 1
    # This finite-sample correction matches boottest/fwildclusterboot convention.
    order = np.sort(Tmax)
    k_idx = int(np.ceil((1.0 - alpha) * (B + 1))) - 1
    k_idx = min(max(k_idx, 0), B - 1)
    c = float(order[k_idx])

    return (
        se.astype(np.float64),
        T.astype(np.float64),
        Tmax.astype(np.float64),
        float(c),
    )


def bootstrap_se(beta_star: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute bootstrap standard errors from bootstrap coefficient draws.

    Parameters
    ----------
    beta_star : (K x B) array of bootstrapped coefficients.

    Returns
    -------
    se : (K,) array of bootstrap standard errors (ddof=1 when B>1).

    """
    B = None
    arr = np.asarray(beta_star, dtype=np.float64)
    if arr.ndim != 2:
        msg = "beta_star must be a 2-D array of shape (K, B)"
        raise ValueError(msg)
    K, B = arr.shape
    if B <= 1:
        # With a single draw, provide zeros but keep dtype
        return np.zeros((K,), dtype=np.float64)
    # Unbiased sample std (ddof=1) per bootstrap convention
    se = np.std(arr, axis=1, ddof=1)
    return se.astype(np.float64)


def uniform_confidence_band(  # noqa: PLR0913
    theta: NDArray[np.float64],
    theta_star: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    vcov_plugin: callable | None = None,
    studentize: str = "bootstrap",
    zero_se_tol: float = 1e-12,
    zero_se_rel: float = 1e-12,
    context: str | None = None,
    family: Sequence[Sequence[int]] | None = None,
    scale: str = "sd",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sup-t uniform confidence band for a vector of parameters from bootstrap replicates.

    This function implements sup-t uniform confidence bands using bootstrap
    studentization only. For theory-strict policy, analytic or per-replicate
    plug-in VCV estimation is prohibited in public estimator code. If callers
    pass a ``vcov_plugin`` callable this function will raise an error. Callers
    that need to perform internal diagnostic VCV re-estimation must do so
    outside of public-band construction and explicitly document the use.
    """
    # Context guard: only allow uniform bands for a restricted set of estimands
    # per project policy (DID / event-study / synthetic_control / RCT). This
    # prevents accidental use of sup-t bands for OLS/IV/GMM/QR where CI
    # construction is disallowed by policy.
    allowed = {"did", "eventstudy", "synthetic_control", "rct"}
    if context is None or context not in allowed:
        msg = f"uniform_confidence_band is only allowed for {allowed} by project policy; received context={context!r}."
        raise ValueError(msg)

    # Disallow analytic or per-replicate VCV plugin usage to enforce bootstrap-only
    # inference in public APIs. Requesters who must run per-replicate VCVs for
    # diagnostics should implement that logic separately and not pass a plugin
    # here.
    if vcov_plugin is not None:
        msg = "vcov_plugin is not allowed: uniform_confidence_band only supports bootstrap studentization."
        raise ValueError(msg)

    # Otherwise, use bootstrap-based studentization (policy-compliant default)
    if studentize != "bootstrap":
        msg = "Only studentize='bootstrap' is supported when vcov_plugin is None."
        raise ValueError(msg)
    se, T, _Tmax, _c = _sup_t_distribution_internal(
        theta,
        theta_star,
        alpha=alpha,
        studentize=studentize,
        zero_se_tol=zero_se_tol,
        zero_se_rel=zero_se_rel,
        scale=scale,
    )
    # If RCT, allow `family` to partition coordinates into hypothesis families.
    # family: iterable of index sequences; compute per-draw family-wise sup|T|
    # and pool by taking the maximum across families to obtain pooled Tmax.
    if context == "rct" and family is not None:
        T = np.asarray(T)
        if T.ndim != 2:
            raise ValueError("internal sup-t results have unexpected shape")
        B = T.shape[1]
        Tmax_fam = np.full((B,), -np.inf, dtype=np.float64)
        for idx in family:
            cols = np.asarray(idx, dtype=int)
            if cols.size == 0:
                continue
            if cols.max() >= T.shape[0]:
                raise ValueError("family indices out of range for parameters")
            fam_max = np.max(np.abs(T[cols, :]), axis=0)
            Tmax_fam = np.maximum(Tmax_fam, fam_max)
        _c = float(finite_sample_quantile(Tmax_fam, 1.0 - float(alpha)))
    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    se_arr = np.asarray(se, dtype=np.float64).reshape(-1)
    finite = np.isfinite(se_arr)
    delta = np.zeros_like(se_arr, dtype=np.float64)
    if np.any(finite):
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            delta[finite] = _c * se_arr[finite]
    lo = theta_arr - delta
    hi = theta_arr + delta
    return lo.astype(np.float64), hi.astype(np.float64)


# ---------------------------------------------------------------------
# Spatial / distance-based cluster construction
# ---------------------------------------------------------------------


def radius_graph_clusters(  # noqa: PLR0913
    coords: np.ndarray,
    *,
    radius: float,
    metric: str = "euclidean",
    earth_radius_km: float = 6371.0,
    time: Sequence[float] | None = None,
    time_window: float | None = None,
) -> np.ndarray:
    """Build cluster ids from coordinates by connecting i,j when dist(i,j) <= radius
    (and optionally |t_i - t_j| <= time_window), then returning connected components.

    Returns integer codes (n,) with values in {0,...,G-1}.
    """
    C = np.asarray(coords, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] == 0:
        msg = "coords must be a non-empty (n x d) array."
        raise ValueError(msg)
    if not (isinstance(radius, (int, float)) and float(radius) > 0.0):
        msg = "radius must be a positive scalar."
        raise ValueError(msg)
    n = C.shape[0]
    t = None if time is None else np.asarray(time).reshape(-1)
    if t is not None and t.shape[0] != n:
        msg = "time must have the same length as coords."
        raise ValueError(msg)
    use_time = (
        (t is not None) and (time_window is not None) and (float(time_window) >= 0.0)
    )

    metr = metric.lower()
    if metr == "euclidean":
        # Pairwise squared Euclidean distances (exact O(n^2)). Use dot-product
        # identity to avoid explicit broadcasting where possible.
        ss = np.sum(C * C, axis=1, keepdims=True)
        D2 = ss + ss.T - 2.0 * la.dot(C, C.T)
        np.maximum(D2, 0.0, out=D2)
        D = np.sqrt(D2, dtype=np.float64)
    elif metr in {"haversine", "greatcircle", "great-circle"}:
        # coords expected as (lon_deg, lat_deg); radius is in same units as
        # earth_radius_km when using haversine (default kilometers).
        if C.shape[1] < 2:
            msg = "haversine metric requires coords with longitude and latitude columns"
            raise ValueError(msg)
        lon = np.radians(C[:, 0].astype(np.float64))
        lat = np.radians(C[:, 1].astype(np.float64))
        n = C.shape[0]
        D = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            dlat = lat - lat[i]
            dlon = lon - lon[i]
            a = (np.sin(dlat * 0.5) ** 2) + np.cos(lat[i]) * np.cos(lat) * (
                np.sin(dlon * 0.5) ** 2
            )
            # numerical guard
            a = np.clip(a, 0.0, 1.0)
            c = 2.0 * np.arcsin(np.sqrt(a))
            D[i, :] = earth_radius_km * c
    else:
        msg = "metric must be 'euclidean' or 'haversine'"
        raise ValueError(msg)

    within = float(radius) >= D

    if use_time:
        Td = np.abs(t.reshape(-1, 1) - t.reshape(1, -1))
        within &= Td <= float(time_window)

    # Union-Find to extract connected components
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int64)

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

    for i in range(n - 1):
        row = within[i, (i + 1) :]
        js = np.nonzero(row)[0]
        if js.size:
            for jj in js:
                union(i, i + 1 + int(jj))

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, inv = np.unique(roots, return_inverse=True)
    return inv.astype(np.int64, copy=False)


def spatial_distance_multipliers(  # noqa: PLR0913
    coords: np.ndarray,
    *,
    radius: float,
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    dist: WildDist | str = "rademacher",
    enumeration: bool = True,
    enumeration_mode: str = "boottest",
    enum_max_g: int = ENUM_THRESH_BOOTTEST,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    policy: str = "boottest",
    time: Sequence[float] | None = None,
    time_window: float | None = None,
    metric: str = "euclidean",
    earth_radius_km: float = 6371.0,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Build cluster ids as connected components of the radius graph and call
    `cluster_multipliers` to obtain multipliers. Returns (W, log, clusters).
    """
    clusters = radius_graph_clusters(
        coords,
        radius=radius,
        time=time,
        time_window=time_window,
        metric=metric,
        earth_radius_km=earth_radius_km,
    )
    W, log = cluster_multipliers(
        clusters,
        n_boot=n_boot,
        dist=dist,
        use_enumeration=enumeration,
        enumeration_mode=enumeration_mode,
        enum_max_g=enum_max_g,
        seed=seed,
        rng=rng,
        policy=policy,
    )
    log = dict(log)
    log.update(
        {
            "method": "distance_based_clusters",
            "radius": float(radius),
            "G": int(np.max(clusters) + 1),
            "time_window": (None if time_window is None else float(time_window)),
            "metric": metric,
            "earth_radius_km": float(earth_radius_km)
            if metric.lower() in {"haversine", "greatcircle", "great-circle"}
            else None,
        },
    )
    return W, log, clusters


# ---------------------------------------------------------------------
# Simple bootstrap summaries
# ---------------------------------------------------------------------


def wild_bootstrap_betas(  # noqa: PLR0913
    y: NDArray[np.float64],
    X: Matrix,
    beta_hat: NDArray[np.float64],
    multipliers: NDArray[np.float64],
    *,
    weights: Sequence[float] | None = None,
    allow_weights: bool = False,
) -> NDArray[np.float64]:
    """Apply wild bootstrap to linear model: y* = X beta_hat + u * W, where u = y - X beta_hat.

    Parameters
    ----------
    y : (n x 1) observed outcome.
    X : (n x K) design matrix.
    beta_hat : (K x 1) estimated coefficients.
    multipliers : (n x B) wild multipliers (distribution mean 0, variance 1).
    weights : optional (n x 1) observation weights.

    Returns
    -------
    beta_star : (K x B) bootstrapped coefficients.

    """
    W = np.asarray(multipliers, dtype=np.float64)
    # Enforce weight policy: by default weights are forbidden for OLS/IV/QR
    if (weights is not None) and (not allow_weights):
        msg = "Weights are forbidden for OLS/IV/QR/IV-QR per project policy. Use GLS/GMM with allow_weights=True."
        raise ValueError(msg)
    # NOTE: Do not perform column-wise recentring of W here.
    # The wild/multiplier distributions used in this library are defined to
    # have mean zero in expectation. Re-centering each bootstrap column would
    # change the reference distribution for finite B; boottest/fwildclusterboot
    # therefore avoid per-column recentering. We document this choice explicitly
    # so all modules maintain the same behavior.
    _n, B = W.shape
    y_hat = dot(X, beta_hat)
    u = y - y_hat
    y_star = y_hat + hadamard(u, W)  # (n x B)
    # Precompute a pivoted QR of the (possibly weighted) design once and reuse it
    beta_star = np.empty((beta_hat.shape[0], B), dtype=np.float64)
    if weights is not None:
        w = _validate_weights(weights, X.shape[0]).reshape(-1, 1)
        sqrt_w = np.sqrt(w)
        Xf = hadamard(X, sqrt_w)
    else:
        Xf = X

    # Compute pivoted QR once (Q, R, P) and determine numerical rank using
    # the library's rank policy to match Stata/R behaviour.
    Q, R, P = la.qr(Xf, pivoting=True)
    diagR = np.abs(np.diag(R))
    r = la._rank_from_diag(diagR, Xf.shape[1], mode="stata")  # noqa: SLF001

    # Vectorized multi-RHS solve: compute Qt @ Y for all bootstrap draws at once
    Yrhs = y_star
    if weights is not None:
        Yrhs = hadamard(Yrhs, sqrt_w)
    QtY = la.dot(Q.T, Yrhs)  # shape (ncols, B) in economic mode
    # Prepare output and fill identified rows via a single triangular solve
    out = np.zeros((X.shape[1], B), dtype=np.float64)
    if r > 0:
        sol = la.solve(R[:r, :r], QtY[:r, :], method="qr", rank_policy="stata")
        out[P[:r], :] = np.asarray(sol)
    beta_star[:, :] = out
    return beta_star


def wald_test_wild_bootstrap(  # noqa: PLR0913
    X: np.ndarray,
    y: np.ndarray,
    *,
    R: NDArray[np.float64] | None = None,
    r: NDArray[np.float64] | None = None,
    residual_type: str = "WCU",  # semantic default: unrestricted recentering semantics
    variant: str | None = "33",
    B: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    dist: WildDist | str = "rademacher",
    clusters: Sequence[int] | None = None,
    multipliers: np.ndarray | None = None,
    enumeration: bool = True,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    vcov_kind: str = "auto_strict",
    policy: str | None = None,
    enum_max_g: int | None = None,
    enumeration_mode: str | None = None,
    weights: Sequence[float] | None = None,
    allow_weights: bool = False,
    ssc: dict[str, str | int | float | bool] | None = None,
    vcov_fix: dict[str, object] | None = None,
    denominator: str | None = "boottest_strict",
    **kwargs,
) -> dict:
    # Silence unused-argument warnings (kept for public API compatibility)
    _ = (vcov_fix, kwargs)
    """
    Complete Wald test with wild bootstrap, supporting WCR/WCU, strict bootstrap,
    enumeration, and multiway clustering. Aligns with boottest/fwildclusterboot.

    - residual_type: "WCR" (restricted residuals), "WCU" (unrestricted, recentered on residuals), "WCU_score" (unrestricted, recentered on scores).
      Alternatively, pass canonical variant strings "11", "13", "31", "33" (see _parse_bootstrap_variant). Default "33" (MNW recommended).
    - vcov_kind: "CRV1", "CRV3", "HC3", "auto_strict" for cluster-robust VCE.
        - enumeration: Default policy is 'boottest' style enumeration. Exact enumeration
            (2^G) or Webb promotion is used according to the 'boottest' policy. Webb must
            be requested explicitly by dist='webb'.
    - ssc: Small-sample corrections per fixest::ssc().
    - vcov_fix: Clip negative eigenvalues in VCV.
        - denominator: "bootstrap" uses B-1 (unbiased sample covariance, default);
            "boottest_strict" uses B (exact R boottest/fwildclusterboot equivalence).
    - Returns stat, the bootstrap distribution (wald_star), df, and bootstrap details.
    """
    # Variant handling: prefer explicit `variant` arg (canonical MNW codes),
    # otherwise fall back to `residual_type` when it encodes a canonical code.
    variant_arg = variant or (
        residual_type
        if isinstance(residual_type, str)
        and residual_type.strip().replace("-", "").replace("_", "").upper()
        in {"11", "13", "31", "33", "33J"}
        else None
    )
    if variant_arg is not None:
        _resid, _vcov = _parse_bootstrap_variant(variant_arg)
        if clusters is None:
            residual_type = "WCU"
            vcov_kind = "HC3"
        else:
            residual_type, vcov_kind = _resid, _vcov

    # Enumeration defaults: use boottest convention unless caller overrides
    eff_policy = "boottest" if policy is None else policy
    eff_enum_max_g = 11 if enum_max_g is None else int(enum_max_g)
    eff_enum_mode = enumeration_mode or "boottest"

    n = X.shape[0]
    k = X.shape[1]

    # Normalize ssc to canonical internal keys matching R fixest/fwildclusterboot
    ssc = _normalize_ssc(ssc)

    # Build or accept multipliers
    # Block weights unless explicitly allowed (GLS/GMM only)
    if (weights is not None) and (not allow_weights):
        msg = "Weights are forbidden for OLS/IV/QR/IV-QR per project policy. Use GLS/GMM with allow_weights=True."
        raise ValueError(msg)

    # allow callers to override enumeration/policy thresholds; fall back to caller args when provided
    eff_policy = "boottest" if policy is None else policy
    eff_enum_max_g = 11 if enum_max_g is None else int(enum_max_g)
    eff_enum_mode = enumeration_mode or "boottest"

    if multipliers is None:
        if clusters is None:
            W = wild_multipliers(n, n_boot=B, dist=dist, seed=seed, rng=rng)
            log = {
                "enumerated": False,
                "effective_dist": str(dist),
                "effective_B": W.shape[1],
                "method": "iid",
            }
        else:
            W, log = cluster_multipliers(
                clusters,
                n_boot=B,
                dist=dist,
                use_enumeration=enumeration,
                enumeration_mode=eff_enum_mode,
                enum_max_g=eff_enum_max_g,
                seed=seed,
                rng=rng,
                policy=eff_policy,
            )
    else:
        W = np.asarray(multipliers, dtype=np.float64)
        log = {"enumerated": False, "provided": True, "effective_B": W.shape[1]}

    B_actual = W.shape[1]

    # Original unrestricted fit (QR-first, SVD fallback). Use sqrt-weight preprocessing when weights provided.
    if weights is None:
        beta0 = la.solve(X, y, method="qr")
    else:
        w0 = _validate_weights(weights, X.shape[0]).reshape(-1, 1)
        sqrt_w0 = np.sqrt(w0)
        Xw0 = hadamard(X, sqrt_w0)
        yw0 = hadamard(y, sqrt_w0)
        beta0 = la.solve(Xw0, yw0, method="qr")

    # Build yhat and residuals depending on residual_type
    if residual_type == "WCR":
        yhat, u = _wls_fit_restricted(y, X, R, r, weights=weights, method="svd")
    else:
        yhat = dot(X, beta0)
        u = y - yhat
        # Recenter under null for WCU variants
        if residual_type == "WCU_score":
            # Delegate to the centralized recentering implementation to avoid
            # duplicated code paths and ensure numerical parity with
            # apply_wild_bootstrap. Record that score recentering was used for
            # reproducibility in logs returned to the caller.
            if clusters is None:
                msg = "WCU_score recentering requires clusters to be provided."
                raise ValueError(msg)
            u = _score_recentering(u, X, clusters, weights=weights)
            # mark in log that recentering was applied and record an orthogonality
            # diagnostic (||X' W z||) so users can inspect numeric quality.
            log["score_recentering"] = True
            try:
                w_chk = (
                    _validate_weights(weights, X.shape[0])
                    if weights is not None
                    else np.ones(X.shape[0], dtype=float)
                )
                # compute weighted cross-product norm || X' (W * z) ||_F
                z_vec = u.reshape(-1, 1)
                W_sqrt = np.sqrt(w_chk).reshape(-1, 1)
                Xw = hadamard(X, W_sqrt)
                z_w = hadamard(z_vec, W_sqrt)
                cross = la.crossprod(Xw, z_w)
                log["score_recentering_norm"] = float(la.norm(cross))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                # best-effort: do not fail the bootstrap because of the diagnostic
                log["score_recentering_norm"] = None

    # Build bootstrap coefficient draws and use plug-in empirical covariance
    # of R * beta_star to avoid any analytic VCV dependence. This mirrors the
    # IV ar_test strategy: compute Rb_hat and Rb_star, build empirical cov and
    # invert it in a rank-aware way using core.linalg helpers.
    betas_star = np.empty((k, B_actual), dtype=np.float64)
    for b in range(B_actual):
        yb = yhat + u * W[:, b : b + 1]
        if weights is None:
            beta_b = la.solve(X, yb, method="qr")
        else:
            wb = _validate_weights(weights, X.shape[0]).reshape(-1, 1)
            sqrt_wb = np.sqrt(wb)
            Xwb = hadamard(X, sqrt_wb)
            ybb = hadamard(yb, sqrt_wb)
            beta_b = la.solve(Xwb, ybb, method="qr")
        betas_star[:, b] = beta_b.reshape(-1)

    # R * beta deviations: observed and bootstrap draws (q x 1) and (q x B)
    Rb_hat = la.dot(R, beta0) - r
    Rb_star = la.dot(R, betas_star) - r  # shape (q x B)

    # Compute empirical covariance of Rb_star with configurable denominator.
    # "bootstrap" (default): unbiased denominator (B-1) for standard bootstrap inference.
    # "boottest_strict": biased denominator (B) for exact R boottest/fwildclusterboot equivalence.
    # Determine denominator policy: if caller passed None, choose default
    # based on the effective bootstrap policy. For boottest policy we use the
    # boottest convention (denominator = B), otherwise use the standard
    # bootstrap unbiased denominator (B-1).
    if denominator is None:
        # Default to boottest_strict to maintain boottest parity by default.
        denominator = "boottest_strict"
    if denominator not in {"bootstrap", "boottest_strict"}:
        msg = "denominator must be 'bootstrap' (B-1, unbiased) or 'boottest_strict' (B, biased)."
        raise ValueError(msg)

    # Clarify naming: provide a cov_denom alias for return metadata while
    # preserving the function signature 'denominator' for backward compatibility.
    cov_denom = denominator

    q = Rb_star.shape[0]
    B_eff = max(1, Rb_star.shape[1])
    denom = float(B_eff - 1) if denominator == "bootstrap" else float(B_eff)
    # Guard against zero denominator when B=1 and denominator="bootstrap"
    if denom <= 0:
        denom = 1.0

    if q == 1:
        # 1-d case: variance scalar
        mean_rb = float(Rb_star.mean(axis=1))
        var_rb = float(((Rb_star.ravel() - mean_rb) ** 2).sum() / denom)
        Rb_cov = np.atleast_2d(np.array(var_rb, dtype=np.float64))
    else:
        # Center the draws and form covariance using core.linalg helpers to
        # ensure consistent dense/sparse behavior and centralized numeric
        # policies. Use denominator chosen above (B-1 or B).
        # Use core.linalg col_mean for consistent, sparse-safe column means
        mean_rb = la.col_mean(Rb_star, ignore_nan=False).reshape(-1, 1)
        centered = Rb_star - mean_rb
        # Use la.dot instead of the @ operator so multiplication routes through
        # the canonical linear-algebra implementation (sparse-aware, batched
        # optimizations and consistent tolerance policies).
        Rb_cov = (la.dot(centered, centered.T) / denom).astype(np.float64)

    # Regularize / invert covariance in a rank-aware manner using eigendecomposition
    evals, Q = la.eigh(Rb_cov)
    tol_cov = la.eig_tol(Rb_cov)
    keep = evals > tol_cov
    if not np.any(keep):
        # fallback to pseudo-inverse if no eigenvalue is above tolerance
        Rb_cov_inv = la.pinv(Rb_cov)
    else:
        Qk = Q[:, keep]
        Rk = la.dot(Qk.T, la.dot(Rb_cov, Qk))
        try:
            Lk = la.safe_cholesky(Rk)
            Rk_inv = la.chol_solve(Lk, la.eye(Rk.shape[0]))
            Rb_cov_inv = la.dot(Qk, la.dot(Rk_inv, Qk.T))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            # If Cholesky fails (numerical), fall back to pseudo-inverse on reduced rank
            Rb_cov_inv = la.pinv(Rb_cov)

    # Observed Wald statistic and bootstrap replicates using the plug-in inverse
    W_stat = float(la.dot(la.dot(Rb_hat.T, Rb_cov_inv), Rb_hat))
    W_star = np.empty(B_actual, dtype=float)
    for b in range(B_actual):
        rb = Rb_star[:, b : b + 1]
        W_star[b] = float(la.dot(la.dot(rb.T, Rb_cov_inv), rb))

    # Per project policy: do not compute or return analytic p-values or critical values.
    # Return the observed statistic and the full bootstrap distribution so callers
    # may compute any desired summaries externally (e.g., exact enumeration-based p-values).
    # Standardize return metadata keys and types for reproducibility/logging.
    return {
        "wald_stat": W_stat,
        "wald_star": W_star.tolist(),
        "df": R.shape[0],
        "B": int(B_actual),
        "enumerated": bool(log.get("enumerated", False)),
        "dist": str(dist),
        "vcov_kind": vcov_kind,
        "residual_type": residual_type,
        "effective_dist": log.get("effective_dist"),
        "effective_B": int(log.get("effective_B", B_actual)),
        "bootcluster": log.get("bootcluster"),
        "selected_dim": log.get("selected_dim"),
        "G": log.get("G"),
        "method": log.get("method"),
        # clarified cov denom naming
        "cov_denom": cov_denom,
    }


# boottest_exact() REMOVED per project policy: bootstrap-only inference, no analytical VCV.
# All estimators use bootstrap_wald_test() which re-samples from bootstrap draws
# without re-computing analytical covariance matrices per draw.


def _cluster_meat(
    Xw: NDArray[np.float64],
    uw: NDArray[np.float64],
    inv_XtWX: NDArray[np.float64],
    groups: NDArray[np.int64],
    *,
    kind: str,  # "CRV1" | "CRV2" | "CRV3"
) -> NDArray[np.float64]:
    """Compute cluster 'meat' for CRV1/CRV2/CRV3.
    Xw   : (n x k) weighted design,  uw : (n x 1) weighted residuals
    inv_XtWX : (k x k) inverse of X'WX
    groups   : (n,) integer codes 0..G-1
    """
    k = Xw.shape[1]
    meat = np.zeros((k, k), dtype=np.float64)
    G = int(groups.max()) + 1
    Icache = {}  # cache per cluster for small matrices
    for g in range(G):
        idx = groups == g
        if not np.any(idx):
            continue
        Xg = Xw[idx, :]  # (n_g x k)
        ug = uw[idx, :]  # (n_g x 1)

        if kind == "CRV1":
            Sg = crossprod(Xg, ug)  # (k x 1)
            meat += dot(Sg, Sg.T)
            continue

        # H_gg = Xg (X'WX)^{-1} Xg'
        Hgg = dot(dot(Xg, inv_XtWX), Xg.T)
        # M_gg = I - H_gg (symmetric PSD). Build symmetric guard.
        Igg = Icache.get(g)
        if Igg is None:
            Igg = la.eye(Hgg.shape[0])
            Icache[g] = Igg
        Mgg = (Igg - Hgg + (Igg - Hgg).T) * 0.5

        # Adjustment A_g
        if kind == "CRV2":
            # A_g = (I - H_gg)^{-1/2} via eigendecomposition (BRL, Bell-McCaffrey 2002)
            evals, evecs = la.eigh(Mgg)
            # clubSandwich-style clipping at 0 (g-inverse sqrt)
            neg = int(np.sum(evals < 0.0))
            if neg:
                warnings.warn(
                    f"Cluster PSD correction: clipped {neg} negative eigenvalues to zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            evals = np.maximum(evals, 0.0)
            with np.errstate(divide="ignore"):
                inv_sqrt = np.where(evals > 0.0, 1.0 / np.sqrt(evals), 0.0)
            Ag = dot(evecs * inv_sqrt, evecs.T)
        elif kind == "CRV3":
            # A_g = (I - H_gg)^{-1} (JK-equivalent)
            # Use pseudo-inverse if singular (as in clubSandwich)
            evals, evecs = la.eigh(Mgg)
            neg = int(np.sum(evals < 0.0))
            if neg:
                warnings.warn(
                    f"Cluster PSD correction: clipped {neg} negative eigenvalues to zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            evals = np.maximum(evals, 0.0)
            with np.errstate(divide="ignore"):
                inv_ = np.where(evals > 0.0, 1.0 / evals, 0.0)
            Ag = dot(evecs * inv_, evecs.T)
        else:
            msg = "kind must be CRV1/CRV2/CRV3"
            raise ValueError(msg)

        ug_adj = dot(Ag, ug)  # (n_g x 1)
        Sg = crossprod(Xg, ug_adj)  # (k x 1)
        meat += dot(Sg, Sg.T)
    return meat


def _cluster_meat_contrib(
    Xw: NDArray[np.float64],
    uw: NDArray[np.float64],
    inv_XtWX: NDArray[np.float64],
    groups: NDArray[np.int64],
    *,
    kind: str,
) -> tuple[NDArray[np.float64], list]:
    """Compute per-cluster meat contributions for CRV1/CRV2/CRV3 (clubSandwich/BRL).
    Returns total meat and list of per-cluster contributions.
    """
    k = Xw.shape[1]
    total = np.zeros((k, k), dtype=np.float64)
    contribs = []
    uniq_G = int(groups.max()) + 1 if groups.size else 0
    Icache = {}
    for g in range(uniq_G):
        m = groups == g
        if not np.any(m):
            contribs.append(np.zeros((k, k), dtype=np.float64))
            continue
        Xg = Xw[m, :]
        ug = uw[m, :]
        if kind == "CRV1":
            Sg = crossprod(Xg, ug)
            Cg = dot(Sg, Sg.T)
        else:
            Hgg = dot(dot(Xg, inv_XtWX), Xg.T)
            Igg = Icache.get(g) or la.eye(Hgg.shape[0])
            Icache[g] = Igg
            Mgg = (Igg - Hgg + (Igg - Hgg).T) * 0.5
            if kind == "CRV2":
                evals, evecs = la.eigh(Mgg)
                evals = np.maximum(evals, 0.0)
                with np.errstate(divide="ignore"):
                    inv_sqrt = np.where(evals > 0.0, 1.0 / np.sqrt(evals), 0.0)
                Ag = dot(evecs * inv_sqrt, evecs.T)
            elif kind == "CRV3":
                evals, evecs = la.eigh(Mgg)
                evals = np.maximum(evals, 0.0)
                with np.errstate(divide="ignore"):
                    inv_vals = np.where(evals > 0.0, 1.0 / evals, 0.0)
                Ag = dot(evecs * inv_vals, evecs.T)
            else:
                Ag = la.eye(Mgg.shape[0])
            tmp = dot(Xg.T, dot(Ag, ug))
            Cg = dot(tmp, tmp.T)
        contribs.append(Cg)
        total += Cg
    return total, contribs


def _score_recentering(
    u: NDArray[np.float64],
    X: Matrix,
    clusters: Sequence[int],
    *,
    weights: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Centralized MNW-style score recentering used by WCU_score.

    The Wild Cluster Score Bootstrap (WCU-score) adjusts residuals to satisfy
    **cluster-wise** score orthogonality: X_g' W_g u*_g = 0 for each cluster g.

    This is distinct from global orthogonality X' W u* = 0. The cluster-wise
    condition ensures that when we multiply by cluster-level wild multipliers v_g,
    the bootstrap scores X_g' W_g (v_g u*_g) have mean zero (over bootstrap draws).

    Implementation
    --------------
    Following MacKinnon, Nielsen, Webb (2023) and boottest/fwildclusterboot,
    for each cluster g we solve:

        minimize ||u*_g - u_g||^2  subject to  X_g' W_g u*_g = 0

    The solution is a projection using **cluster-specific** (X_g' W_g X_g)^{-1}:

        u*_g = (I - H_g) u_g  where  H_g = X_g (X_g' W_g X_g)^{-1} X_g' W_g

    For numerical stability, we compute the cluster-specific inverse via
    pseudoinverse when the cluster design matrix is rank-deficient.

    Parameters
    ----------
    u : ndarray
        Unrestricted residuals (n, 1) or (n,).
    X : Matrix
        Design matrix (n, k).
    clusters : array-like
        Cluster assignment (n,).
    weights : array-like, optional
        Observation weights for WLS (n,).

    Returns
    -------
    u_adj : ndarray
        Score-recentered residuals (n,) with X_g' W_g u_adj_g = 0 for each g.

    """
    if clusters is None:
        msg = "WCU_score recentering requires clusters to be provided."
        raise ValueError(msg)

    clusters_arr = np.asarray(clusters, dtype=np.int64).reshape(-1)
    Xd = to_dense(X)
    n, _k = Xd.shape

    if weights is not None:
        w = _validate_weights(weights, n)
    else:
        w = np.ones(n, dtype=np.float64)

    u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
    u_adj = u.copy()

    for cl in np.unique(clusters_arr):
        mask = clusters_arr == cl
        if not np.any(mask):
            continue
        Xg = Xd[mask, :]
        wg = w[mask].reshape(-1, 1)
        ug = u[mask, :]
        ng = Xg.shape[0]
        XgTWg = Xg.T * wg.T
        XgTWgXg = la.dot(XgTWg, Xg)
        try:
            inv_XgTWgXg = np.linalg.pinv(XgTWgXg)
        except np.linalg.LinAlgError:
            inv_XgTWgXg = np.linalg.pinv(XgTWgXg + 1e-12 * np.eye(XgTWgXg.shape[0]))
        # Hg = X_g (X_g' W_g X_g)^{-1} X_g' W_g is the oblique projection and
        # is NOT symmetric when W ≠ I.  Do not symmetrize; it would break the
        # score orthogonality property X_g' W_g (I - H_g) u_g = 0.
        Hg = la.dot(la.dot(Xg, inv_XgTWgXg), XgTWg)
        Mg = np.eye(ng, dtype=np.float64) - Hg
        u_adj[mask, :] = la.dot(Mg, ug)

    for cl in np.unique(clusters_arr):
        mask = clusters_arr == cl
        Xg = Xd[mask, :]
        wg = w[mask].reshape(-1, 1)
        u_g_adj = u_adj[mask, :]
        check_vec = la.dot(Xg.T, (wg * u_g_adj))
        scale = max(la.norm(Xg * wg, ord=2) * la.norm(u_g_adj, ord=2), 1.0)
        tol = 1e-8 * scale
        if not np.all(np.isfinite(check_vec)) or la.norm(check_vec) > tol:
            raise RuntimeError(
                f"Score recentering failed for cluster {cl}; check numerical stability.",
            )

    return u_adj.reshape(-1)


def _score_recentering_iv(
    u: NDArray[np.float64],
    Z: Matrix,
    clusters: Sequence[int] | None = None,
    *,
    eig_tol: float | None = None,
) -> NDArray[np.float64]:
    """Recenter IV scores so that, within each cluster g, Z_g' u_g = 0.

    Achieved by u_g <- u_g - Z_g a_g where a_g = (Z_g'Z_g)^+ Z_g' u_g (range-restricted
    pseudo-inverse). When `clusters` is None the operation is global.
    """
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1, 1)
    Zd = to_dense(Z)

    if clusters is None:
        # Global recentering: solve a = (Z'Z)^+ Z' u and return u - Z a
        ZtZ = la.crossprod(Zd, Zd)
        try:
            evals, U = la.eigh(ZtZ)
            tol = la.eig_tol(ZtZ) if eig_tol is None else float(eig_tol)
            keep = evals > tol
            if not np.any(keep):
                return u_arr.reshape(-1)
            # Build range-restricted pseudoinverse via eigenbasis
            Ur = U[:, keep]
            inv_vals = 1.0 / evals[keep]
            Zt_u = la.dot(Zd.T, u_arr)
            a = la.dot(Ur, la.dot(np.diag(inv_vals), la.dot(Ur.T, Zt_u)))
            u_new = u_arr - la.dot(Zd, a)
            return u_new.reshape(-1)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            # Fallback: use Moore-Penrose pseudoinverse
            try:
                a = la.pinv(la.dot(Zd.T, Zd))
                a = la.dot(a, la.dot(Zd.T, u_arr))
                return (u_arr - la.dot(Zd, a)).reshape(-1)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return u_arr.reshape(-1)

    # Clustered recentering: operate cluster-by-cluster
    clusters_arr = np.asarray(clusters, dtype=np.int64).reshape(-1)
    u_new = u_arr.copy()
    for g in np.unique(clusters_arr):
        mask = clusters_arr == g
        if not np.any(mask):
            continue
        Zg = Zd[mask, :]
        ug = u_arr[mask, :]
        if Zg.size == 0:
            continue
        ZgT_Zg = la.crossprod(Zg, Zg)
        try:
            evals, U = la.eigh(ZgT_Zg)
            tol = la.eig_tol(ZgT_Zg) if eig_tol is None else float(eig_tol)
            keep = evals > tol
            if not np.any(keep):
                # skip if no informative columns in this cluster
                continue
            Ur = U[:, keep]
            inv_vals = 1.0 / evals[keep]
            Zt_u = la.dot(Zg.T, ug)
            ag = la.dot(Ur, la.dot(np.diag(inv_vals), la.dot(Ur.T, Zt_u)))
            u_new[mask, :] = ug - la.dot(Zg, ag)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            try:
                ag = la.pinv(la.dot(Zg.T, Zg))
                ag = la.dot(ag, la.dot(Zg.T, ug))
                u_new[mask, :] = ug - la.dot(Zg, ag)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                # leave cluster unchanged on failure, but surface a warning for diagnostics
                warnings.warn(
                    f"Score recentering fallback failed for cluster {g}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
    return u_new.reshape(-1)


def score_recentering_iv(
    u: NDArray[np.float64],
    Z: Matrix,
    clusters: Sequence[int] | None = None,
    *,
    eig_tol: float | None = None,
) -> NDArray[np.float64]:
    """Public wrapper for :func:`_score_recentering_iv`."""
    return _score_recentering_iv(u, Z, clusters, eig_tol=eig_tol)


def cluster_robust_vcov(  # noqa: PLR0913
    X: np.ndarray,
    u: np.ndarray,
    clusters: Sequence[int] | None = None,
    *,
    kind: str = "CRV1",
    weights: Sequence[float] | None = None,
    y: np.ndarray | None = None,
    ssc: dict[str, str | int | float | bool] | None = None,
    vcov_fix: dict[str, object] | None = None,
    return_df: bool = False,
    bootcluster: str | int = "max",
    _internal: bool = False,
    _internal_allow: bool = False,
) -> np.ndarray:
    """Compute (cluster-robust) VCV with optional small-sample corrections.

    This function supports non-clustered HC2/HC3, single- and multi-way clustered
    CRV variants, and optional K adjustments to match fixest semantics.
    """
    # Policy: analytical VCV is disabled for inference paths. This function may
    # still be used internally for diagnostics when explicitly allowed by the
    # caller via _internal_allow=True. External callers should use bootstrap
    # studentization instead of analytic VCV.
    if not _internal_allow:
        raise RuntimeError(
            "Analytical VCOV is disabled for inference. Use bootstrap studentization.",
        )
    # prepare weighted design and residuals
    Xd = to_dense(X)
    u = np.asarray(u, float).reshape(-1, 1)
    n, k = Xd.shape
    w = (
        _validate_weights(weights, n)
        if weights is not None
        else np.ones(n, dtype=float)
    )
    sw = np.sqrt(w).reshape(-1, 1)
    Xw = hadamard(Xd, sw).astype(np.float64)  # (n x k)
    uw = hadamard(u, sw).astype(np.float64)  # (n x 1)
    # A = (X' W X)^{-1} via pivoted QR / SVD fallback (centralized helper)
    A = la.xtwx_inv_via_qr(X, weights=w)

    # Guard: analytic VCV computations are restricted to internal callers only
    if not _internal:
        msg = (
            "Analytical VCOV is disabled by design. "
            "Use bootstrap-based studentization / uniform bands for DID/event-study/synth only. "
            "Public APIs must not compute analytic standard errors or p-values. "
            "If calling from inside the package for diagnostics, pass _internal=True."
        )
        raise ValueError(msg)

    # Normalize ssc for consistent key access across branches
    ssc = _normalize_ssc(ssc)

    if kind == "auto_strict":
        # Non-clustered: HC3, Clustered: CRV3 (Imbens-Kolesar/Bell-McCaffrey small-sample robustness)
        kind = "HC3" if clusters is None else "CRV3"

    # Non-clustered branch: HC2/HC3
    if clusters is None:
        # EHW (HC2/HC3) for i.i.d. wild bootstrap (boottest/fwildclusterboot standard)
        if kind not in {"HC2", "HC3"}:
            msg = "Non-clustered VCV must be 'HC2' or 'HC3'. 'CRV*' is invalid without clusters."
            raise ValueError(msg)
        # Compute leverage (hat diagonal) using core.linalg.hat_diag_r
        # (R convention for wild bootstrap)
        hat = hat_diag_r(Xd, weights=w)
        h = 1.0 / (1.0 - hat) if kind == "HC2" else 1.0 / ((1.0 - hat) ** 2)
        # Meat: X' diag(h * u^2) X
        u2 = uw**2  # (n x 1)
        # FIX: ensure elementwise (n,) vector multiplication without accidental outer product
        h_u2 = h.reshape(-1) * u2.reshape(-1)
        meat = gram(Xd, h_u2)  # (k x k)
        V = dot(dot(A, meat), A)
        # optional K adjustment (fixest compatibility)
        if ssc.get("adj", False):
            K_fixef = ssc.get("fixef.K", "none")
            fe_dof = int(ssc.get("fe_dof", 0))
            N, Kp = Xd.shape
            K_eff = Kp + (fe_dof if K_fixef in {"full", "nonnested"} else 0)
            if K_eff >= N:
                msg = "K adjustment would lead to non-positive df."
                raise ValueError(msg)
            # fixest: (N - 1) / (N - K_eff)
            V *= float(N - 1) / float(N - K_eff)
        if return_df:
            return V.astype(np.float64), (Xd.shape[0] - Xd.shape[1])
        return V.astype(np.float64)

    # single-way vs multi-way
    if (
        isinstance(clusters, (list, tuple))
        and len(clusters) > 0
        and isinstance(clusters[0], (list, tuple, np.ndarray))
    ):
        # Multi-way clustering (inclusion-exclusion / CGM). Strict mode: only
        # CRV1 is supported for multi-way. CRV2/CRV3/CRV3J require single-way
        # derivations and are therefore disallowed here.
        clusters_list = [np.asarray(c, dtype=int).reshape(-1) for c in clusters]
        R = len(clusters_list)
        n_check = len(clusters_list[0])
        for c in clusters_list:
            if len(c) != n_check:
                msg = "All cluster arrays must have the same length."
                raise ValueError(msg)

        # If kind was requested as auto_strict, ensure we do not promote to CRV3
        # for multi-way: force CRV1 to preserve theoretical correctness.
        if kind == "auto_strict":
            # Prefer CRV1 for pure multi-way by default to be conservative,
            # but allow CRV2/CRV3 if the caller explicitly requests them.
            kind = "CRV1"
        if kind in {"CRV2", "CRV3", "CRV3J"}:
            # Strict theory: disallow CRV2/CRV3/CRV3J under multi-way clustering.
            msg = "Multi-way clustering with CRV2/CRV3/CRV3J is not supported under strict theory. Use CRV1 for multi-way clustering."
            raise ValueError(
                msg,
            )

        # Inclusion-exclusion: sum over non-empty subsets with alternating signs
        from itertools import combinations

        meat = np.zeros((k, k), dtype=np.float64)
        G_min = int(np.min([np.unique(c).size for c in clusters_list]))
        p = Xd.shape[1]
        if kind == "CRV1" and G_min < max(p + 1, 10):
            warnings.warn(
                "CRV1 is unreliable with few clusters; consider increasing clusters.",
                RuntimeWarning,
                stacklevel=2,
            )

        for r in range(1, R + 1):
            for S in combinations(range(R), r):
                sign = +1 if (r % 2 == 1) else -1
                cols = [np.asarray(clusters_list[j]).reshape(-1, 1) for j in S]
                codes_S = la.column_stack(cols)
                _, inv_S = np.unique(codes_S, axis=0, return_inverse=True)
                inv_S = inv_S.astype(np.int64, copy=False)
                if kind == "CRV1":
                    Gs = int(np.max(inv_S) + 1)
                    if Gs < max(p + 1, 10):
                        warnings.warn(
                            f"CRV1 may be unreliable with few clusters on subset {S}.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                meat += sign * _cluster_meat(Xw, uw, A, inv_S, kind=kind)

        V = dot(dot(A, meat), A)
        # Small-sample corrections (fixest/boottest semantics)
        df = None
        G_correction = ssc.get("cluster.adj", True)
        t_df = ssc.get("cluster.df", "conventional")

        # K adjustment (FE effective parameter count)
        if ssc.get("adj", False):
            K_fixef = ssc.get("fixef.K", "none")
            fe_dof = int(ssc.get("fe_dof", 0))
            N, Kp = Xd.shape
            K_eff = Kp + (fe_dof if K_fixef in {"full", "nonnested"} else 0)
            if K_eff >= N:
                msg = "K adjustment would lead to non-positive df."
                raise ValueError(msg)
            V *= float(N) / float(N - K_eff)

        if G_correction == "min":
            V *= G_min / (G_min - 1.0) if G_min > 1 else 1.0
        elif G_correction == "conventional":
            pass
        else:
            msg = "G_correction must be 'min' or 'conventional'"
            raise ValueError(msg)

        # Helper: convert codes -> inverse indices
        # ---- Satterthwaite / df handling (reachable scope) ----
        def _codes_to_inv(codes_arr: np.ndarray) -> np.ndarray:
            codes = np.asarray(codes_arr).reshape(-1)
            _, inv_local = np.unique(codes, return_inverse=True)
            return inv_local.astype(np.int64, copy=False)

        if isinstance(t_df, str) and t_df.lower() == "satterthwaite":
            N, Kp = Xd.shape
            # choose a single dimension according to bootcluster
            G_list = [int(np.unique(c).size) for c in clusters_list]
            if isinstance(bootcluster, str):
                if bootcluster == "max":
                    idx = int(np.argmax(G_list))
                elif bootcluster == "min":
                    idx = int(np.argmin(G_list))
                else:
                    msg = "bootcluster must be 'max','min', or an integer index."
                    raise ValueError(msg)
            elif isinstance(bootcluster, int) and 0 <= int(bootcluster) < len(
                clusters_list,
            ):
                idx = int(bootcluster)
            else:
                msg = "bootcluster must be 'max','min', or a valid dimension index."
                raise ValueError(msg)
            inv_for_df = _codes_to_inv(clusters_list[idx])

            # per-cluster contributions under CRV1
            _, contribs = _cluster_meat_contrib(Xw, uw, A, inv_for_df, kind="CRV1")
            AGAdiags = []
            for Cg in contribs:
                Vg = dot(dot(A, Cg), A)
                AGAdiags.append(np.diag(Vg).astype(np.float64))
            AGAdiags = (
                np.vstack(AGAdiags) if AGAdiags else np.zeros((0, Kp), dtype=np.float64)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                sum_v = np.sum(AGAdiags, axis=0)
                sum_v_sq = np.sum(AGAdiags**2, axis=0)
                df_j = 2.0 * (sum_v**2) / sum_v_sq
            df_j = np.where(np.isfinite(df_j) & (df_j > 0.0), df_j, float(N - Kp))
            df = float(np.min(df_j))
            global _LAST_DF_INFO  # noqa: PLW0603
            _LAST_DF_INFO = {
                "method": "satterthwaite",
                "bootcluster": bootcluster,
                "selected_dim": int(idx),
                "selected_G": int(G_list[idx]),
            }
        elif isinstance(t_df, int):
            df = float(t_df)
        elif t_df == "min":
            df = float(G_min - 1)
        elif t_df == "conventional":
            N, Kp = Xd.shape
            df = float(N - Kp)
        else:
            msg = "t_df must be int, 'satterthwaite', 'min', or 'conventional'"
            raise ValueError(msg)

        # vcov_fix
        if vcov_fix:
            evals, evecs = la.eigh(V)
            evals = np.maximum(evals, 0.0)
            V = dot(evecs * evals, evecs.T)
        if return_df:
            return V.astype(np.float64), (
                df if df is not None else float(Xd.shape[0] - Xd.shape[1])
            )
        return V.astype(np.float64)
    # Single-way
    codes = np.asarray(clusters).reshape(-1)
    _, inv = np.unique(codes, return_inverse=True)
    inv = inv.astype(np.int64, copy=False)
    G = int(inv.max()) + 1
    G_min = G  # for unified warning
    p = Xd.shape[1]
    if kind == "CRV1" and G_min < max(p + 1, 10):
        warnings.warn(
            "CRV1 is unreliable with few clusters; consider CRV2/CRV3.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Special-case: Jackknife-by-cluster (CRV3J) exact implementation
    if kind == "CRV3J":
        # Enforce single-way clusters only for CRV3J (jackknife-by-cluster).
        if (
            isinstance(clusters, (list, tuple))
            and len(clusters) > 0
            and isinstance(clusters[0], (list, tuple, np.ndarray))
        ):
            msg = "CRV3J (jackknife-by-cluster) is only supported for single-way clustering."
            raise ValueError(msg)
        # Jackknife-by-cluster (CRV3J) following MNW (2023, JAE).
        # Two equivalent implementations exist:
        #  (A) Leave-one-cluster-out re-estimation: compute beta_{(-g)} for each
        #      cluster g omission and form V = (G-1)/G * sum_g (beta_{(-g)} - mean_beta)**2
        #  (B) Pseudovalue representation (numerically equivalent):
        #      p_g = G * beta_hat - (G - 1) * beta_{(-g)}
        #      V = 1 / (G * (G - 1)) * sum_g (p_g - p_bar) (p_g - p_bar)'
        # We implement (A) directly for clarity and to match MNW notation,
        # but include comments and unit tests should verify equivalence to (B).
        if G <= 1:
            msg = "CRV3J requires at least two clusters."
            raise ValueError(msg)
        k_dim = Xd.shape[1]
        betas_minus = np.zeros((k_dim, G), dtype=np.float64)
        # For exact LOO, we must re-fit beta on the sample with cluster g removed
        if y is None:
            msg = "CRV3J requires the original outcome y for leave-one-cluster-out refits."
            raise ValueError(msg)
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        for g in range(G):
            mask_g = inv == g
            if not np.any(mask_g):
                msg = f"Cluster {g} has no observations."
                raise ValueError(msg)
            mask_minus = ~mask_g
            # Use weighted least-squares re-fit on leave-one-out sample when weights provided
            if weights is None:
                beta_m = la.solve(Xd[mask_minus, :], y_arr[mask_minus, :], method="qr")
            else:
                w_full = _validate_weights(weights, Xd.shape[0]).reshape(-1, 1)
                w_minus = w_full[mask_minus, :]
                sqrt_w = np.sqrt(w_minus)
                Xw_minus = hadamard(Xd[mask_minus, :], sqrt_w)
                yw_minus = hadamard(y_arr[mask_minus, :], sqrt_w)
                beta_m = la.solve(Xw_minus, yw_minus, method="qr")
            betas_minus[:, g] = np.asarray(beta_m).reshape(-1)
        mean_beta_minus = la.col_mean(betas_minus, ignore_nan=False).reshape(-1, 1)
        diffs = betas_minus - mean_beta_minus
        # Jackknife variance estimate (MNW / standard jackknife scaling)
        V = ((G - 1.0) / G) * la.dot(diffs, diffs.T)
    else:
        meat = _cluster_meat(Xw, uw, A, inv, kind=kind)
        V = dot(dot(A, meat), A)
    # Small-sample corrections (fixest/boottest semantics)
    df = None
    if ssc is not None:
        # Normalize ssc keys for compatibility
        ssc = _normalize_ssc(ssc)
        G_correction = ssc.get("cluster.adj", True)
        t_df = ssc.get("cluster.df", "conventional")
        # K adjustment
        if ssc.get("adj", False):
            K_fixef = ssc.get("fixef.K", "none")
            fe_dof = int(ssc.get("fe_dof", 0))
            N, K = Xd.shape
            K_eff = K + (fe_dof if K_fixef in {"full", "nonnested"} else 0)
            if K_eff >= N:
                msg = "K adjustment would lead to non-positive df."
                raise ValueError(msg)
            V *= float(N) / float(N - K_eff)
        if G_correction == "min":
            V *= G / (G - 1.0) if G > 1 else 1.0
        elif G_correction == "conventional":
            pass
        else:
            msg = "G_correction must be 'min' or 'conventional'"
            raise ValueError(msg)
        if isinstance(t_df, int):
            N, K = Xd.shape
            df = t_df
        elif t_df == "min":
            N, K = Xd.shape
            df_min = G - 1
            df = df_min
        elif t_df == "conventional":
            df = N - K  # conventional df
        else:
            msg = "t_df must be int, 'min', or 'conventional'"
            raise ValueError(msg)
    # vcov_fix
    if vcov_fix:
        evals, evecs = la.eigh(V)
        evals = np.maximum(evals, 0.0)
        V = dot(evecs * evals, evecs.T)
    if return_df:
        return V.astype(np.float64), df if df is not None else (
            Xd.shape[0] - Xd.shape[1]
        )
    return V.astype(np.float64)


def parallel_bootstrap_se(
    beta_star: NDArray[np.float64],
    *,
    n_jobs: int | None = None,
    batch_size: int = 1000,
) -> NDArray[np.float64]:
    """Compute bootstrap standard errors in parallel for large B.

    Vectorized and parallelized computation of bootstrap SE from bootstrap
    replicates beta_star (n_boot, K). Uses ThreadPoolExecutor for CPU-bound
    tasks. For B > 10000, consider batching to avoid memory issues.

    Parameters
    ----------
    beta_star : array-like
        Bootstrap replicates of shape (B, K).
    n_jobs : int, optional
        Number of parallel jobs. Defaults to min(cpu_count, 4).
    batch_size : int, optional
        Batch size for processing large B. Defaults to 1000.

    Returns
    -------
    se : ndarray
        Bootstrap standard errors of shape (K,).

    """
    beta_star = np.asarray(beta_star, dtype=np.float64)
    B, K = beta_star.shape
    if n_jobs is None:
        n_jobs = min(multiprocessing.cpu_count(), 4)

    # Vectorized computation for small B
    if batch_size >= B:
        return np.std(beta_star, axis=0, ddof=1)

    # Parallel batch processing for large B
    def _batch_stats(
        batch: NDArray[np.float64],
    ) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
        # Return (n_i, sum_i, sumsq_i) for the batch along axis 0
        n_i = batch.shape[0]
        sum_i = np.sum(batch, axis=0, dtype=np.float64)
        sumsq_i = np.sum(batch * batch, axis=0, dtype=np.float64)
        return int(n_i), sum_i, sumsq_i

    batches = [beta_star[i : i + batch_size] for i in range(0, B, batch_size)]
    stats_batches = []

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_batch_stats, batch) for batch in batches]
        stats_batches = [future.result() for future in as_completed(futures)]

    # Aggregate totals from batch summaries to compute unbiased variance
    total_B = 0
    total_sum = np.zeros(K, dtype=np.float64)
    total_sumsq = np.zeros(K, dtype=np.float64)
    for n_i, sum_i, sumsq_i in stats_batches:
        total_B += n_i
        total_sum += sum_i
        total_sumsq += sumsq_i

    if total_B <= 1:
        return np.zeros(K, dtype=np.float64)

    mean = total_sum / float(total_B)
    # unbiased variance: (Σ x^2 - N * mean^2) / (N - 1)
    var = (total_sumsq - float(total_B) * (mean * mean)) / float(total_B - 1)
    # numerical guard: clip tiny negatives
    var = np.maximum(var, 0.0)
    return np.sqrt(var)


def vectorized_bootstrap_multipliers(
    n: int,
    B: int,
    *,
    dist: str = "rademacher",
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Vectorized generation of bootstrap multipliers.

    Generates (n, B) array of multipliers in a single vectorized operation,
    avoiding loops for better performance. Supports all wild distributions.

    Parameters
    ----------
    n : int
        Number of observations.
    B : int
        Number of bootstrap replicates.
    dist : str, optional
        Distribution: 'rademacher', 'mammen', 'norm', 'webb'. Default 'rademacher'.
    seed : int, optional
        Random seed.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    multipliers : ndarray
        Shape (n, B) multipliers.

    """
    if rng is None:
        rng = np.random.default_rng(seed)

    dist = dist.lower()
    if dist in {"rademacher", "rad"}:
        return rng.choice([-1.0, 1.0], size=(n, B), p=[0.5, 0.5])
    if dist == "mammen":
        # Mammen (1993) two-point support with mean 0 and var 1
        a = (1.0 - np.sqrt(5.0)) / 2.0
        b = (1.0 + np.sqrt(5.0)) / 2.0
        pa = (np.sqrt(5.0) + 1.0) / (2.0 * np.sqrt(5.0))
        pb = 1.0 - pa
        return rng.choice([a, b], size=(n, B), p=[pa, pb])
    if dist in {"norm", "normal", "gaussian"}:
        return rng.normal(0.0, 1.0, size=(n, B))
    if dist == "webb":
        # Webb (2014) canonical six-point: {±sqrt(3/2), ±1, ±1/sqrt(2)} each prob 1/6
        points = np.array(
            [
                -np.sqrt(1.5),
                -1.0,
                -1.0 / np.sqrt(2.0),
                1.0 / np.sqrt(2.0),
                1.0,
                np.sqrt(1.5),
            ],
        )
        probs = np.array([1.0 / 6.0] * 6)
        return rng.choice(points, size=(n, B), p=probs)
    msg = f"Unknown distribution: {dist}"
    raise ValueError(msg)


def sparse_bootstrap_meat(
    X: Matrix,
    u: NDArray[np.float64],
    *,
    cluster: NDArray[np.intp] | None = None,
    weights: NDArray[np.float64] | None = None,
) -> Matrix:
    """Compute bootstrap meat matrix with sparse-aware operations.

    Handles sparse X matrices efficiently using sparse matrix operations
    where possible. For clustered data, uses grouped operations.

    Parameters
    ----------
    X : Matrix
        Design matrix (dense or sparse).
    u : ndarray
        Residuals of shape (n,).
    cluster : ndarray, optional
        Cluster indices of shape (n,).
    weights : ndarray, optional
        Observation weights of shape (n,).

    Returns
    -------
    meat : Matrix
        Meat matrix for bootstrap VCV.

    """
    n, K = X.shape
    u = np.asarray(u, dtype=np.float64).reshape(-1, 1)

    if weights is not None:
        weights = _validate_weights(weights, n).reshape(-1, 1)
        u = hadamard(u, np.sqrt(weights))

    if cluster is not None:
        G = len(np.unique(cluster))
        meat = np.zeros((K, K), dtype=np.float64)
        for g in range(G):
            mask = cluster == g
            if not np.any(mask):
                continue
            X_g = X[mask] if not la._is_sparse(X) else X[mask].toarray()  # noqa: SLF001
            u_g = u[mask]
            if weights is not None:
                w_g = weights[mask]
                X_g = hadamard(X_g, np.sqrt(w_g))
                u_g = hadamard(u_g, np.sqrt(w_g))
            s_g = dot(X_g.T, u_g)
            meat += dot(s_g, s_g.T)
    elif sparse.issparse(X):
        X_sparse = sparse.csr_matrix(X) if not sparse.issparse(X) else X
        Xu = X_sparse.multiply(u)
        meat = to_dense(dot(Xu.T, Xu))
    else:
        Xu = hadamard(X, u)
        meat = dot(Xu.T, Xu)

    return meat


def resample_units_block(df, id_name: str, rng: np.random.Generator):
    raise RuntimeError(
        "Policy violation: pairs/block resampling bootstrap is forbidden in lineareg. "
        "Use wild/multiplier bootstrap or placebo/permutation depending on estimator."
    )


def unit_multipliers(
    df,
    id_name: str,
    rng: np.random.Generator,
    dist: str = "rademacher",
) -> np.ndarray:
    import pandas as pd
    ids = pd.Index(df[id_name].to_numpy())
    uniq = ids.unique()
    n_units = len(uniq)
    dist_lower = dist.lower()
    if dist_lower in {"rademacher", "sign"}:
        draws = rng.choice(np.array([-1.0, 1.0]), size=n_units, replace=True)
    elif dist_lower in {"standard_normal", "normal", "gaussian"}:
        draws = rng.standard_normal(size=n_units)
    elif dist_lower in {"mammen"}:
        s5 = np.sqrt(5.0)
        w1, w2 = -(s5 - 1.0) / 2.0, (s5 + 1.0) / 2.0
        p1, p2 = (s5 + 1.0) / (2.0 * s5), (s5 - 1.0) / (2.0 * s5)
        draws = rng.choice(np.array([w1, w2]), size=n_units, p=np.array([p1, p2]))
    else:
        raise ValueError(f"dist must be 'rademacher', 'standard_normal', or 'mammen', got '{dist}'")
    mp = dict(zip(uniq.to_numpy(), draws))
    return ids.map(mp).to_numpy(dtype=float)


def two_way_demean(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu_i = np.mean(M, axis=1, keepdims=True)
    mu_t = np.mean(M, axis=0, keepdims=True)
    mu = float(np.mean(M))
    Md = M - mu_i - mu_t + mu
    return Md, mu_i, mu_t, mu


def pca_low_rank(Md: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return np.zeros_like(Md)
    U, s, Vt = np.linalg.svd(Md, full_matrices=False)
    r = int(min(r, s.size))
    return (U[:, :r] * s[:r]) @ Vt[:r, :]


def select_r_ic1(Md: np.ndarray, rmax: int = 8) -> int:
    N, T = Md.shape
    rmax = int(min(rmax, min(N, T) - 1))
    if rmax <= 0:
        return 0
    NT = N * T
    pen = (N + T) / NT * np.log(NT / (N + T))
    best_r, best = 0, np.inf
    for r in range(0, rmax + 1):
        Lr = pca_low_rank(Md, r)
        sigma2 = float(np.mean((Md - Lr) ** 2))
        ic = np.log(max(sigma2, 1e-12)) + r * pen
        if ic < best:
            best = ic
            best_r = r
    return int(best_r)


def fit_ife_dgp(
    Y: np.ndarray,
    W: np.ndarray,
    tau_it: np.ndarray,
    *,
    control_mask: np.ndarray,
    r: int | None = None,
    rmax: int = 8,
    ridge: float = 1e-8,
) -> dict:
    Yc = Y[control_mask, :].astype(float)
    Md, _, _, _ = two_way_demean(Yc)
    r_sel = select_r_ic1(Md, rmax=rmax) if r is None else int(r)

    if r_sel > 0:
        U, s, Vt = np.linalg.svd(Md, full_matrices=False)
        F = Vt[:r_sel, :].T
        X_full = np.column_stack([np.ones(Y.shape[1]), F])
    else:
        X_full = np.ones((Y.shape[1], 1))

    delta_t = np.mean(Yc - np.mean(Yc, axis=1, keepdims=True), axis=0)

    Y0_hat = np.zeros_like(Y, dtype=float)
    for i in range(Y.shape[0]):
        untreated = (W[i, :] == 0)
        if not np.any(untreated):
            Y0_hat[i, :] = float(np.mean(Y[i, :]))
            continue
        y_tilde = Y[i, :] - delta_t
        X = X_full[untreated, :]
        y_u = y_tilde[untreated]
        XtX = X.T @ X + ridge * np.eye(X.shape[1])
        beta = np.linalg.solve(XtX, X.T @ y_u)
        Y0_hat[i, :] = (X_full @ beta) + delta_t

    resid = Y - (Y0_hat + tau_it * W)
    return {"Y0_hat": Y0_hat, "resid": resid, "r": r_sel}


def wild_unit_multiplier(rng: np.random.Generator, n: int, dist: str = "rademacher") -> np.ndarray:
    """Generate wild bootstrap multipliers for unit-level resampling.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n : int
        Number of units.
    dist : str, default "rademacher"
        Distribution for multipliers: "rademacher" or "normal"/"gaussian".

    Returns
    -------
    np.ndarray
        Array of multipliers of shape (n,).
    """
    dist = (dist or "rademacher").lower()
    if dist in {"rademacher", "r"}:
        return rng.choice(np.array([-1.0, 1.0]), size=n, replace=True)
    if dist in {"normal", "gaussian"}:
        return rng.standard_normal(size=n)
    raise ValueError(f"Unknown dist: {dist}")
