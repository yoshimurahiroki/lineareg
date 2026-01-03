"""Wild bootstrap and multiplier inference.

This module implements wild and multiplier bootstrap methods for both IID and
clustered errors, supporting multi-way clustering.
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



# Enumeration thresholds
ENUM_THRESH_BOOTTEST: int = 11
ENUM_THRESH_STRICT: int = 12

# Default bootstrap replications
DEFAULT_BOOTSTRAP_ITERATIONS: int = 2000

# Note: Webb enumeration (6-point) is used only when explicitly requested.

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
    """Compute HAC kernel weights (Andrews 1991 family)."""
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
    """Compute Newey-West (1994) bandwidth.

    Uses NW94 formula for Bartlett kernels and delegates to Andrews (1991)
    for Parzen and QS kernels.
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
    """Compute Andrews (1991) optimal bandwidth."""
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
    """Construct a Toeplitz covariance matrix based on rank order."""
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
    """Compute Quadratic-Spectral (QS) kernel (Andrews 1991).

    Uses exact formula with series expansion for small values.
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
    """Compute finite-sample corrected bootstrap quantile (B+1 rule)."""
    arr = np.asarray(t_stats, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("t_stats must contain at least one bootstrap draw")
    B = int(arr.shape[0])
    k = int(np.ceil((B + 1) * float(alpha)))
    k = min(max(k, 1), B)
    quantile_value = np.partition(arr, k - 1)[k - 1]
    return float(quantile_value)


_NORM_IQR = 1.3489795003921634


def robust_sigma_iqr_normal(x, axis=0):
    # Use R type-1 quantile for robustness and centralization in core.linalg
    q75 = la.quantile_type1_r(x, 0.75) if axis == 0 else np.apply_along_axis(lambda v: la.quantile_type1_r(v, 0.75), axis, x)
    q25 = la.quantile_type1_r(x, 0.25) if axis == 0 else np.apply_along_axis(lambda v: la.quantile_type1_r(v, 0.25), axis, x)
    return (q75 - q25) / _NORM_IQR


def uniform_band_did_mboot(theta_hat, theta_star, alpha=0.05, n_eff=None):
    if n_eff is None:
        n_eff = theta_star.shape[0]
    b = np.sqrt(n_eff) * (theta_star - theta_hat.reshape(1, -1))
    bSigma = robust_sigma_iqr_normal(b, axis=0)
    bT = np.max(np.abs(b / bSigma.reshape(1, -1)), axis=1)
    crit = finite_sample_quantile(bT, 1.0 - alpha)
    half = crit * bSigma / np.sqrt(n_eff)
    return theta_hat - half, theta_hat + half, crit, bSigma


def rademacher_enumeration(G: int) -> np.ndarray:
    """Enumerate full Rademacher sign grid {-1, 1}^G."""
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
    """Enumerate full Webb 6-point grid."""
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
        D = np.sqrt(D2)
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
    # Accept common prefixes used in the literature / wrappers.
    # Examples: 'mnw_33j' -> 'MNW33J' -> '33J'
    if v.startswith("MNW"):
        v = v[3:]
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


def _normalize_residual_type(residual_type: str) -> str:
    """Normalize residual type names to canonical internal strings.

    Canonical residual types used internally:
    - "WCR"
    - "WCU"
    - "WCU_score"

    The library historically used mixed conventions (e.g. "WCU_SCORE" in
    estimator code). This helper makes parsing case- and separator-insensitive.
    """
    rt = str(residual_type).strip()
    # unify separators and case
    rt_u = rt.upper().replace("-", "_").replace(" ", "")
    rt_key = rt_u.replace("_", "")

    if rt_key in {"WCR", "RESTRICTED"}:
        return "WCR"
    if rt_key in {"WCU", "UNRESTRICTED"}:
        return "WCU"
    if rt_key in {"WCUSCORE", "WCU_SCORE"}:
        return "WCU_score"
    # Leave unknown strings untouched so the caller can raise a targeted error.
    return rt


def _normalize_ssc(ssc: dict[str, object] | None) -> dict[str, object]:
    """Normalize ssc dictionaries to canonical keys.

    Accepts only {'adj', 'fixef.K', 'cluster.adj', 'cluster.df', 'fe_dof'}.
    Any other keys will raise an error.

    Notes
    -----
    This module uses an explicit, minimal SSC policy:
    - "adj": toggles the (N-1)/(N-K_eff) component.
    - "cluster.adj": toggles the G/(G-1) component.
    - "cluster.df": selects how to map multiway clustering to an effective G.
    - "fe_dof": absorbed fixed-effect dof used for K_eff accounting.
    """
    defaults = {
        "adj": True,
        "fixef.K": "none",
        "cluster.adj": "conventional",
        "cluster.df": "conventional",
        "fe_dof": 0,
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

    _ALLOWED_DISTS = frozenset({
        "rademacher",
        "rad",
        "webb",
        "mammen",
        "normal",
        "gaussian",
        "standard_normal",
        "wgb",
    })

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
        if self.name == "wgb":
            # Wild Gradient Bootstrap (WGB) convention: use Mammen two-point multipliers.
            # We implement this as an alias of the Mammen distribution.
            s5 = np.sqrt(5.0)
            a = (1.0 - s5) / 2.0
            b = (1.0 + s5) / 2.0
            pa = (s5 + 1.0) / (2.0 * s5)
            pb = 1.0 - pa
            return rng.choice(
                np.array([a, b], dtype=np.float64),
                size=(r, c),
                p=np.array([pa, pb], dtype=np.float64),
            ).astype(np.float64, copy=False)
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
    # If the user explicitly requested Webb but the number of clusters is
    # too small to support the six-point design reliably (practical guidance
    # suggests at least G >= 6), fall back to Rademacher and emit a warning.
    if D.name == "webb":
        if G < 6:
            pol = getattr(D, "policy", "strict")
            warnings.warn(
                "Webb multipliers requested with G<6 clusters; falling back to Rademacher.",
                RuntimeWarning,
                stacklevel=2,
            )
            return WildDist("rademacher", policy=pol)
        return D
    # No automatic Webb promotion: the distribution must be selected explicitly
    # by the caller. This keeps behavior transparent and matches the function
    # docstrings in the public bootstrap APIs.
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

    def _encode_cluster_ids(x: Any) -> np.ndarray:
        """Encode cluster identifiers into integer codes without silent loss.

        - Integer/bool arrays are accepted as-is.
        - Float arrays are accepted only if all values are finite and integer-valued.
        - Object/string arrays are factorized deterministically via np.unique.
        - Missing values are forbidden.
        """
        a = np.asarray(x).reshape(-1)
        if a.size == 0:
            raise ValueError("Cluster ids must be non-empty")
        # Missing check (covers object NaN/None, float NaN)
        if np.any(a == None):  # noqa: E711
            raise ValueError("Cluster ids contain missing (None) values")
        if a.dtype.kind in {"f"}:
            if not np.all(np.isfinite(a)):
                raise ValueError("Cluster ids contain non-finite values (NaN/Inf)")
            r = np.rint(a)
            if np.max(np.abs(a - r)) > 0.0:
                raise ValueError(
                    "Cluster ids are float with non-integer values; refusing lossy cast. "
                    "Provide integer-coded cluster ids or string/object labels.",
                )
            return r.astype(np.int64, copy=False)
        if a.dtype.kind in {"i", "u", "b"}:
            return a.astype(np.int64, copy=False)
        # If we have an object array that is numerically representable, treat it
        # like the float case (this catches np.nan missing values too).
        try:
            af = a.astype(np.float64)
        except (TypeError, ValueError):
            af = None
        if af is not None:
            if not np.all(np.isfinite(af)):
                raise ValueError("Cluster ids contain non-finite values (NaN/Inf)")
            r = np.rint(af)
            if np.max(np.abs(af - r)) > 0.0:
                raise ValueError(
                    "Cluster ids are numeric but contain non-integer values; refusing lossy cast. "
                    "Provide integer-coded cluster ids or string/object labels.",
                )
            return r.astype(np.int64, copy=False)
        # object / string / categorical-like
        # Prefer pandas.factorize: robust for mixed object types.
        try:
            import pandas as _pd

            if _pd.isna(a).any():
                raise ValueError("Cluster ids contain missing (NA) values")
            codes, _uniques = _pd.factorize(a, sort=True)
            if np.any(codes < 0):
                raise ValueError("Cluster ids contain missing (NA) values")
            return codes.astype(np.int64, copy=False)
        except ImportError:
            # Fall back if pandas is not installed.
            pass
        # Without pandas, do a strict missing-value check for common sentinels.
        # Note: object arrays may contain float NaNs.
        try:
            if np.any(a == None):  # noqa: E711
                raise ValueError("Cluster ids contain missing (None) values")
        except Exception:  # noqa: BLE001
            pass
        if a.dtype.kind == "O":
            try:
                is_bad = np.frompyfunc(
                    lambda v: isinstance(v, float) and (not np.isfinite(v)), 1, 1,
                )(a).astype(bool)
                if np.any(is_bad):
                    raise ValueError("Cluster ids contain non-finite float values (NaN/Inf)")
            except Exception:  # noqa: BLE001
                pass
        try:
            _labels, inv = np.unique(a.astype(object), return_inverse=True)
        except TypeError as e:
            raise ValueError(
                "Cluster ids contain non-comparable mixed object types; "
                "install pandas or provide integer-coded cluster ids.",
            ) from e
        return inv.astype(np.int64, copy=False)

    clusters_arr = np.asarray(clusters)
    if clusters_arr.ndim == 2:
        clusters_list = [_encode_cluster_ids(clusters_arr[:, j]) for j in range(clusters_arr.shape[1])]
    elif (
        isinstance(clusters, (list, tuple))
        and len(clusters) > 0
        and hasattr(clusters[0], '__len__')
        and not isinstance(clusters[0], str)
    ):
        clusters_list = [_encode_cluster_ids(c) for c in clusters]
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
            if bk in {"twoway", "product"} and len(clusters_list) == 2:
                ids_0 = clusters_list[0]
                ids_1 = clusters_list[1]
                labels_0, inv_0 = np.unique(ids_0, return_inverse=True)
                labels_1, inv_1 = np.unique(ids_1, return_inverse=True)
                G0 = labels_0.size
                G1 = labels_1.size
                D = dist if isinstance(dist, WildDist) else WildDist(str(dist), policy=policy)
                if D.name in {"rademacher", "rad"}:
                    v0 = rng.choice(np.array([-1.0, 1.0]), size=(G0, n_boot))
                    v1 = rng.choice(np.array([-1.0, 1.0]), size=(G1, n_boot))
                elif D.name == "webb":
                    # Webb six-point distribution (Webb 2013, 2023 CJE):
                    # values = {±√(3/2), ±1, ±1/√2} with equal probabilities 1/6 each.
                    weights_w = np.array([
                        -np.sqrt(1.5),  # -√(3/2)
                        -1.0,
                        -1.0 / np.sqrt(2.0),  # -1/√2
                        1.0 / np.sqrt(2.0),   # +1/√2
                        1.0,
                        np.sqrt(1.5),   # +√(3/2)
                    ], dtype=np.float64)
                    probs_w = np.array([1.0/6] * 6, dtype=np.float64)
                    v0 = rng.choice(weights_w, size=(G0, n_boot), p=probs_w)
                    v1 = rng.choice(weights_w, size=(G1, n_boot), p=probs_w)
                elif D.name == "mammen":
                    s5 = np.sqrt(5.0)
                    w1, w2 = -(s5 - 1.0) / 2.0, (s5 + 1.0) / 2.0
                    p1, p2 = (s5 + 1.0) / (2.0 * s5), (s5 - 1.0) / (2.0 * s5)
                    v0 = rng.choice(np.array([w1, w2]), size=(G0, n_boot), p=np.array([p1, p2]))
                    v1 = rng.choice(np.array([w1, w2]), size=(G1, n_boot), p=np.array([p1, p2]))
                else:
                    v0 = rng.standard_normal((G0, n_boot))
                    v1 = rng.standard_normal((G1, n_boot))
                Wobs = v0[inv_0, :] * v1[inv_1, :]
                log = {
                    "mode": "twoway_product",
                    "G0": G0,
                    "G1": G1,
                    "n_boot": n_boot,
                    "dist": D.name,
                    "enumerated": False,
                }
                return Wobs, log
            elif bk in {"max", "largest"}:
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
                msg = "bootcluster must be 'twoway','intersection','max','min','first', or integer index"
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
        ids = _encode_cluster_ids(clusters_arr)
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
                Wg = None
                total = 0

            if Wg is not None:
                if Wg.shape[0] == G and Wg.shape[1] == total:
                    Wg_proc = Wg
                elif Wg.shape[0] == total and Wg.shape[1] == G:
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
        W_prod = Wk if W_prod is None else la.hadamard(W_prod, Wk)

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
        Xw = la.hadamard(X, sqrt_w)
        yw = la.hadamard(y, sqrt_w)
        beta = la.solve(Xw, yw, method=(method if method in {"qr", "svd"} else "qr"))
    yhat = la.dot(X, beta)
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
    Xd_orig = la.to_dense(X)
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
            Xw = la.hadamard(Xd, sqrt_w)
            yw = la.hadamard(y, sqrt_w)
            beta = la.solve(
                Xw, yw, method=(method if method in {"qr", "svd"} else "qr"),
            )
        if keep_mask is not None:
            beta_full = np.zeros((keep_mask.size, 1), dtype=np.float64)
            beta_full[keep_mask, :] = beta
            beta = beta_full
            yhat = la.dot(Xd_orig, beta)
        else:
            yhat = la.dot(Xd, beta)
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
        particular = la.dot(Vt[:rank, :].T, la.dot(Sr_inv, la.dot(U[:, :rank].T, r)))

    X_proj = la.dot(Xd, null_space)  # (n, p - rank)
    y_proj = y - la.dot(Xd, particular)  # (n, 1)

    if weights is None:
        gamma = la.solve(
            X_proj, y_proj, method=(method if method in {"qr", "svd"} else "qr"),
        )
    else:
        sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float64).reshape(-1, 1))
        X_proj_w = la.hadamard(X_proj, sqrt_w)
        y_proj_w = la.hadamard(y_proj, sqrt_w)
        gamma = la.solve(
            X_proj_w, y_proj_w, method=(method if method in {"qr", "svd"} else "qr"),
        )

    beta_R = particular + la.dot(null_space, gamma)

    res = la.norm(la.dot(R_eff, beta_R) - r)
    if not np.isfinite(res) or res > 1e-10 * (
        la.norm(R_eff) * la.norm(beta_R) + la.norm(r) + 1.0
    ):
        msg = "Infeasible linear restrictions: ||R beta - r|| too large."
        raise ValueError(msg)

    if keep_mask is not None:
        beta_full = np.zeros((keep_mask.size, 1), dtype=np.float64)
        beta_full[keep_mask, :] = beta_R
        beta_R = beta_full
        yhat = la.dot(Xd_orig, beta_R)
    else:
        yhat = la.dot(Xd, beta_R)
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



def compute_ssc_correction(
    n: int,
    k: int,
    clusters: Sequence[int] | None = None,
    ssc: dict[str, object] | None = None,
) -> float:
    """Compute the small-sample correction factor for residuals.

    Factor is implemented as:

        sqrt( f_n * f_g ),

    where f_n is an (N-1)/(N-K_eff) adjustment and f_g is a G/(G-1)
    adjustment.

    For multiway clustering, map to an effective G via ssc['cluster.df']:
    - "conventional" (default): use min(G_j) across dimensions.
    - "min" / "smallest": same as conventional.
    - "max" / "largest": use max(G_j) across dimensions.
    - "none": disable the cluster df adjustment (f_g=1).
    """
    ssc = _normalize_ssc(ssc)
    # 1. Finite Sample Adjustment (N-1)/(N-K)
    f_n = 1.0
    if ssc.get("adj", True):
        # Determine K_eff based on fixef policy
        k_fixef = str(ssc.get("fixef.K", "none")).lower()
        fe_dof = int(ssc.get("fe_dof", 0))
        k_eff = float(k)
        if k_fixef == "full":
            k_eff += fe_dof
        elif k_fixef == "nested":
            # For nested/absorb-only, often treat as partial or zero depending on method
            # Here we follow simple policy: add known FE dof if requested
            k_eff += fe_dof

        # Guard against non-positive denom
        if n > k_eff:
            f_n = float(n - 1) / float(n - k_eff)

    # 2. Cluster Adjustment G/(G-1)
    f_g = 1.0
    if clusters is not None:
        g_adj = ssc.get("cluster.adj", True)
        if g_adj and (g_adj != "none"):
            g_df = str(ssc.get("cluster.df", "conventional")).lower()

            is_multiway = (
                isinstance(clusters, (list, tuple))
                and (len(clusters) > 0)
                and isinstance(clusters[0], (list, tuple, np.ndarray))
            )
            if is_multiway:
                gs = [int(len(np.unique(np.asarray(c)))) for c in clusters]
                if g_df in {"conventional", "min", "smallest"}:
                    g_val: int | None = min(gs)
                elif g_df in {"max", "largest"}:
                    g_val = max(gs)
                elif g_df == "none":
                    g_val = None
                else:
                    raise ValueError(
                        "ssc['cluster.df'] must be one of 'conventional','min','max','none' for multiway clustering",
                    )
            else:
                g_val = int(len(np.unique(np.asarray(clusters))))

            if g_val is not None and g_val > 1:
                f_g = float(g_val) / float(g_val - 1)

    return np.sqrt(f_n * f_g)


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
    ssc: dict[str, object] | None = None,
    x_dof: int | None = None, # k (regressors) for SSC
) -> tuple[np.ndarray, np.ndarray | None]:
    """Construct Y* = yhat + u_adj ⊙ W for all replications at once.

    Residuals
    ---------
    - If (yhat,resid) are provided, they are used as-is (subject to SSC scaling if requested).
    - Otherwise, if (y,X) are provided:
        * residual_type="unrestricted" -> WLS residuals
        * residual_type="restricted"   -> restricted WLS with R beta = r

    For WCU (unrestricted), do NOT recenter residuals under the null (MNW/boottest
    semantics). For WCU_score, perform MNW-style score recentering which solves
    M_gg z = u_g for each cluster (see _score_recentering).

        NA handling
        -----------
        - na_action="drop": drop rows with any non-finite value in the constructed
            bootstrap matrix and return the retained-row mask.
    """
    # Initialize RNG if provided for future extensions (kept for API compatibility)
    if rng is None and seed is not None:
        rng = np.random.default_rng(seed)
    _aliases = {"restricted": "WCR", "unrestricted": "WCU"}
    residual_type = _aliases.get(residual_type, residual_type)
    residual_type = _normalize_residual_type(residual_type)
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
        # Infer X dimensionality for SSC if possible, otherwise rely on x_dof argument
        k_eff = 0
        if x_dof is not None:
            k_eff = int(x_dof)
        elif X is not None:
            k_eff = X.shape[1]
    else:
        if y is None or X is None:
            msg = "If (yhat,resid) are not provided, supply (y,X)."
            raise ValueError(msg)
        k_eff = X.shape[1] if x_dof is None else int(x_dof)
        if residual_type == "WCR":
            if null_R is None or null_r is None:
                msg = "restricted residuals require (null_R, null_r)."
                raise ValueError(msg)
            yh, u = _wls_fit_restricted(y, X, null_R, null_r, weights=weights)
        else:
            yh, u = _wls_fit(y, X, weights=weights)

    # Apply SSC scaling to residuals when clustering is used.
    # We apply this BEFORE score recentering so the base residuals are scaled correctly.
    # Policy: for IID wild bootstrap (clusters is None) do not apply any (N-1)/(N-K)
    # scaling implicitly. SSC is a clustered-inference concept here.
    if clusters is not None:
        ssc_norm = _normalize_ssc(ssc)
        n_obs = u.shape[0]
        factor = compute_ssc_correction(n_obs, k_eff, clusters=clusters, ssc=ssc_norm)
        if abs(factor - 1.0) > 1e-9:
            u *= factor

    # Recenter under null (per-cluster): implement MNW-consistent score recentering.
    # - WCU_score: apply M_gg^{-1} to u_g where M_gg = I - H_g (H_g uses global X (weighted if provided)).
    #   The operation is performed by solving M_gg z = u_g (avoid explicit inverse), stable via cholesky or pinv fallback.
    # - WCU: classic unrestricted bootstrap: do NOT recenter residuals (MNW/boottest semantics).
    if residual_type == "WCU_score" and clusters is not None:
        if X is None:
            msg = "WCU_score recentering requires X to be provided (cannot use yhat/resid alone)."
            raise ValueError(msg)
        u = _score_recentering(u, X, clusters, weights=weights)

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
    if not np.isfinite(thb).all():
        bad = np.argwhere(~np.isfinite(thb))
        head = bad[:10].tolist()
        raise ValueError(
            "Non-finite bootstrap draws detected in theta_star (showing up to 10 [k,b] indices): "
            f"{head}. This indicates numerical failure or an upstream bug."
        )
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
        # DID / event-study parity: use R quantile(type=1) for IQR (no interpolation)
        # and normalize by the standard normal IQR.
        iqr_norm = float(_NORM_IQR)
        se = np.zeros((K,), dtype=np.float64)
        for k in range(K):
            q75 = float(la.quantile_type1_r(thb[k, :], 0.75))
            q25 = float(la.quantile_type1_r(thb[k, :], 0.25))
            iqr_val = q75 - q25
            se[k] = (iqr_val / iqr_norm) if iqr_norm > 0 else 0.0
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

    This function is intentionally strict:

    - Requires at least 2 bootstrap draws.
    - Rejects any non-finite (NaN/Inf) values.
    - Uses ddof=1 (unbiased sample standard deviation).

    Parameters
    ----------
    beta_star : (K, B) array
        Bootstrap draws of coefficients/statistics.

    Returns
    -------
    se : (K,) array
        Bootstrap standard errors.

    """
    arr = np.asarray(beta_star, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("beta_star must be a 2-D array of shape (K, B).")
    K, B = arr.shape
    if B < 2:
        raise ValueError(f"bootstrap_se requires at least 2 draws; got B={B}.")
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        head = bad[:10].tolist()
        raise ValueError(
            "Non-finite bootstrap draws detected (showing up to 10 [k,b] indices): "
            f"{head}. This indicates numerical failure or an upstream bug."
        )
    return np.std(arr, axis=1, ddof=1).astype(np.float64)


def uniform_confidence_band(  # noqa: PLR0913
    theta: NDArray[np.float64],
    theta_star: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    studentize: str = "bootstrap",
    zero_se_tol: float = 1e-12,
    zero_se_rel: float = 1e-12,
    context: str | None = None,
    family: Sequence[Sequence[int]] | None = None,
    scale: str = "sd",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sup-t uniform confidence band for a vector of parameters from bootstrap replicates.

    This function implements sup-t uniform confidence bands using bootstrap
    studentization only.
    """
    # Context guard: only allow uniform bands for a restricted set of estimands
    # per project policy (DID / event-study / synthetic_control / RCT). This
    # prevents accidental use of sup-t bands for OLS/IV/GMM/QR where CI
    # construction is disallowed by policy.
    allowed = {"did", "eventstudy", "synthetic_control", "rct"}
    if context is not None and str(context).lower() not in allowed:
        raise ValueError(
            f"uniform_confidence_band context must be one of {sorted(allowed)}; got {context!r}",
        )

    # Enforce bootstrap studentization policy by coercion.
    if studentize != "bootstrap":
        studentize = "bootstrap"
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
        D = np.sqrt(D2)
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
    y_hat = la.dot(X, beta_hat)
    u = y - y_hat
    y_star = y_hat + la.hadamard(u, W)  # (n x B)
    # Precompute a pivoted QR of the (possibly weighted) design once and reuse it
    beta_star = np.empty((beta_hat.shape[0], B), dtype=np.float64)
    if weights is not None:
        w = _validate_weights(weights, X.shape[0]).reshape(-1, 1)
        sqrt_w = np.sqrt(w)
        Xf = la.hadamard(X, sqrt_w)
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
        Yrhs = la.hadamard(Yrhs, sqrt_w)
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
    multiway_ids: Sequence[Sequence[int]] | None = None,
    space_ids: Sequence[int] | None = None,
    time_ids: Sequence[int] | None = None,
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

    # Normalize residual_type after variant handling so callers may pass
    # values like 'WCU_SCORE' and still trigger score recentering.
    residual_type = _normalize_residual_type(residual_type)

    # Enumeration defaults: use boottest convention unless caller overrides
    eff_policy = "boottest" if policy is None else policy
    eff_enum_max_g = 11 if enum_max_g is None else int(enum_max_g)
    eff_enum_mode = enumeration_mode or "boottest"

    n = X.shape[0]
    k = X.shape[1]

    # Keep SSC normalization lazy: for IID wild bootstrap (no clustering)
    # we do not apply any (N-1)/(N-K) scaling implicitly.

    # Build or accept multipliers
    # Block weights unless explicitly allowed (GLS/GMM only)
    if (weights is not None) and (not allow_weights):
        msg = "Weights are forbidden for OLS/IV/QR/IV-QR per project policy. Use GLS/GMM with allow_weights=True."
        raise ValueError(msg)

    # Validate clustering specification
    if (space_ids is None) ^ (time_ids is None):
        raise ValueError("space_ids and time_ids must be provided jointly.")
    if (multiway_ids is not None) and (clusters is not None):
        raise ValueError("Provide either multiway_ids or clusters, not both.")

    # allow callers to override enumeration/policy thresholds; fall back to caller args when provided
    eff_policy = "boottest" if policy is None else policy
    eff_enum_max_g = 11 if enum_max_g is None else int(enum_max_g)
    eff_enum_mode = enumeration_mode or "boottest"

    if multipliers is None:
        if (multiway_ids is not None) or (space_ids is not None and time_ids is not None):
            cluster_list = (
                list(multiway_ids)
                if multiway_ids is not None
                else [space_ids, time_ids]
            )
            W = multiway_multipliers(
                cluster_list,
                n_boot=B,
                dist=dist,
                seed=seed,
                rng=rng,
                enumeration_mode=eff_enum_mode,
                policy=eff_policy,
            )
            log = {
                "enumerated": False,
                "effective_dist": str(dist),
                "effective_B": W.shape[1],
                "method": "multiway",
                "n_dims": int(len(cluster_list)),
            }
        elif clusters is None:
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
        Xw0 = la.hadamard(X, sqrt_w0)
        yw0 = la.hadamard(y, sqrt_w0)
        beta0 = la.solve(Xw0, yw0, method="qr")

    # Build yhat and residuals depending on residual_type
    if residual_type == "WCR":
        yhat, u = _wls_fit_restricted(y, X, R, r, weights=weights, method="svd")
    else:
        yhat = la.dot(X, beta0).reshape(-1, 1)
        u = (y.reshape(-1, 1) - yhat)
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
            u = u.reshape(-1, 1)
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
                Xw = la.hadamard(X, W_sqrt)
                z_w = la.hadamard(z_vec, W_sqrt)
                cross = la.crossprod(Xw, z_w)
                log["score_recentering_norm"] = float(la.norm(cross))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                # best-effort: do not fail the bootstrap because of the diagnostic
                log["score_recentering_norm"] = None

    # Apply Small Sample Corrections (SSC) to residuals before bootstrap resampling.
    # Policy: apply only when clustering is used (cluster/multiway/space-time).
    clusters_for_ssc = (
        clusters
        if clusters is not None
        else (
            list(multiway_ids)
            if multiway_ids is not None
            else (
                [space_ids, time_ids]
                if (space_ids is not None and time_ids is not None)
                else None
            )
        )
    )
    if clusters_for_ssc is not None:
        ssc_norm = _normalize_ssc(ssc)
        ssc_factor = compute_ssc_correction(n, k, clusters_for_ssc, ssc_norm)
        if ssc_factor != 1.0:
            u *= ssc_factor

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
            Xwb = la.hadamard(X, sqrt_wb)
            ybb = la.hadamard(yb, sqrt_wb)
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
        mean_rb = Rb_star.mean(axis=1).item()
        var_rb = ((Rb_star.ravel() - mean_rb) ** 2).sum() / denom
        Rb_cov = np.atleast_2d(np.array(var_rb, dtype=np.float64))
    else:
        # Center the draws and form covariance using core.linalg helpers to
        # ensure consistent dense/sparse behavior and centralized numeric
        # policies. Use denominator chosen above (B-1 or B).
        # We need MEAN ACROSS DRAWS (axis 1) for the q-vector.
        # la.col_mean typically computes mean of columns (axis 0).
        mean_rb = np.mean(Rb_star, axis=1, keepdims=True)
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
    W_stat = la.dot(la.dot(Rb_hat.T, Rb_cov_inv), Rb_hat).item()
    W_star = np.empty(B_actual, dtype=float)
    for b in range(B_actual):
        rb = Rb_star[:, b : b + 1]
        W_star[b] = la.dot(la.dot(rb.T, Rb_cov_inv), rb).item()

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





def _score_recentering(
    u: NDArray[np.float64],
    X: Matrix,
    clusters: Sequence[int],
    *,
    weights: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """MNW-style score recentering for WCU_score (Wild Cluster Score Bootstrap).

    Implements the WCU_score (wild cluster score) recentering used by
    boottest/fwildclusterboot-style procedures.

    For each cluster g, the adjusted residuals enforce cluster score
    orthogonality on the original design:

        X_g' W_g u*_g = 0.

    A direct way to achieve this is the cluster-specific projection
    (avoids any global inverse):

        u*_g = (I - X_g (X_g' W_g X_g)^{-1} X_g' W_g) u_g.

    This imposes the null in the bootstrap DGP in the MNW-style score bootstrap.

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
        Score-recentered residuals (n,).

    """
    if clusters is None:
        msg = "WCU_score recentering requires clusters to be provided."
        raise ValueError(msg)

    clusters_arr = np.asarray(clusters, dtype=np.int64).reshape(-1)
    Xd = la.to_dense(X)
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
        XgTWgXg = la.dot(Xg.T * wg.T, Xg)
        try:
            XgTWgXg_inv = la.pinv(XgTWgXg)
        except Exception:
            XgTWgXg_inv = la.pinv(XgTWgXg + 1e-12 * la.eye(XgTWgXg.shape[0]))
        Hg = la.dot(la.dot(Xg, XgTWgXg_inv), Xg.T * wg.T)
        Mg = la.eye(ng) - Hg
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
    """Recenter IV scores using cluster-specific (Z_g'Z_g)^{-1}.

    For each cluster g: u*_g = (I - Z_g (Z_g'Z_g)^{-1} Z_g') u_g.
    This ensures Z_g' u*_g = 0 for each cluster, imposing the null in the bootstrap DGP.
    When `clusters` is None the operation is global using (Z'Z)^{-1}.
    """
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1, 1)
    Zd = la.to_dense(Z)

    if clusters is None:
        ZtZ = la.crossprod(Zd, Zd)
        try:
            evals, U = la.eigh(ZtZ)
            tol_eig = la.eig_tol(ZtZ) if eig_tol is None else float(eig_tol)
            keep = evals > tol_eig
            if not np.any(keep):
                return u_arr.reshape(-1)
            Ur = U[:, keep]
            inv_vals = 1.0 / evals[keep]
            ZtZ_inv = la.dot(Ur, la.dot(np.diag(inv_vals), Ur.T))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            try:
                ZtZ_inv = la.pinv(ZtZ)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return u_arr.reshape(-1)
        Zt_u = la.dot(Zd.T, u_arr)
        a = la.dot(ZtZ_inv, Zt_u)
        u_new = u_arr - la.dot(Zd, a)
        return u_new.reshape(-1)

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
        ZgTZg = la.crossprod(Zg, Zg)
        try:
            ZgTZg_inv = la.pinv(ZgTZg)
        except Exception:
            ZgTZg_inv = la.pinv(ZgTZg + 1e-12 * la.eye(ZgTZg.shape[0]))
        Hg = la.dot(la.dot(Zg, ZgTZg_inv), Zg.T)
        Mg = la.eye(Hg.shape[0]) - Hg
        u_new[mask, :] = la.dot(Mg, ug)
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
    msg = f"Unknown distribution: {dist}. Allowed: rademacher, mammen, norm/normal, webb"
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
        u = la.hadamard(u, np.sqrt(weights))

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
                X_g = la.hadamard(X_g, np.sqrt(w_g))
                u_g = la.hadamard(u_g, np.sqrt(w_g))
            s_g = la.dot(X_g.T, u_g)
            meat += la.dot(s_g, s_g.T)
    elif sparse.issparse(X):
        X_sparse = sparse.csr_matrix(X) if not sparse.issparse(X) else X
        Xu = X_sparse.multiply(u)
        meat = la.to_dense(la.dot(Xu.T, Xu))
    else:
        Xu = la.hadamard(X, u)
        meat = la.dot(Xu.T, Xu)

    return meat



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
    U, s, Vt = la.svd(Md, full_matrices=False)
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
        U, s, Vt = la.svd(Md, full_matrices=False)
        F = Vt[:r_sel, :].T
        X_full = la.column_stack([np.ones(Y.shape[1]), F])
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
        XtX = la.dot(X.T, X) + ridge * la.eye(X.shape[1])
        beta = la.solve(XtX, la.dot(X.T, y_u), method="qr")
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
