"""Spatial econometrics helpers.

SAR2SLS estimator and Moran's I statistics. All algebra via core.linalg.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from lineareg.core import linalg as la
from lineareg.estimators.base import (
    BaseEstimator,
    BootConfig,
    EstimationResult,
    attach_formula_metadata,
    prepare_formula_environment,
)
from lineareg.estimators.iv import IV2SLS
from lineareg.utils.auto_constant import add_constant
from lineareg.utils.formula import FormulaParser

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
else:
    Sequence = tuple  # type: ignore[assignment]
    NDArray = np.ndarray  # type: ignore[misc,assignment]

__all__ = ["SAR2SLS", "moran_i", "moran_i_panel", "moran_i_permutation"]


def moran_i(residuals: np.ndarray | pd.Series, W: la.Matrix) -> float:
    """Compute global Moran's I.

    I = (n / S0) * (e' W e) / (e' e). Returns NaN if undefined.
    """
    import warnings

    e = np.asarray(residuals, dtype=np.float64).reshape(-1, 1)
    n = int(e.shape[0]) or 0
    if n == 0:
        return float("nan")
    # Defensive shape checks for W without densifying (sparse-safe)
    if not hasattr(W, "shape") or len(W.shape) != 2:
        msg = "W must be a 2D array-like."
        raise ValueError(msg)
    if W.shape[0] != W.shape[1] or W.shape[0] != n:
        msg = "W must be square (n x n) and conformable with residuals length n."
        raise ValueError(msg)
    # enforce numeric dtype to avoid object-dtype surprises (strings/objects)
    # For sparse matrices, check dtype directly without densifying
    try:
        from scipy import sparse as sp
    except ImportError:  # pragma: no cover - optional dependency
        sp = None  # type: ignore[assignment]

    if sp is not None and sp.issparse(W):
        if not np.issubdtype(W.dtype, np.number):
            raise TypeError("W must have a numeric dtype.")
    else:
        try:
            _ = np.asarray(W, dtype=float)
        except TypeError:
            raise
        except Exception as exc:
            raise TypeError("W must be numeric (coercible to float).") from exc
    if hasattr(W, "diagonal"):
        diagW = np.asarray(W.diagonal()).reshape(-1)
    else:
        diagW = np.asarray(np.diag(np.asarray(W))).reshape(-1)
    if np.any(diagW < -1e-12):
        raise ValueError("Spatial weight matrix has negative diagonal entries.")
    one = np.ones((n, 1), dtype=np.float64)
    row_sums = np.asarray(la.dot(W, one)).reshape(-1)
    if np.any(row_sums < -1e-12):
        raise ValueError("Spatial weight matrix has negative row sums.")
    if np.any(np.isclose(row_sums, 0.0)):
        raise ValueError("Spatial weight matrix has zero-sum rows; Moran's I is undefined.")

    We = la.dot(W, e)
    # Extract scalar from 1x1 matrix explicitly to avoid NumPy deprecation
    num_arr = la.crossprod(e, We)
    num = float(num_arr.item()) if hasattr(num_arr, "item") else float(num_arr)
    den_arr = la.crossprod(e, e)
    den = float(den_arr.item()) if hasattr(den_arr, "item") else float(den_arr)
    # S0 = sum of all entries of W without densifying
    # Use row_sums to avoid forming dense W; works for scipy.sparse/numpy/pandas.
    S0 = float(np.sum(row_sums, dtype=np.float64))
    if den == 0.0 or S0 == 0.0:
        return float("nan")
    return (n / S0) * (num / den)


def moran_i_panel(
    residuals: np.ndarray | pd.Series,
    W: la.Matrix,
    id_vals: np.ndarray | pd.Series,
    time_vals: np.ndarray | pd.Series,
) -> float:
    """Panel Moran's I: average of time-period-specific Moran's I statistics.

    Computes I_t for each time period t and returns the time average.
    This is appropriate for panel data where spatial correlation may vary over time.

    Parameters
    ----------
    residuals : array-like, shape (n_obs,)
        Residuals from panel regression (long format: n_obs = N_units x T_periods).
    W : array-like, shape (N_units, N_units)
        Spatial weight matrix (cross-sectional dimension) row-normalized for interpretability.
    id_vals : array-like, shape (n_obs,)
        Unit identifiers for each observation.
    time_vals : array-like, shape (n_obs,)
        Time identifiers for each observation.

    Returns
    -------
    float
        Time-averaged Moran's I statistic.

    Notes
    -----
    - W must be conformable with the cross-sectional dimension (number of unique units).
    - For each period t, computes I_t = (N / S0) * (e_t' W e_t) / (e_t' e_t).
    - Returns the average across all periods: mean(I_t).
    - Returns NaN if any period has insufficient data or W is incompatible.

    References
    ----------
    - Elhorst, J.P. (2014). "Spatial Econometrics: From Cross-Sectional Data to Spatial Panels." Springer Briefs in Regional Science.

    """
    import pandas as pd

    # Convert to arrays
    resid_arr = np.asarray(residuals, dtype=np.float64).reshape(-1)
    id_arr = np.asarray(id_vals).reshape(-1)
    time_arr = np.asarray(time_vals).reshape(-1)

    if len(resid_arr) != len(id_arr) or len(resid_arr) != len(time_arr):
        return float("nan")

    # Create DataFrame for convenient grouping
    df = pd.DataFrame(
        {
            "resid": resid_arr,
            "id": id_arr,
            "time": time_arr,
        },
    )

    # Get unique times and units
    unique_times = sorted(df["time"].unique())
    unique_units = sorted(df["id"].unique())
    n_units = len(unique_units)

    # Validate W dimensions
    if not hasattr(W, "shape") or len(W.shape) != 2:
        return float("nan")
    if W.shape[0] != W.shape[1] or W.shape[0] != n_units:
        return float("nan")

    # Compute Moran's I for each time period
    moran_values = []
    for t in unique_times:
        df_t = df[df["time"] == t].copy()

        # Check if we have all units in this period
        if len(df_t) != n_units:
            continue  # Skip incomplete periods

        # Sort by unit ID to ensure alignment with W
        df_t = df_t.sort_values("id")
        resid_t = df_t["resid"].to_numpy().reshape(-1, 1)

        # Compute Moran's I for this period
        try:
            I_t = moran_i(resid_t, W)
            if np.isfinite(I_t):
                moran_values.append(I_t)
        except (ValueError, TypeError):
            continue
    # Return average Moran's I across periods
    if len(moran_values) == 0:
        return float("nan")

    return float(np.mean(moran_values))


def moran_i_permutation(
    y: NDArray[np.float64],
    W: NDArray[np.float64],
    *,
    B: int = 1999,
    seed: int | None = None,
    alpha: float = 0.05,
) -> dict[str, float | bool | tuple[float, float]]:
    """Permutation inference for Moran's I.

    Returns critical interval (lo, hi) and a boolean `reject` if the observed
    Moran's I lies outside the (1-alpha) central permutation band. Does not
    return p-values per project policy.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size

    I_obs = float(moran_i(y, W))
    stats = np.empty(B, dtype=float)

    idx = np.arange(n)
    for b in range(B):
        perm = rng.permutation(idx)
        stats[b] = float(moran_i(y[perm], W))

    # Use finite-sample B+1 quantile rule from core.linalg for permutation tests
    lo = float(la.finite_sample_quantile_bplus1(stats, float(alpha / 2.0)))
    hi = float(la.finite_sample_quantile_bplus1(stats, float(1.0 - alpha / 2.0)))
    reject = bool(I_obs < lo or I_obs > hi)

    return {
        "bootstrap_quantile": (lo, hi),
        "ci": (lo, hi),
        "reject": reject,
    }


class SAR2SLS(BaseEstimator):
    """Spatial autoregressive (lag) model via 2SLS.

        y = rho * W y + X beta + u

    Inference: wild bootstrap SE only (no CIs).
    """

    def __init__(  # noqa: PLR0913
        self,
        y: np.ndarray | pd.Series,
        X: np.ndarray | pd.DataFrame,
        W: la.Matrix,
        *,
        add_const: bool = True,
        var_names: Sequence | None = None,
        include_W2: bool = True,
        dialect: str = "stata",  # {"r","stata","statsmodels"} — pass through to add_constant (default 'stata' for unified 'cons')
        ids_align: Sequence
        | None = None,  # if provided, must match W ordering strictly
        rank_policy: str = "stata",  # {"stata","r"} unified cutoff via la.rank_from_diag
    ) -> None:
        super().__init__()
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        X_arr = np.asarray(X, dtype=np.float64)
        # Store spatial weight matrix and configuration
        self.W = W
        # R spatialreg::stsls defaults to using W^2 X in addition to W X.
        self.include_W2 = bool(include_W2)

        if add_const:
            X_aug, names_out, const_name = add_constant(
                X_arr, var_names, dialect=dialect,
            )
            self._const_name = const_name
            names = list(names_out)
            # Safety: ensure an intercept is present even if upstream parsing
            # provided a disguised/near-constant column that was dropped.
            if const_name not in names:
                # Inject intercept per dialect positioning contract
                n = X_arr.shape[0]
                ones = np.ones((n, 1), dtype=np.float64)
                if str(dialect).lower() == "stata":
                    X_aug = np.c_[X_aug, ones]
                    names = [*list(names), const_name]
                else:
                    # R/statsmodels: intercept at front
                    X_aug = np.c_[ones, X_aug]
                    names = [const_name, *list(names)]
            # Final guard: verify a true ones column exists; if not, force-add
            if X_aug.size == 0:
                raise RuntimeError("Empty design matrix unexpectedly.")
            n, k = X_aug.shape
            has_intercept = False
            for j in range(k):
                col = np.asarray(X_aug[:, j]).reshape(-1)
                if (
                    np.all(np.isfinite(col))
                    and float(np.max(np.abs(col - 1.0))) <= 1e-12
                ):
                    has_intercept = True
                    break
            if not has_intercept:
                ones = np.ones((n, 1), dtype=np.float64)
                if str(dialect).lower() == "stata":
                    X_aug = np.c_[X_aug, ones]
                    names = [*list(names), const_name]
                else:
                    X_aug = np.c_[ones, X_aug]
                    names = [const_name, *list(names)]
        else:
            X_aug = X_arr
            self._const_name = None
            names = (
                list(var_names)
                if var_names is not None
                else [f"x{i}" for i in range(X_aug.shape[1])]
            )
        if len(names) != X_aug.shape[1]:
            names = [f"x{i}" for i in range(X_aug.shape[1])]
        self._x_names = names

        self.y_orig = y_arr
        self.X_aug = X_aug
        self._n_obs_raw = y_arr.shape[
            0
        ]  # Store as private; BaseEstimator.n_obs is a property
        self._param_names = ["rho", *self._x_names]
        # store ids_align for later strict validation in `fit`
        self._ids_align = None if ids_align is None else np.asarray(ids_align)
        # Rank decision policy for QR-based rank detection ('stata' or 'r')
        self._rank_policy = str(rank_policy).lower()
        if self._rank_policy not in {"stata", "r"}:
            raise ValueError("rank_policy must be one of {'stata','r'}.")

    @classmethod
    def from_formula(  # noqa: PLR0913
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        W,
        id_name: str | None = None,
        time: str | None = None,
        options: str | None = None,
        ids_align: object | None = None,
        include_W2: bool = True,
        W_dict: dict[str, object] | None = None,
        boot: BootConfig | None = None,
    ) -> SAR2SLS:
        """SAR(2SLS) from formula. Structural spatial weight W must be provided explicitly.
        """
        parser = FormulaParser(data, id_name=id_name, t_name=time, W_dict=W_dict)
        parsed = parser.parse(formula, iv=None, options=options)
        _, boot_eff, meta = prepare_formula_environment(
            formula=formula,
            data=data,
            parsed=parsed,
            boot=boot,
            attr_keys={
                "_row_mask_valid": "row_mask_valid",
                "_fe_codes_from_formula": "fe_codes_list",
            },
        )
        if boot_eff is not None:
            meta.attrs.setdefault("_boot_from_formula", boot_eff)
        model = cls(
            parsed["y"],
            parsed["X"],
            W,
            add_const=True,
            var_names=parsed["var_names"],
            include_W2=include_W2,
            ids_align=ids_align,
        )
        attach_formula_metadata(model, meta)
        return model

    def _drop_near_constant_columns(
        self, M: NDArray[np.float64], tol: float | None = None,
    ) -> NDArray[np.bool_]:
        """Return boolean mask of columns to keep using QR-based rank detection.

        Strict behavior: drop only columns that are (numerically) linearly dependent.
        We use pivoted-QR and determine rank from diag(R) via the unified policy
        (mode in {'stata','r'}), matching R/Stata tolerance conventions.

        Parameters
        ----------
        M : (n x p) array
            Candidate matrix whose columns are evaluated for linear dependence.
        tol : Optional[float]
            Backwards-compatible absolute tolerance (not recommended). If
            None, ``la.eig_tol`` is used on the Gram matrix.

        """
        if M.size == 0:
            return np.zeros((0,), dtype=bool)

        # Work on a dense representation for QR/SVD procedures via core.linalg
        Md = la.to_dense(M)
        # Pivoted QR only (R/Stata parity). If QR fails, raise an exception.
        qr_res = la.qr(Md, pivoting=True, mode="economic")
        if len(qr_res) == 3:
            _Q, R, piv = qr_res
        else:
            _Q, R = qr_res
            piv = np.arange(R.shape[1], dtype=np.int64)
        # Determine numerical rank from diag(R) using unified policy
        diagR = np.abs(np.diag(la.to_dense(R))) if R.size else np.array([])
        if tol is not None:
            rank = int(np.sum(diagR > float(tol)))
        else:
            rank = la.rank_from_diag(diagR, Md.shape[1], mode=self._rank_policy)
        if rank <= 0:
            raise ValueError(
                "All candidate instrument columns are (near) zero or collinear; no valid instruments remain.",
            )
        piv = np.asarray(piv, dtype=np.int64)
        keep_cols = np.sort(piv[:rank])

        mask = np.zeros(Md.shape[1], dtype=bool)
        mask[keep_cols] = True
        return mask

    def fit(
        self,
        *,
        boot: BootConfig | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        spatial_coords: np.ndarray | None = None,
        spatial_radius: float | None = None,
        ssc: dict[str, str | int | float | bool] | None = None,
    ) -> EstimationResult:
        """Estimate SAR via 2SLS; return EstimationResult with bootstrap SE.

        Parameters
        ----------
        boot : BootConfig | None
            Bootstrap configuration. If None, default spatial block bootstrap
            is used when spatial_coords is provided.
        spatial_coords : np.ndarray | None
            Spatial coordinates (n x 2) for each observation. Required for
            proper spatial inference via spatial block bootstrap.
        spatial_radius : float | None
            Radius for spatial clustering in coordinate units. Required when
            spatial_coords is provided. Observations within this distance
            form spatial clusters for block bootstrap.
        """
        # Import spatial bootstrap utility
        from lineareg.core import bootstrap as bt

        # PATCH: Strict W validation at fit() entry (R spatialreg::lagsarlm compatibility)
        # Enforce squareness, dimension matching, and optional row-normalization checks
        # to prevent silent misalignment errors that violate Kelejian-Prucha assumptions.
        W0 = self.W
        if not hasattr(W0, "shape") or len(W0.shape) != 2:
            msg = "W must be a 2-dimensional matrix-like object (ndarray/DataFrame/sparse)."
            raise ValueError(msg)
        if W0.shape[0] != W0.shape[1]:
            msg = f"W must be square (n x n); found shape={W0.shape}."
            raise ValueError(msg)
        n_raw = self._n_obs_raw
        if int(W0.shape[0]) != n_raw:
            msg = (
                f"W dimension {W0.shape[0]} must match number of observations {n_raw}."
            )
            raise ValueError(msg)

        # Optional: row-normalization sanity check (tolerant; metadata only)
        # R spatialreg assumes row-standardized W for valid spatial instruments;
        # we detect (not enforce) to populate metadata for diagnostics.
        try:
            one = np.ones((n_raw, 1), dtype=np.float64)
            rsum = np.asarray(la.dot(W0, one)).reshape(-1)
            frac_unit = float(np.mean(np.isclose(rsum, 1.0, rtol=1e-10, atol=1e-12)))
            self._W_is_row_normalized = frac_unit > 0.95
            # Fraction of rows whose sum is (approximately) 1.0; diagnostic for auditors
            self._W_row_sum_fraction_close_to_one = frac_unit
        except (ValueError, TypeError, AttributeError):
            self._W_is_row_normalized = None
            self._W_row_sum_fraction_close_to_one = None
        # --- ① Preprocessing: finite-data mask and aligned W/ID trimming (R/Stata style) ---
        y0 = self.y_orig
        X0 = self.X_aug
        W0 = self.W
        # PATCH-IDS: strict validation of ids_align provided at construction
        if getattr(self, "_ids_align", None) is not None:
            # Require labeled W for deterministic alignment (as in R's listw/spatialreg)
            if not isinstance(W0, pd.DataFrame):
                raise ValueError(
                    "ids_align requires a labeled W (pandas.DataFrame with matching index/columns).",
                )
            ids_arr = np.asarray(self._ids_align)
            if ids_arr.shape[0] != y0.shape[0]:
                msg = "ids_align length must equal number of rows in y/X/W."
                raise ValueError(msg)
            if len(np.unique(ids_arr)) != ids_arr.shape[0]:
                msg = "ids_align must contain unique identifiers (no duplicates)."
                raise ValueError(msg)
        # If W is labeled (DataFrame) but no ids_align is given, ensure uniqueness of labels
        elif isinstance(W0, pd.DataFrame):
            if W0.index.duplicated().any() or W0.columns.duplicated().any():
                raise ValueError(
                    "W has duplicated index/columns; provide unique labels or ids_align to disambiguate.",
                )

        # Create a base finite mask for rows where y and all X columns are finite
        base_mask = np.isfinite(y0).reshape(-1) & np.all(np.isfinite(X0), axis=1)

        # upfront length checks for IDs before any masking: match original sample size
        def _check_len(name, ids):
            if ids is not None and len(np.asarray(ids)) != n_raw:
                raise ValueError(
                    f"{name} length must equal original sample size n={n_raw}.",
                )

        _check_len("cluster_ids", cluster_ids)
        _check_len("space_ids", space_ids)
        _check_len("time_ids", time_ids)
        if spatial_coords is not None:
            spatial_coords_arr = np.asarray(spatial_coords)
            if spatial_coords_arr.ndim != 2 or spatial_coords_arr.shape[0] != n_raw:
                raise ValueError(
                    f"spatial_coords must be (n x 2) array with n={n_raw}; got shape {spatial_coords_arr.shape}."
                )

        if not np.all(base_mask):
            # Subset y and X to the effective sample
            y = y0[base_mask].reshape(-1, 1)
            X = X0[base_mask, :]
            # Subset W to the effective rows/cols WITHOUT densifying.
            if (
                not hasattr(W0, "shape")
                or len(W0.shape) != 2
                or W0.shape[0] != W0.shape[1]
                or W0.shape[0] != y0.shape[0]
            ):
                msg = "W must be square n x n with n equal to original sample size."
                raise ValueError(msg)
            Wbase = W0  # keep original type (sparse/ndarray/DF)
            W = Wbase[np.ix_(base_mask, base_mask)]
            # If ids_align is provided and W has labeled index/columns (DataFrame),
            # enforce exact ordering; if the same set but different order, reorder W.
            if getattr(self, "_ids_align", None) is not None:
                ids_eff = np.asarray(self._ids_align)[base_mask]
                if isinstance(W, pd.DataFrame):
                    idx = np.asarray(W.index)
                    col = np.asarray(W.columns)
                    if set(idx) != set(ids_eff) or set(col) != set(ids_eff):
                        msg = (
                            "W's index/columns must match ids_align (effective sample)."
                        )
                        raise ValueError(msg)
                    # reorder both axes to match ids_eff
                    if not (
                        np.array_equal(idx, ids_eff) and np.array_equal(col, ids_eff)
                    ):
                        W = W.loc[ids_eff, ids_eff]

            # Mask IDs consistently with the same base_mask so bootstrap multipliers align
            def _mask_ids(ids):
                return None if ids is None else np.asarray(ids)[base_mask]

            cluster_ids = _mask_ids(cluster_ids)
            space_ids = _mask_ids(space_ids)
            time_ids = _mask_ids(time_ids)
            if spatial_coords is not None:
                spatial_coords = np.asarray(spatial_coords, dtype=np.float64)[base_mask, :]
            # Enforce paired specification for (space_ids, time_ids)
            if (space_ids is None) ^ (time_ids is None):
                raise ValueError("space_ids and time_ids must be provided jointly.")
            n_eff = int(y.shape[0])
            if getattr(self, "_ids_align", None) is not None:
                # store effective ids consistent with base_mask for later use
                self._ids_effective = np.asarray(self._ids_align)[base_mask]
        else:
            y = y0
            X = X0
            W = W0
            n_eff = int(y.shape[0])
            if getattr(self, "_ids_align", None) is not None:
                self._ids_effective = np.asarray(self._ids_align)

        # ---- STRICT: enforce zero diagonal (no self-loops) for SAR lag model ----
        if hasattr(W, "diagonal"):
            diagW = np.asarray(W.diagonal()).reshape(-1)
        else:
            diagW = np.asarray(np.diag(np.asarray(W))).reshape(-1)
        # allow only tiny jitter: combined absolute+relative tolerance
        if np.any(
            np.abs(diagW)
            > (np.finfo(float).eps * max(1, la.norm(diagW, ord=np.inf)) + 1e-12),
        ):
            msg = "W must have (numerically) zero diagonal (no self-loops) for SAR lag 2SLS (Kelejian-Prucha)."
            raise ValueError(msg)

        Wy = la.dot(W, y)
        # Use la.hstack to remain within core.linalg abstraction (no raw np.hstack)
        X_struct = la.hstack([Wy, X])

        # Build full instrument matrix and perform pivoted-QR / rank selection
        # to deterministically drop only linearly dependent columns. This
        # replaces ad-hoc variance thresholds with a QR/SVD-based rank test.
        # Build instrument blocks explicitly and drop dependence only within WX(,W2X)
        WX = la.dot(W, X)
        instr_blocks: list[la.Matrix] = []
        instr_names_all: list[str] = []
        # PATCH-WCONST: detect constant column in X and drop corresponding W:const / W2:const
        const_j = None
        w_is_row_std = False
        if self._const_name is not None and self._const_name in self._x_names:
            const_j = self._x_names.index(self._const_name)
            one = np.ones((n_eff, 1), dtype=np.float64)
            row_sums = np.asarray(la.dot(W, one)).reshape(-1)
            # Treat 'row-sums ≈ constant c' (including c≠1) as constant-duplicate case.
            # Use a scale-aware relative tolerance to avoid false positives on very
            # large or very small weight matrices. The base scale accounts for the
            # maximum absolute row sum and the mean magnitude.
            rs_mean = float(np.mean(row_sums))
            rs_dev = float(np.max(np.abs(row_sums - rs_mean)))
            base = max(1.0, float(np.max(np.abs(row_sums))), abs(rs_mean))
            tol = 10.0 * np.finfo(float).eps * base  # slightly conservative multiplier
            if rs_dev <= tol:
                w_is_row_std = True  # generalized 'constant-row-sum' flag
        if self.include_W2:
            W2X = la.dot(W, WX)
            # include W2X but drop the constant column if W is row-standardized
            if const_j is None or not w_is_row_std:
                instr_blocks.append(W2X)
                instr_names_all += [f"W2:{nm}" for nm in self._x_names]
            else:
                keep_cols = [j for j in range(W2X.shape[1]) if j != const_j]
                if keep_cols:
                    instr_blocks.append(la.to_dense(W2X)[:, keep_cols])
                    instr_names_all += [f"W2:{self._x_names[j]}" for j in keep_cols]
        # WX block (drop const column if W is row-standardized)
        if const_j is None or not w_is_row_std:
            instr_blocks.append(WX)
            instr_names_all += [f"W:{nm}" for nm in self._x_names]
        else:
            keep_cols = [j for j in range(WX.shape[1]) if j != const_j]
            if keep_cols:
                instr_blocks.append(la.to_dense(WX)[:, keep_cols])
                instr_names_all += [f"W:{self._x_names[j]}" for j in keep_cols]
        Z_nonX_full = (
            la.hstack(instr_blocks)
            if len(instr_blocks) > 1
            else (instr_blocks[0] if len(instr_blocks) == 1 else None)
        )
        if Z_nonX_full is None or (
            hasattr(Z_nonX_full, "shape") and Z_nonX_full.shape[1] == 0
        ):
            msg = "No valid spatial instruments after constant-column elimination; SAR model is not identified."
            raise ValueError(msg)
        # Rank-selection on Z_nonX_full unified via QR/SVD helper to avoid duplicated logic
        Znf = la.to_dense(Z_nonX_full)
        mask_keep = self._drop_near_constant_columns(Znf)
        if not np.any(mask_keep):
            raise ValueError(
                "No valid spatial instruments (WX/W2X) after rank selection; identification fails.",
            )
        Z_nonX = Znf[:, np.where(mask_keep)[0]]
        instr_names = [n for n, keep in zip(instr_names_all, mask_keep) if keep]
        # metadata counts per block
        W_order1 = int(np.sum([1 for name in instr_names if name.startswith("W:")]))
        W_order2 = int(np.sum([1 for name in instr_names if name.startswith("W2:")]))
        # PATCH: Record pre-selection instrument count for weak-IV diagnostics
        W_full_count = len(instr_names_all)
        spatial_instr_meta = {
            "W_order1": W_order1,
            "W_order2": W_order2,
            "W_rank_selected": len(instr_names),  # after QR rank selection
            "W_full_instruments": W_full_count,  # before rank selection
            "W_names_used": instr_names,
            "W_names_all": instr_names_all,
        }
        struct_names = ["Wy", *list(self._x_names)]

        # Build preliminary Z and then perform full-rank column selection by pivoted-QR
        # (R/Stata parity). First, assemble candidate Z and drop exact duplicates.
        Z_pre = la.hstack([la.to_dense(Z_nonX), la.to_dense(X)])
        Z_pre = la.drop_duplicate_cols(la.to_dense(Z_pre))  # cheap duplicate pass first
        # Final rank selection on the full Z using pivoted-QR to choose a deterministic
        # independent column set that will be used both for IV and for projection.
        qr_all = la.qr(Z_pre, pivoting=True, mode="economic")
        if len(qr_all) == 3:
            _Qall, R_all, pall = qr_all
        else:
            _Qall, R_all = qr_all
            pall = np.arange(R_all.shape[1], dtype=np.int64)
        diagR_all = np.abs(np.diag(la.to_dense(R_all))) if R_all.size else np.array([])
        rankZ = int(
            la.rank_from_diag(diagR_all, Z_pre.shape[1], mode=self._rank_policy),
        )
        # Keep the leading pivot columns (deterministic independent set)
        keep_cols_Z = (
            np.asarray(pall[:rankZ], dtype=np.int64)
            if rankZ > 0
            else np.array([], dtype=np.int64)
        )
        if keep_cols_Z.size == 0:
            raise ValueError("Instrument matrix has rank 0 after full QR selection.")
        Z_input = la.to_dense(Z_pre)[:, keep_cols_Z]
        # --- Re-map: figure out which kept columns came from Z_nonX (excluded instruments) ---
        n_nonx = la.to_dense(Z_nonX).shape[1]
        # positions in the KEPT set that originate from the first block (Z_nonX)
        z_excluded_idx_kept = [
            j for j, col in enumerate(keep_cols_Z) if int(col) < n_nonx
        ]
        # Build KEPT instrument names aligned to Z_input
        names_pre = list(instr_names) + list(self._x_names)
        instr_names_kept = [names_pre[int(col)] for col in keep_cols_Z]
        # Sanity check: require enough instruments to identify structural params
        p_struct = int(la.to_dense(X_struct).shape[1])
        if rankZ < p_struct:
            raise ValueError(f"Under-identified: rank(Z)={rankZ} < #params={p_struct}.")
        iv = IV2SLS(
            y,
            X_struct,
            Z_input,
            endog_idx=[0],
            # exclude only the spatial part AMONG THE KEPT COLUMNS
            z_excluded_idx=z_excluded_idx_kept,
            # IMPORTANT: set add_const=True so IV2SLS recognizes and protects
            # the constant column from zero-variance/collinearity screening.
            # The helper add_constant() will normalize an existing ones column
            # and will not duplicate the intercept.
            add_const=True,
            var_names=struct_names,
            instr_names=instr_names_kept,
        )
        # Store rank_policy for internal use in weak IV diagnostics
        iv._rank_policy = self._rank_policy  # noqa: SLF001

        # Build/normalize BootConfig for spatial inference with spatial block bootstrap
        # For spatial estimators, we REQUIRE spatial_coords for proper spatial inference.
        # When provided, use spatial_distance_multipliers to build spatially-clustered
        # bootstrap multipliers. This is the correct approach for spatial correlation.
        if spatial_coords is not None:
            spatial_coords = np.asarray(spatial_coords, dtype=np.float64)
            if spatial_coords.ndim != 2 or spatial_coords.shape[0] != n_eff:
                raise ValueError(
                    f"spatial_coords must be (n x 2) array; got shape {spatial_coords.shape}, expected ({n_eff}, 2)."
                )
            if spatial_radius is None:
                raise ValueError(
                    "spatial_radius is required when spatial_coords is provided."
                )
            # Spatial inference: approximate block structure by clustering units based on
            # a distance radius, then run a standard cluster wild bootstrap on those
            # spatial clusters. (BootConfig does not accept precomputed multipliers.)
            n_boot_target = (
                getattr(boot, "n_boot", bt.DEFAULT_BOOTSTRAP_ITERATIONS)
                if boot
                else bt.DEFAULT_BOOTSTRAP_ITERATIONS
            )
            seed_target = getattr(boot, "seed", None) if boot else None
            policy_target = getattr(boot, "policy", "boottest") if boot else "boottest"

            # We only need the cluster labels; keep this call cheap.
            _W1, _log1, spatial_clusters = bt.spatial_distance_multipliers(
                spatial_coords,
                radius=spatial_radius,
                n_boot=1,
                seed=seed_target,
                policy=policy_target,
            )

            if boot is None:
                boot = BootConfig(
                    n_boot=int(n_boot_target),
                    cluster_ids=spatial_clusters,
                    policy=policy_target,
                    seed=seed_target,
                )
            else:
                boot = dataclasses.replace(
                    boot,
                    n_boot=int(n_boot_target),
                    cluster_ids=spatial_clusters,
                    multiway_ids=None,
                    space_ids=None,
                    time_ids=None,
                )
        elif boot is None:
            # No spatial coords provided: use simple clustering if available
            boot = BootConfig(
                cluster_ids=None if cluster_ids is None else np.asarray(cluster_ids),
                space_ids=None if space_ids is None else np.asarray(space_ids),
                time_ids=None if time_ids is None else np.asarray(time_ids),
                policy="boottest",
                enumeration_mode="boottest",
                dist="wgb",
            )
        else:
            # Respect user-supplied policy/enumeration fields. Only replace missing IDs
            # with the passed-in IDs (masked to arrays). Do not overwrite policy/enumeration.
            b_kwargs = {f: getattr(boot, f) for f in boot.__dataclass_fields__}
            if b_kwargs.get("cluster_ids") is None and cluster_ids is not None:
                b_kwargs["cluster_ids"] = np.asarray(cluster_ids)
            if b_kwargs.get("space_ids") is None and space_ids is not None:
                b_kwargs["space_ids"] = np.asarray(space_ids)
            if b_kwargs.get("time_ids") is None and time_ids is not None:
                b_kwargs["time_ids"] = np.asarray(time_ids)
            boot = dataclasses.replace(boot, **b_kwargs)

        iv_res = iv.fit(boot=boot, ssc=ssc)

        # Retrieve structural coefficients from IV result
        # Use IV param names directly since struct_names may have counted columns
        # differently than IV2SLS (which handles add_const internally)
        beta_vec = iv_res.params.to_numpy().reshape(-1, 1)
        # Rename first parameter to "rho" for SAR interpretation and normalize
        # intercept aliases to a canonical 'cons' so downstream summaries display it.
        param_names_sar = list(iv_res.params.index)
        if param_names_sar:
            param_names_sar[0] = "rho"
        # Normalize intercept naming: {(Intercept), Intercept, _cons, const} -> 'cons'
        for j, nm in enumerate(param_names_sar):
            if isinstance(nm, str) and nm.strip() in {
                "(Intercept)",
                "Intercept",
                "_cons",
                "const",
            }:
                param_names_sar[j] = "cons"
        params = pd.Series(beta_vec.reshape(-1), index=param_names_sar, name="coef")
        # Extract bootstrap SE from IV result
        bootstrap_se_out = None
        if iv_res.se is not None:
            # Reindex to match our SAR parameter names
            try:
                if isinstance(iv_res.se, pd.Series):
                    bs = iv_res.se.copy()
                    # Apply the same intercept-name normalization to the SE index
                    bs.index = [
                        (
                            "cons"
                            if (
                                isinstance(ix, str)
                                and ix.strip()
                                in {"(Intercept)", "Intercept", "_cons", "const"}
                            )
                            else ("rho" if (i == 0) else ix)
                        )
                        for i, ix in enumerate(list(iv_res.params.index))
                    ]
                    # Ensure first entry is labeled 'rho' if lengths match
                    if len(bs.index) == len(param_names_sar):
                        bs.index = param_names_sar
                    bootstrap_se_out = bs
                else:
                    bs_arr = np.asarray(iv_res.se).reshape(-1)
                    bootstrap_se_out = pd.Series(bs_arr, index=param_names_sar)
            except (ValueError, TypeError, AttributeError):
                bootstrap_se_out = None

        # Build Z_used for projection: use the exact Z_input used above.
        Z_used = la.to_dense(Z_input)
        Zd = la.to_dense(Z_used)
        qr_res = la.qr(Zd, pivoting=True, mode="economic")
        if len(qr_res) == 3:
            Qz, Rz, _ = qr_res
        else:
            Qz, Rz = qr_res
        # Numerical rank via pivoted QR diag(R) using the unified project policy
        diagRz = np.abs(np.diag(la.to_dense(Rz))) if Rz.size else np.array([])
        rnk = int(la.rank_from_diag(diagRz, Zd.shape[1], mode=self._rank_policy))
        if rnk == 0:
            msg = "Instrument matrix (used for projection) is rank-deficient (rank=0)."
            raise RuntimeError(msg)
        Qr = Qz[:, :rnk]
        # Project X_struct onto instrument space, using exactly the columns from IV result
        # IV2SLS may have dropped duplicate constant columns internally; match by name, not position
        X_struct_d = la.to_dense(X_struct)
        iv_param_names = list(iv_res.params.index)  # actual columns IV retained
        # struct_names defined above as ["Wy"] + list(self._x_names)
        keep_cols = [i for i, nm in enumerate(struct_names) if nm in iv_param_names]
        if len(keep_cols) != len(iv_param_names):
            msg = f"IV result has {len(iv_param_names)} params, but struct_names has {len(keep_cols)} matches."
            raise RuntimeError(msg)
        X_struct_used = X_struct_d[:, keep_cols]
        X_hat = la.dot(Qr, la.dot(Qr.T, X_struct_used))
        yhat = la.dot(X_hat, beta_vec)
        resid = y - yhat

        mI = moran_i(resid, W)

        self._results = EstimationResult(
            params=params,
            se=bootstrap_se_out,  # Bootstrap SE stored directly in .se
            n_obs=int(getattr(iv_res, "n_obs", n_eff)),
            model_info={
                "Estimator": "SAR-2SLS (spatial lag)",
                "B": iv_res.model_info.get("B", None),
                "MoranI": mI,
                "SpatialInstruments": spatial_instr_meta
                | {"IncludeW2": self.include_W2},
                "W_RowNormalized": self._W_is_row_normalized,  # NEW: validation metadata
                "OverID_df": iv_res.model_info.get("OverID_df", 0),
            },
            extra={
                "yhat": yhat,
                "resid": resid,
                "moran_I": mI,
                "se_source": "wild_bootstrap",  # Bootstrap-based SE per policy
                "first_stage_stats": iv_res.extra.get("first_stage_stats", {}),
                "OverID_stat": iv_res.extra.get("OverID_stat", None),
                "J_stat": iv_res.extra.get("J_stat", None),
                "OverID_label": iv_res.extra.get("OverID_label", None),
                "boot_betas": iv_res.extra.get("boot_betas", None),
                # reproducibility metadata
                "clusters_inference": (
                    np.asarray(cluster_ids) if cluster_ids is not None else None
                ),
                "space_ids_inference": (
                    np.asarray(space_ids) if space_ids is not None else None
                ),
                "time_ids_inference": (
                    np.asarray(time_ids) if time_ids is not None else None
                ),
                "W_shape": tuple(W.shape),
                "include_W2": self.include_W2,
            },
        )
        return self._results
