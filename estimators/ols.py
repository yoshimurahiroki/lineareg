"""Ordinary Least Squares (OLS) estimator.

This module implements OLS with support for high-dimensional fixed effects,
linear constraints, and wild bootstrap inference.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lineareg.core import bootstrap as bt
from lineareg.core import fe as fe_core
from lineareg.core import linalg as la
from lineareg.utils.auto_constant import add_constant
from lineareg.utils.constraints import solve_constrained, solve_constrained_batch
from lineareg.utils.formula import FormulaParser

from .base import (
    BaseEstimator,
    BootConfig,
    EstimationResult,
    attach_formula_metadata,
    prepare_formula_environment,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


ArrayLike = Union[pd.Series, np.ndarray[Any, np.dtype[np.float64]]]
MatrixLike = Union[pd.DataFrame, np.ndarray[Any, np.dtype[np.float64]]]


class OLS(BaseEstimator):
    """Ordinary Least Squares regression.

    Estimates y = Xβ + u with optional fixed effects absorption and
    Inference is via wild bootstrap.
    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        Dependent variable (outcome).
    X : array-like, shape (n, p)
        Independent variables (covariates). Can be numpy array or pandas DataFrame.
    add_const : bool, default=True
        If True, automatically adds a constant term to X. The constant is always placed
        as the LAST column in the design matrix.
    var_names : Sequence[str], optional
        Variable names for X columns. If None and X is a DataFrame, uses X.columns.
        If None and X is array, generates names as ['x0', 'x1', ...].

    Attributes
    ----------
    y_orig : ndarray, shape (n, 1)
        Original outcome variable before any transformations.
    X_orig : ndarray, shape (n, p+1) or (n, p)
        Original design matrix (with constant if add_const=True).
    _var_names : list of str
        Variable names including constant (e.g., ['x1', 'x2', '_cons']).
    _const_name : str or None
        Name of the constant term if added, else None.

    Methods
    -------
    from_formula(formula, data, id=None, time=None, options=None, W_dict=None)
        Create OLS model from Patsy-style formula with lineareg extensions.
    fit(boot=None, fe_codes_list=None, cluster_ids=None, constraints=None, constraint_vals=None)
        Fit the OLS model with optional bootstrap inference.

    Returns
    -------
    EstimationResult
        Object containing:
        - params : pd.Series - Estimated coefficients β̂
        - se : pd.Series - Bootstrap standard errors
        - extra : dict - Additional statistics (R², N, estimator name, etc.)

    Examples
    --------
    Basic OLS regression:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from lineareg.estimators.ols import OLS
    >>> from lineareg.estimators.base import BootConfig
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.random.randn(n, 2)
    >>> y = 1.0 + 2.0 * X[:, 0] + 1.5 * X[:, 1] + np.random.randn(n) * 0.5
    >>>
    >>> # Fit OLS
    >>> model = OLS(y, X, var_names=['x1', 'x2'])
    >>> result = model.fit(boot=BootConfig(n_boot=2000, seed=42))
    >>> print(result.params)
    >>> print(result.se)

    OLS with fixed effects (formula API):

    >>> df = pd.DataFrame({
    ...     'y': y, 'x1': X[:, 0], 'x2': X[:, 1],
    ...     'firm_id': np.repeat(np.arange(20), 10),
    ...     'year': np.tile(np.arange(10), 20)
    ... })
    >>>
    >>> model = OLS.from_formula("y ~ x1 + x2 + fe(firm_id) + fe(year)", df)
    >>> result = model.fit(boot=BootConfig(n_boot=2000))

    OLS with clustered standard errors:

    >>> model = OLS.from_formula("y ~ x1 + x2 + cluster(firm_id)", df)
    >>> result = model.fit(boot=BootConfig(n_boot=2000))

    OLS with linear equality constraints (β_x1 + β_x2 = 5):

    >>> import numpy as np
    >>> R = np.array([[1, 1, 0]])  # [x1, x2, _cons]
    >>> q = np.array([5.0])
    >>> model = OLS.from_formula("y ~ x1 + x2", df)
    >>> result = model.fit(constraints=R, constraint_vals=q)

    Notes
    -----
    - All linear algebra uses QR decomposition (no explicit matrix inversions).
    - Fixed effects are absorbed via Frisch-Waugh-Lovell (within transformation).
    - Singleton observations (groups with only 1 observation) are iteratively removed.
    - Bootstrap distributions use Rademacher or Mammen weights for wild bootstrap.
    - Constant term is ALWAYS the last column in X_orig and var_names.

    References
    ----------
    .. [1] Cameron, A. C., & Trivedi, P. K. (2005). Microeconometrics: Methods and
           Applications. Cambridge University Press.
    .. [2] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data
           (2nd ed.). MIT Press.
    .. [3] Correia, S. (2017). "Linear Models with High-Dimensional Fixed Effects:
           An Efficient and Feasible Estimator." Working Paper.
           (Reference for reghdfe algorithm)
    .. [4] Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). "Bootstrap-Based
           Improvements for Inference with Clustered Errors." Review of Economics and
           Statistics, 90(3), 414-427.

    See Also
    --------
    IV2SLS : Two-stage least squares for instrumental variables
    GLS : Generalized least squares for heteroskedasticity/autocorrelation

    """

    _constraints_from_formula: tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ] | None

    def __init__(
        self,
        y: ArrayLike,
        X: MatrixLike,
        *,
        add_const: bool = True,
        var_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        # Ensure Optional typing is inferred correctly by type checkers.
        self._const_name = None
        self._constraints_from_formula = None
        # Original scale storage (before any FE absorption or row dropping)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if var_names is None and isinstance(X, pd.DataFrame):
            var_names = list(X.columns)
        X_arr = np.asarray(X, dtype=np.float64, order="C")

        # Authoritative column ordering & names via add_constant (single source of truth)
        if add_const:
            X_aug, names_out, const_name = add_constant(X_arr, var_names)
            self._const_name = const_name
            self._var_names = list(names_out)
        else:
            X_aug = X_arr
            self._const_name = None
            if isinstance(X, pd.DataFrame):
                self._var_names = list(X.columns)
            else:
                self._var_names = (
                    list(var_names)
                    if var_names is not None
                    else [f"x{i}" for i in range(X_arr.shape[1])]
                )

        self.y_orig: NDArray[np.float64] = y_arr
        self.X_orig: NDArray[np.float64] = X_aug
        self._n_obs, self._n_features = self.X_orig.shape
        if len(self._var_names) != self._n_features:  # final guard
            self._var_names = [f"x{i}" for i in range(self._n_features)]

    @classmethod
    def from_formula(  # noqa: PLR0913
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        id: str | None = None,  # noqa: A002 - keep API name for compatibility
        time: str | None = None,
        options: str | None = None,
        W_dict: dict[str, object] | None = None,
        boot: BootConfig | None = None,
    ) -> OLS:
        """Build OLS model from formula."""
        parser = FormulaParser(data, id_name=id, t_name=time, W_dict=W_dict)
        parsed = parser.parse(formula, iv=None, options=options)

        # Normalize parser metadata and bootstrap configuration
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
            add_const=bool(parsed.get("include_intercept", True)),
            var_names=parsed["var_names"],
        )

        attach_formula_metadata(model, meta)

        # Build linear equality constraints if specified in formula options
        if parsed.get("constraints_raw"):
            from lineareg.utils.constraints import build_rq_from_string

            const_aliases = tuple(
                [
                    nm
                    for nm in (
                        getattr(model, "_const_name", None),
                        "const",
                        "Intercept",
                        "_cons",
                    )
                    if nm
                ],
            )
            R, q, _lbl = build_rq_from_string(
                parsed["constraints_raw"],
                list(model._var_names),
                const_aliases=const_aliases,
            )
            model._constraints_from_formula = (R, q)
        return model

    # ------------------------------------------------------------------
    def fit(  # noqa: PLR0913
        self,
        *,
        absorb_fe: pd.DataFrame | np.ndarray[Any, np.dtype[np.float64]] | None = None,
        device: str | None = None,
        fe_backend: str = "reghdfe",
        fe_tol: float | None = None,
        fe_max_iter: int | None = None,
        rank_policy: str | None = None,
        constraints: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        constraint_vals: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        cluster_ids: Sequence[Any] | None = None,
        space_ids: Sequence[Any] | None = None,
        time_ids: Sequence[Any] | None = None,
        multiway_ids: Sequence[Sequence[Any]] | None = None,
        residual_type: str = "unrestricted",  # "unrestricted" | "restricted"
        null_R: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        null_r: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        method: str = "qr",
        ssc: dict[str, str | int] | None = None,
        boot: BootConfig | None = None,
        weights: Sequence[float] | None = None,
    ) -> EstimationResult:
        """Fit OLS model.

        Performs estimation with optional fixed effects, clustering, and bootstrap inference.
        """
        # across estimators (message and behavior are centralized).
        self._enforce_weight_policy("ols", weights)

        # SEs for OLS are always WCU (unrestricted) per project policy: if the
        # caller requests a different residual_type, fail fast to avoid silent
        # mismatches between fit-time options and inference semantics.
        # local stats/metadata containers (always reachable)
        dropped_stats = {"na": 0, "fe_id_na": 0, "singleton": 0}
        extra_local: dict[str, Any] = {}

        if str(residual_type).lower() not in {"unrestricted", "wcu"}:
            msg = (
                "For OLS.fit(), bootstrap SEs are always WCU (unrestricted, Stata/R practice). "
                "Use EstimationResult.wald_test() to choose WCR/WCU/WCU_SCORE for tests "
                "(p-values/critical values remain forbidden)."
            )
            raise ValueError(msg)

        absorb_fe = self._absorb_fe_from_formula(absorb_fe)

        # If constructed via from_formula() and a BootConfig was captured,
        # use it as the default unless the caller explicitly overrides.
        if boot is None:
            boot = getattr(self, "_boot_from_formula", None)

        # 1) Row pre-filtering: strict NA removal and zero-weight dropping BEFORE FE absorption
        #    This ensures FE codes and singleton removal are computed on the same sample
        #    that will be used downstream. We do not allow NaN propagation.
        # Optionally enforce device preference across dense LA; public API remains NumPy.
        with self._device_context(device):
            # Base finite mask on original data
            base_mask = np.isfinite(self.y_orig).reshape(-1) & np.all(
                np.isfinite(self.X_orig),
                axis=1,
            )

        # If FE provided, run demean on the pre-filtered sample so FE codes do not include dropped rows
        # Decide numeric rank policy (Stata or R) before FE block so it's
        # available even when no FE are provided. Default to 'stata' for
        # reghdfe parity when unspecified.
        rp_global = (rank_policy or "stata").lower()
        if rp_global not in {"stata", "r"}:
            msg = "rank_policy must be one of {'stata','r',None}."
            raise ValueError(msg)

        if absorb_fe is not None:
            dummy_Z = np.empty((int(np.sum(base_mask)), 0), dtype=np.float64)
            X_sub = self.X_orig[base_mask]
            y_sub = self.y_orig[base_mask]
            # Robustly subset absorb_fe to the pre-filtered rows. Support pandas Series/DataFrame
            # and numpy arrays (1D or 2D). This ensures FE routines see the same sample as X_sub/y_sub.
            try:
                if isinstance(absorb_fe, (pd.DataFrame, pd.Series)):
                    fe_sub = absorb_fe.loc[base_mask]
                else:
                    arr_fe = np.asarray(absorb_fe)
                    fe_sub = (
                        arr_fe[base_mask] if arr_fe.ndim == 1 else arr_fe[base_mask, :]
                    )
            except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                arr_fe = np.asarray(absorb_fe)
                fe_sub = arr_fe[base_mask] if arr_fe.ndim == 1 else arr_fe[base_mask, :]

            # Select FE backend and apply canonical defaults for strict parity
            be = str(fe_backend).lower()
            if be not in {"reghdfe", "fixest"}:
                msg = "fe_backend must be one of {'reghdfe','fixest'}."
                raise ValueError(msg)
            # Defaults: reghdfe -> tol=1e-8, max_iter=16000; fixest -> tol=1e-6, max_iter=10000
            # Align fixest defaults with public documentation: fixef.tol=1e-6, fixef.iter=10000
            fe_tol_eff = (
                (1e-8 if be == "reghdfe" else 1e-6) if fe_tol is None else float(fe_tol)
            )
            fe_iter_eff = (
                (16000 if be == "reghdfe" else 10000)
                if fe_max_iter is None
                else int(fe_max_iter)
            )

            # Determine rank policy: if unspecified, follow FE backend
            rp = (rank_policy or ("stata" if be == "reghdfe" else "r")).lower()
            if rp not in {"stata", "r"}:
                msg = "rank_policy must be one of {'stata','r',None}."
                raise ValueError(msg)

            # Call demean_xyz WITHOUT weights (OLS policy: no observation weighting)
            X_proc, _dummy_Z_proc, y_proc, mask_sub, dropped_stats = fe_core.demean_xyz(
                X_sub,
                dummy_Z,
                y_sub,
                fe_sub,
                weights=None,  # OLS forbids weights
                na_action="drop",
                drop_na_fe_ids=True,
                drop_singletons=True,
                return_mask=True,
                return_dropped_stats=True,
                backend=be,
                tol=fe_tol_eff,
                max_iter=fe_iter_eff,
            )
            # mask_sub indexes into X_sub; build global mask
            mask = np.zeros(self._n_obs, dtype=bool)
            orig_idx = np.nonzero(base_mask)[0]
            mask[orig_idx[mask_sub]] = True
            # Compute FE degrees-of-freedom for small-sample corrections (fe_dof)
            if isinstance(absorb_fe, pd.DataFrame):
                fe_ids_masked = [
                    absorb_fe[col].to_numpy()[mask] for col in absorb_fe.columns
                ]
            elif isinstance(absorb_fe, pd.Series):
                fe_ids_masked = absorb_fe.to_numpy()[mask]
            else:
                arr_fe = np.asarray(absorb_fe)
                if arr_fe.ndim == 1:
                    fe_ids_masked = arr_fe[mask]
                else:
                    fe_ids_masked = [arr_fe[:, j][mask] for j in range(arr_fe.shape[1])]
            # Compute FE degrees-of-freedom with explicit intercept inclusion
            fe_dof_info: dict[str, Any] | None = fe_core.compute_fe_dof(
                fe_ids_masked,
                include_intercept=(self._const_name is not None),
            )
        else:
            # No FE: apply base_mask directly and record NA drops
            mask = base_mask
            X_proc = self.X_orig[mask]
            y_proc = self.y_orig[mask]
            fe_dof_info = {}
            dropped_stats["na"] = int(self._n_obs - int(np.sum(mask)))
            # Adopt global rank policy when no FE are present
            rp = rp_global

        boot, cluster_spec = self._coerce_bootstrap(
            boot=boot,
            n_obs_original=self._n_obs,
            row_mask=mask,
            cluster_ids=cluster_ids
            if cluster_ids is not None
            else getattr(self, "_cluster_ids_from_formula", None),
            space_ids=space_ids,
            time_ids=time_ids,
            multiway_ids=multiway_ids,
        )
        cluster_ids_proc = cluster_spec["cluster_ids"]
        space_ids_proc = cluster_spec["space_ids"]
        time_ids_proc = cluster_spec["time_ids"]
        multiway_ids_proc = cluster_spec["multiway_ids"]
        # After FE absorption, perform a single Stata-consistent collinearity screen (QRCP)
        # and remap constraints/null_R accordingly. This ensures deterministic Stata-style
        # rank decisions and avoids mixing rank rules. The screening is applied whether
        # or not FE absorption was performed.

        # --- Column keep by rank policy (Stata vs R) ---
        # Centralized rank policy tolerances (keep in sync with constraints.py)
        _STATA_ETA_SCALE = 1e-13
        _R_TOL_SCALE = 1e-7

        def _keep_columns_stata(
            A: np.ndarray[Any, np.dtype[np.float64]],
        ) -> np.ndarray[Any, np.dtype[np.bool_]]:
            _Qm, Rm, piv = la.qr(A, mode="economic", pivoting=True)
            Rm_dense = la.to_dense(Rm) if getattr(Rm, "size", None) else np.array([])
            diagR = np.abs(np.diag(Rm_dense)) if Rm_dense.size else np.array([])
            if diagR.size == 0:
                return np.zeros(A.shape[1], dtype=bool)
            # Mata qrsolve cutoff: eta = 1e-13 * trace(|R|) / rows(R)
            eta = _STATA_ETA_SCALE * float(np.sum(diagR)) / float(diagR.size)
            rkeep = int(np.sum(diagR > eta))
            keep = np.zeros(A.shape[1], dtype=bool)
            if rkeep > 0:
                keep[np.asarray(piv[:rkeep], dtype=int)] = True
            return keep

        def _keep_columns_r(
            A: np.ndarray[Any, np.dtype[np.float64]],
        ) -> np.ndarray[Any, np.dtype[np.bool_]]:
            _Qm, Rm, piv = la.qr(A, mode="economic", pivoting=True)
            Rm_dense = la.to_dense(Rm) if getattr(Rm, "size", None) else np.array([])
            diagR = np.abs(np.diag(Rm_dense)) if Rm_dense.size else np.array([])
            if diagR.size == 0:
                return np.zeros(A.shape[1], dtype=bool)
            # R lm.fit cutoff: tol = 1e-7 * max|diag(R)|
            tol = _R_TOL_SCALE * float(np.max(diagR))
            rkeep = int(np.sum(diagR > tol))
            keep = np.zeros(A.shape[1], dtype=bool)
            if rkeep > 0:
                keep[np.asarray(piv[:rkeep], dtype=int)] = True
            return keep

        keep_cols = (_keep_columns_stata if rp == "stata" else _keep_columns_r)(X_proc)
        if int(np.sum(keep_cols)) == 0:
            msg = "All regressors dropped by collinearity screen (rank=0). Check FE specification and multicollinearity."
            raise RuntimeError(msg)
        if not np.all(keep_cols):
            dropped_vars = [
                self._var_names[j] for j in range(len(keep_cols)) if not keep_cols[j]
            ]
            diag = extra_local.setdefault("diagnostics", {})
            if not isinstance(diag, dict):
                diag = {}
                extra_local["diagnostics"] = diag
            diag["dropped_collinear"] = list(dropped_vars)
            diag["keep_mask"] = keep_cols.copy()
        X_proc = X_proc[:, keep_cols]
        self._var_names = [nm for (nm, k) in zip(self._var_names, keep_cols) if k]
        if (self._const_name is not None) and (self._const_name not in self._var_names):
            if absorb_fe is not None:
                try:
                    if isinstance(absorb_fe, pd.DataFrame):
                        fe_ids_masked = [
                            absorb_fe[col].to_numpy()[mask] for col in absorb_fe.columns
                        ]
                    elif isinstance(absorb_fe, pd.Series):
                        fe_ids_masked = absorb_fe.to_numpy()[mask]
                    else:
                        arr_fe = np.asarray(absorb_fe)
                        if arr_fe.ndim == 1:
                            fe_ids_masked = arr_fe[mask]
                        else:
                            fe_ids_masked = [
                                arr_fe[:, j][mask] for j in range(arr_fe.shape[1])
                            ]
                    fe_dof_info = fe_core.compute_fe_dof(
                        fe_ids_masked,
                        include_intercept=False,
                    )
                except (TypeError, ValueError, AttributeError, RuntimeError):
                    fe_dof_info = None
            self._const_name = None

        # --- Remap linear equality constraints to surviving columns ---
        def _remap_matrix(
            M: np.ndarray[Any, np.dtype[np.float64]] | None,
            keep: np.ndarray[Any, np.dtype[np.bool_]],
        ) -> np.ndarray[Any, np.dtype[np.float64]] | None:
            if M is None:
                return None
            M = np.asarray(M, dtype=np.float64)
            if M.ndim != 2 or M.shape[1] != keep.size:
                msg = (
                    "Constraint/null_R must have width equal to X's column count BEFORE applying the keep mask "
                    "(the collinearity screen re-maps rows afterward)."
                )
                raise ValueError(
                    msg,
                )
            M_red = M[:, keep]
            nonzero = np.any(np.abs(M_red) > 0, axis=1)
            return M_red[nonzero, :] if M_red.size else None

        # Auto-inject formula-derived constraints if user did not provide any
        cf = self._constraints_from_formula
        if constraints is None and cf is not None:
            constraints, constraint_vals = cf

        # Strict contract: equality constraints must include RHS values.
        if (constraints is not None) and (constraint_vals is None):
            raise ValueError(
                "constraint_vals must be provided when constraints (R) are supplied.",
            )
        if (constraints is None) and (constraint_vals is not None):
            raise ValueError(
                "constraints (R) must be provided when constraint_vals (q) are supplied.",
            )

        if constraints is not None:
            C = _remap_matrix(constraints, keep_cols)
            if C is not None:
                r_vec = np.asarray(constraint_vals, dtype=np.float64).reshape(-1)
                if r_vec.shape[0] != np.asarray(constraints).shape[0]:
                    msg = "constraint_vals length must match number of constraint rows."
                    raise ValueError(msg)
                # infeasible 0 = nonzero check after remap
                nonzero_row = np.any(
                    np.abs(np.asarray(constraints, dtype=np.float64)[:, keep_cols]) > 0,
                    axis=1,
                )
                bad = (~nonzero_row) & (np.abs(r_vec) > 0)
                if np.any(bad):
                    msg = "A constraint became infeasible after collinearity screen (0 = nonzero RHS)."
                    raise ValueError(msg)
                constraints = C if (C is not None and C.size) else None
                constraint_vals = (
                    r_vec[nonzero_row].reshape(-1, 1)
                    if (constraints is not None)
                    else None
                )
            else:
                constraints = C if (C is not None and C.size) else None

        # --- Remap fit-time null hypothesis restrictions (if provided) ---
        # These are not used for SE construction (OLS uses WCU draws), but we store
        # them for reproducible downstream Wald/bootstrap calls. They must match
        # the FINAL parameter vector after collinearity screening.
        if null_R is not None:
            null_R_arr = np.asarray(null_R, dtype=np.float64)
            if null_R_arr.ndim != 2 or null_R_arr.shape[1] != keep_cols.size:
                raise ValueError(
                    "null_R must have width equal to X's column count BEFORE applying the keep mask.",
                )
            m0 = int(null_R_arr.shape[0])
            if null_r is None:
                null_r_vec = np.zeros((m0,), dtype=np.float64)
            else:
                null_r_vec = np.asarray(null_r, dtype=np.float64).reshape(-1)
                if null_r_vec.size == 1 and m0 > 1:
                    null_r_vec = np.repeat(null_r_vec, m0)
                if null_r_vec.shape[0] != m0:
                    raise ValueError("null_r length must match number of rows in null_R.")
            nonzero_row0 = np.any(np.abs(null_R_arr[:, keep_cols]) > 0, axis=1)
            bad0 = (~nonzero_row0) & (np.abs(null_r_vec) > 0)
            if np.any(bad0):
                raise ValueError(
                    "A null restriction became infeasible after collinearity screen (0 = nonzero RHS).",
                )
            null_R_red = null_R_arr[:, keep_cols]
            null_R_red = null_R_red[nonzero_row0, :] if null_R_red.size else None
            null_R = null_R_red if (null_R_red is not None and null_R_red.size) else None
            null_r = (
                null_r_vec[nonzero_row0].reshape(-1, 1)
                if (null_R is not None)
                else None
            )

        # Analytic (diagonal) observation weights are not supported for OLS.
        # Use unweighted QR/SVD solves routed through core.linalg for point estimates.
        # Enforce QR-only solver for OLS (project policy: prefer QR stability,
        # disallow SVD branch for determinism/parity). Ignore caller `method`.
        # Solve for coefficients under the requested device preference
        with self._device_context(device):
            beta_hat = self._solve_ols(
                X_proc,
                y_proc,
                constraints=constraints,
                constraint_vals=constraint_vals,
                method="qr",
                _rank_policy_internal=rp,
            )

        with self._device_context(device):
            yhat_proc = la.dot(X_proc, beta_hat)
        resid_proc = y_proc - yhat_proc

        # Defensive shape validation: ensure residuals are column vectors
        if resid_proc.ndim != 2 or resid_proc.shape[1] != 1:
            msg = (
                f"Internal error: resid_proc should have shape (n, 1) but has shape {resid_proc.shape}. "
                f"y_proc shape: {y_proc.shape}, yhat_proc shape: {yhat_proc.shape}, beta_hat shape: {beta_hat.shape}"
            )
            raise RuntimeError(msg)

        # Analytic VCV is forbidden by project policy: do not compute or store.
        # All inference is performed via multiplier bootstrap only.

        ssc_local: dict[str, object] = (
            {} if ssc is None else {str(k): v for k, v in dict(ssc).items()}
        )
        if (
            ("fe_dof" not in ssc_local)
            and isinstance(fe_dof_info, dict)
            and ("fe_dof" in fe_dof_info)
        ):
            ssc_local["fe_dof"] = int(fe_dof_info["fe_dof"])

        # Bootstrap SEs for inference (analytic VCV completely forbidden)
        Wmult, boot_log = self._bootstrap_multipliers(X_proc.shape[0], boot=boot)
        # For SE construction we always use unrestricted bootstrap draws (WCU semantics).
        # Null hypotheses (null_R/null_r) are handled by Wald test routines which
        # will re-use the estimator's recorded BootConfig to replay enumeration.
        Ystar, _ = bt.apply_wild_bootstrap(
            yhat_proc,
            resid_proc,
            Wmult.to_numpy(),
            residual_type="WCU",
            null_R=None,
            null_r=None,
            ssc=ssc_local,
            x_dof=X_proc.shape[1],
            clusters=cluster_spec["clusters_inference"],
        )
        # Solve bootstrap systems WITHOUT weights (OLS policy: unweighted estimation)
        if constraints is None or constraint_vals is None:
            # Unconstrained: use standard solve for efficiency
            with self._device_context(device):
                try:
                    boot_betas = la.solve(
                        X_proc,
                        Ystar,
                        method="qr",
                        rank_policy=("r" if rp == "r" else "stata"),
                    )
                except TypeError:
                    boot_betas = la.solve(
                        X_proc,
                        Ystar,
                        method="qr",
                    )
        else:
            # Constrained estimation: attempt a vectorized batch solve to avoid Python loops.
            B_actual = Ystar.shape[1]
            try:
                # Prepare q_batch: repeat constraint_vals across bootstrap replicates if provided
                if constraint_vals is None:
                    q_batch = None
                else:
                    q_arr = np.asarray(constraint_vals, dtype=np.float64).reshape(-1, 1)
                    q_batch = np.repeat(q_arr, B_actual, axis=1)
                betas = solve_constrained_batch(X_proc, Ystar, constraints, q_batch)
            except (TypeError, ValueError, RuntimeError):
                # Fallback to safe per-rep constrained solve if batch path fails
                betas = np.empty((X_proc.shape[1], B_actual), dtype=np.float64)
                for b in range(B_actual):
                    betas[:, b : b + 1] = solve_constrained(
                        X_proc,
                        Ystar[:, b : b + 1],
                        constraints,
                        constraint_vals,
                    )
            boot_betas = betas

        # Compute bootstrap SE and store in result.se (not in extra)
        se = pd.Series(
            bt.bootstrap_se(boot_betas).reshape(-1),
            index=self._var_names,
            name="se",
        )

        # Preserve exact clustering scheme for reproducible Wald bootstrap
        clusters_inference_obj = cluster_spec["clusters_inference"]

        extra = {
            "X_inference": X_proc,
            "y_inference": y_proc,
            "u_inference": resid_proc,
            "weights_inference": None,  # OLS forbids weights
            "clusters_inference": clusters_inference_obj,
            "multiway_ids_inference": (
                clusters_inference_obj if multiway_ids_proc is not None else None
            ),
            "space_ids_inference": (
                np.asarray(space_ids_proc) if space_ids_proc is not None else None
            ),
            "time_ids_inference": (
                np.asarray(time_ids_proc) if time_ids_proc is not None else None
            ),
            "beta0_inference": beta_hat,
            "yhat": yhat_proc,
            "boot_betas": boot_betas,
            "W_multipliers_inference": Wmult,
            "multipliers_log": boot_log,
            # Record a nominal variant tag for diagnostics only (does not control computation):
            # '33' when any clustering is used, else '11'. Actual small-G rules, enumeration,
            # Webb promotion, and WCR/WCU/WCU_SCORE selection are decided centrally by bootstrap code.
            "boot_variant_used": (
                "33" if (clusters_inference_obj is not None) else "11"
            ),
            # Record actual residual semantics used for SE construction (WCU by design)
            "boot_residual_default": "WCU",
            "boot_policy_used": getattr(boot, "policy", None),
            # Record numeric rank policy used for reproducibility ("stata" or "R")
            "rank_policy_used": ("r" if rp == "r" else "stata"),
            # Save the BootConfig used at fit-time so Wald/test replay is exact
            "boot_config": boot,
            # If the caller provided null hypothesis restrictions at fit time,
            # preserve them as defaults for later Wald/bootstrap calls.
            "null_R_default": (null_R if null_R is not None else None),
            "null_r_default": (null_r if null_r is not None else None),
            # Analytic VCV is intentionally not available by design
            "vcov_matrix": None,
            "se_vcov": None,
            "se_source": "bootstrap",  # enforce: SE are bootstrap-based (analytic SE forbidden)
            "ssc_effective": ssc_local,
        }
        # Record final design-matrix metadata for reproducibility/debuggability
        extra.setdefault("design_info", {})
        extra["design_info"].update(
            {
                "var_names_final": list(self._var_names),
                "n_features_final": int(X_proc.shape[1]),
                "constant_name": (
                    self._const_name if self._const_name is not None else None
                ),
                "constant_position": (
                    "last"
                    if (self._const_name is not None
                        and self._var_names
                        and self._const_name == self._var_names[-1])
                    else None
                ),
            },
        )
        # Merge any local extra metadata (e.g., dropped collinear variables)
        if isinstance(extra_local, dict) and extra_local:
            extra.update(extra_local)

        # Compute effective number of free parameters (= k - rank(R)) if constraints provided
        rank_R_effective = 0
        if constraints is not None and np.asarray(constraints).size:
            R_arr = np.asarray(constraints, dtype=float, order="C")
            qr_res = la.qr(R_arr.T, pivoting=True, mode="economic")
            # qr_res expected (Q, R, P); guard shape
            R_up = qr_res[1] if len(qr_res) >= 2 else None
            R_up_d = la.to_dense(R_up) if getattr(R_up, "size", None) else np.array([])
            diagR = np.abs(np.diag(R_up_d)) if R_up_d.size else np.array([])
            tol = (1e-10 * float(np.max(diagR))) if diagR.size else 0.0
            rank_R_effective = int(np.sum(diagR > tol))

        self._results = EstimationResult(
            params=pd.Series(beta_hat.flatten(), index=self._var_names),
            se=se,  # Bootstrap SE stored directly in .se
            bands=None,
            n_obs=X_proc.shape[0],
            model_info={
                "Estimator": "OLS",
                # Report that analytic vcov is unavailable; inference is bootstrap-only
                "vcov": "none",
                "ssc": ssc,
                "boot_config": boot,
                # Record effective bootstrap reps and provenance for summary display
                "B": int(getattr(Wmult, "shape", (0, 0))[1])
                if "Wmult" in locals()
                else (int(getattr(boot, "n_boot", 0))),
                "SE_Origin": "bootstrap",
                "dropped_stats": dropped_stats,

                "NoAnalyticSE": True,
                "n_params": int(X_proc.shape[1]),
                # Effective number of free parameters = k - rank(R) when constraints provided
                "n_params_effective": int(X_proc.shape[1] - rank_R_effective),
            },
            extra=extra,
        )
        return self._results

    def _solve_ols(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        constraints: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        constraint_vals: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        method: str = "qr",  # noqa: ARG002 - retained for API compatibility
        _rank_policy_internal: str = "stata",
    ) -> NDArray[np.float64]:
        """Solve least squares with optional linear constraints (no analytic weights).

        Constrained problems are delegated to `solve_constrained`. For unconstrained
        problems, OLS enforces a QR-only solve for determinism and parity with
        project policy (SVD branch is disallowed).
        """
        if constraints is not None and constraint_vals is not None:
            beta = solve_constrained(X, y, constraints, constraint_vals)
            return np.asarray(beta, dtype=np.float64).reshape(-1, 1)
        # Enforce QR-only solver for OLS regardless of caller-supplied `method`.
        try:
            beta = la.solve(
                X,
                y,
                method="qr",
                rank_policy=("r" if _rank_policy_internal.lower() == "r" else "stata"),
            )
        except TypeError:
            # Older linalg implementations may not accept rank_policy; fall back.
            beta = la.solve(X, y, method="qr")
        if beta.ndim == 1:
            beta = beta.reshape(-1, 1)
        return beta


# Module-level convenience function removed: use the OLS class directly.
