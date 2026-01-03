"""Generalized Method of Moments (GMM) estimator.

This module implements one-step and two-step efficient GMM estimators for linear models
with high-dimensional fixed effects and supports custom weighting matrices.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

try:
    import scipy.linalg as sla
except ImportError:
    sla = None
from lineareg.core.bootstrap import compute_ssc_correction, _normalize_ssc
from lineareg.core import bootstrap as bt
from lineareg.core import fe as fe_core
from lineareg.core import linalg as la
import warnings
from lineareg.estimators.base import (
    BaseEstimator,
    BootConfig,
    EstimationResult,
    attach_formula_metadata,
    prepare_formula_environment,
)
from lineareg.estimators.iv import _cd_kp_stats
from lineareg.utils.auto_constant import add_constant
from lineareg.utils.constraints import build_rq_from_string, solve_constrained
from lineareg.utils.formula import FormulaParser

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

ArrayLike = pd.Series | np.ndarray
MatrixLike = pd.DataFrame | np.ndarray

__all__ = ["GMM"]


def _moment_covariance_zu(  # noqa: PLR0913
    Z: np.ndarray,
    u: np.ndarray,
    *,
    n_eff: int,
    cluster_ids=None,
    multiway_ids=None,
    space_ids=None,
    time_ids=None,
    obs_weights: np.ndarray | None = None,
    adj: bool = True,
    n_features: int | None = None,
    fixefK: str = "nested",
    cluster_df: str = "min",
    fe_count: int = 0,
    fe_nested_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate moment covariance S = E[g_i g_i'].

    Computes covariance of moments g_i = z_i * u_i, accommodating clustering
    and observation weights.
    """
    u_col = u.reshape(-1, 1) if u.ndim == 1 else u
    Zd = np.asarray(Z, dtype=np.float64)

    if obs_weights is not None:
        w_full = np.asarray(obs_weights, dtype=np.float64).reshape(-1, 1)
        zu = la.hadamard(la.hadamard(Zd, w_full), u_col)
    else:
        zu = la.hadamard(Zd, u_col)

    if multiway_ids is not None:
        code_dims = [np.asarray(c).astype(np.int64, copy=False).reshape(-1) for c in multiway_ids]
        R = len(code_dims)
        Ldim = zu.shape[1]
        S = np.zeros((Ldim, Ldim), dtype=np.float64)
        for r in range(1, R + 1):
            sign = +1.0 if (r % 2 == 1) else -1.0
            for idxs in itertools.combinations(range(R), r):
                if r == 1:
                    sums = la.group_sum(zu, code_dims[idxs[0]])
                else:
                    codes = np.column_stack([code_dims[j] for j in idxs])
                    sums = la.group_sum_multi(zu, codes)
                S += sign * la.tdot(sums)
    elif space_ids is not None and time_ids is not None:
        code_space = np.asarray(space_ids).astype(np.int64, copy=False).reshape(-1)
        code_time = np.asarray(time_ids).astype(np.int64, copy=False).reshape(-1)
        S_space = la.tdot(la.group_sum(zu, code_space))
        S_time = la.tdot(la.group_sum(zu, code_time))
        codes_inter = np.column_stack([code_space, code_time])
        S_inter = la.tdot(la.group_sum_multi(zu, codes_inter))
        S = S_space + S_time - S_inter
    elif cluster_ids is not None:
        S = la.tdot(la.group_sum(zu, cluster_ids))
    else:
        S = la.tdot(zu)

    if adj:
        if n_features is None:
            raise ValueError("n_features is required when adj=True")
        if fixefK == "none":
            Kstar = int(n_features)
        elif fixefK == "full":
            Kstar = int(n_features) + int(fe_count)
        else:
            Kstar = (
                int(n_features) + int(np.sum(~fe_nested_mask))
                if fe_nested_mask is not None
                else int(n_features)
            )
        if int(n_eff) > int(Kstar):
            S *= (float(n_eff) - 1.0) / (float(n_eff) - float(Kstar))

    if multiway_ids is not None:
        if str(cluster_df).lower() == "min":
            Gmin = min(len(np.unique(c)) for c in multiway_ids)
            S *= (Gmin / (Gmin - 1.0)) if Gmin > 1 else 1.0
        else:
            for c in multiway_ids:
                G = len(np.unique(c))
                S *= (G / (G - 1.0)) if G > 1 else 1.0
    elif cluster_ids is not None:
        G = len(np.unique(cluster_ids))
        S *= (G / (G - 1.0)) if G > 1 else 1.0
    elif space_ids is not None and time_ids is not None:
        G_space = len(np.unique(np.asarray(space_ids)))
        G_time = len(np.unique(np.asarray(time_ids)))
        Gmin = min(G_space, G_time)
        S *= (Gmin / (Gmin - 1.0)) if Gmin > 1 else 1.0

    return S


class GMM(BaseEstimator):
    """Generalized Method of Moments (GMM) estimator.

    Estimates linear IV models using optimal weighting. Supports one-step,
    two-step efficient GMM, and custom weight matrices.

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        Dependent variable (outcome).
    X : array-like, shape (n, p)
        Regressors (endogenous and exogenous variables).
    Z : array-like, shape (n, L)
        Instrument matrix with L ≥ p instruments. Should include all exogenous
        regressors plus excluded instruments.
    endog_idx : Sequence[int]
        Indices of endogenous variables in X (0-indexed).
    z_excluded_idx : Sequence[int]
        Indices of excluded instruments in Z (0-indexed).
    add_const : bool, default=True
        If True, adds constant term to X and Z.
    var_names : Sequence[str], optional
        Variable names for X.
    instr_names : Sequence[str], optional
        Instrument names for Z.

    Attributes
    ----------
    y_orig : ndarray, shape (n, 1)
        Original outcome variable.
    X_orig : ndarray, shape (n, p)
        Original design matrix.
    Z_orig : ndarray, shape (n, L)
        Original instrument matrix.
    _var_names : list of str
        Variable names.
    _instr_names : list of str
        Instrument names.

    Methods
    -------
    from_formula(formula, data, iv=None, options=None, W_dict=None, id=None, time=None)
        Create GMM model from formula with optional weight matrix specification.
    fit(boot=None, W=None, fe_codes_list=None, cluster_ids=None, weights=None,
        constraints=None, constraint_vals=None, two_step=False)
        Fit GMM model.
    Returns
    -------
    EstimationResult
        Object containing:
        - params : pd.Series - GMM coefficient estimates β̂
        - se : pd.Series - Bootstrap standard errors
        - extra : dict - Diagnostics (Hansen J-stat, Sargan stat, weak IV tests)

    Examples
    --------
    Basic 1-step GMM (identity weighting):

    >>> import numpy as np
    >>> import pandas as pd
    >>> from lineareg.estimators.gmm import GMM
    >>> from lineareg.estimators.base import BootConfig
    >>>
    >>> # Generate IV data
    >>> np.random.seed(42)
    >>> n = 300
    >>> z1 = np.random.randn(n)
    >>> z2 = np.random.randn(n)
    >>> z3 = np.random.randn(n)
    >>> x_exog = np.random.randn(n)
    >>> x_endog = 0.7 * z1 + 0.5 * z2 + 0.4 * z3 + np.random.randn(n) * 0.3
    >>> y = 2.0 + 1.8 * x_endog + 1.2 * x_exog + np.random.randn(n) * 0.6
    >>>
    >>> df = pd.DataFrame({
    ...     'y': y, 'x_endog': x_endog, 'x_exog': x_exog,
    ...     'z1': z1, 'z2': z2, 'z3': z3
    ... })
    >>>
    >>> # Fit 1-step GMM
    >>> model = GMM.from_formula(
    ...     "y ~ x_exog + x_endog",
    ...     df,
    ...     iv="(x_endog ~ z1 + z2 + z3)"
    ... )
    >>> result = model.fit(boot=BootConfig(n_boot=2000, seed=42))
    >>> print(result.params)

    Efficient 2-step GMM:

    >>> result_2step = model.fit(
    ...     boot=BootConfig(n_boot=2000),
    ...     two_step=True  # Use optimal weighting matrix
    ... )
    >>> print("Hansen J-stat:", result_2step.extra.get('J_stat'))

    GMM with custom weight matrix:

    >>> # Specify weight matrix via formula options
    >>> Omega = np.eye(3)  # 3x3 identity for 3 instruments
    >>> W_dict = {'W_custom': Omega}
    >>>
    >>> model = GMM.from_formula(
    ...     "y ~ x_exog + x_endog",
    ...     df,
    ...     iv="(x_endog ~ z1 + z2 + z3)",
    ...     options="gmm(W=W_custom)",
    ...     W_dict=W_dict
    ... )
    >>> result = model.fit(boot=BootConfig(n_boot=2000))

    GMM with fixed effects:

    >>> df['firm_id'] = np.repeat(np.arange(30), 10)
    >>> model = GMM.from_formula(
    ...     "y ~ x_exog + x_endog + fe(firm_id)",
    ...     df,
    ...     iv="(x_endog ~ z1 + z2 + z3)"
    ... )
    >>> result = model.fit(boot=BootConfig(n_boot=2000))

    Notes
    -----
    - **1-step GMM**: Uses identity weighting matrix W = I (equivalent to 2SLS)
    - **2-step GMM**:
        1. Estimate β̂₁ with W = I
        2. Compute moment covariance Ŝ = (1/n) Σᵢ ĝᵢĝᵢ' where ĝᵢ = zᵢûᵢ
        3. Re-estimate with W = Ŝ⁻¹ for efficiency
    - **Custom weights**: Specify via `gmm(W=key)` in formula options
    - **Overidentification tests**:
        * Sargan (1-step, homoskedastic): J = n x û'Z(Z'Z)⁻¹Z'û / sigma_hat^2
        * Hansen (2-step, robust): J = n x g_bar'W_hatg_bar ~ chi^2(L - p)
    - **Weak IV diagnostics**: Same as IV2SLS (Cragg-Donald, Kleibergen-Paap, SW)
    - All inference via wild bootstrap (no analytical p-values)

    References
    ----------
    .. [1] Hansen, L. P. (1982). "Large Sample Properties of Generalized Method of
           Moments Estimators." Econometrica, 50(4), 1029-1054.
    .. [2] Newey, W. K., & McFadden, D. (1994). "Large Sample Estimation and
           Hypothesis Testing." In Handbook of Econometrics (Vol. 4, pp. 2111-2245).
           Elsevier.
    .. [3] Hayashi, F. (2000). Econometrics. Princeton University Press, Chapter 3.
    .. [4] Cameron, A. C., & Trivedi, P. K. (2005). Microeconometrics: Methods and
           Applications. Cambridge University Press, Chapter 6.

    See Also
    --------
    IV2SLS : Two-stage least squares (special case of 1-step GMM)
    OLS : Ordinary least squares (no instruments)

    """

    def __init__(  # noqa: PLR0913
        self,
        y: ArrayLike,
        X: MatrixLike,
        Z: MatrixLike,
        *,
        endog_idx: Sequence[int],
        add_const: bool = True,
        var_names: Sequence[str] | None = None,
        instr_names: Sequence[str] | None = None,
    ) -> None:
        """Initialize GMM estimator with data.

        Parameters
        ----------
        y : array-like
            Dependent variable (n x 1).
        X : array-like
            Regressors (n x k).
        Z : array-like
            Instruments (n x L). Must include all exogenous regressors from X.
        endog_idx : Sequence[int]
            Column indices of endogenous regressors in X (before add_const).
            Required for explicit specification of endogenous variables, matching
            IV2SLS interface and R/Stata conventions.
        add_const : bool, default True
            Whether to add a constant to X and Z.
        var_names : sequence of str, optional
            Names for regressors in X.
        instr_names : sequence of str, optional
            Names for instruments in Z.

        """
        super().__init__()

        # First, convert input arrays to numpy
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if var_names is None and isinstance(X, pd.DataFrame):
            var_names = list(X.columns)
        if instr_names is None and isinstance(Z, pd.DataFrame):
            instr_names = list(Z.columns)
        X_arr = np.asarray(X, dtype=np.float64)
        Z_arr = np.asarray(Z, dtype=np.float64)

        # Build original var names for mapping before add_const
        orig_var_names = (
            list(var_names) if var_names is not None else [f"x{i}" for i in range(X_arr.shape[1])]
        )
        endog_names = [orig_var_names[int(i)] for i in endog_idx]

        if add_const:
            X_aug, x_names_out, const_name = add_constant(X_arr, var_names)
            # For Stata/R dialect (default), DO NOT use force_name parameter
            # Just call add_constant normally; it will use dialect default (_cons for Stata)
            Z_aug, z_names_out, _ = add_constant(Z_arr, instr_names)
            self._const_name = const_name
            self._x_names = list(x_names_out)
            self._z_names = list(z_names_out)
        else:
            X_aug, Z_aug = X_arr, Z_arr
            self._const_name = None
            self._x_names = (
                list(X.columns)
                if isinstance(X, pd.DataFrame)
                else (
                    list(var_names)
                    if var_names is not None
                    else [f"x{i}" for i in range(X_arr.shape[1])]
                )
            )
            self._z_names = (
                list(Z.columns)
                if isinstance(Z, pd.DataFrame)
                else (
                    list(instr_names)
                    if instr_names is not None
                    else [f"z{i}" for i in range(Z_arr.shape[1])]
                )
            )

        if len(self._x_names) != X_aug.shape[1]:
            self._x_names = [f"x{i}" for i in range(X_aug.shape[1])]
        if len(self._z_names) != Z_aug.shape[1]:
            self._z_names = [f"z{i}" for i in range(Z_aug.shape[1])]

        # Map provided endogenous indices (which reference original X columns)
        # into the augmented design `X_aug` by name. This mirrors IV2SLS logic
        # and avoids index-shift bugs when `add_const=True` inserts an intercept.
        self.endog_idx = [i for i, nm in enumerate(self._x_names) if nm in endog_names]
        if not self.endog_idx:
            raise ValueError("endog_idx cannot be empty after mapping to augmented design; check var_names and indices.")

        # ---- Ensure exogenous regressors are included in Z (column-space) ----
        exog_idx_work = [j for j in range(len(self._x_names)) if j not in self.endog_idx]
        if exog_idx_work:
            # prepend exogenous columns to Z so they are preserved by rank screening
            # Use centralized horizontal stack to avoid direct numpy column ops
            try:
                Z_aug = la.hstack([la.to_dense(X_aug[:, exog_idx_work]), la.to_dense(Z_aug)])
            except Exception:
                # Fallback to dense concatenation if la.hstack fails for any reason
                Z_aug = la.to_dense(la.hstack([la.to_dense(X_aug[:, exog_idx_work]), la.to_dense(Z_aug)]))
            z_names_aug = [self._x_names[j] for j in exog_idx_work] + list(self._z_names)
            # Run order-preserving rank screening (Stata-like) using core.linalg
            Z_aug_d, keep = la.drop_rank_deficient_cols_stable(la.to_dense(Z_aug), mode="stata")
            # If a prepended included exogenous regressor is dropped here, it is
            # redundant in the instrument column space under the current numerical
            # rank rule. Identification relies on span(Z), not literal column inclusion.
            if exog_idx_work and not all(keep[: len(exog_idx_work)]):
                self._rank_screening_dropped_included_exog_as_instrument = [
                    z_names_aug[i]
                    for i in range(len(exog_idx_work))
                    if not bool(keep[i])
                ]
            if Z_aug_d.shape[1] < Z_aug.shape[1]:
                dropped = [nm for nm, k in zip(z_names_aug, keep) if not k]
                self._dropped_instr_names = list(dropped)
                # keep only surviving names
                self._z_names = [nm for nm, k in zip(z_names_aug, keep) if k]
            else:
                self._z_names = list(z_names_aug)
            Z_aug = Z_aug_d

        self.y_orig = y_arr
        self.X_orig = X_aug
        self.Z_orig = Z_aug
        self._n_obs = y_arr.shape[0]
        self._n_features = X_aug.shape[1]
        self._n_instr = Z_aug.shape[1]

    @classmethod
    def from_formula(  # noqa: PLR0913
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        iv: str,
        id_name: str | None = None,
        time: str | None = None,
        options: str | None = None,
        W_dict: dict[str, object] | None = None,
        boot: BootConfig | None = None,
    ) -> GMM:
        """Linear GMM from formula + IV clause.
        Example: GMM.from_formula("y ~ x1 + x2", df, iv="(x2 ~ z1 + z2)")
        """
        parser = FormulaParser(data, id_name=id_name, t_name=time, W_dict=W_dict)
        parsed = parser.parse(formula, iv=iv, options=options)
        if not parsed.get("iv_endog"):
            raise ValueError(
                "GMM.from_formula requires an IV clause specifying endogenous regressors.",
            )
        df_use, boot_eff, meta = prepare_formula_environment(
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
        Z_full, z_names, _ = parser.patsy_matrix(
            df_use, parsed.get("iv_instr_full") or "0",
        )
        if parsed.get("iv_instr_user"):
            _Z_user, z_user_names, _ = parser.patsy_matrix(
                df_use,
                " + ".join(parsed["iv_instr_user"]),
            )
            excluded_set = set(z_user_names)
            z_excluded_idx = [j for j, nm in enumerate(z_names) if nm in excluded_set]
        else:
            z_excluded_idx = []
        endog_idx = [parsed["var_names"].index(nm) for nm in parsed["iv_endog"]]
        model = cls(
            parsed["y"],
            parsed["X"],
            Z_full,
            endog_idx=endog_idx,
            add_const=bool(parsed.get("include_intercept", True)),
            var_names=parsed["var_names"],
            instr_names=z_names,
        )
        gmm_W = parsed.get("gmm_W", None)
        if gmm_W is not None:
            meta.attrs["_weight_matrix_from_formula"] = gmm_W
        meta.attrs["_z_excluded_idx_from_formula"] = z_excluded_idx
        attach_formula_metadata(model, meta)
        model._iv_clause = iv
        # Build constraints matrix from formula options if present
        if parsed.get("constraints_raw"):
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
            R, q, _ = build_rq_from_string(
                parsed["constraints_raw"],
                list(model._x_names),
                const_aliases=const_aliases,
            )
            model._constraints_from_formula = (R, q)
        return model

    def _mop_effective_f(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        clusters: Sequence | None = None,
        multiway_ids: Sequence[Sequence] | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        rank_policy: str = "stata",
        ssc: dict[str, Any] | None = None,
    ) -> float:
        """Montiel-Olea & Pflueger effective F (strict; IID / 1-way / multiway / space-time).

        F_eff = (pi' Qzz pi) / tr(Sigma Qzz)
        where pi is the first-stage coefficient vector on excluded instruments and
        Sigma is its robust VCV. Supports observation weights and 1-way cluster-robust VCV.

        Multiway and space-time clustering follow the same Cameron-Gelbach-Miller
        inclusion-exclusion logic used elsewhere in this estimator.
        """
        try:
            xnames = list(self._x_names)
            znames = list(self._z_names)
            included = sorted(set(xnames) & set(znames))
            endogenous = [nm for nm in xnames if nm not in included]
            excluded_instr = [nm for nm in znames if nm not in included]
            if len(endogenous) != 1 or not excluded_instr:
                return float("nan")

            x_idx = {nm: j for j, nm in enumerate(xnames)}
            z_idx = {nm: j for j, nm in enumerate(znames)}
            j = x_idx[endogenous[0]]
            X0_cols = [x_idx[nm] for nm in included if nm in x_idx]
            Z2_cols = [z_idx[nm] for nm in excluded_instr if nm in z_idx]
            X0 = X[:, X0_cols] if X0_cols else np.zeros((X.shape[0], 0))
            Z2 = Z[:, Z2_cols]
            W_fs = la.hstack([X0, Z2]) if X0.shape[1] > 0 else Z2
            x1 = X[:, [j]]

            if weights is not None:
                w_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
                scale = float(np.sum(w_arr))
                if scale <= 0:
                    return float("nan")
                sqrt_w = np.sqrt(w_arr).reshape(-1, 1)
                Wl = la.hadamard(W_fs, sqrt_w)
                x1l = la.hadamard(x1, sqrt_w)
                Z2l = la.hadamard(Z2, sqrt_w)
            else:
                scale = float(Z2.shape[0])
                if scale <= 0:
                    return float("nan")
                Wl = W_fs
                x1l = x1
                Z2l = Z2

            rank_flag = "R" if rank_policy.lower() == "r" else "stata"
            b = la.solve(Wl, x1l, method="qr", rank_policy=rank_flag)
            e = x1l - la.dot(Wl, b)

            try:
                XtX = la.crossprod(Wl, Wl)
                XtX_inv = la.solve(XtX, la.eye(XtX.shape[0]), sym_pos=True)
            except (np.linalg.LinAlgError, ValueError):
                XtX_inv = la.pinv(la.crossprod(Wl, Wl))

            We = la.hadamard(Wl, e)
            if multiway_ids is not None:
                code_dims = [np.asarray(c).reshape(-1) for c in multiway_ids]
                R = len(code_dims)
                M = np.zeros((We.shape[1], We.shape[1]), dtype=np.float64)
                for r in range(1, R + 1):
                    sign = +1.0 if (r % 2 == 1) else -1.0
                    for idxs in itertools.combinations(range(R), r):
                        if r == 1:
                            sums = la.group_sum(We, code_dims[idxs[0]])
                        else:
                            codes = np.column_stack([code_dims[j] for j in idxs])
                            sums = la.group_sum_multi(We, codes)
                        M += sign * la.tdot(sums)
            elif space_ids is not None and time_ids is not None:
                code_space = np.asarray(space_ids).reshape(-1)
                code_time = np.asarray(time_ids).reshape(-1)
                M_space = la.tdot(la.group_sum(We, code_space))
                M_time = la.tdot(la.group_sum(We, code_time))
                codes_inter = np.column_stack([code_space, code_time])
                M_inter = la.tdot(la.group_sum_multi(We, codes_inter))
                M = M_space + M_time - M_inter
            elif clusters is not None:
                g = np.asarray(clusters).reshape(-1)
                Gsum = la.group_sum(We, g)
                M = la.tdot(Gsum)
            else:
                M = la.crossprod(We, We)

            # Apply SSC to M (meat)
            if ssc is not None:
                ssc_eval = _normalize_ssc(ssc)
                k_eff = X.shape[1]
                clusters_for_ssc = (
                    list(multiway_ids)
                    if multiway_ids is not None
                    else (
                        [space_ids, time_ids]
                        if (space_ids is not None and time_ids is not None)
                        else clusters
                    )
                )
                factor = compute_ssc_correction(
                    X.shape[0],
                    k_eff,
                    clusters=clusters_for_ssc,
                    ssc=ssc_eval,
                )
                if abs(factor - 1.0) > 1e-9:
                    M *= factor

            V = la.to_dense(la.dot(la.dot(XtX_inv, M), XtX_inv))
            q = X0.shape[1]
            pi = b[q:, :].reshape(-1, 1)
            Sigma = V[q:, q:]

            try:
                return float(la.effective_f_from_first_stage(pi, Sigma, Z2l))
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                Qzz = la.crossprod(Z2l, Z2l)
                num = float(la.dot(pi.T, la.dot(Qzz, pi)))
                den = float(np.trace(la.to_dense(la.dot(Sigma, Qzz))))
                k = pi.shape[0]
                return (num / den / k) if den > 0 else float("nan")
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, IndexError, RuntimeError):
            return float("nan")

    def fit(  # noqa: PLR0913
        self,
        *,
        absorb_fe: pd.DataFrame | np.ndarray | None = None,
        weights: Sequence[float] | None = None,
        weight_type: str = "identity",  # default to identity; options: {"identity","2sls","2step"}
        constraints: np.ndarray | None = None,
        constraint_vals: np.ndarray | None = None,
        boot: BootConfig | None = None,
        device: str | None = None,
        # FE backend selection for strict reproducibility: 'reghdfe' (default) or 'fixest'
        fe_backend: str = "reghdfe",
        # Rank policy controls QR rank thresholding behavior: 'stata' or 'R'
        rank_policy: str = "stata",
        ssc: dict[str, str | int] | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        s_center: bool = False,
        s_dof: str = "none",
        adj: bool = True,
        fixefK: str = "nested",
        cluster_df: str = "conventional",
        weight_cluster_ids: Sequence | None = None,
        weight_multiway_ids: Sequence[Sequence] | None = None,
        weight_space_ids: Sequence | None = None,
        weight_time_ids: Sequence | None = None,
        method: str = "qr",
    ):
        """Fit GMM and return an EstimationResult (bootstrap SE only).

            Parameters
            ----------
            adj : bool, default True
                Apply (n-1)/(n-K*) adjustment for residual degrees of freedom.
            fixefK : str, default "nested"
                How to count fixed effects in K*: "none", "full", "nested".
            cluster_df : str, default "conventional"
                Cluster degrees of freedom adjustment: "min" or "conventional".

            DOF Scaling (s_dof)
            -------------------
            For theoretical consistency, DoF scaling is applied to display values only (e.g., reported S).
            Point estimates use raw S without scaling. s_dof_used is forced to "none" for estimates.

        Weight Clustering
        -----------------
        Weight clustering uses the intersection (Cameron-Gelbach-Miller inclusion-exclusion) method by design.
        The factorized alternative is deliberately excluded because it departs from standard GMM asymptotics.

        """
        # Enforce unsupported options explicitly to avoid ambiguity.
        if s_center:
            msg = "s_center is not supported (moment centering prohibited by policy)."
            raise ValueError(msg)
        s_dof_used = s_dof.lower()
        if s_dof_used != "none":
            msg = "s_dof must be 'none' (estimation uses raw S only)."
            raise ValueError(msg)

        absorb_fe = self._absorb_fe_from_formula(absorb_fe)

        # FE absorption (synchronized) -------------------------------------------------
        if absorb_fe is not None:
            # Use the backend keyword to match BaseEstimator / fe_core signature
            X, Z, y, mask, dropped_stats = fe_core.demean_xyz(
                self.X_orig,
                self.Z_orig,
                self.y_orig,
                absorb_fe,
                na_action="drop",
                drop_na_fe_ids=True,
                drop_singletons=True,
                backend=fe_backend,
                return_mask=True,
                return_dropped_stats=True,
            )
            fe_count = absorb_fe.shape[1]
            fe_dof_info: dict[str, object] = {}
            try:
                if isinstance(absorb_fe, (pd.DataFrame, pd.Series)):
                    if isinstance(absorb_fe, pd.DataFrame):
                        fe_ids_masked = [
                            absorb_fe[col].to_numpy()[mask] for col in absorb_fe.columns
                        ]
                    else:
                        fe_ids_masked = absorb_fe.to_numpy()[mask]
                else:
                    arr_fe = np.asarray(absorb_fe)
                    if arr_fe.ndim == 1:
                        fe_ids_masked = arr_fe[mask]
                    else:
                        fe_ids_masked = [
                            arr_fe[:, j][mask] for j in range(arr_fe.shape[1])
                        ]
                fe_dof_info = fe_core.compute_fe_dof(fe_ids_masked)
            except (ValueError, TypeError, AttributeError, KeyError):
                fe_dof_info = {}
        else:
            X, Z, y = self.X_orig, self.Z_orig, self.y_orig
            mask = np.ones(self._n_obs, dtype=bool)
            dropped_stats = {"na_dropped": 0, "singleton_dropped": 0}
            fe_count = 0
            fe_dof_info = {}  # Initialize empty for no-FE case
        n_eff = X.shape[0]

        # Prepare ssc_local and inject fe_dof if available (for downstream VCV builders)
        ssc_local = dict(ssc) if ssc is not None else {}
        if "fe_dof_info" in locals() and isinstance(fe_dof_info, dict) and fe_dof_info:
            ssc_local["fe_dof"] = int(fe_dof_info.get("fe_dof", 0))

        # Strict collinearity screening via QR with column pivoting (relative threshold)
        def _keep_columns_strict(A: np.ndarray) -> np.ndarray:
            """Column keep using QR column-pivoting and a centralized rank policy.

            This delegates the numerical cutoff to la.rank_from_diag(..., mode=rank_policy)
                so that 'stata' or 'R' behaviors are controlled by the `rank_policy` argument
                passed to `fit`.
            """
            _Q, R, piv = la.qr(A, mode="economic", pivoting=True)
            diagR = np.abs(np.diag(la.to_dense(R))) if R.size else np.array([])
            if diagR.size == 0:
                return np.zeros(A.shape[1], dtype=bool)
            r = la.rank_from_diag(diagR, A.shape[1], mode=rank_policy)
            keep = np.zeros(A.shape[1], dtype=bool)
            if r > 0:
                keep[np.asarray(piv[:r], dtype=int)] = True
            return keep

        # preserve original names to remap endogenous indices after column pruning
        orig_x_names = list(self._x_names)
        orig_z_names = list(self._z_names)
        keepX = _keep_columns_strict(X)
        keepZ = _keep_columns_strict(Z)
        X = X[:, keepX]
        Z = Z[:, keepZ]
        self._x_names = [self._x_names[j] for j in range(len(keepX)) if keepX[j]]
        self._z_names = [self._z_names[j] for j in range(len(keepZ)) if keepZ[j]]
        self.n_features = X.shape[1]  # Update after FE absorption

        # Constraints must be specified as a pair and must reference only
        # coefficients that survive collinearity screening.
        if (constraints is None) ^ (constraint_vals is None):
            raise ValueError(
                "constraints and constraint_vals must be provided together (or both omitted).",
            )
        if constraints is not None:
            R_full = np.asarray(constraints, dtype=np.float64)
            if R_full.ndim != 2:
                raise ValueError("constraints must be a 2D array")
            q_full = np.asarray(constraint_vals, dtype=np.float64).reshape(-1, 1)
            if q_full.shape[0] != R_full.shape[0]:
                raise ValueError("constraint_vals must have one entry per constraint row")

            k_before = int(len(keepX))
            k_after = int(X.shape[1])
            if R_full.shape[1] == k_before:
                dropped_mask = ~np.asarray(keepX, dtype=bool)
                if dropped_mask.any():
                    dropped_block = R_full[:, dropped_mask]
                    if np.any(np.abs(dropped_block) > 1e-12):
                        raise ValueError(
                            "Constraints reference regressors dropped by collinearity screening.",
                        )
                constraints = R_full[:, np.asarray(keepX, dtype=bool)]
                constraint_vals = q_full
            elif R_full.shape[1] == k_after:
                constraints = R_full
                constraint_vals = q_full
            else:
                raise ValueError(
                    "constraints has incompatible number of columns for this model.",
                )
        # remap endogenous indices and ensure no endogenous regressor was dropped
        orig_to_new_x = {old: new for new, old in enumerate(np.flatnonzero(keepX))}
        endog_idx_new = [orig_to_new_x[i] for i in self.endog_idx if i in orig_to_new_x]
        if len(endog_idx_new) != len(self.endog_idx):
            msg = "One or more endogenous regressors were removed by collinearity screening; cannot proceed."
            raise ValueError(msg)

        # Ensure included exogenous regressors are present in Z (column-space).
        # Avoid name-based checks: default names (x0/x1 vs z0/z1) can differ even when
        # the underlying columns are identical, which would spuriously duplicate
        # instruments and break Z'Z.
        exog_cols = [j for j in range(X.shape[1]) if j not in endog_idx_new]

        def _col_matches_any(col: np.ndarray, M: np.ndarray) -> bool:
            if M.size == 0:
                return False
            v = np.asarray(col, dtype=np.float64).reshape(-1)
            Md = np.asarray(M, dtype=np.float64)
            atol = 1e-12 * max(1.0, float(np.linalg.norm(v)))
            for jj in range(Md.shape[1]):
                if np.allclose(v, Md[:, jj].reshape(-1), rtol=1e-10, atol=atol):
                    return True
            return False

        add_cols = [j for j in exog_cols if not _col_matches_any(X[:, j], Z)]
        if add_cols:
            Z = la.hstack([la.to_dense(X[:, add_cols]), la.to_dense(Z)])
            self._z_names = [self._x_names[j] for j in add_cols] + list(self._z_names)
            # After augmenting Z with exogenous X, perform order-preserving rank screening.
            try:
                Z_aug = la.to_dense(Z)
                Z_aug_d, keep_Zaug = la.drop_rank_deficient_cols_stable(Z_aug, mode="stata")
                n_prepended = len(add_cols)
                if n_prepended and not np.all(keep_Zaug[:n_prepended]):
                    # Not fatal: prepended exogenous regressor can be redundant under
                    # the numerical rank rule.
                    self._rank_screening_dropped_included_exog_as_instrument = [
                        nm
                        for nm, k in zip(self._z_names[:n_prepended], keep_Zaug[:n_prepended])
                        if not bool(k)
                    ]
                if Z_aug_d.shape[1] < Z_aug.shape[1]:
                    dropped = [nm for nm, k in zip(self._z_names, keep_Zaug) if not k]
                    self._dropped_instr_names = list(dropped)
                    self._z_names = [nm for nm, k in zip(self._z_names, keep_Zaug) if k]
                Z = Z_aug_d
            except Exception:
                Z = np.asarray(Z, dtype=np.float64)

        # >>> Re-identification check after collinearity screening (R/Stata alignment)
        # We partial out exogenous X0 and require rank(Z2' X1) >= k_endo_eff where
        # k_endo_eff = #endogenous - rank of constraints effective on endogenous block.
        x_names = list(self._x_names)
        z_names = list(self._z_names)
        included_exog = sorted(set(x_names) & set(z_names))
        endogenous = [nm for nm in x_names if nm not in included_exog]
        excluded_instr = [nm for nm in z_names if nm not in included_exog]
        x_idx = {nm: j for j, nm in enumerate(x_names)}
        z_idx = {nm: j for j, nm in enumerate(z_names)}
        exog_idx_work = [x_idx[nm] for nm in included_exog]
        endog_idx_work = [x_idx[nm] for nm in endogenous]
        z2_idx_work = [z_idx[nm] for nm in excluded_instr]
        X0w = (
            X[:, exog_idx_work]
            if exog_idx_work
            else np.zeros((X.shape[0], 0), dtype=np.float64)
        )
        X1w = (
            X[:, endog_idx_work]
            if endog_idx_work
            else np.zeros((X.shape[0], 0), dtype=np.float64)
        )
        Z2w = (
            Z[:, z2_idx_work]
            if z2_idx_work
            else np.zeros((Z.shape[0], 0), dtype=np.float64)
        )
        if X0w.shape[1] > 0:
            X1_t = X1w - la.dot(X0w, la.solve(X0w, X1w, method="qr"))
            Z2_t = Z2w - la.dot(X0w, la.solve(X0w, Z2w, method="qr"))
        else:
            X1_t, Z2_t = X1w, Z2w
        # rank of constraints on endogenous block
        rank_R = 0
        if (
            constraints is not None
            and constraint_vals is not None
            and getattr(constraints, "size", 0) > 0
        ):
            try:
                R_dense = np.asarray(constraints, dtype=np.float64)
                qrR = la.qr(R_dense, mode="economic", pivoting=True)
                if len(qrR) == 3:
                    _Qr_c, Rr_c, _pr_c = qrR
                else:
                    _Qr_c, Rr_c = qrR
                diagRr_c = (
                    np.abs(np.diag(la.to_dense(Rr_c))) if Rr_c.size else np.array([])
                )
                rank_R = (
                    la.rank_from_diag(diagRr_c, R_dense.shape[1], mode=rank_policy)
                    if diagRr_c.size
                    else 0
                )
            except (np.linalg.LinAlgError, ValueError):
                rank_R = 0
        # effective endogenous constraint rank
        rank_R_endo = 0
        if (
            constraints is not None
            and getattr(constraints, "size", 0) > 0
            and endog_idx_work
        ):
            try:
                R_endog = np.asarray(constraints, np.float64)[:, endog_idx_work]
                Rqr = la.qr(la.to_dense(R_endog), mode="economic", pivoting=True)
                if len(Rqr) == 3:
                    _Qre, Rre, _pre = Rqr
                else:
                    _Qre, Rre = Rqr
                diagRe = np.abs(np.diag(la.to_dense(Rre))) if Rre.size else np.array([])
                rank_R_endo = (
                    la.rank_from_diag(diagRe, R_endog.shape[1], mode=rank_policy)
                    if diagRe.size
                    else 0
                )
            except (np.linalg.LinAlgError, ValueError, IndexError):
                rank_R_endo = 0
        k_endo_eff = max(0, int(len(endog_idx_work) - rank_R_endo))
        try:
            Z2X1 = la.crossprod(Z2_t, X1_t)
            R_id = la.qr(la.to_dense(Z2X1), mode="economic", pivoting=True)
            if len(R_id) == 3:
                _Qi, Ri, _pi = R_id
            else:
                _Qi, Ri = R_id
            diagRi = np.abs(np.diag(la.to_dense(Ri))) if Ri.size else np.array([])
            rank_Z2X1 = (
                la.rank_from_diag(diagRi, Z2X1.shape[1], mode=rank_policy)
                if diagRi.size
                else 0
            )
        except (np.linalg.LinAlgError, ValueError):
            rank_Z2X1 = 0
        if rank_Z2X1 < k_endo_eff:
            msg = "Underidentified after partialling: rank(Z2'X1|X0) < effective endogenous parameters."
            raise ValueError(msg)
        K_eff = int(X.shape[1] - rank_R)

        # ---- (A) Mask IDs early ----
        def _mask_seq(seq):
            if seq is None:
                return None
            arr = np.asarray(seq)
            if arr.shape[0] != self._n_obs:
                msg = "Cluster/space/time ids length mismatch original n."
                raise ValueError(msg)
            return arr[mask]

        def _mask_seq_multiway(multiway):
            if multiway is None:
                return None
            return [_mask_seq(arr) for arr in multiway]

        # Inference IDs
        cluster_ids_proc = _mask_seq(cluster_ids)
        space_ids_proc = _mask_seq(space_ids)
        time_ids_proc = _mask_seq(time_ids)
        multiway_ids_proc = (
            None
            if (boot is None or boot.multiway_ids is None)
            else [_mask_seq(a) for a in boot.multiway_ids]
        )
        # FE nested mask
        if fe_count > 0 and multiway_ids_proc:
            if isinstance(absorb_fe, (pd.DataFrame, pd.Series)):

                def _fe_col_j(j: int) -> np.ndarray:
                    if isinstance(absorb_fe, pd.DataFrame):
                        return absorb_fe.iloc[:, j].to_numpy()[mask]
                    return absorb_fe.to_numpy()[mask]
            else:

                def _fe_col_j(j: int) -> np.ndarray:
                    arr = np.asarray(absorb_fe)
                    if arr.ndim == 1:
                        return arr[mask]
                    return arr[:, j][mask]

            fe_nested_mask = np.array(
                [
                    fe_core.is_nested(_fe_col_j(j), multiway_ids_proc)
                    for j in range(fe_count)
                ],
            )
        else:
            fe_nested_mask = None
        # Weight IDs
        weight_cluster_ids_proc = _mask_seq(weight_cluster_ids)
        weight_space_ids_proc = _mask_seq(weight_space_ids)
        weight_time_ids_proc = _mask_seq(weight_time_ids)
        weight_multiway_ids_proc = (
            None
            if weight_multiway_ids is None
            else [_mask_seq(a) for a in weight_multiway_ids]
        )

        # Inherit inference clustering into weight clustering when user did not
        # explicitly specify weight_* arguments. This ensures 2-step weight
        # construction uses the same clustering granularity as inference by default.
        if weight_cluster_ids_proc is None and cluster_ids_proc is not None:
            weight_cluster_ids_proc = cluster_ids_proc
        if weight_multiway_ids_proc is None and multiway_ids_proc is not None:
            weight_multiway_ids_proc = multiway_ids_proc
        if (
            weight_space_ids_proc is None
            and space_ids_proc is not None
            and time_ids_proc is not None
        ):
            weight_space_ids_proc = space_ids_proc
            weight_time_ids_proc = time_ids_proc

        # Mask weights
        if weights is not None:
            w_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w_arr.shape[0] != self._n_obs:
                msg = "weights length mismatch original n_obs."
                raise ValueError(msg)
            if np.any(w_arr <= 0):
                msg = "weights must be positive for GMM weighting."
                raise ValueError(msg)
            weights_proc = w_arr[mask].reshape(-1, 1)
        else:
            weights_proc = None

        # Determine if multiway G-1 approximation is used
        s_dof_internal = "none"  # Fixed to exact; distinct from user-facing s_dof

        # 1-step initializer: default is identity; allow explicit 2sls initializer
        init_w = "2sls" if weight_type.lower() == "2sls" else "identity"

        # Propagate fixed effects DoF to SSC
        ssc_local = dict(ssc) if ssc is not None else {}
        if (
            ("fe_dof" not in ssc_local)
            and isinstance(fe_dof_info, dict)
            and ("fe_dof" in fe_dof_info)
        ):
            ssc_local["fe_dof"] = int(fe_dof_info["fe_dof"])

        with self._device_context(device):
            beta_1, W_1 = self._gmm_1step(
                X,
                Z,
                y,
                weights_proc,
                rank_policy=rank_policy,
                init_weight=init_w,
                constraints=constraints,
                constraint_vals=constraint_vals,
            )
        # If a weight matrix was provided via formula options, validate and use it.
        # Note: GMM.fit does not accept a direct weight_matrix parameter; instead,
        # from_formula may attach one under the private attribute below.
        weight_matrix_attached = getattr(self, "_weight_matrix_from_formula", None)
        if weight_matrix_attached is not None:
            W_used = np.asarray(weight_matrix_attached, dtype=np.float64)
            if W_used.shape[0] != W_used.shape[1]:
                msg = "weight_matrix must be square (L x L)."
                raise ValueError(msg)
            # Symmetry check (numerical tolerance) before eigen-decomposition
            if not np.allclose(W_used, W_used.T, atol=la.eig_tol(W_used)):
                raise ValueError("weight_matrix must be symmetric.")
            # SPD check via eigenvalues
            with self._device_context(device):
                evals, _Q = la.eigh(W_used)
            tol = la.eig_tol(W_used)
            if not np.all(evals > tol):
                msg = "weight_matrix must be SPD (positive eigenvalues)."
                raise ValueError(msg)
            with self._device_context(device):
                beta_hat = self._gmm_solve(
                    X,
                    Z,
                    y,
                    W_used,
                    weights_proc,
                    constraints,
                    constraint_vals,
                    rank_policy=rank_policy,
                )
        elif weight_type.lower() in {"2sls", "1step", "one-step", "identity"}:
            beta_hat = beta_1
            W_used = W_1
        else:
            with self._device_context(device):
                u1 = y.reshape(-1) - la.dot(X, beta_1).reshape(-1)
            # For estimation use raw S (no small-sample adj) and then form S_bar = S / n
            S = self._moment_covariance(
                Z,
                u1,
                n_eff=X.shape[0],
                cluster_ids=weight_cluster_ids_proc,
                multiway_ids=weight_multiway_ids_proc,
                space_ids=weight_space_ids_proc,
                time_ids=weight_time_ids_proc,
                obs_weights=weights_proc,
                adj=False,  # raw S for estimation (adj scaling is for display only)
                fixefK=fixefK,
                cluster_df=cluster_df,
                fe_count=fe_count,
                fe_nested_mask=fe_nested_mask,
            )
            # Standard 2-step GMM weight uses S_bar = (1/scale) * sum g_i g_i'
            # where scale = n (no obs weights) or sum(weights) when analytic weights
            scale = (
                float(X.shape[0])
                if (weights_proc is None)
                else float(np.sum(weights_proc))
            )
            S_bar = S / scale
            # rank-aware inversion of S_bar
            with self._device_context(device):
                evals, Q = la.eigh(S_bar)
                keep = evals > la.eig_tol(S_bar)
                if not np.any(keep):
                    msg = "S_bar is singular; 2-step GMM weight undefined."
                    raise ValueError(msg)
                Sk = la.dot(Q[:, keep].T, la.dot(S_bar, Q[:, keep]))
                # Use safe_cholesky with fallback to pseudoinverse for near-singular cases
                try:
                    Lk = la.safe_cholesky(Sk)
                    Wk = la.chol_solve(Lk, la.eye(Sk.shape[0]))
                except RuntimeError:
                    # Fallback to pseudoinverse for numerically unstable cases
                    Wk = la.pinv(Sk)
                W = la.dot(Q[:, keep], la.dot(Wk, Q[:, keep].T))
                beta_hat = self._gmm_solve(
                    X,
                    Z,
                    y,
                    W,
                    weights_proc,
                    constraints,
                    constraint_vals,
                    rank_policy=rank_policy,
                )
            W_used = W

        with self._device_context(device):
            yhat = la.dot(X, beta_hat)
            u = y.reshape(-1) - yhat.reshape(-1)

        # Weak-IV (values only)
        weak_iv = self._weak_iv_first_stage_stats(X, Z)
        # MOP effective F (weights & clusters aware)
        try:
            weak_iv["MOP_effective_F"] = self._mop_effective_f(
                X,
                Z,
                weights=weights_proc,
                clusters=cluster_ids_proc,
                multiway_ids=multiway_ids_proc,
                space_ids=space_ids_proc,
                time_ids=time_ids_proc,
                rank_policy=rank_policy,
                ssc=ssc_local,
            )
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, RuntimeError):
            weak_iv["MOP_effective_F"] = float("nan")

        try:
            sw_partial_list = self._sw_partial_f_list(X, Z)
        except (
            np.linalg.LinAlgError,
            ValueError,
        ):  # pragma: no cover - robustness path
            sw_partial_list = []
        weak_iv["SW_partial_F_list"] = sw_partial_list
        for name, val, _msg in sw_partial_list:
            key = f"F_SW_partial_{name}"
            weak_iv[key] = float(val) if np.isfinite(val) else float("nan")

        # Cragg-Donald and Kleibergen-Paap minimal generalized eigenvalues
        # Compute using the helper imported from iv and pass residuals and clusters
        try:
            z_names_post = list(self._z_names)
            x_names_post = list(self._x_names)
            included_exog = sorted(set(x_names_post) & set(z_names_post))
            z_excluded_idx_new = [
                j for j, nm in enumerate(z_names_post) if nm not in included_exog
            ]
            cdkp = _cd_kp_stats(
                X,
                Z,
                endog_idx_new,
                z_excluded_idx_new,
                u=u,
                clusters=cluster_ids_proc,
                multiway_ids=(list(multiway_ids_proc) if multiway_ids_proc is not None else None),
                space_ids=(np.asarray(space_ids_proc) if space_ids_proc is not None else None),
                time_ids=(np.asarray(time_ids_proc) if time_ids_proc is not None else None),
                ssc=ssc_local,
                k_params=X.shape[1],
            )
            lam_cd = float(cdkp.get("cd_min_eig", np.nan))
            lam_kp = float(cdkp.get("kp_min_eig", np.nan))
            weak_iv["cd_min_eig"] = lam_cd
            weak_iv["kp_min_eig"] = lam_kp
            weak_iv["cd_wald_F"] = float(cdkp.get("cd_wald_F", np.nan))
            weak_iv["kp_rk_LM"] = float(cdkp.get("kp_rk_LM", np.nan))
            weak_iv["kp_rk_Wald_F"] = float(cdkp.get("kp_rk_Wald_F", np.nan))
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            weak_iv["cd_min_eig"] = float("nan")
            weak_iv["kp_min_eig"] = float("nan")
            weak_iv["cd_wald_F"] = float("nan")
            weak_iv["kp_rk_LM"] = float("nan")
            weak_iv["kp_rk_Wald_F"] = float("nan")

        # Bootstrap SEs (no analytic CIs) -------------------------------------------------------
        cluster_ids_proc = _mask_seq(cluster_ids)
        space_ids_proc = _mask_seq(space_ids)
        time_ids_proc = _mask_seq(time_ids)
        weight_cluster_ids_proc = _mask_seq(weight_cluster_ids)
        weight_space_ids_proc = _mask_seq(weight_space_ids)
        weight_time_ids_proc = _mask_seq(weight_time_ids)
        weight_multiway_ids_proc = _mask_seq_multiway(weight_multiway_ids)
        multiway_proc = _mask_seq_multiway(boot.multiway_ids if boot else None)

        # Require (space_ids,time_ids) jointly; error if only one is provided
        if (space_ids_proc is None) ^ (time_ids_proc is None):
            raise ValueError("space_ids and time_ids must be provided jointly.")
        # Early exclusivity check for clustering schemes (estimator-level)
        provided = sum(
            int(x is not None)
            for x in (
                cluster_ids_proc,
                (
                    space_ids_proc
                    if (space_ids_proc is not None and time_ids_proc is not None)
                    else None
                ),
                multiway_proc,
            )
        )
        if provided > 1:
            msg = "Specify at most one clustering scheme among {multiway_ids, (space_ids,time_ids), cluster_ids}."
            raise ValueError(msg)

        # Enforce WGB policy
        # We do NOT force a specific distribution (boottest/fwildclusterboot
        # compatibility typically uses Rademacher by default; Webb may be used
        # under small-G rules inside core.bootstrap).

        if boot is None:
            # Default: if any clustering scheme is provided, prefer 'boottest'
            # policy to mimic small-G enumeration and Webb promotion semantics.
            any_cluster = (
                (cluster_ids_proc is not None)
                or ((space_ids_proc is not None) and (time_ids_proc is not None))
                or (multiway_proc is not None)
            )
            boot = BootConfig(
                dist="rademacher",
                cluster_ids=cluster_ids_proc,
                space_ids=space_ids_proc,
                time_ids=time_ids_proc,
                multiway_ids=multiway_proc,
                policy=("boottest" if any_cluster else "strict"),
                enumeration_mode="boottest",
            )
        else:
            # Enforce strict bootstrap-only enumeration policy for any clustering
            # scenarios: do not honor user-supplied enumeration flags that would
            # enable non-bootstrapped enumeration. Always prefer 'boottest'
            # semantics for clustered/multiway cases and disable use_enumeration.
            any_cluster = (
                (cluster_ids_proc is not None)
                or ((space_ids_proc is not None) and (time_ids_proc is not None))
                or (multiway_proc is not None)
            )
            boot = BootConfig(
                n_boot=int(
                    getattr(boot, "n_boot", bt.DEFAULT_BOOTSTRAP_ITERATIONS)
                    or bt.DEFAULT_BOOTSTRAP_ITERATIONS
                ),
                dist=getattr(boot, "dist", "rademacher"),
                seed=getattr(boot, "seed", None),
                enum_max_g=getattr(boot, "enum_max_g", None),
                use_enumeration=getattr(boot, "use_enumeration", True),
                cluster_ids=cluster_ids_proc,
                multiway_ids=multiway_proc,
                space_ids=space_ids_proc,
                time_ids=time_ids_proc,
                cluster_method=getattr(boot, "cluster_method", None),
                policy=(
                    "boottest"
                    if any_cluster
                    else (getattr(boot, "policy", None) or "strict")
                ),
                enumeration_mode=getattr(boot, "enumeration_mode", "boottest"),
            )

        # SSC needs the same clustering structure used for inference.
        clusters_for_ssc = (
            list(multiway_proc)
            if multiway_proc is not None
            else (
                [space_ids_proc, time_ids_proc]
                if (space_ids_proc is not None and time_ids_proc is not None)
                else cluster_ids_proc
            )
        )
        Wmult, boot_log = self._bootstrap_multipliers(n_eff, boot=boot)
        _variant = None
        if isinstance(boot_log, dict):
            _variant = boot_log.get("mnw_variant_used") or boot_log.get("variant")

        W_arr = Wmult.to_numpy()
        B_boot = W_arr.shape[1]

        # Use the post-screening endogenous indices to keep X column references consistent.
        endog_idx_local = list(endog_idx_new)

        # Robust first-stage F (value only; no p-values/criticals) ------------------------
        # Definition matches IV: run a wild-bootstrap Wald test in the first stage and
        # scale to an F-style statistic by dividing by q (#excluded instruments).
        try:
            z_names_post = list(self._z_names)
            x_names_post = list(self._x_names)
            included_exog = sorted(set(x_names_post) & set(z_names_post))
            z_excluded_idx_new = [
                j for j, nm in enumerate(z_names_post) if nm not in included_exog
            ]
            boot_rf = float("nan")
            if len(endog_idx_local) == 1 and len(z_excluded_idx_new) > 0:
                y_endog = X[:, endog_idx_local].reshape(-1, 1)
                L_total = Z.shape[1]
                q = int(len(z_excluded_idx_new))
                R_fs = np.zeros((q, L_total), dtype=np.float64)
                for i, idx in enumerate(z_excluded_idx_new):
                    R_fs[i, int(idx)] = 1.0
                r_fs = np.zeros((q, 1), dtype=np.float64)
                wt = bt.wald_test_wild_bootstrap(
                    Z,
                    y_endog,
                    R=R_fs,
                    r=r_fs,
                    multipliers=W_arr,
                    ssc=ssc_local,
                    residual_type="WCU",
                    clusters=cluster_ids_proc,
                    multiway_ids=multiway_proc,
                    space_ids=space_ids_proc,
                    time_ids=time_ids_proc,
                )
                boot_rf = float(wt.get("wald_stat", float("nan"))) / float(q)
            weak_iv["bootstrap_robust_f"] = boot_rf
        except Exception:  # noqa: BLE001
            weak_iv["bootstrap_robust_f"] = float("nan")
        if len(endog_idx_local) > 0:
            Qz_rf_gmm = la.dot(Z, la.pinv(la.crossprod(Z, Z)))
            Qz_rf_gmm = la.dot(Qz_rf_gmm, la.crossprod(Z, Z))
            y_rf_hat_gmm = la.dot(Z, la.dot(la.pinv(la.crossprod(Z, Z)), la.crossprod(Z, y)))
            e_y_gmm = y - y_rf_hat_gmm
            X_endog_gmm = X[:, endog_idx_local]
            X_rf_hat_gmm = la.dot(Z, la.dot(la.pinv(la.crossprod(Z, Z)), la.crossprod(Z, X_endog_gmm)))
            e_x_gmm = X_endog_gmm - X_rf_hat_gmm

            # Apply SSC scaling to reduced-form residuals so the bootstrap DGP
            # matches apply_wild_bootstrap semantics used elsewhere.
            try:
                ssc_norm = _normalize_ssc(ssc_local)
                ssc_factor = compute_ssc_correction(
                    n_eff,
                    int(X.shape[1]),
                    clusters=clusters_for_ssc,
                    ssc=ssc_norm,
                )
                if abs(float(ssc_factor) - 1.0) > 1e-9:
                    e_y_gmm = e_y_gmm * float(ssc_factor)
                    e_x_gmm = e_x_gmm * float(ssc_factor)
            except Exception:  # noqa: BLE001
                pass
        else:
            Ystar, _ = bt.apply_wild_bootstrap(
                yhat,
                u,
                W_arr,
                residual_type="unrestricted",
                clusters=clusters_for_ssc,
                ssc=ssc_local,
                x_dof=int(X.shape[1]),
            )

        if (
            weight_type.lower() in {"2sls", "1step", "one-step", "identity"}
            and constraints is None
            and constraint_vals is None
            and len(endog_idx_local) == 0
        ):
            with self._device_context(device):
                G1 = la.crossprod(X, la.dot(Z, la.dot(W_1, la.crossprod(Z, X))))
                ZtYstar = la.crossprod(Z, Ystar)
                g_star = la.dot(la.crossprod(X, Z), la.dot(W_1, ZtYstar))
                boot_betas = la.solve(G1, g_star, sym_pos=True)
        else:
            B = B_boot if len(endog_idx_local) > 0 else Ystar.shape[1]
            boot_betas = np.empty((self.n_features, B), dtype=np.float64)
            with self._device_context(device):
                for b in range(B):
                    if len(endog_idx_local) > 0:
                        w_b = W_arr[:, b:b+1]
                        y_star_b = y_rf_hat_gmm + e_y_gmm * w_b
                        X_star_b = X.copy() if isinstance(X, np.ndarray) else la.to_dense(X).copy()
                        for j_enum, j_col in enumerate(endog_idx_local):
                            X_star_b[:, j_col:j_col+1] = X_rf_hat_gmm[:, j_enum:j_enum+1] + e_x_gmm[:, j_enum:j_enum+1] * w_b
                        yb = y_star_b
                    else:
                        yb = Ystar[:, b : b + 1]
                        X_star_b = X
                    if weight_type.lower() in {"2sls", "1step", "one-step", "identity"}:
                        boot_betas[:, b] = self._gmm_solve(
                            X_star_b, Z, yb, W_1, weights_proc, constraints, constraint_vals,
                        ).reshape(-1)
                    else:
                        beta1_b, _ = self._gmm_1step(X_star_b, Z, yb, weights_proc)
                        u1_b = yb.reshape(-1) - la.dot(X_star_b, beta1_b).reshape(-1)
                        S_b = self._moment_covariance(
                            Z,
                            u1_b,
                            n_eff=X.shape[0],
                            cluster_ids=weight_cluster_ids_proc,
                            multiway_ids=weight_multiway_ids_proc,
                            space_ids=weight_space_ids_proc,
                            time_ids=weight_time_ids_proc,
                            obs_weights=weights_proc,
                            adj=False,
                            fixefK=fixefK,
                            cluster_df=cluster_df,
                            fe_count=fe_count,
                            fe_nested_mask=fe_nested_mask,
                        )
                        scale_b = (
                            float(X.shape[0])
                            if (weights_proc is None)
                            else float(np.sum(weights_proc))
                        )
                        S_bar_b = S_b / scale_b
                        evals_b, Q_b = la.eigh(S_bar_b)
                        keep_b = evals_b > la.eig_tol(S_bar_b)
                        if not np.any(keep_b):
                            msg = "S_bar in replicate is singular; 2-step GMM weight undefined."
                            raise ValueError(msg)
                        Sk_b = la.dot(Q_b[:, keep_b].T, la.dot(S_bar_b, Q_b[:, keep_b]))
                        Lk_b = la.safe_cholesky(Sk_b)
                        Wk_b = la.chol_solve(Lk_b, la.eye(Sk_b.shape[0]))
                        W_b = la.dot(Q_b[:, keep_b], la.dot(Wk_b, Q_b[:, keep_b].T))
                        boot_betas[:, b] = self._gmm_solve(
                            X_star_b, Z, yb, W_b, weights_proc, constraints, constraint_vals,
                        ).reshape(-1)

        se_hat = bt.bootstrap_se(boot_betas)

        # ---- (B) Overidentification statistic (ensure S definition and scheme match) ----
        L, k = Z.shape[1], X.shape[1]
        j_stat: float | None = None
        if K_eff < L:
            # Compute gbar using the same moment weighting convention as S
            if weights_proc is None:
                scale = float(X.shape[0])
                Z_eff = Z
            else:
                sw = np.sqrt(weights_proc).reshape(-1, 1)
                scale = float(np.sum(weights_proc))
                Z_eff = la.hadamard(Z, sw)
            gbar = la.crossprod(Z_eff, u) / scale
            if weight_type.lower() in {"2sls", "1step", "one-step", "identity"}:
                # Determine strict i.i.d. eligibility in one place for auditability
                iid_ok = (
                    (weights_proc is None)
                    and (weight_cluster_ids_proc is None)
                    and (weight_multiway_ids_proc is None)
                    and not (
                        (weight_space_ids_proc is not None)
                        and (weight_time_ids_proc is not None)
                    )
                )
                if not iid_ok:
                    msg = "Sargan is only defined for i.i.d. (no analytic weights / no clustering). Use robust 2-step (Hansen J)."
                    raise ValueError(msg)
                # Standard Sargan (value-only):
                #   J = (n / sigma^2) * gbar' Qzz^{-1} gbar,
                # where gbar = (Z'u)/n and Qzz = (Z'Z)/n.
                # Use constraint-aware effective parameter count for sigma^2 DOF.
                k_eff = int(locals().get("K_eff", X.shape[1]))
                denom = max(1, int(X.shape[0] - k_eff))
                u_col = u.reshape(-1, 1) if u.ndim == 1 else u
                sigma2 = float(la.to_dense(la.dot(u_col.T, u_col) / denom).squeeze())
                Qzz = la.tdot(Z_eff) / scale
                try:
                    evals, Q = la.eigh(Qzz)
                    keep = evals > la.eig_tol(Qzz)
                    if not np.any(keep):
                        j_stat = float("nan")
                    else:
                        Qk = Q[:, keep]
                        Qzz_k = la.dot(Qk.T, la.dot(Qzz, Qk))
                        Lk = la.safe_cholesky(Qzz_k)
                        Qzz_k_inv = la.chol_solve(Lk, la.eye(Qzz_k.shape[0]))
                        gk = la.dot(Qk.T, gbar)
                        j_val = la.crossprod(gk, la.dot(Qzz_k_inv, gk))
                        j_stat = float(la.to_dense(j_val).squeeze()) * scale / sigma2
                except (np.linalg.LinAlgError, ValueError):
                    j_stat = float("nan")
            else:
                # Hansen J (robust): use S_bar = S / scale, and J = scale * gbar' S_bar^{-1} gbar
                S = self._moment_covariance(
                    Z,
                    u,
                    n_eff=X.shape[0],
                    cluster_ids=weight_cluster_ids_proc,
                    multiway_ids=weight_multiway_ids_proc,
                    space_ids=weight_space_ids_proc,
                    time_ids=weight_time_ids_proc,
                    obs_weights=weights_proc,
                    adj=False,  # build raw S first
                    fixefK=fixefK,
                    cluster_df=cluster_df,
                    fe_count=fe_count,
                    fe_nested_mask=fe_nested_mask,
                )
                # scale must match gbar's denominator (n or sum(weights))
                scale = (
                    float(X.shape[0])
                    if (weights_proc is None)
                    else float(np.sum(weights_proc))
                )
                S_bar = S / scale
                try:
                    evals, Q = la.eigh(S_bar)
                    keep = evals > la.eig_tol(S_bar)
                    if not np.any(keep):
                        j_stat = float("nan")
                    else:
                        Sk = la.dot(Q[:, keep].T, la.dot(S_bar, Q[:, keep]))
                        Lk = la.safe_cholesky(Sk)
                        Sk_inv = la.chol_solve(Lk, la.eye(Sk.shape[0]))
                        gk = la.dot(Q[:, keep].T, gbar)
                        j_val = la.crossprod(gk, la.dot(Sk_inv, gk))
                        j_stat = float(la.to_dense(j_val).squeeze()) * scale
                except (np.linalg.LinAlgError, ValueError):
                    j_stat = float("nan")
                # If user requested adjusted/display S, compute and store it separately
                S_display = None
                if adj:
                    try:
                        S_display = self._moment_covariance(
                            Z,
                            u,
                            n_eff=X.shape[0],
                            cluster_ids=weight_cluster_ids_proc,
                            multiway_ids=weight_multiway_ids_proc,
                            space_ids=weight_space_ids_proc,
                            time_ids=weight_time_ids_proc,
                            obs_weights=weights_proc,
                            adj=True,
                            fixefK=fixefK,
                            cluster_df=cluster_df,
                            fe_count=fe_count,
                            fe_nested_mask=fe_nested_mask,
                        )
                    except (np.linalg.LinAlgError, ValueError):
                        S_display = None

        params = pd.Series(beta_hat.reshape(-1), index=self._x_names, name="coef")
        se = pd.Series(se_hat.reshape(-1), index=self._x_names, name="se")
        # Per policy: do not attach se to EstimationResult for core linear estimators.
        extra_local = {
            "yhat_within": yhat,
            "resid_within": u,
        }
        # merge with later extra dict built below (we'll collect values later)
        self._results = EstimationResult(
            params=params,
            se=se,  # Bootstrap SE stored directly in .se
            bands=None,
            n_obs=int(n_eff),
            model_info={
                "Estimator": "GMM",
                "WeightType": (
                    "Identity (1-step)"
                    if weight_type.lower() in {"identity", "1step", "one-step"}
                    else (
                        "2SLS (1-step)"
                        if weight_type.lower() == "2sls"
                        else "Robust 2-step"
                    )
                ),
                "FixedEffects": absorb_fe is not None,
                "Constraints": constraints is not None and constraint_vals is not None,
                "Bootstrap": "wild",
                "B": Wmult.shape[1]
                if "Wmult" in locals()
                else (
                    boot.n_boot if boot is not None else bt.DEFAULT_BOOTSTRAP_ITERATIONS
                ),
                "OverID_df": int(L - K_eff) if K_eff < L else 0,
                "HasConstraints": constraints is not None
                and constraint_vals is not None,
                "WeakIV_diagnostic": "SW partial F (values only)",
                "Dropped": dropped_stats,
                "n_eff": n_eff,
                "WeightClusterMethod": "CGM(inclusion-exclusion)",
                **boot_log,
            },
            extra={
                "yhat_within": yhat,
                "resid_within": u,
                "yhat": yhat,
                "boot_betas": boot_betas,
                "OverID_stat": j_stat,
                "J_stat": j_stat,
                "OverID_label": "Sargan"
                if weight_type.lower() in {"2sls", "1step", "one-step", "identity"}
                else "Hansen J",
                "first_stage_stats": weak_iv,
                "mask_used": mask,
                "X_inference": X,
                "u_inference": u,
                "clusters_inference": cluster_ids_proc,
                "multiway_ids_inference": multiway_proc,
                "space_ids_inference": space_ids_proc,
                "time_ids_inference": time_ids_proc,
                "weights_inference": weights_proc,
                "vcov_kind_inference": "auto_strict",
                "beta0_inference": beta_hat,
                # Preserve exact multiplier layout/order for strict reproducibility
                "W_multipliers_inference": Wmult,
                "boot_config": boot,
                "multipliers_log": boot_log,
                "ssc_effective": ssc_local,
                "S_display": (S_display if "S_display" in locals() else None),
                # Diagnostics only (does not control computation): prefer the actual
                # variant returned by core.bootstrap if present; otherwise fall back
                # to the nominal '33'/'11' label based on clustering presence.
                "boot_variant_used": (
                    _variant
                    if (_variant is not None)
                    else (
                        "33"
                        if (
                            multiway_proc is not None
                            or (
                                space_ids_proc is not None and time_ids_proc is not None
                            )
                            or (cluster_ids_proc is not None)
                        )
                        else "11"
                    )
                ),
                "boot_residual_default": "WCU",
                "boot_policy_used": getattr(boot, "policy", None),
                "se_source": "bootstrap",
            },
        )
        return self._results

    def _gmm_1step(  # noqa: PLR0913
        self,
        X: NDArray[np.float64],
        Z: NDArray[np.float64],
        y: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
        rank_policy: str = "stata",
        init_weight: str = "identity",
        constraints: np.ndarray | None = None,
        constraint_vals: np.ndarray | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute 1-step GMM beta and initial weight W_1.

        init_weight:
          - "identity": W_1 = I_L  (project default)
          - "2sls":     W_1 = (Z'Z)^{-1}  (2SLS-equivalent init, no /n scaling)

        This implementation enforces a strict QR-based rank check on the
        (possibly weighted) instrument matrix and requires a true Cholesky
        factorization when computing (Z'Z)^{-1}.
        """
        if weights is None:
            ZtZ = la.tdot(Z)  # No division by n for 2SLS weight
            Z_for_qr = Z
        else:
            sqrt_w = np.sqrt(weights)
            Z_w = la.hadamard(Z, sqrt_w)
            ZtZ = la.tdot(Z_w)  # No division by sum(weights) for 2SLS weight
            Z_for_qr = Z_w

        # QR pivot-based rank check on instrument space (weighted if applicable)
        _Qr, Rr, _p = la.qr(Z_for_qr, mode="economic", pivoting=True)
        diagR = np.abs(np.diag(la.to_dense(Rr))) if Rr.size else np.array([])
        r = (
            la.rank_from_diag(diagR, Z_for_qr.shape[1], mode=rank_policy)
            if diagR.size
            else 0
        )
        if r == 0:
            msg = "Underidentified: Z has no column rank."
            raise ValueError(msg)

        # Construct initial W depending on init_weight
        L = Z.shape[1]
        if init_weight.lower() in {"identity", "i", "eye"}:
            W = la.eye(L)
            # >>> Strict: even at 1-step, compute beta with W = I (and honor constraints if provided)
            beta = self._gmm_solve(
                X,
                Z,
                y,
                W,
                weights=weights,
                constraints=constraints,
                constraint_vals=constraint_vals,
                rank_policy=rank_policy,
            )
            return beta, W
        # Strict Cholesky on Z'Z using core.linalg helpers for numerical safety
        # NOTE: For 2SLS-equivalent GMM, W = (Z'Z)^{-1}, NOT (Z'Z/n)^{-1}
        ZtZ_dense = la.to_dense(ZtZ)
        Lc = la.safe_cholesky(ZtZ_dense)
        W = la.chol_solve(Lc, la.eye(ZtZ_dense.shape[0]))
        beta = self._gmm_solve(
            X,
            Z,
            y,
            W,
            weights=weights,
            constraints=constraints,
            constraint_vals=constraint_vals,
            rank_policy=rank_policy,
        )
        return beta, W

    def _gmm_solve(  # noqa: PLR0913
        self,
        X: NDArray[np.float64],
        Z: NDArray[np.float64],
        y: NDArray[np.float64],
        W: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
        constraints: np.ndarray | None = None,
        constraint_vals: np.ndarray | None = None,
        rank_policy: str = "stata",
    ) -> NDArray[np.float64]:
        """Solve GMM beta = (X' Z W Z' X)^{-1} X' Z W Z' y, with optional constraints."""
        if weights is not None:
            sw = np.sqrt(weights).reshape(-1, 1)
            Xw = la.hadamard(X, sw)
            Zw = la.hadamard(Z, sw)
            yw = la.hadamard(y, sw)
        else:
            Xw = X
            Zw = Z
            yw = y

        A = la.crossprod(Zw, Xw)
        b = la.crossprod(Zw, yw)

        if W is None:
            A_star = A
            b_star = b
        else:
            Wd = la.to_dense(W)
            Lw = la.safe_cholesky(Wd)
            A_star = la.dot(Lw.T, A)
            b_star = la.dot(Lw.T, b)

        if constraints is not None and constraint_vals is not None:
            return la.to_dense(
                solve_constrained(A_star, b_star, constraints, constraint_vals),
            )

        return la.solve(A_star, b_star, method="qr", rank_policy=rank_policy)

    @staticmethod
    def _solve_kkt(
        G: NDArray[np.float64], g: NDArray[np.float64], R: np.ndarray, q: np.ndarray,
    ) -> NDArray[np.float64]:
        """KKT via Schur complement (SPD route preferred; no '@')."""
        Gd = la.to_dense(G)
        Rd = np.asarray(R, dtype=np.float64)
        qd = np.asarray(q, dtype=np.float64).reshape(-1, 1)

        Lg = la.safe_cholesky(Gd)
        Ginv_RT = la.chol_solve(Lg, Rd.T)
        Ginv_g = la.chol_solve(Lg, g)
        M = la.dot(Rd, Ginv_RT)
        rhs = la.dot(Rd, Ginv_g) - qd
        Lm = la.safe_cholesky(M)
        lam = la.chol_solve(Lm, rhs)
        return Ginv_g - la.dot(Ginv_RT, lam)

    def _moment_covariance(  # noqa: PLR0913
        self,
        Z: NDArray[np.float64],
        u: NDArray[np.float64],
        *,
        n_eff: int,
        cluster_ids: Sequence | None = None,
        multiway_ids: Sequence[Sequence] | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        obs_weights: NDArray[np.float64] | None = None,
        adj: bool = True,
        fixefK: str = "nested",
        cluster_df: str = "min",
        fe_count: int = 0,
        fe_nested_mask: np.ndarray | None = None,
    ) -> NDArray[np.float64]:
        """Estimate S = E[g_i g_i'] with g_i = z_i u_i."""
        return _moment_covariance_zu(
            la.to_dense(Z),
            la.to_dense(u),
            n_eff=n_eff,
            cluster_ids=cluster_ids,
            multiway_ids=multiway_ids,
            space_ids=space_ids,
            time_ids=time_ids,
            obs_weights=obs_weights,
            adj=adj,
            n_features=int(self.n_features),
            fixefK=fixefK,
            cluster_df=cluster_df,
            fe_count=fe_count,
            fe_nested_mask=fe_nested_mask,
        )

    def moment_covariance(  # noqa: PLR0913
        self,
        Z: NDArray[np.float64],
        u: NDArray[np.float64],
        *,
        n_eff: int,
        cluster_ids: Sequence | None = None,
        multiway_ids: Sequence[Sequence] | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        obs_weights: NDArray[np.float64] | None = None,
        adj: bool = True,
        fixefK: str = "nested",
        cluster_df: str = "min",
        fe_count: int = 0,
        fe_nested_mask: np.ndarray | None = None,
    ) -> NDArray[np.float64]:
        """Public wrapper for :meth:`_moment_covariance`."""
        return self._moment_covariance(
            Z,
            u,
            n_eff=n_eff,
            cluster_ids=cluster_ids,
            multiway_ids=multiway_ids,
            space_ids=space_ids,
            time_ids=time_ids,
            obs_weights=obs_weights,
            adj=adj,
            fixefK=fixefK,
            cluster_df=cluster_df,
            fe_count=fe_count,
            fe_nested_mask=fe_nested_mask,
        )

    def _weak_iv_first_stage_stats(
        self,
        X: NDArray[np.float64],
        Z: NDArray[np.float64],
        *,
        endog_names: Sequence[str] | None = None,
        exog_names: Sequence[str] | None = None,
        excluded_instr_names: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Values-only SW partial F diagnostic (minimum across endogenous)."""
        n = X.shape[0]
        x_names = self._x_names
        z_names = self._z_names

        if exog_names is not None:
            included_exog = list(exog_names)
        else:
            included_exog = sorted(set(x_names) & set(z_names))
        if endog_names is not None:
            endogenous = list(endog_names)
        else:
            endogenous = [nm for nm in x_names if nm not in included_exog]
        if excluded_instr_names is not None:
            excluded_instr = list(excluded_instr_names)
        else:
            excluded_instr = [nm for nm in z_names if nm not in included_exog]

        if not endogenous or not excluded_instr:
            return {"min_partial_F": np.nan}

        x_idx = {nm: j for j, nm in enumerate(x_names)}
        z_idx = {nm: j for j, nm in enumerate(z_names)}
        exog_cols = [x_idx[nm] for nm in included_exog if nm in x_idx]
        z2_cols = [z_idx[nm] for nm in excluded_instr if nm in z_idx]
        K1 = len(exog_cols)
        L2 = len(z2_cols)
        if L2 == 0 or n - 1 <= K1:
            return {"min_partial_F": np.nan}

        def _ssr(y: np.ndarray, Xmat: np.ndarray) -> float:
            try:
                b = la.solve(Xmat, y, method="qr")
            except (np.linalg.LinAlgError, ValueError):
                b = la.solve(Xmat, y, method="svd")
            r = y - la.dot(Xmat, b)
            return float(la.to_dense(la.crossprod(r, r)).squeeze())

        minF = np.inf
        for nm in endogenous:
            if nm not in x_idx:
                continue
            y1 = X[:, [x_idx[nm]]]
            X0 = X[:, exog_cols] if K1 > 0 else np.zeros((n, 0), dtype=np.float64)
            Z2 = Z[:, z2_cols]
            X1 = la.hstack([X0, Z2]) if K1 > 0 else Z2
            ssr_u = _ssr(y1, X1)
            ssr_r = (
                _ssr(y1, X0)
                if K1 > 0
                else float(la.to_dense(la.crossprod(y1, y1)).squeeze())
            )
            df_u = n - (K1 + L2)
            if df_u <= 0:
                continue
            # Sanderson-Windmeijer partial F: numerator df is exactly L2
            F = ((ssr_r - ssr_u) / L2) / (ssr_u / df_u)
            minF = min(minF, F)
        return {"min_partial_F": float(minF) if np.isfinite(minF) else np.nan}

    # _mop_effective_f is defined once above; no duplicate definition retained.

    def _sw_partial_f_list(
        self,
        X: NDArray[np.float64],
        Z: NDArray[np.float64],
        *,
        endog_names: Sequence[str] | None = None,
        exog_names: Sequence[str] | None = None,
        excluded_instr_names: Sequence[str] | None = None,
    ) -> list[tuple[str, float, str | None]]:
        """Sanderson-Windmeijer partial F statistics for each endogenous variable."""
        x_names = self._x_names
        z_names = self._z_names
        if exog_names is not None:
            included_exog = list(exog_names)
        else:
            included_exog = sorted(set(x_names) & set(z_names))
        if endog_names is not None:
            endogenous = list(endog_names)
        else:
            endogenous = [nm for nm in x_names if nm not in included_exog]
        if excluded_instr_names is not None:
            excluded_instr = list(excluded_instr_names)
        else:
            excluded_instr = [nm for nm in z_names if nm not in included_exog]

        if not endogenous:
            return []

        x_idx = {nm: j for j, nm in enumerate(x_names)}
        z_idx = {nm: j for j, nm in enumerate(z_names)}
        exog_cols = [x_idx[nm] for nm in included_exog if nm in x_idx]
        z2_cols = [z_idx[nm] for nm in excluded_instr if nm in z_idx]
        K1 = len(exog_cols)
        L2 = len(z2_cols)
        if L2 == 0:
            return [(nm, float("nan"), "No excluded instruments") for nm in endogenous]

        def _ssr(y: np.ndarray, Xmat: np.ndarray) -> float:
            try:
                b = la.solve(Xmat, y, method="qr")
            except (np.linalg.LinAlgError, ValueError):
                b = la.solve(Xmat, y, method="svd")
            r = y - la.dot(Xmat, b)
            return float(la.to_dense(la.crossprod(r, r)).squeeze())

        f_list: list[tuple[str, float, str | None]] = []
        for nm in endogenous:
            if nm not in x_idx:
                f_list.append((nm, float("nan"), "Not in design matrix"))
                continue
            y1 = X[:, [x_idx[nm]]]
            X0 = (
                X[:, exog_cols]
                if K1 > 0
                else np.zeros((X.shape[0], 0), dtype=np.float64)
            )
            Z2 = Z[:, z2_cols]
            X1 = la.hstack([X0, Z2]) if K1 > 0 else Z2
            ssr_u = _ssr(y1, X1)
            ssr_r = (
                _ssr(y1, X0) if K1 > 0 else float(la.to_dense(la.tdot(y1)).squeeze())
            )
            # Sanderson-Windmeijer F: [(SSR_r - SSR_u) / L2] / [SSR_u / (n - K1 - L2)]
            # where L2 = number of excluded instruments (numerator df matches R ivreg)
            df_u = X.shape[0] - (K1 + L2)
            if df_u <= 0:
                f_list.append((nm, float("nan"), "Insufficient df"))
                continue
            # Numerator df is exactly L2 (not max(L2,1); R ivreg/Stata ivregress convention)
            F = ((ssr_r - ssr_u) / L2) / (ssr_u / df_u)
            f_list.append((nm, float(F), None))
        return f_list

    def _residualize(
        self, Y: NDArray[np.float64], X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Residualize Y on X."""
        if X.shape[1] == 0:
            return Y
        # Numerically stable residualization via QR-first solve (avoid X'X)
        beta = la.solve(X, Y, method="qr")
        return Y - la.dot(X, beta)

    # ------------------------------------------------------------------
