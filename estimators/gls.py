"""Generalized Least Squares (GLS) estimator.

This module implements GLS, Feasible GLS (FGLS), and Iterated FGLS algorithms
handling autocorrelation (AR(1), PSAR(1)) and heteroskedasticity.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from lineareg.core import bootstrap as bt
from lineareg.core import linalg as la
from lineareg.core.bootstrap import _normalize_ssc, compute_ssc_correction
from lineareg.estimators.base import (
    BaseEstimator,
    BootConfig,
    EstimationResult,
    attach_formula_metadata,
    prepare_formula_environment,
)
from lineareg.utils.auto_constant import add_constant
from lineareg.utils.constraints import solve_constrained
from lineareg.utils.formula import FormulaParser

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

ArrayLike = pd.Series | np.ndarray
MatrixLike = pd.DataFrame | np.ndarray

__all__ = ["GLS"]


class GLS(BaseEstimator):
    """Generalized Least Squares estimator.

    Supports diagonal weighting and AR(1)/PSAR(1) correlation.
    Inference is via wild bootstrap.
    """

    def __init__(
        self,
        y: ArrayLike,
        X: MatrixLike,
        *,
        add_const: bool = True,
        var_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if var_names is None and isinstance(X, pd.DataFrame):
            var_names = list(X.columns)
        X_arr = np.asarray(X, dtype=np.float64, order="C")
        if add_const:
            X_aug, names_out, _ = add_constant(X_arr, var_names)
            self._var_names = list(names_out)
        else:
            X_aug = X_arr
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
        self._n_obs_init, self._n_features_init = self.X_orig.shape

    @classmethod
    def from_formula(  # noqa: PLR0913
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        id_name: str | None = None,
        time_name: str | None = None,
        options: str | None = None,
        W_dict: dict[str, object] | None = None,
        boot: BootConfig | None = None,
    ) -> GLS:
        """Initialize GLS model from formula."""
        parser = FormulaParser(data, id_name=id_name, t_name=time_name, W_dict=W_dict)
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
            parsed["y"], parsed["X"], add_const=bool(parsed.get("include_intercept", True)), var_names=parsed["var_names"],
        )
        attach_formula_metadata(model, meta)
        return model

    def fit(  # noqa: PLR0913
        self,
        *,
        # R/Stata-style compatibility aliases
        corr: str | None = None,
        cluster_ids: Sequence | None = None,
        omega: str | np.ndarray | None = None,
        weights: Sequence[float] | None = None,
        time_ids: Sequence | None = None,
        series_ids: Sequence | None = None,
        pw_mode: str = "iterated",  # {"iterated","two-step"}
        # Stata Prais defaults: tol ~ 1e-3, maxiter 100 for parity with prais/xtgls
        pw_tol: float = 1e-3,
        pw_max_iter: int = 100,
        device: str | None = None,
        rho_method: str = "dw",
        # ---- Constraint support (GLS constrained estimation) ----
        constraints: np.ndarray | None = None,
        constraint_vals: np.ndarray | None = None,
        # NOTE: only QR solver is supported when solving transformed GLS (keep parity with R/Stata)
        method: str = "qr",
        # Stata xtgls 'force' analogue: allow irregular spacing by using time order only
        force_irregular: bool = False,
        # If True, require exact equal spacing within series for xtgls (Stata-like strictness)
        strict_xtgls: bool = True,
        # If True, re-estimate rho inside each bootstrap replication (Stata-like)
        boot_reestimate_rho: bool = True,
        # Transformation for AR1 head handling: 'prais' scales first obs, 'corc' drops first obs
        ar1_transform: str = "prais",
        rank_policy: str | None = None,
        boot: BootConfig | None = None,
        boot_cluster: Sequence | None = None,
        ssc: dict[str, str | int | float | bool] | None = None,
    ) -> EstimationResult:

        """Fit GLS model.

        Perform estimation with optional weights and correlation structure.
        """

        # ----- R/Stata corr() alias mapping -----
        if corr is not None:
            c = str(corr).strip().lower()
            if c in {"independent", "none"}:
                omega = None
            elif c == "ar1":
                omega = "AR1"
            elif c == "psar1":
                omega = "PSAR1"
            elif c in {"exchangeable", "exch", "equicorrelated"}:
                omega = "EXCH"
            else:
                raise ValueError(
                    "corr() supports: 'independent'/'none', 'ar1', 'psar1', 'exchangeable'.",
                )

        # Rank policy selection for QR rank-detection and solver tolerances.
        # Default: strict Stata parity unless caller requests 'R'. This value
        # is used throughout GLS transforms and bootstrap re-estimation.
        rp = "stata" if rank_policy is None else str(rank_policy).lower()
        if rp not in {"stata", "r"}:
            raise ValueError("rank_policy must be one of {'stata','r',None}.")

        # Strict pre-filtering: drop any row with non-finite X or y
        finite_mask = np.isfinite(self.y_orig).reshape(-1) & np.all(
            np.isfinite(self.X_orig), axis=1,
        )
        X = self.X_orig[finite_mask]
        y = self.y_orig[finite_mask]
        n_proc = y.shape[0]

        # Mutual exclusivity: full Omega (user-provided full covariance) cannot be combined
        # with analytic diagonal `weights`. However, correlation structures like "AR1"
        # may be combined with diagonal variance weights (R nlme::gls semantics).
        if isinstance(omega, np.ndarray) and (weights is not None):
            raise ValueError(
                "Full `omega` (covariance matrix) cannot be combined with `weights` "
                "(use correlation structure + variance weights, or provide full Omega alone).",
            )

        # defaults (transformed design)
        X_t = X
        y_t = y
        transform: tuple[str, object | None] = ("none", None)

        # Prais-Winsten transform is implemented as a class-private helper
        # `self._prais_winsten_by_segments` to allow reuse by bootstrap reestimate
        # helpers defined below. See that implementation for exact semantics.

        # 1) Full Omega provided as ndarray
        if isinstance(omega, np.ndarray):
            Omega = np.asarray(omega, dtype=np.float64)
            if Omega.shape != (self._n_obs_init, self._n_obs_init):
                raise ValueError(
                    "omega must be shape (n_obs, n_obs) matching original sample",
                )
            # Strict symmetry requirement (do not symmetrize silently)
            if not np.allclose(Omega, Omega.T, rtol=0, atol=1e-12):
                raise ValueError(
                    "omega must be symmetric (no symmetrization performed).",
                )
            Omega_m = Omega[np.ix_(finite_mask, finite_mask)]
            L = la.safe_cholesky(Omega_m)
            if L is None:
                raise RuntimeError(
                    "Provided Omega is not positive-definite after masking.",
                )
            X_t = la.triangular_solve(L, X)
            y_t = la.triangular_solve(L, y)
            transform = ("full", None)

        # 2) AR(1) Prais-Winsten (single-pass estimate of rho)
        elif isinstance(omega, str) and omega.upper() in {"AR1", "PSAR1"}:
            psar1 = omega.upper() == "PSAR1"
            if time_ids is None:
                raise ValueError(
                    "AR1 requires time_ids to determine ordering for Prais-Winsten.",
                )
            # Order by series (if provided) then time to preserve panel segments.
            # If `series_ids` is None, treat all observations as the same single series
            # (fixes previous bug where each observation was its own series).
            t = np.asarray(time_ids)[finite_mask]
            s = (
                np.zeros_like(t)
                if series_ids is None
                else np.asarray(series_ids)[finite_mask]
            )
            # Store filtered time/series ids for bootstrap reestimation
            time_ids_filtered = t
            series_ids_filtered = s
            # lexsort keys: (t, s) with s as primary key (last key in lexsort tuple)
            order = np.lexsort((t, s))  # primary: s, secondary: t
            Xs = X[order]
            ys = y[order]
            ts = t[order]
            ss = s[order]
            # ---- STRICT (default): enforce constant spacing per series (no heuristics) ----
            # For each series, the set of positive time differences must be a single value (delta_g).
            # Otherwise AR(1) with equal spacing is ill-defined -> raise.
            # If force_irregular=True, relax spacing checks and treat adjacency by ordering
            # (series-only adjacency), matching xtgls 'force' semantics.
            d: float = 1.0  # default: unit spacing (overwritten if constant Δ≠1)
            seg_break = np.ones(ys.shape[0], dtype=bool)
            if ys.shape[0] > 1:
                if force_irregular:
                    # Only break when the series id changes (ignore time gaps)
                    seg_break[1:] = ss[1:] != ss[:-1]
                else:
                    # compute per-series deltas using contiguous blocks of series ids in sorted order
                    delta_by_series: dict = {}
                    starts = np.where(np.r_[True, ss[1:] != ss[:-1]])[0]
                    ends = np.r_[starts[1:], ss.shape[0]]
                    for a, b in zip(starts, ends):
                        if b - a <= 1:
                            # single observation block: nothing to check
                            continue
                        diffs = np.diff(ts[a:b])
                        pos = diffs[diffs > 0]
                        uniq = np.unique(pos)
                        if uniq.size == 0:
                            # all non-positive diffs (shouldn't happen for valid time ordering)
                            continue
                        if uniq.size != 1:
                            # If strict_xtgls is requested, fail hard when spacing is irregular.
                            if strict_xtgls and (not force_irregular):
                                raise ValueError(
                                    "xtgls exact replication requires equally spaced time within each series when strict_xtgls=True.",
                                )
                            # Otherwise, fall back to the existing behavior (raise) for clarity
                            raise ValueError(
                                "AR1 requires constant time spacing within each series.",
                            )
                        delta_by_series[ss[a]] = float(uniq[0])
                    # Compute common d if all series share the same spacing
                    if len(delta_by_series) > 0:
                        deltas = np.array(list(delta_by_series.values()), dtype=float)
                        if np.allclose(deltas, deltas[0]) and not np.isclose(
                            deltas[0], 1.0,
                        ):
                            d = float(deltas[0])
                    # contiguous adjacency: within-series and delta matches the per-series delta
                    same_series = ss[1:] == ss[:-1]
                    inside = np.zeros(ys.shape[0] - 1, dtype=bool)
                    if same_series.any():
                        idxs = np.where(same_series)[0]
                        expected = np.array(
                            [delta_by_series.get(ss[i], np.nan) for i in idxs],
                            dtype=float,
                        )
                        inside[idxs] = np.isclose(
                            ts[1:][idxs] - ts[:-1][idxs], expected, equal_nan=False,
                        )
                    seg_break[1:] = (~inside) | (ss[1:] != ss[:-1])
            seg_id = np.cumsum(seg_break) - 1

            # --- STRICT: compute residuals on ORIGINAL scale for DW/OLS ---
            # Even if beta_ols is obtained from weighted regression, residuals used
            # for initial rho (DW or OLS/Yule–Walker) must be computed on the original
            # (unweighted) scale for strict Stata/R parity. This matches Stata prais
            # "Durbin-Watson statistic (original)" and EViews developer guidance.
            if weights is not None:
                w_full = np.asarray(weights, dtype=np.float64).reshape(-1)
                if w_full.shape[0] != self._n_obs_init:
                    raise ValueError("weights length must match original n_obs")
                w_sorted = w_full[finite_mask][order].reshape(-1, 1)
                sqrtw = np.sqrt(w_sorted)
                Xw = la.hadamard(Xs, sqrtw)
                yw = la.hadamard(ys, sqrtw)
                beta_ols = la.solve(
                    Xw, yw, method="qr", rank_policy=("R" if rp == "r" else "stata"),
                )
            else:
                beta_ols = la.solve(
                    Xs, ys, method="qr", rank_policy=("R" if rp == "r" else "stata"),
                )
            # residuals on original scale (critical for DW/YW adjacency estimators)
            e_ols = (ys - la.dot(Xs, beta_ols)).ravel()
            valid = seg_id[1:] == seg_id[:-1]
            rho = 0.0
            if e_ols.size >= 2 and np.any(valid):
                u0 = e_ols[:-1][valid]
                u1 = e_ols[1:][valid]
                # Strict: accept only canonical names {'dw','ols'}
                method_raw = rho_method.lower() if isinstance(rho_method, str) else "dw"
                if method_raw not in {"dw", "ols"}:
                    raise ValueError("rho_method must be one of {'dw','ols'}.")
                method = method_raw
                if method == "ols":
                    num = float(la.dot(u0.reshape(-1, 1).T, u1.reshape(-1, 1)))
                    den = float(la.dot(u0.reshape(-1, 1).T, u0.reshape(-1, 1)))
                    rho = 0.0 if den <= 0 else num / den
                elif method == "dw":
                    # rho ≈ 1 - DW/2 with DW = Σ_{t≥2} (e_t - e_{t-1})^2 / Σ_{t=1}^T e_t^2
                    num = float(np.sum((u1 - u0) ** 2))
                    den = float(np.sum(e_ols**2))
                    rho = 0.0 if den <= 0 else 1.0 - (num / den) / 2.0
                else:
                    raise ValueError(
                        "rho_method must be one of {'dw','durbin-watson','ols','regress'}",
                    )
                rho = float(np.clip(rho, -0.999999, 0.999999))

            # Check spacing within series: if constant but !=1, xtgls semantics require caution.
            for g in np.unique(ss):
                idx = np.where(ss == g)[0]
                if idx.size >= 3:
                    dif = np.diff(ts[idx])
                    uniq = np.unique(dif)
                    if uniq.size == 1 and not np.isclose(uniq[0], 1.0):
                        if strict_xtgls:
                            raise ValueError(
                                "AR1 requires constant unit spacing within each series (strict_xtgls=True).",
                            )

            if pw_mode.lower() in {"two", "two-step", "twostep"}:
                # Two-step: compute rho on full sorted data, apply transform, then restore
                # If diagonal weights provided, compose Σ = Λ^{1/2} R(ρ) Λ^{1/2} with Λ=diag(1/w_i).
                if psar1:
                    # Two-step PSAR1: estimate series-specific rho, then apply one-shot transform.
                    rho_by_series = self._estimate_rho_by_series(
                        Xs,
                        ys,
                        ss,
                        ts,
                        seg_id,
                        weights=(w_full[finite_mask][order] if weights is not None else None),
                        method=rho_method,
                        _rank_policy_internal=rp,
                    )
                    # Clip for PD safety
                    rho_by_series = {
                        k: float(np.clip(v, -0.999999, 0.999999))
                        for k, v in rho_by_series.items()
                    }
                    if weights is not None:
                        w_full = np.asarray(weights, dtype=np.float64).reshape(-1)
                        if w_full.shape[0] != self._n_obs_init:
                            raise ValueError("weights length must match original n_obs")
                        w = w_full[finite_mask][order]
                        if np.any(~np.isfinite(w)) or np.any(w <= 0):
                            raise ValueError("weights must be positive and finite")
                        Sigma = self._psar1_cov_from_weights(ts, ss, seg_id, rho_by_series, w)
                        L = la.safe_cholesky(Sigma)
                        if L is None:
                            raise RuntimeError("Σ not PD for PSAR1+weights; check data/weights.")
                        Xpw = la.triangular_solve(L, Xs)
                        ypw = la.triangular_solve(L, ys)
                        inv = np.empty_like(order)
                        inv[order] = np.arange(order.size)
                        X_t = Xpw[inv]
                        y_t = ypw[inv]
                    else:
                        Xp, yp = self._prais_winsten_psar1(
                            Xs, ys, ss, seg_id, rho_by_series, transform=ar1_transform,
                        )
                        if ar1_transform == "corc":
                            X_t, y_t = Xp, yp
                        else:
                            inv = np.empty_like(order)
                            inv[order] = np.arange(order.size)
                            X_t = Xp[inv]
                            y_t = yp[inv]
                    transform = ("pw_two_step_psar1", {"rho": rho_by_series})
                elif weights is not None:
                    # Inverse-variance weights: w_i = 1 / Var(ε_i).
                    # Exact nlme::gls semantics (no approximation):
                    #   Σ = Λ^{-1/2} R(ρ) Λ^{-1/2},  Λ = diag(w_i).
                    # We factor Σ via Cholesky and apply Σ^{-1/2} exactly (whitening).
                    w_full = np.asarray(weights, dtype=np.float64).reshape(-1)
                    if w_full.shape[0] != self._n_obs_init:
                        raise ValueError("weights length must match original n_obs")
                    w = w_full[finite_mask][order]
                    if np.any(~np.isfinite(w)) or np.any(w <= 0):
                        raise ValueError("weights must be positive and finite")
                    # Use central helper to assemble AR1+weights covariance
                    # ar1_covariance_from_weights_by_groups uses actual time differences
                    # in the matrix: R_ij = rho^|t_i - t_j|.  rho is the per-unit-time
                    # autocorrelation.  If panel spacing is Δ≠1 and rho was estimated
                    # from adjacent residuals, we need the Δ-th root.
                    rho = float(np.clip(rho, -0.999999, 0.999999))
                    if d != 1 and rho != 0:
                        rho = np.sign(rho) * (abs(rho) ** (1.0 / d))
                    Sigma = la.ar1_covariance_from_weights_by_groups(
                        ts, seg_id, rho=rho, weights=w,
                    )
                    L = la.safe_cholesky(Sigma)
                    if L is None:
                        raise RuntimeError(
                            "Σ not PD for AR1+weights; check data/weights.",
                        )
                    Xpw = la.triangular_solve(L, Xs)
                    ypw = la.triangular_solve(L, ys)
                    inv = np.empty_like(order)
                    inv[order] = np.arange(order.size)
                    X_t = Xpw[inv]
                    y_t = ypw[inv]
                else:
                    # Unweighted Prais-Winsten operates on observation adjacency.
                    # If time spacing is constant but != 1 and strict_xtgls is disabled,
                    # treat the estimated rho as the adjacency correlation between
                    # successive observed points (no Δ-th root adjustment here).
                    tau = rho
                    Xp, yp = self._prais_winsten_by_segments(
                        Xs, ys, seg_id, tau, transform=ar1_transform,
                    )
                    if ar1_transform == "corc":
                        # Cochrane-Orcutt drops first obs in each segment; do not attempt
                        # to inverse-reorder dropped rows (they no longer exist).
                        X_t, y_t = Xp, yp
                    else:
                        inv = np.empty_like(order)
                        inv[order] = np.arange(order.size)
                        X_t = Xp[inv]
                        y_t = yp[inv]
                transform = ("pw_two_step", float(rho))
            else:
                # Iterated Prais-Winsten until convergence (Stata-like defaults)
                # For PSAR1, rho_old will be a dict {series_id: rho_value}; for AR1 it's scalar
                rho_old = (
                    rho
                    if not psar1
                    else self._estimate_rho_by_series(
                        Xs,
                        ys,
                        ss,
                        ts,
                        seg_id,
                        weights=(
                            w_full[finite_mask][order] if weights is not None else None
                        ),
                        method=rho_method,
                        _rank_policy_internal=rp,
                    )
                )
                it = 0
                while it < pw_max_iter:
                    if weights is not None:
                        # iterate with Sigma(rho_old) whitening within segments
                        w_full = np.asarray(weights, dtype=np.float64).reshape(-1)
                        w = w_full[finite_mask][order]
                        # For PSAR1 with weights, build segment-specific block-diagonal Σ
                        if psar1:
                            # --- STRICT implementation: series-specific block-diagonal Σ ---
                            # Σ = blockdiag_g{ D_g R_g(ρ_g) D_g }, D_g = diag(1/sqrt(w_i))
                            # Ensure per-series rho values are clipped for PD
                            rho_old_clipped = (
                                {
                                    k: float(np.clip(v, -0.999999, 0.999999))
                                    for k, v in rho_old.items()
                                }
                                if isinstance(rho_old, dict)
                                else rho_old
                            )
                            Sigma = self._psar1_cov_from_weights(
                                ts,
                                ss,
                                seg_id,
                                rho_old_clipped,
                                w,
                            )
                        else:
                            # Assemble covariance for current rho via centralized helper
                            # Apply Δ-th root if spacing != 1 (rho was estimated from
                            # adjacent observations but matrix uses actual time diffs)
                            rho_old_eff = (
                                float(np.clip(rho_old, -0.999999, 0.999999))
                                if not isinstance(rho_old, dict)
                                else rho_old
                            )
                            if d != 1 and rho_old_eff != 0:
                                rho_old_eff = np.sign(rho_old_eff) * (
                                    abs(rho_old_eff) ** (1.0 / d)
                                )
                            Sigma = la.ar1_covariance_from_weights_by_groups(
                                ts,
                                seg_id,
                                rho=rho_old_eff,
                                weights=w,
                            )
                        L = la.safe_cholesky(Sigma)
                        if L is None:
                            raise RuntimeError("Sigma not PD during PW iteration.")
                        X_tmp_sorted = la.triangular_solve(L, Xs)
                        y_tmp_sorted = la.triangular_solve(L, ys)
                        inv = np.empty_like(order)
                        inv[order] = np.arange(order.size)
                        X_tmp = X_tmp_sorted[inv]
                        y_tmp = y_tmp_sorted[inv]
                        # compute beta on transformed (weighted) design every iteration
                        try:
                            beta = la.solve(
                                X_tmp,
                                y_tmp,
                                method="qr",
                                rank_policy=("R" if rp == "r" else "stata"),
                            )
                        except TypeError:
                            beta = la.solve(X_tmp, y_tmp, method="qr")
                    else:
                        # For PSAR1 we need to apply series-specific PW scaling
                        if psar1:
                            # PSAR1 with no weights: need to transform each rho_g
                            # if Δ != 1 (rho_old is adjacency correlation)
                            if d != 1:
                                rho_old_adj = {
                                    k: np.sign(v) * (abs(v) ** (1.0 / d))
                                    for k, v in rho_old.items()
                                }
                            else:
                                rho_old_adj = rho_old
                            X_tmp_sorted, y_tmp_sorted = self._prais_winsten_psar1(
                                Xs, ys, ss, seg_id, rho_old_adj, transform=ar1_transform,
                            )
                        else:
                            # AR1 with no weights: transform rho for Δ != 1
                            rho_pw = rho_old
                            if d != 1 and rho_old != 0:
                                rho_pw = np.sign(rho_old) * (abs(rho_old) ** (1.0 / d))
                            X_tmp_sorted, y_tmp_sorted = (
                                self._prais_winsten_by_segments(
                                    Xs, ys, seg_id, rho_pw, transform=ar1_transform,
                                )
                            )
                        # restore to original masked order before solving so solver sees original order
                        inv = np.empty_like(order)
                        inv[order] = np.arange(order.size)
                        X_tmp = X_tmp_sorted[inv]
                        y_tmp = y_tmp_sorted[inv]

                        beta = la.solve(
                            X_tmp,
                            y_tmp,
                            method="qr",
                            rank_policy=("R" if rp == "r" else "stata"),
                        )
                    # rho update must use original-scale residuals (sorted) to match
                    # Stata prais/xtgls 'Durbin-Watson (original)' semantics. Use
                    # the sorted original design Xs/ys so adjacency calculations
                    # (Yule–Walker / DW) operate on the original scale.
                    e = (ys - la.dot(Xs, beta)).ravel()

                    # Update rho for next iteration
                    if psar1:
                        # Re-estimate series-specific rho from the current residuals.
                        # NOTE: Xs/ys/ss/seg_id are in sorted order; e is aligned with them.
                        rho_new = self._rho_by_series_from_resid(
                            e,
                            ss,
                            seg_id,
                            method=rho_method,
                        )
                        # Check convergence: all series rho values must have converged
                        converged = True
                        for sid in rho_old:
                            if (
                                sid not in rho_new
                                or abs(rho_new[sid] - rho_old[sid]) > pw_tol
                            ):
                                converged = False
                                break
                        if converged:
                            rho_old = rho_new
                            break
                        rho_old = rho_new
                    else:
                        # compute Yule-Walker on adjacent valid pairs *in sorted order*
                        # re-create sorted residuals for adjacency calculation
                        e_sorted = e[order]
                        if e_sorted.size >= 2 and np.any(valid):
                            e_lag = e_sorted[:-1][valid].reshape(-1, 1)
                            e_lead = e_sorted[1:][valid].reshape(-1, 1)
                            # Convert 1x1 arrays to Python scalars explicitly to avoid NumPy deprecation warnings
                            num = float(la.dot(e_lag.T, e_lead).squeeze())
                            den = float(la.dot(e_lag.T, e_lag).squeeze())
                            rho_new = 0.0 if den <= 0 else num / den
                        else:
                            rho_new = 0.0
                        rho_new = float(np.clip(rho_new, -0.999999, 0.999999))
                        if abs(rho_new - rho_old) <= pw_tol:
                            rho_old = rho_new
                            break
                        rho_old = rho_new
                    it += 1
                # After convergence, apply final transform on sorted data and restore order
                if weights is not None:
                    # Final whitening at convergence.
                    w_full = np.asarray(weights, dtype=np.float64).reshape(-1)
                    w = w_full[finite_mask][order]
                    if psar1:
                        rho_old_clipped = {
                            k: float(np.clip(v, -0.999999, 0.999999))
                            for k, v in rho_old.items()
                        }
                        Sigma = self._psar1_cov_from_weights(ts, ss, seg_id, rho_old_clipped, w)
                    else:
                        rho_eff = float(np.clip(rho_old, -0.999999, 0.999999))
                        if d != 1 and rho_eff != 0:
                            rho_eff = np.sign(rho_eff) * (abs(rho_eff) ** (1.0 / d))
                        Sigma = la.ar1_covariance_from_weights_by_groups(
                            ts,
                            seg_id,
                            rho=rho_eff,
                            weights=w,
                        )
                    L = la.safe_cholesky(Sigma)
                    if L is None:
                        raise RuntimeError("Σ not PD at convergence.")
                    Xp = la.triangular_solve(L, Xs)
                    yp = la.triangular_solve(L, ys)
                elif psar1:
                    # Unweighted PSAR1 transform: adjacency rho per series (no Δ-th root adjustment).
                    Xp, yp = self._prais_winsten_psar1(
                        Xs, ys, ss, seg_id, rho_old, transform=ar1_transform,
                    )
                else:
                    # Unweighted AR1 transform: adjacency rho (no Δ-th root adjustment).
                    Xp, yp = self._prais_winsten_by_segments(
                        Xs, ys, seg_id, float(rho_old), transform=ar1_transform,
                    )
                if ar1_transform == "corc":
                    X_t, y_t = Xp, yp
                else:
                    inv = np.empty_like(order)
                    inv[order] = np.arange(order.size)
                    X_t = Xp[inv]
                    y_t = yp[inv]
                # For PSAR1, rho_old is a dict; for AR1, it's a scalar
                transform = (
                    "pw_iterated",
                    {
                        "rho": (
                            rho_old if isinstance(rho_old, dict) else float(rho_old)
                        ),
                        "iterations": int(it),
                    },
                )

        # 3) Cluster-EXCHANGEABLE correlation (equicorrelated within clusters)
        elif isinstance(omega, str) and omega.upper() in {"EXCH", "EXCHANGEABLE"}:
            # Select cluster ids: explicit cluster_ids first, else fall back to series_ids
            if (cluster_ids is None) and (series_ids is None):
                raise ValueError(
                    "corr(exchangeable) requires cluster_ids= (or series_ids= as fallback).",
                )
            cl_full = np.asarray(
                cluster_ids if (cluster_ids is not None) else series_ids,
            )
            if cl_full.shape[0] != self._n_obs_init:
                raise ValueError("cluster_ids length must match original n_obs")
            cl = cl_full[finite_mask]
            # For bootstrap re-estimation, if boot_cluster isn't explicitly
            # provided, fix it deterministically to the original cluster ids
            # so bootstrap re-estimation uses the same clustering granularity.
            if boot_cluster is None and (
                boot is None or getattr(boot, "cluster_ids", None) is None
            ):
                boot_cluster = cl_full

            # Initial residuals for rho (OLS or weighted OLS) computed on original scale
            if weights is not None:
                w_all = np.asarray(weights, dtype=np.float64).reshape(-1)
                if w_all.shape[0] != self._n_obs_init:
                    raise ValueError("weights length must match original n_obs")
                w = w_all[finite_mask].reshape(-1, 1)
                if np.any(~np.isfinite(w)) or np.any(w <= 0):
                    raise ValueError("weights must be positive and finite")
                Xw = la.hadamard(X, np.sqrt(w))
                yw = la.hadamard(y.reshape(-1, 1), np.sqrt(w))
                beta0 = la.solve(Xw, yw, method="qr", rank_policy="stata")
                e0 = yw.reshape(-1) - la.dot(Xw, beta0).reshape(-1)
            else:
                beta0 = la.solve(X, y.reshape(-1, 1), method="qr", rank_policy="stata")
                e0 = y.reshape(-1) - la.dot(X, beta0).reshape(-1)

            # Two-step or iterated ICC-type MoM estimate of rho
            if pw_mode.lower() in {"two", "two-step", "twostep"}:
                # Use residuals on the original (unweighted) scale for rho,
                # matching Stata/nlme parity (scale-free ICC/MoM).
                e0_orig = y.reshape(-1) - la.dot(X, beta0).reshape(-1)
                rho = self._exch_rho_from_resid(e0_orig, cl)
            elif str(pw_mode).strip().lower() == "iterated":
                rho_old = self._exch_rho_from_resid(e0, cl)
                for _ in range(int(pw_max_iter)):
                    X_t_tmp, y_t_tmp = self._exchangeable_transform(
                        X, y.reshape(-1, 1), cl, rho_old, weights=weights,
                    )
                    beta_it = la.solve(
                        X_t_tmp, y_t_tmp, method="qr", rank_policy="stata",
                    )
                    # EXCH: ρ is defined on the original scale. Always recompute residuals on
                    # the original (untransformed) space: e = y - X β_it. No shortcuts.
                    e_it = y.reshape(-1) - la.dot(X, beta_it).reshape(-1)
                    rho_new = self._exch_rho_from_resid(e_it, cl)
                    if abs(rho_new - rho_old) <= float(pw_tol):
                        rho_old = rho_new
                        break
                    rho_old = rho_new
                rho = float(rho_old)
            else:
                raise ValueError("pw_mode must be 'two-step' or 'iterated'")

            # Final transform with estimated rho
            X_t, y_t = self._exchangeable_transform(
                X, y.reshape(-1, 1), cl, rho, weights=weights,
            )
            y_t = y_t.reshape(-1)
            transform = (
                "exch_iterated" if pw_mode == "iterated" else "exch_two-step",
                {"rho": float(rho)},
            )

        # 4) Analytic diagonal weights (WLS)
        elif weights is not None:
            w_full = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w_full.shape[0] != self._n_obs_init:
                raise ValueError("weights length must match original n_obs")
            w = w_full[finite_mask]
            if np.any(~np.isfinite(w)) or np.any(w <= 0):
                raise ValueError("weights must be positive and finite")
            # If AR1 correlation structure was requested earlier, we should have
            # handled the combined AR1+diag case in the omega=="AR1" branch by
            # constructing Σ = Λ^{-1/2} R(ρ) Λ^{-1/2} (with Λ = diag(w_i)).
            # Here handle pure diagonal WLS (no correlation structure).
            sqrtw = np.sqrt(w).reshape(-1, 1)
            # Use core.linalg.hadamard to preserve sparse/dtype semantics
            X_t = la.hadamard(X, sqrtw)
            # y is 1-d; ensure consistent shape then apply hadamard and flatten
            y_t = la.hadamard(y.reshape(-1, 1), sqrtw).reshape(-1)
            transform = ("diag", w)

        # ---- Solve on transformed data (after any transform) ----
        # Rank policy selection: default is strict Stata parity unless caller
        # supplied rank_policy='R'. This controls both column-drop and solver
        # tolerances so that users can reproduce either Stata (Mata) or R lm.fit
        # behavior deterministically.
        rp = "stata" if rank_policy is None else str(rank_policy).lower()
        if rp not in {"stata", "r"}:
            raise ValueError("rank_policy must be one of {'stata','r',None}.")
        # Remove zero-variance / collinear columns on transformed design per policy
        if rp == "stata":
            X_t_red, keep_cols = la.drop_rank_deficient_cols_stata(X_t)
        else:
            X_t_red, keep_cols = la.drop_rank_deficient_cols_r(X_t)
        # keep_mask is required downstream for constraint remapping and coefficient expansion
        keep_mask = np.asarray(keep_cols, dtype=bool).reshape(-1)
        # Solve in reduced space using selected rank policy for reproducibility
        # Ensure y_t is column vector for consistency
        y_t_col = y_t.reshape(-1, 1) if y_t.ndim == 1 else y_t

        # Helper to remap constraint matrix R to the reduced column space
        def _remap_matrix(M: np.ndarray | None, keep: np.ndarray) -> np.ndarray | None:
            if M is None:
                return None
            M = np.asarray(M, dtype=np.float64)
            if M.ndim != 2 or M.shape[1] != keep.size:
                raise ValueError(
                    "Constraint/null_R width must match the number of columns in X at the screening point.",
                )
            M_red = M[:, keep]
            nonzero = np.any(np.abs(M_red) > 0, axis=1)
            return M_red[nonzero, :] if M_red.size else None

        # If constraints supplied, remap to reduced space and solve constrained GLS
        # Wrap dense solve operations under device context for optional GPU acceleration
        with self._device_context(device):
            if (
                constraints is None
                or constraint_vals is None
                or getattr(constraints, "size", 0) == 0
            ):
                # Unconstrained: use existing solver on transformed design
                beta_hat_red = self._solve_gls_transformed(
                    X_t_red, y_t_col, _rank_policy_internal=rp,
                )
                constraints_info = None
            else:
                # Remap constraints/null to surviving columns and prune infeasible zero rows
                C = _remap_matrix(np.asarray(constraints, dtype=np.float64), keep_mask)
                q_vec = None
                if constraint_vals is not None:
                    q_raw = np.asarray(constraint_vals, dtype=np.float64).reshape(-1)
                    # keep rows that remained after remap
                    nonzero_row = np.any(
                        np.abs(np.asarray(constraints, dtype=np.float64)[:, keep_mask]) > 0,
                        axis=1,
                    )
                    if q_raw.shape[0] != np.asarray(constraints).shape[0]:
                        raise ValueError(
                            "constraint_vals length must match number of constraint rows.",
                        )
                    q_vec = q_raw[nonzero_row].reshape(-1, 1) if (C is not None) else None
                # Solve constrained problem on whitened/transformed design
                # Use solve_constrained utility; no additional W (we work on whitened X_t_red/y_t_col)
                try:
                    beta_hat_red = solve_constrained(
                        X_t_red,
                        y_t_col,
                        C if C is not None else None,
                        q_vec if q_vec is not None else None,
                        weight_policy="allow",
                        rank_policy=("R" if rp == "r" else "stata"),
                    )
                except TypeError:
                    # Backwards compatibility fallback if signature differs
                    beta_hat_red = solve_constrained(
                        X_t_red,
                        y_t_col,
                        C if C is not None else None,
                        q_vec if q_vec is not None else None,
                    )
                constraints_info = {
                    "provided_rows": int(np.asarray(constraints).shape[0])
                    if constraints is not None
                    else 0,
                    "active_rows": int(C.shape[0]) if (C is not None) else 0,
                }

            # transformed residuals on reduced design
            yhat_t = la.dot(X_t_red, beta_hat_red).reshape(-1)
            resid_t = y_t.reshape(-1) - yhat_t

            # expand coefficients back to original design dimension for reporting
            beta_full = np.zeros((X_t.shape[1], 1), dtype=np.float64)
            # fill surviving coefficients into full vector using keep_mask created earlier
            beta_full[keep_mask, 0] = beta_hat_red.reshape(-1)

            yhat_orig = la.dot(X, beta_full).reshape(-1)
            resid_orig = y.reshape(-1) - yhat_orig

        # bootstrap multipliers via BootConfig. If boot_cluster provided, propagate
        def _mask_ids(ids):
            if ids is None:
                return None
            arr = np.asarray(ids)
            if arr.shape[0] != self._n_obs_init:
                raise ValueError("bootstrap id arrays must match original n_obs")
            return arr[finite_mask]

        def _mask_multiway(mw):
            if mw is None:
                return None
            return [_mask_ids(a) for a in mw]

        if boot is None:
            # Default bootstrap config (IID unless boot_cluster supplied)
            boot_cfg = BootConfig(
                cluster_ids=_mask_ids(boot_cluster) if boot_cluster is not None else None,
            )
        elif boot_cluster is not None:
            # Override clustering scheme deterministically for bootstrap draws.
            # IMPORTANT: enforce BootConfig exclusivity by clearing other schemes.
            boot_cfg = BootConfig(
                n_boot=boot.n_boot,
                dist=boot.dist,
                seed=boot.seed,
                policy=boot.policy,
                enum_max_g=boot.enum_max_g,
                use_enumeration=boot.use_enumeration,
                enumeration_mode=boot.enumeration_mode,
                cluster_method=boot.cluster_method,
                bootcluster=boot.bootcluster,
                cluster_ids=_mask_ids(boot_cluster),
                multiway_ids=None,
                space_ids=None,
                time_ids=None,
            )
        else:
            # Preserve the caller-provided bootstrap config but mask ids to the
            # estimation sample (finite_mask) so lengths match n_proc.
            boot_cfg = BootConfig(
                n_boot=boot.n_boot,
                dist=boot.dist,
                seed=boot.seed,
                policy=boot.policy,
                enum_max_g=boot.enum_max_g,
                use_enumeration=boot.use_enumeration,
                enumeration_mode=boot.enumeration_mode,
                cluster_method=boot.cluster_method,
                bootcluster=boot.bootcluster,
                cluster_ids=_mask_ids(getattr(boot, "cluster_ids", None)),
                multiway_ids=_mask_multiway(getattr(boot, "multiway_ids", None)),
                space_ids=_mask_ids(getattr(boot, "space_ids", None)),
                time_ids=_mask_ids(getattr(boot, "time_ids", None)),
            )

        Wdf, boot_log = boot_cfg.make_multipliers(n_proc)

        Wnp = Wdf.to_numpy()

        # --- SSC Correction ---
        # Compute ssc_factor for residuals scaling
        # K = number of parameters (full design)
        # N = number of observations
        # Clusters = boot_cluster (if available, or from boot settings)

        # Resolve clusters for SSC
        clusters_ssc = (
            list(getattr(boot_cfg, "multiway_ids", None))
            if getattr(boot_cfg, "multiway_ids", None) is not None
            else (
                [boot_cfg.space_ids, boot_cfg.time_ids]
                if (
                    getattr(boot_cfg, "space_ids", None) is not None
                    and getattr(boot_cfg, "time_ids", None) is not None
                )
                else getattr(boot_cfg, "cluster_ids", None)
            )
        )

        ssc_local = _normalize_ssc(ssc)
        # Use the effective number of estimated parameters after rank screening
        # (matches the solve performed on X_t_red).
        k_params = int(X_t_red.shape[1])

        ssc_factor = compute_ssc_correction(
            n=int(n_proc),
            k=k_params,
            clusters=clusters_ssc,
            ssc=ssc_local
        )

        # Scale residuals if factor != 1.0 needed
        # We need to scale resid_orig and resid_t used in bootstrap
        # Note: we do NOT modify the 'resid_orig' stored in extra, only the one used for bootstrap DGP.
        resid_boot_orig = resid_orig
        resid_boot_t = resid_t
        if ssc_factor != 1.0:
            resid_boot_orig = resid_orig * ssc_factor
            resid_boot_t = resid_t * ssc_factor

        # apply wild multiplier bootstrap
        # If transform is a Prais-Winsten or EXCHANGEABLE variant and boot_reestimate_rho requested,
        # perform bootstrap on the original scale and re-estimate rho per replication
        if (transform[0].startswith("pw") or transform[0].startswith("exch")) and (
            boot_reestimate_rho is True
        ):
            # original-scale bootstrap on residuals then recompute transform per draw
            Ystar_orig, _ = bt.apply_wild_bootstrap(
                yhat_orig.reshape(-1, 1), resid_boot_orig.reshape(-1, 1), Wnp,
            )
            B = Ystar_orig.shape[1] if Ystar_orig.ndim == 2 else 1
            boot_betas = np.zeros((X_t.shape[1], B), dtype=np.float64)
            # For each draw, re-run appropriate pipeline to produce transformed X,y
            # Wrap bootstrap solves under device context for optional GPU acceleration
            # Precompute Prais–Winsten ordering/segments once for reuse across draws
            precomp_pw: dict | None = None
            if transform[0].startswith("pw"):
                if ("order" in locals()) and ("seg_id" in locals()):
                    precomp_pw = {
                        "order": order,
                        "ts": time_ids_filtered if "time_ids_filtered" in locals() else t,
                        "ss": series_ids_filtered if "series_ids_filtered" in locals() else s,
                        "seg_id": seg_id,
                        "d": float(d) if "d" in locals() else 1.0,
                    }
                else:
                    precomp_pw = None
            # For EXCH, fix clustering once outside the loop
            cl_for_boot: np.ndarray | None = None
            if transform[0].startswith("exch"):
                if boot_cluster is not None:
                    cl_for_boot = np.asarray(boot_cluster)[finite_mask]
                elif (boot is not None) and (
                    getattr(boot, "cluster_ids", None) is not None
                ):
                    cl_for_boot = np.asarray(boot.cluster_ids)[finite_mask]
                else:
                    cl_for_boot = None

            with self._device_context(device):
                # Optional threading across bootstrap draws when re-estimating rho per draw
                workers_env = os.getenv("LINEAREG_GLS_BOOTSTRAP_WORKERS", "").strip()
                try:
                    workers = int(workers_env) if workers_env else 0
                except ValueError:
                    workers = 0
                do_parallel = (workers and workers > 1) and (
                    device is None or str(device).lower() == "cpu"
                )

                def _one_draw(b: int) -> tuple[int, np.ndarray]:
                    yb = Ystar_orig[:, b : b + 1]
                    if transform[0].startswith("pw"):
                        # Pass filtered time_ids and series_ids to match filtered X and yb
                        Xb_t, yb_t = self._pw_reestimate_transform(
                            X,
                            yb,
                            time_ids=time_ids_filtered
                            if "time_ids_filtered" in locals()
                            else time_ids,
                            series_ids=series_ids_filtered
                            if "series_ids_filtered" in locals()
                            else series_ids,
                            weights=weights,
                            mode=pw_mode,
                            rho_method=rho_method,
                            pw_tol=pw_tol,
                            pw_max_iter=pw_max_iter,
                            ar1_transform=ar1_transform,
                            psar1=(omega.upper() == "PSAR1")
                            if isinstance(omega, str)
                            else False,
                            force_irregular=force_irregular,
                            _rank_policy_internal=rp,
                            _precomputed=precomp_pw,
                        )
                    else:
                        # Exchangeable: re-estimate rho_b from yb on original scale, then transform
                        if cl_for_boot is None:
                            raise RuntimeError(
                                "exchangeable requires cluster_ids for bootstrap reestimation.",
                            )
                        # compute OLS beta for this bootstrap draw on original X
                        try:
                            beta_b = la.solve(
                                X,
                                yb,
                                method="qr",
                                rank_policy=("R" if rp == "r" else "stata"),
                            )
                        except TypeError:
                            beta_b = la.solve(X, yb, method="qr")
                        e_b = yb.reshape(-1) - la.dot(X, beta_b).reshape(-1)
                        rho_b = self._exch_rho_from_resid(e_b, cl_for_boot)
                        Xb_t, yb_t = self._exchangeable_transform(
                            X, yb, cl_for_boot, rho_b, weights=weights,
                        )
                    # Use Stata-style QR-based screening inside bootstrap re-estimation
                    # so that column-dropping is deterministic and matches main fit.
                    if rp == "stata":
                        Xb_t_red, keep_cols_b = la.drop_rank_deficient_cols_stata(Xb_t)
                    else:
                        Xb_t_red, keep_cols_b = la.drop_rank_deficient_cols_r(Xb_t)
                    betab_red = la.solve(
                        Xb_t_red,
                        yb_t,
                        method="qr",
                        rank_policy=("R" if rp == "r" else "stata"),
                    )
                    betab = np.zeros((X_t.shape[1],), dtype=np.float64)
                    betab[np.asarray(keep_cols_b, bool)] = betab_red.reshape(-1)
                    return b, betab

                if do_parallel:
                    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                        for b_idx, betab in ex.map(_one_draw, range(B)):
                            boot_betas[:, b_idx] = betab
                else:
                    for b in range(B):
                        _, betab = _one_draw(b)
                        boot_betas[:, b] = betab
            se_hat = bt.bootstrap_se(boot_betas)
        else:
            # legacy path: fixed transform solved in transformed domain
            # Wrap under device context
            with self._device_context(device):
                Ystar_t, _ = bt.apply_wild_bootstrap(
                    yhat_t.reshape(-1, 1),
                    resid_boot_t.reshape(-1, 1),
                    Wnp,
                )
                # solve in reduced X space then map back to full param space
                boot_betas_red = la.solve(
                    X_t_red,
                    Ystar_t,
                    method="qr",
                    rank_policy=("R" if rp == "r" else "stata"),
                )
            boot_betas = np.zeros(
                (X_t.shape[1], boot_betas_red.shape[1]), dtype=np.float64,
            )
            boot_betas[keep_mask, :] = boot_betas_red
            se_hat = bt.bootstrap_se(boot_betas)

        # present full-dimension parameter vector (consistent across estimators)
        params = pd.Series(beta_full.reshape(-1), index=self._var_names, name="coef")
        se = pd.Series(se_hat.reshape(-1), index=self._var_names, name="se")

        # record transform metadata for reproducibility (time/series used in transform and rho estimates)
        time_ids_transform = None
        series_ids_transform = None
        rho_est = None
        if isinstance(omega, str) and omega.upper() in {"AR1", "PSAR1"}:
            time_ids_transform = t if "t" in locals() else None
            series_ids_transform = s if "s" in locals() else None
            if omega.upper() == "PSAR1":
                rho_est = locals().get("rho_by_series", None)
            else:
                rho_est = locals().get("rho_old", locals().get("rho", None))

        extra_local = {
            "yhat": yhat_orig,
            "resid": resid_orig,
            "beta": beta_full,
            # Store multipliers as a DataFrame for strict reproducibility in EstimationResult.wald_test
            "W_multipliers_inference": Wdf,
            # Keep a numpy copy for estimator-internal/debug convenience
            "W": Wnp,
            "boot_log": boot_log,
            # For Wald replay, provide the full transformed design (same column dimension as params)
            "X_inference": X_t,
            "y_inference": y_t,
            "transform_details": transform,
            "time_ids_transform": time_ids_transform,
            "series_ids_transform": series_ids_transform,
            "rho_est": rho_est,
            "boot_config": boot_cfg,
            # preserve clustering scheme for strict reproducibility (OLS parity)
            "clusters_inference": (
                np.asarray(boot_cluster)[finite_mask]
                if boot_cluster is not None
                else (
                    getattr(boot_cfg, "cluster_ids", None)
                    if (boot_cfg is not None and getattr(boot_cfg, "cluster_ids", None) is not None)
                    else None
                )
            ),
            "multiway_ids_inference": (
                getattr(boot_cfg, "multiway_ids", None)
                if (boot_cfg is not None and getattr(boot_cfg, "multiway_ids", None) is not None)
                else None
            ),
            "space_ids_inference": (
                getattr(boot_cfg, "space_ids", None) if boot_cfg is not None else None
            ),
            "time_ids_inference": (
                getattr(boot_cfg, "time_ids", None) if boot_cfg is not None else None
            ),
            "weights_inference": (
                np.asarray(weights)[finite_mask] if weights is not None else None
            ),
            # variant/residual/policy snapshot for Wald bootstrap replay
            "boot_variant_used": (
                "33"
                if (
                    boot_cluster is not None
                    or (
                        boot is not None
                        and (
                            getattr(boot, "cluster_ids", None) is not None
                            or getattr(boot, "multiway_ids", None) is not None
                            or (
                                getattr(boot, "space_ids", None) is not None
                                and getattr(boot, "time_ids", None) is not None
                            )
                        )
                    )
                )
                else "11"
            ),
            "boot_residual_default": "WCU",
            "boot_policy_used": (
                getattr(boot, "policy", None)
                if boot is not None
                else getattr(boot_cfg, "policy", None)
            ),
        }

        # Ensure SE provenance is explicit so EstimationResult.validate() can
        # enforce analytic-SE prohibition.
        extra_local = {**extra_local, "se_source": "bootstrap"}
        self._results = EstimationResult(
            params=params,
            se=se,  # Bootstrap SE stored directly in .se
            bands=None,
            n_obs=int(n_proc),
            model_info={
                "Estimator": "GLS",
                "Transform": transform,
                "Bootstrap": "wild",
                "B": (int(Wnp.shape[1]) if Wnp.ndim == 2 else 1),
                "NoAnalyticSE": True,
                # document weight semantics in combined AR1+diag case
                "WeightsAreInverseVar": weights is not None,
            },
            extra=extra_local,
        )

        return self._results

    def _solve_gls_transformed(
        self,
        X_t: NDArray[np.float64],
        y_t: NDArray[np.float64],
        *,
        _rank_policy_internal: str = "stata",
    ) -> NDArray[np.float64]:
        """Solve transformed least squares using core.linalg QR/SVD path.

        This delegates to la.solve which implements QR/SVD choice and tolerances.
        """
        return la.solve(
            X_t,
            y_t,
            method="qr",
            rank_policy=("R" if _rank_policy_internal.lower() == "r" else "stata"),
        )

    def _rho_by_series_from_resid(
        self,
        e_sorted: NDArray[np.float64],
        ss_sorted: NDArray[np.int64],
        seg_id_sorted: NDArray[np.int64],
        *,
        method: str = "dw",
    ) -> dict[int, float]:
        """Estimate series-specific AR(1) rho from residuals in sorted order.

        Parameters are assumed to be aligned in the same (sorted) order.
        Only adjacent pairs within the same segment (gap-restart) are used.
        """
        e = np.asarray(e_sorted, dtype=np.float64).reshape(-1)
        ss = np.asarray(ss_sorted)
        seg_id = np.asarray(seg_id_sorted)
        if e.shape[0] != ss.shape[0] or e.shape[0] != seg_id.shape[0]:
            raise ValueError("e_sorted, ss_sorted, seg_id_sorted must have same length")

        method_raw = method.lower() if isinstance(method, str) else "dw"
        m = {"regress": "ols", "durbin-watson": "dw"}.get(method_raw, method_raw)
        if m not in {"dw", "ols"}:
            raise ValueError(
                "rho_method must be one of {'dw','durbin-watson','ols','regress'}",
            )

        out: dict[int, float] = {}
        unique_series = np.unique(ss)
        for g in unique_series:
            idx = np.where(ss == g)[0]
            if idx.size <= 1:
                out[g] = 0.0
                continue
            eg = e[idx]
            seg_g = seg_id[idx]

            # Adjacent valid pairs only when segment continues
            valid = np.r_[False, seg_g[1:] == seg_g[:-1]]
            if not np.any(valid):
                out[g] = 0.0
                continue
            u0 = eg[:-1][valid[1:]]
            u1 = eg[1:][valid[1:]]
            if u0.size == 0:
                out[g] = 0.0
                continue

            if m == "ols":
                num = float(np.sum(u0 * u1))
                den = float(np.sum(u0 * u0))
                rho_g = 0.0 if den <= 0.0 else num / den
            else:
                num = float(np.sum((u1 - u0) ** 2))
                den = float(np.sum(eg**2))
                rho_g = 0.0 if den <= 0.0 else 1.0 - (num / den) / 2.0
            out[g] = float(np.clip(rho_g, -0.999999, 0.999999))
        return out

    # --- Helpers for exchangeable correlation --------------------------------
    def _exch_rho_from_resid(self, e: NDArray[np.float64], cl: NDArray) -> float:
        """MoM/ICC-type rho estimate using cluster residuals; PD-clipped.

        Formula:
            rho = sum_c [ (sum_i e_ci)^2 - sum_i e_ci^2 ] / sum_c [ (m_c - 1) sum_i e_ci^2 ]

        The returned rho is clipped to ensure positive-definiteness across clusters:
        rho ∈ (max_c{-1/(m_c-1)} + eps, 1 - eps)
        """
        uniq = np.unique(cl)
        num = 0.0
        den = 0.0
        min_lb = -0.999999
        for g in uniq:
            idx = cl == g
            eg = np.asarray(e[idx], dtype=np.float64).reshape(-1)
            m = eg.size
            if m <= 1:
                continue
            s1 = float(np.sum(eg))
            s2 = float(np.sum(eg * eg))
            num += s1 * s1 - s2
            den += (m - 1) * s2
            # Lower bound for rho that ensures the m x m equicorrelation matrix
            # R(ρ) = (1-ρ) I_m + ρ 1_m 1_m^T is positive-definite is ρ > -1/(m-1).
            # (See standard result for equicorrelation matrices: smallest eigenvalue
            # = 1 - ρ - ρ/(m-1) = 1 - ρ * m/(m-1) => requires ρ > -1/(m-1)). We compute
            # per-cluster bounds to clip rho into a PD-preserving interval.
            lb = -1.0 / (m - 1.0)
            min_lb = max(min_lb, lb)
        rho = 0.0 if den <= 0.0 else (num / den)
        eps = 1e-8
        return float(np.clip(rho, min_lb + eps, 1.0 - eps))

    def _exchangeable_transform(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        cl: NDArray,
        rho: float,
        *,
        weights: Sequence[float] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply blockwise Sigma^{-1/2} based on exchangeable R within clusters (and optional diag weights).

        Sigma_g = D_g^{-1/2} R_g(rho) D_g^{-1/2} where D_g = diag(w_i) if weights provided.

        Note: weights are treated as inverse-variance weights (w_i = 1/Var(e_i)).
        rho is clipped per-cluster to ensure positive-definiteness of each block
        before attempting Cholesky decomposition.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        _n, _k = X.shape
        X_out = np.empty_like(X)
        y_out = np.empty_like(y)
        # diagonal inverse-variance weights if provided
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != self._n_obs_init:
                raise ValueError("weights length must match original n_obs")
            w = w[
                np.isfinite(self.y_orig).reshape(-1)
                & np.all(np.isfinite(self.X_orig), axis=1)
            ]
        else:
            w = None
        for g in np.unique(cl):
            idx = np.where(cl == g)[0]
            m = idx.size
            # Clip rho to PD-safe interval for this cluster: rho ∈ ( -1/(m-1) + eps, 1 - eps )
            eps = 1e-8
            min_lb = -1.0 / max(1.0, (m - 1.0))
            rho_safe = float(np.clip(rho, min_lb + eps, 1.0 - eps))
            Rg = (1.0 - rho_safe) * np.eye(m, dtype=np.float64) + rho_safe * np.ones(
                (m, m), dtype=np.float64,
            )
            if w is not None:
                Dg = 1.0 / np.sqrt(np.asarray(w[idx], dtype=np.float64).reshape(-1))
                Sig = (Dg[:, None] * Rg) * Dg[None, :]
            else:
                Sig = Rg
            L = la.safe_cholesky(Sig)
            if L is None:
                raise RuntimeError("Exchangeable Σ block is not PD; check rho/weights.")
            X_out[idx, :] = la.triangular_solve(L, X[idx, :])
            y_out[idx, :] = la.triangular_solve(L, y[idx, :])
        return X_out, y_out

    def _pw_reestimate_transform(  # noqa: PLR0913
        self,
        X_orig: NDArray[np.float64],
        y_orig: NDArray[np.float64],
        *,
        time_ids: Sequence[object],
        series_ids: Sequence[object] | None,
        weights: Sequence[float] | None,
        mode: str,
        rho_method: str,
        pw_tol: float,
        pw_max_iter: int,
        ar1_transform: str = "prais",
        psar1: bool = False,
        force_irregular: bool = False,
        _rank_policy_internal: str = "stata",
        _precomputed: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """For a single bootstrap draw `y_orig` (already drawn on original scale),
        re-estimate AR(1) parameter(s) and return transformed X,y for inference.

        This reproduces the same pipeline used in the main fit: spacing checks,
        weighted or unweighted iteration, and Prais/Corc transform semantics.

        NOTE: X_orig and y_orig are assumed to be already filtered (finite_mask applied).
        """
        # Keep full parity with main fit: honor mode (two-step vs iterated)
        mode_raw = str(mode).strip().lower() if isinstance(mode, str) else "iterated"
        is_two_step = mode_raw in {"two", "two-step", "twostep"}
        if not (is_two_step or mode_raw == "iterated"):
            raise ValueError("pw_mode must be 'two-step' or 'iterated'")

        # weights can be passed either already filtered (len == n_proc) or at full length
        # (len == n_obs_init). Normalize to filtered length to match X_orig/y_orig.
        weights_filt: NDArray[np.float64] | None
        if weights is None:
            weights_filt = None
        else:
            w_full = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w_full.shape[0] == X_orig.shape[0]:
                weights_filt = w_full
            elif w_full.shape[0] == self._n_obs_init:
                finite_mask = np.isfinite(self.y_orig).reshape(-1) & np.all(
                    np.isfinite(self.X_orig), axis=1,
                )
                weights_filt = w_full[finite_mask]
            else:
                raise ValueError(
                    "weights length must match either filtered sample size or original n_obs",
                )

        # NOTE: Do NOT recompute finite_mask - X_orig and y_orig are already filtered
        # in the bootstrap loop before calling this function.
        if _precomputed is not None:
            # Use precomputed ordering and segmentation when provided
            order = np.asarray(_precomputed.get("order"))
            # IMPORTANT: ts/ss/seg_id stored in _precomputed are already in sorted order.
            ts = np.asarray(_precomputed.get("ts"))
            ss = np.asarray(_precomputed.get("ss"))
            # seg_id is already defined in the sorted (order) space
            seg_id = np.asarray(_precomputed.get("seg_id"))
            d = float(_precomputed.get("d", 1.0))
            if ts.shape[0] != X_orig.shape[0] or ss.shape[0] != X_orig.shape[0] or seg_id.shape[0] != X_orig.shape[0]:
                raise ValueError("precomputed ordering arrays must match filtered sample length")
            Xs = X_orig[order]
            ys = y_orig[order]
        else:
            t = np.asarray(time_ids)
            s = np.zeros_like(t) if series_ids is None else np.asarray(series_ids)
            order = np.lexsort((t, s))
            Xs = X_orig[order]
            ys = y_orig[order]
            ts = t[order]
            ss = s[order]

            # compute a common spacing d if all series share it (used for rho^{1/d} adjustment)
            d = 1.0

            # recompute seg_id and spacing checks (same as fit)
            seg_break = np.ones(ys.shape[0], dtype=bool)
            if ys.shape[0] > 1:
                if force_irregular:
                    seg_break[1:] = ss[1:] != ss[:-1]
                else:
                    starts = np.where(np.r_[True, ss[1:] != ss[:-1]])[0]
                    ends = np.r_[starts[1:], ss.shape[0]]
                    delta_by_series: dict[int, float] = {}
                    for a, b in zip(starts, ends):
                        if b - a <= 1:
                            continue
                        diffs = np.diff(ts[a:b])
                        pos = diffs[diffs > 0]
                        uniq = np.unique(pos)
                        if uniq.size == 0:
                            continue
                        if uniq.size != 1:
                            raise ValueError(
                                "AR1 requires constant time spacing within each series.",
                            )
                        delta_by_series[ss[a]] = float(uniq[0])
                    if len(delta_by_series) > 0:
                        deltas = np.array(list(delta_by_series.values()), dtype=float)
                        if np.allclose(deltas, deltas[0]) and not np.isclose(deltas[0], 1.0):
                            d = float(deltas[0])
                    same_series = ss[1:] == ss[:-1]
                    inside = np.zeros(ys.shape[0] - 1, dtype=bool)
                    if same_series.any():
                        idxs = np.where(same_series)[0]
                        expected = np.array(
                            [delta_by_series.get(ss[i], np.nan) for i in idxs], dtype=float,
                        )
                        inside[idxs] = np.isclose(
                            ts[1:][idxs] - ts[:-1][idxs], expected, equal_nan=False,
                        )
                    seg_break[1:] = (~inside) | (ss[1:] != ss[:-1])
            seg_id = np.cumsum(seg_break) - 1

        if psar1:
            if weights_filt is not None:
                w_sorted = weights_filt[order]
            else:
                w_sorted = None
            # Initialize rho_by_series
            rho_old = self._estimate_rho_by_series(
                Xs,
                ys,
                ss,
                ts,
                seg_id,
                weights=w_sorted,
                method=rho_method,
                _rank_policy_internal=_rank_policy_internal,
            )
            if (not is_two_step) and (pw_max_iter > 0):
                it = 0
                while it < int(pw_max_iter):
                    # apply current transform and solve
                    # adjust each rho_g if common spacing d != 1
                    rho_use = (
                        {k: (np.sign(v) * (abs(v) ** (1.0 / d)) if (d != 1 and v != 0) else v) for k, v in rho_old.items()}
                        if d != 1
                        else rho_old
                    )
                    X_tmp_sorted, y_tmp_sorted = self._prais_winsten_psar1(
                        Xs,
                        ys,
                        ss,
                        seg_id,
                        rho_use,
                        transform=ar1_transform,
                    )
                    if ar1_transform == "corc":
                        X_tmp = X_tmp_sorted
                        y_tmp = y_tmp_sorted
                    else:
                        inv = np.empty_like(order)
                        inv[order] = np.arange(order.size)
                        X_tmp = X_tmp_sorted[inv]
                        y_tmp = y_tmp_sorted[inv]
                    beta = la.solve(
                        X_tmp,
                        y_tmp,
                        method="qr",
                        rank_policy=(
                            "R" if _rank_policy_internal.lower() == "r" else "stata"
                        ),
                    )
                    e = (ys - la.dot(Xs, beta)).ravel()
                    rho_new = self._rho_by_series_from_resid(
                        e,
                        ss,
                        seg_id,
                        method=rho_method,
                    )
                    converged = True
                    for sid in rho_old:
                        if (sid not in rho_new) or (abs(rho_new[sid] - rho_old[sid]) > float(pw_tol)):
                            converged = False
                            break
                    rho_old = rho_new
                    if converged:
                        break
                    it += 1
            # final transform
            rho_final = (
                {k: (np.sign(v) * (abs(v) ** (1.0 / d)) if (d != 1 and v != 0) else v) for k, v in rho_old.items()}
                if d != 1
                else rho_old
            )
            Xp, yp = self._prais_winsten_psar1(
                Xs, ys, ss, seg_id, rho_final, transform=ar1_transform,
            )
        else:
            # estimate a common rho from the bootstrap draw
            if weights_filt is not None:
                w_sorted = weights_filt[order].reshape(-1, 1)
                sqrtw = np.sqrt(w_sorted)
                Xw = la.hadamard(Xs, sqrtw)
                yw = la.hadamard(ys, sqrtw)
                beta_ols = la.solve(
                    Xw,
                    yw,
                    method="qr",
                    rank_policy=("R" if _rank_policy_internal == "r" else "stata"),
                )
                # Residuals for rho estimation must be on the original (unweighted) scale
                e_ols = (ys - la.dot(Xs, beta_ols)).ravel()
            else:
                beta_ols = la.solve(
                    Xs,
                    ys,
                    method="qr",
                    rank_policy=("R" if _rank_policy_internal == "r" else "stata"),
                )
                e_ols = (ys - la.dot(Xs, beta_ols)).ravel()
            valid = seg_id[1:] == seg_id[:-1]
            rho_old = 0.0
            if e_ols.size >= 2 and np.any(valid):
                method_raw = rho_method.lower() if isinstance(rho_method, str) else "dw"
                if method_raw not in {"dw", "ols"}:
                    raise ValueError("rho_method must be one of {'dw','ols'}")
                method = method_raw
                if method == "ols":
                    u0 = e_ols[:-1][valid]
                    u1 = e_ols[1:][valid]
                    num = float(la.dot(u0.reshape(-1, 1).T, u1.reshape(-1, 1)))
                    den = float(la.dot(u0.reshape(-1, 1).T, u0.reshape(-1, 1)))
                    rho_old = 0.0 if den <= 0 else num / den
                elif method == "dw":
                    u0 = e_ols[:-1][valid]
                    u1 = e_ols[1:][valid]
                    num = float(np.sum((u1 - u0) ** 2))
                    den = float(np.sum(e_ols**2))
                    rho_old = 0.0 if den <= 0 else 1.0 - (num / den) / 2.0
                else:
                    raise ValueError(
                        "rho_method must be one of {'dw','durbin-watson','ols','regress'}",
                    )
            rho_old = float(np.clip(rho_old, -0.999999, 0.999999))

            if (not is_two_step) and (pw_max_iter > 0):
                it = 0
                while it < int(pw_max_iter):
                    rho_use = rho_old
                    X_tmp_sorted, y_tmp_sorted = self._prais_winsten_by_segments(
                        Xs,
                        ys,
                        seg_id,
                        float(rho_use),
                        transform=ar1_transform,
                    )
                    if ar1_transform == "corc":
                        X_tmp = X_tmp_sorted
                        y_tmp = y_tmp_sorted
                    else:
                        inv = np.empty_like(order)
                        inv[order] = np.arange(order.size)
                        X_tmp = X_tmp_sorted[inv]
                        y_tmp = y_tmp_sorted[inv]
                    beta = la.solve(
                        X_tmp,
                        y_tmp,
                        method="qr",
                        rank_policy=(
                            "R" if _rank_policy_internal.lower() == "r" else "stata"
                        ),
                    )
                    e = (ys - la.dot(Xs, beta)).ravel()
                    # Yule–Walker on adjacent valid pairs (sorted order)
                    if e.size >= 2 and np.any(valid):
                        e_lag = e[:-1][valid].reshape(-1, 1)
                        e_lead = e[1:][valid].reshape(-1, 1)
                        num = float(la.dot(e_lag.T, e_lead).squeeze())
                        den = float(la.dot(e_lag.T, e_lag).squeeze())
                        rho_new = 0.0 if den <= 0 else num / den
                    else:
                        rho_new = 0.0
                    rho_new = float(np.clip(rho_new, -0.999999, 0.999999))
                    if abs(rho_new - rho_old) <= float(pw_tol):
                        rho_old = rho_new
                        break
                    rho_old = rho_new
                    it += 1

            rho_final = rho_old
            Xp, yp = self._prais_winsten_by_segments(
                Xs, ys, seg_id, float(rho_final), transform=ar1_transform,
            )

        if ar1_transform == "corc":
            # Cochrane–Orcutt drops the first obs per segment; keep sorted order
            return Xp, yp

        # restore original order
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        X_t = Xp[inv]
        y_t = yp[inv]
        return X_t, y_t

    def _estimate_rho_by_series(  # noqa: PLR0913
        self,
        Xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        ss: NDArray[np.int64],
        ts: NDArray[np.int64],
        seg_id: NDArray[np.int64],
        *,
        weights: Sequence[float] | None = None,
        method: str = "dw",
        _rank_policy_internal: str = "stata",
    ) -> dict[int, float]:
        """Estimate AR(1) rho for each series separately.

        Returns a dict mapping series id -> rho_g.
        """
        del ts  # retained for compatibility with earlier API

        out: dict[int, float] = {}
        unique_series = np.unique(ss)
        for g in unique_series:
            idx = np.where(ss == g)[0]
            if idx.size <= 1:
                out[g] = 0.0
                continue
            Xg = Xs[idx]
            yg = ys[idx]
            if weights is not None:
                # select corresponding weights slice if provided
                w_full = np.asarray(weights, dtype=np.float64).reshape(-1)
                # mapping back to global mask is not trivial here; caller should pass ordered weights
                # assume caller passed ordered weights already
                wg = w_full[idx].reshape(-1, 1)
                sqrtw = np.sqrt(wg)
                Xgw = la.hadamard(Xg, sqrtw)
                ygw = la.hadamard(yg, sqrtw)
                beta = la.solve(
                    Xgw,
                    ygw,
                    method="qr",
                    rank_policy=("R" if _rank_policy_internal == "r" else "stata"),
                )
                # CRITICAL: residuals must be on original scale for DW/adjacency
                e = (yg - la.dot(Xg, beta)).ravel()
            else:
                beta = la.solve(
                    Xg,
                    yg,
                    method="qr",
                    rank_policy=("R" if _rank_policy_internal == "r" else "stata"),
                )
                e = (yg - la.dot(Xg, beta)).ravel()
            # adjacency valid pairs: same segment only (STRICT gap-restart Prais-Winsten)
            if e.size < 2:
                out[g] = 0.0
                continue
            seg_g = seg_id[idx]
            valid = np.r_[False, seg_g[1:] == seg_g[:-1]]
            if not np.any(valid):
                out[g] = 0.0
                continue
            u0 = e[:-1][valid[1:]]
            u1 = e[1:][valid[1:]]
            method_raw = method.lower() if isinstance(method, str) else "dw"
            m = {"regress": "ols", "durbin-watson": "dw"}.get(method_raw, method_raw)
            if m == "ols":
                num = float(la.dot(u0.reshape(-1, 1).T, u1.reshape(-1, 1)))
                den = float(la.dot(u0.reshape(-1, 1).T, u0.reshape(-1, 1)))
                rho_g = 0.0 if den <= 0 else num / den
            else:
                num = float(np.sum((u1 - u0) ** 2))
                den = float(np.sum(e**2))
                rho_g = 0.0 if den <= 0 else 1.0 - (num / den) / 2.0
            out[g] = float(np.clip(rho_g, -0.999999, 0.999999))
        return out

    def _prais_winsten_psar1(  # noqa: PLR0913
        self,
        Xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        ss: NDArray[np.int64],
        seg_id: NDArray[np.int64],
        rho_by_series: dict[int, float],
        *,
        transform: str = "prais",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply Prais-Winsten or Cochrane-Orcutt per-series transformation.

        If `transform=='corc'`, the first observation in each segment is dropped
        (Cochrane-Orcutt). If 'prais', the first obs is scaled by sqrt(1-rho^2).
        """
        Xp = Xs.copy()
        yp = ys.copy()
        # identify first index per segment
        first = np.empty(seg_id.shape[0], dtype=bool)
        first[0] = True
        if seg_id.shape[0] > 1:
            first[1:] = seg_id[1:] != seg_id[:-1]
        # iterate series-wise applying local rho
        for i in range(1, ys.shape[0]):
            if seg_id[i] == seg_id[i - 1]:
                # Use series id as canonical lookup key for rho; do not fall back to
                # numeric segment index which may change ordering. Default to 0.0.
                rho_g = rho_by_series.get(ss[i], 0.0)
                Xp[i] = Xp[i] - rho_g * Xp[i - 1]
                yp[i] = yp[i] - rho_g * yp[i - 1]
        # handle first obs scaling or dropping
        if transform == "prais":
            for j in np.where(first)[0]:
                # pick a representative rho for the segment start using series id
                rho_g = rho_by_series.get(ss[j], 0.0)
                s = float(np.sqrt(max(0.0, 1.0 - rho_g**2)))
                Xp[j] = Xp[j] * s
                yp[j] = yp[j] * s
        elif transform == "corc":
            # Cochrane-Orcutt: drop first obs in each segment
            keep = ~first
            Xp = Xp[keep]
            yp = yp[keep]
        return Xp, yp

    def _prais_winsten_by_segments(
        self,
        X_in: NDArray[np.float64],
        y_in: NDArray[np.float64],
        seg_id: NDArray[np.int64],
        rho_val: float,
        *,
        transform: str = "prais",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply Prais-Winsten transform restarting at segment boundaries.

        Each segment's first observation is scaled by sqrt(1 - rho^2) as per P-W.
        Differences are applied only within contiguous observations of the same segment.
        """
        Xp = X_in.copy()
        yp = y_in.copy()
        # Identify segment starts (first row or seg change)
        first = np.empty(seg_id.shape[0], dtype=bool)
        first[0] = True
        if seg_id.shape[0] > 1:
            first[1:] = seg_id[1:] != seg_id[:-1]

        # Apply differences within segments
        idx = np.arange(1, yp.shape[0])
        inside = seg_id[idx] == seg_id[idx - 1]
        if np.any(inside):
            yp[idx[inside]] = yp[idx[inside]] - rho_val * yp[idx[inside] - 1]
            Xp[idx[inside]] = Xp[idx[inside]] - rho_val * Xp[idx[inside] - 1]

        # Handle first-observation behavior according to transform
        if transform == "prais":
            s = float(np.sqrt(max(0.0, 1.0 - rho_val**2)))
            Xp[first] *= s
            yp[first] *= s
        elif transform == "corc":
            # Cochrane-Orcutt: drop first obs in each segment
            keep = ~first
            Xp = Xp[keep]
            yp = yp[keep]
        else:
            # unknown transform: be conservative and raise
            raise ValueError("ar1_transform must be one of {'prais','corc'}")

        return Xp, yp

    def _psar1_cov_from_weights(
        self,
        ts: NDArray[np.int64],
        ss: NDArray[np.int64],
        seg_id: NDArray[np.int64],
        rho_by_series: dict,
        w_sorted: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Build block-diagonal covariance Σ for PSAR1 with diagonal weights.

        Σ = blockdiag_g{ D_g R_g(rho_g) D_g }, D_g = diag(1/sqrt(w_i)),
        R_g is Toeplitz AR(1) correlation matrix. Strictly restarts at segment
        boundaries for strict Stata xtgls corr(psar1) panels(het) equivalence.
        """
        n = ts.shape[0]
        Sigma = np.zeros((n, n), dtype=np.float64)
        # Process each segment (contiguous block)
        starts = np.where(np.r_[True, seg_id[1:] != seg_id[:-1]])[0]
        ends = np.r_[starts[1:], n]
        for a, b in zip(starts, ends):
            sid = ss[a]
            rho_g = float(rho_by_series.get(sid, 0.0))
            m = b - a
            # Toeplitz AR(1) correlation matrix R_g
            idx = np.arange(m, dtype=int)
            Rg = rho_g ** np.abs(idx[:, None] - idx[None, :])
            # D_g = diag(1/sqrt(w_i))
            wg = np.asarray(w_sorted[a:b], dtype=np.float64).reshape(-1)
            if np.any(~np.isfinite(wg)) or np.any(wg <= 0):
                msg = "weights must be positive and finite within each segment"
                raise ValueError(msg)
            Dg = 1.0 / np.sqrt(wg)
            # Sigma_g = D_g R_g D_g
            Sigma[a:b, a:b] = (Dg[:, None] * Rg) * Dg[None, :]
        return Sigma
