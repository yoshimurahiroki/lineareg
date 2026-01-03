"""Two-Stage Least Squares (2SLS) estimator.

This module implements 2SLS estimation with comprehensive weak instrument diagnostics
(Cragg-Donald, Kleibergen-Paap, Sanderson-Windmeijer) and wild bootstrap inference.
"""

from __future__ import annotations

import concurrent.futures as cf
import itertools
import os
import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from lineareg.core import bootstrap as bt
from lineareg.core.bootstrap import (
    compute_ssc_correction,
    _normalize_ssc,
    wald_test_wild_bootstrap,
)
from lineareg.core import fe as fe_core
from lineareg.core import linalg as la
from lineareg.utils.auto_constant import add_constant
from lineareg.utils.constraints import (
    build_rq_from_string,
    solve_constrained,
    solve_constrained_2sls,
    solve_constrained_batch,
)
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

    from numpy.typing import NDArray

ArrayLike = pd.Series | np.ndarray
MatrixLike = pd.DataFrame | np.ndarray


def _cd_kp_stats(
    X: np.ndarray,
    Z: np.ndarray,
    endog_idx: list[int] | tuple[int, ...],
    z_excluded_idx: list[int] | tuple[int, ...],
    *,
    u: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
    multiway_ids: list[np.ndarray] | None = None,
    space_ids: np.ndarray | None = None,
    time_ids: np.ndarray | None = None,
    rank_policy: str = "stata",
    ssc: dict[str, Any] | None = None,
    k_params: int | None = None,
) -> dict:
    """Compute Cragg-Donald and Kleibergen-Paap weak IV diagnostics."""
    out: dict[str, float] = {
        "cd_min_eig": float("nan"),
        "kp_min_eig": float("nan"),
        "cd_wald_F": float("nan"),
        "kp_rk_LM": float("nan"),
        "kp_rk_Wald_F": float("nan"),
    }
    try:
        # Prepare matrices for excluded instruments and endogenous block
        if len(z_excluded_idx) == 0 or len(endog_idx) == 0:
            return out
        Z2 = la.to_dense(Z)[:, list(z_excluded_idx)]
        X1 = la.to_dense(X)[:, list(endog_idx)]
        n = int(X1.shape[0])
        L2 = int(Z2.shape[1])

        # Cross-products
        Qxz = la.crossprod(Z2, X1)  # (L2 x K)
        Qzz = la.tdot(Z2)  # (L2 x L2)
        Qxx = la.tdot(X1)  # (K x K)

        # Cragg-Donald minimal eigenvalue: generalized eigenvalues of (Qxz Qxz', Qzz)
        try:
            Qzz_inv = la.pinv(Qzz, rcond=1e-12)
            M_cd = la.dot(Qzz_inv, la.dot(Qxz, Qxz.T))
            M_cd = 0.5 * (M_cd + M_cd.T)
            evals_cd = la.eigvalsh(M_cd)
            lam_cd = float(np.min(evals_cd)) if evals_cd.size else float("nan")
            out["cd_min_eig"] = lam_cd
            out["cd_wald_F"] = float(n * lam_cd) if np.isfinite(lam_cd) else float("nan")
        except Exception:
            out["cd_min_eig"] = float("nan")

        # Robust S (meat)
        if u is not None:
            u_vec = np.asarray(u, dtype=np.float64).reshape(-1)
            zu = la.hadamard(la.to_dense(Z2), u_vec.reshape(-1, 1))

            if multiway_ids is not None:
                code_dims = [np.asarray(c).reshape(-1) for c in multiway_ids]
                R = len(code_dims)
                S = np.zeros((L2, L2), dtype=np.float64)
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
                code_space = np.asarray(space_ids).reshape(-1)
                code_time = np.asarray(time_ids).reshape(-1)
                S_space = la.tdot(la.group_sum(zu, code_space))
                S_time = la.tdot(la.group_sum(zu, code_time))
                codes_inter = np.column_stack([code_space, code_time])
                S_inter = la.tdot(la.group_sum_multi(zu, codes_inter))
                S = S_space + S_time - S_inter
            elif clusters is not None:
                S = la.tdot(la.group_sum(zu, np.asarray(clusters).reshape(-1)))
            else:
                S = la.tdot(zu)

            if ssc is not None:
                ssc_eval = _normalize_ssc(ssc)
                k_eff = k_params if k_params is not None else X.shape[1]
                clusters_for_ssc = (
                    multiway_ids
                    if multiway_ids is not None
                    else (
                        [space_ids, time_ids]
                        if (space_ids is not None and time_ids is not None)
                        else clusters
                    )
                )
                factor = compute_ssc_correction(n, k_eff, clusters=clusters_for_ssc, ssc=ssc_eval)
                if abs(factor - 1.0) > 1e-9:
                    S *= factor
        else:
            S = np.zeros((L2, L2), dtype=np.float64)

        # Kleibergen-Paap minimal generalized eigenvalue of Qxx^{-1} Qxz' S^{+} Qxz
        try:
            S_dense = la.to_dense(S)
            eig_tol = getattr(la, "eig_tol", None)
            eig_tol = eig_tol(S_dense) if callable(eig_tol) else 1e-10
            eigvals_S, eigvecs_S = la.eigh(S_dense)
            keep_S = eigvals_S > eig_tol
            if np.any(keep_S):
                eigvals_S_inv = np.zeros_like(eigvals_S)
                eigvals_S_inv[keep_S] = 1.0 / eigvals_S[keep_S]
                S_plus = la.dot(eigvecs_S, la.dot(np.diag(eigvals_S_inv), eigvecs_S.T))
            else:
                S_plus = np.zeros_like(S_dense)

            Qxx_dense = la.to_dense(Qxx).astype(np.float64)
            Qxx_dense = 0.5 * (Qxx_dense + Qxx_dense.T)
            eig_tol_qxx = getattr(la, "eig_tol", None)
            eig_tol_qxx = eig_tol_qxx(Qxx_dense) if callable(eig_tol_qxx) else 1e-10
            eigvals_qxx, eigvecs_qxx = la.eigh(Qxx_dense)
            keep_qxx = eigvals_qxx > eig_tol_qxx
            if np.any(keep_qxx):
                eigvals_qxx_inv = np.zeros_like(eigvals_qxx)
                eigvals_qxx_inv[keep_qxx] = 1.0 / eigvals_qxx[keep_qxx]
                Qxx_inv = la.dot(eigvecs_qxx, la.dot(np.diag(eigvals_qxx_inv), eigvecs_qxx.T))
            else:
                Qxx_inv = np.zeros_like(Qxx_dense)

            G = la.dot(S_plus, Qxz)
            M = la.dot(Qxz.T, G)
            kp_mat = la.dot(Qxx_inv, M)
            kp_mat = 0.5 * (kp_mat + kp_mat.T)
            lam_kp = float(np.min(la.eigvalsh(kp_mat))) if kp_mat.size else float("nan")
            out["kp_min_eig"] = lam_kp
            out["kp_rk_LM"] = float(n * lam_kp) if n > 0 else float("nan")
            df_denom_kp = max(1, n - L2)
            out["kp_rk_Wald_F"] = float(lam_kp * df_denom_kp / max(1, L2)) if L2 > 0 else float("nan")
        except Exception:
            out.setdefault("kp_min_eig", float("nan"))
        return out
    except Exception:
        return out

def _two_sls_qr(  # noqa: PLR0913
    Z: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "qr",
    R: np.ndarray | None = None,
    q: np.ndarray | None = None,
    rank_policy: str = "stata",
) -> np.ndarray:
    """Compute 2SLS coefficients via QR projection."""
    Zd = la.to_dense(Z)
    qr_res = la.qr(Zd, mode="economic", pivoting=True)
    # qr may return (Q, R, piv) or (Q, R) depending on implementation
    if len(qr_res) == 3:
        Qz, Rz, _p = qr_res
    else:
        Qz, Rz = qr_res
    # --- strict: trim to rank(Z) (Stata-compatible) ---
    diagR = np.abs(np.diag(la.to_dense(Rz))) if Rz.size else np.array([])
    # Use centralized rank-from-diagonal routine with supplied rank policy so callers can request R-like behavior.
    r = la.rank_from_diag(diagR, Zd.shape[1], mode=rank_policy) if diagR.size else 0
    if r == 0:
        raise RuntimeError("Instrument matrix Z is rank-deficient (rank=0).")
    Qz = Qz[:, :r]

    # If constraints supplied, delegate to the helper which builds and solves
    # the KKT system on the projected problem.
    if R is not None and q is not None:
        return solve_constrained_2sls(Qz, X, y, R, q)

    X_t = la.dot(Qz.T, X)
    y_t = la.dot(Qz.T, y)
    solver = method if method in {"qr", "svd"} else "qr"
    return la.solve(X_t, y_t, method=solver, rank_policy=rank_policy)


def _partial_f_sw_qr(  # noqa: PLR0913
    X: np.ndarray,
    Z: np.ndarray,
    *,
    endog_idx: Sequence[int],
    z_excluded_idx: Sequence[int],
    var_names: Sequence[str],
    rank_policy: str = "stata",
) -> tuple[float, list[tuple[str, float, str | None]]]:
    """Compute Sanderson-Windmeijer partial F-statistics."""
    endog_names = [var_names[i] for i in endog_idx]
    z2_cols = list(z_excluded_idx)
    # exogenous indices (those X columns not listed as endogenous)
    exog_idx = [i for i in range(len(var_names)) if i not in endog_idx]
    Fj_tuples: list[tuple[str, float, str | None]] = []
    for j, nm in enumerate(endog_names):
        y1 = X[:, [endog_idx[j]]]
        # Always control for exogenous regressors and the other endogenous
        # regressors (Sanderson-Windmeijer style conditioning on X_{-j}).
        other_endog = [i for i in endog_idx if i != endog_idx[j]]
        included_controls = exog_idx + other_endog
        X0 = (
            X[:, included_controls]
            if included_controls
            else np.zeros((X.shape[0], 0), dtype=np.float64)
        )
        Z2 = (
            Z[:, z2_cols]
            if Z.shape[1] > 0
            else np.zeros((X.shape[0], 0), dtype=np.float64)
        )
        X1 = la.hstack([X0, Z2]) if X0.shape[1] > 0 else Z2
        if X1.shape[1] == 0:
            Fj_tuples.append((nm, float("nan"), "No excluded instruments"))
            continue
        # Restricted vs Unrestricted (AER/ivreg style SSR-difference F)
        try:
            b_u = la.solve(X1, y1, method="qr")
            e_u = (y1 - la.dot(X1, b_u)).reshape(-1)
            rss_u = float(la.dot(e_u.reshape(1, -1), e_u.reshape(-1, 1)).squeeze())
        except Exception:  # noqa: BLE001
            Fj_tuples.append((nm, float("nan"), "Unrestricted fit failed"))
            continue
        try:
            if X0.shape[1] > 0:
                b_r = la.solve(X0, y1, method="qr")
                e_r = (y1 - la.dot(X0, b_r)).reshape(-1)
                rss_r = float(la.dot(e_r.reshape(1, -1), e_r.reshape(-1, 1)).squeeze())
            else:
                rss_r = float(la.dot(y1.T, y1).squeeze())
        except Exception:  # noqa: BLE001
            Fj_tuples.append((nm, float("nan"), "Restricted fit failed"))
            continue

        # ---- Sanderson-Windmeijer partial F: (RSS_R - RSS_U)/q_eff divided by MSE_U ----
        num = max(rss_r - rss_u, 0.0)
        n = y1.shape[0]
        # q_eff = rank(Z2 after residualizing on X0) using Stata-style rank
        try:
            if Z2.shape[1] > 0:
                Mx0Z2 = (
                    Z2
                    if X0.shape[1] == 0
                    else (Z2 - la.dot(X0, la.solve(X0, Z2, method="qr")))
                )
                qr_res = la.qr(la.to_dense(Mx0Z2), mode="economic", pivoting=True)
                if len(qr_res) == 3:
                    _Qz2, Rz2, _p2 = qr_res
                else:
                    _Qz2, Rz2 = qr_res
                diagR = np.abs(np.diag(la.to_dense(Rz2))) if Rz2.size else np.array([])
                q_eff = (
                    la.rank_from_diag(diagR, Mx0Z2.shape[1], mode=rank_policy)
                    if diagR.size
                    else 0
                )
            else:
                q_eff = 0
        except Exception as exc:
            raise RuntimeError(
                "Failed to compute q_eff (rank of excluded instruments after partialling).",
            ) from exc
        if q_eff <= 0:
            Fj_tuples.append(
                (nm, float("nan"), "No excluded instruments after partialling"),
            )
            continue

        # p_u: effective parameters in unrestricted model = rank(X1)
        try:
            qr_x1 = la.qr(la.to_dense(X1), mode="economic", pivoting=True)
            if len(qr_x1) == 3:
                _Qx1, Rx1, _p1 = qr_x1
            else:
                _Qx1, Rx1 = qr_x1
            diagRx1 = np.abs(np.diag(la.to_dense(Rx1))) if Rx1.size else np.array([])
            p_u = (
                la.rank_from_diag(diagRx1, X1.shape[1], mode=rank_policy)
                if diagRx1.size
                else 0
            )
        except Exception as exc:
            raise RuntimeError("Failed to compute rank(X1) for df_U.") from exc
        # Common df and F computation for both try success/failure
        df_u_raw = int(n - p_u)
        if df_u_raw <= 0:
            Fj_tuples.append(
                (
                    nm,
                    float("nan"),
                    "Insufficient degrees of freedom in unrestricted model",
                ),
            )
            continue
        df_u = max(1, df_u_raw)
        F_j = (num / float(q_eff)) / (rss_u / float(df_u))
        Fj_tuples.append((nm, float(F_j), None))
    finite_F = [f for _, f, _ in Fj_tuples if np.isfinite(f)]
    minF = min(finite_F) if finite_F else float("nan")
    return minF, Fj_tuples


def _cluster_robust_sw_f(  # noqa: PLR0913
    X: np.ndarray,
    Z: np.ndarray,
    *,
    endog_idx: Sequence[int],
    z_excluded_idx: Sequence[int],
    var_names: Sequence[str],
    clusters: np.ndarray,
    rank_policy: str = "stata",
    ssc: dict[str, Any] | None = None,
    n_original: int | None = None,
    k_params: int | None = None,
) -> tuple[float, list[tuple[str, float, str | None]]]:
    """Cluster-robust Sanderson-Windmeijer-style F for the first stage.

    For each endogenous regressor j (after residualizing on X0), run x̃_j = Z̃2 π̃_j + ẽ_j.
    Let S = Σ_g (Z̃2_g' ẽ_g)(Z̃2_g' ẽ_g)' be the cluster-robust "meat".
    Then Var(π̃̂_j) = (Z̃2'Z̃2)^{-1} S (Z̃2'Z̃2)^{-1} and the Wald statistic for H0: π̃_j=0 is
        W_j = π̃̂_j' [(Z̃2'Z̃2) S^{+} (Z̃2'Z̃2)] π̃̂_j ,
    where S^{+} is the Moore-Penrose pseudo-inverse restricted to Range(S).
    The reported F_j is W_j divided by the effective number of excluded instruments q_eff.

    This statistic is robust to arbitrary heteroskedasticity and clustering,
    providing valid weak-instrument inference in clustered settings.

    Parameters
    ----------
    X : np.ndarray (n x K)
        Design matrix including all regressors (endog + exog).
    Z : np.ndarray (n x L)
        Full instrument matrix.
    endog_idx : Sequence[int]
        Column indices in X of endogenous regressors.
    z_excluded_idx : Sequence[int]
        Column indices in Z of excluded instruments.
    var_names : Sequence[str]
        Names of variables in X (for reporting).
    clusters : np.ndarray (n,)
        Cluster identifiers.

    Returns
    -------
    minF : float
        Minimum effective F across endogenous regressors.
    Fj_tuples : list[tuple[str, float, str | None]]
        Per-regressor diagnostics: (name, F-stat, message-or-None).

    References
    ----------
    Montiel Olea, J. L., & Pflueger, C. (2013). A Robust Test for Weak
    Instruments. Journal of Business & Economic Statistics, 31(3), 358-369.

    Sanderson, E., & Windmeijer, F. (2016). A weak instrument F-test in linear
    IV models with multiple endogenous variables. Journal of Econometrics,
    190(2), 212-221.

    """
    endog_names = [var_names[i] for i in endog_idx]
    z2_cols = list(z_excluded_idx)
    exog_idx = [i for i in range(len(var_names)) if i not in endog_idx]

    # Validate cluster input
    arr = np.asarray(clusters, dtype=object).reshape(-1)
    try:
        import pandas as pd_module  # type: ignore[import]  # noqa: ICN001
    except ImportError:
        pd_module = None
    if pd_module is not None and pd_module.isna(arr).any():
        raise ValueError("clusters contains NA")
    if pd_module is None:
        try:
            arr_float = arr.astype(float, copy=False)
        except (TypeError, ValueError):
            arr_float = None
        if arr_float is not None and np.isnan(arr_float).any():
            raise ValueError("clusters contains NA")
    _, clusters = np.unique(arr, return_inverse=True)
    clusters = clusters.astype(int, copy=False)
    if clusters.shape[0] != X.shape[0]:
        raise ValueError("clusters length must match number of rows in X.")

    Fj_tuples: list[tuple[str, float, str | None]] = []

    for j, nm in enumerate(endog_names):
        x_j = X[:, endog_idx[j]].reshape(-1, 1)  # (n, 1)

        # Conditioning variables: other exog + other endog (Sanderson-Windmeijer)
        other_endog = [i for i in endog_idx if i != endog_idx[j]]
        included_controls = exog_idx + other_endog
        X0 = (
            X[:, included_controls]
            if included_controls
            else np.zeros((X.shape[0], 0), dtype=np.float64)
        )

        # Excluded instruments
        Z2 = Z[:, z2_cols] if z2_cols else np.zeros((X.shape[0], 0), dtype=np.float64)

        if Z2.shape[1] == 0:
            Fj_tuples.append((nm, float("nan"), "No excluded instruments"))
            continue

        # --- Step 1: Residualize w.r.t. included controls (M_{X0} projection) ---
        if X0.shape[1] > 0:
            try:
                # x̃_j = M_{X0} x_j = x_j - X0 (X0'X0)^{-1} X0' x_j
                X0tX0_inv_X0t_xj = la.solve(X0, x_j, method="qr")
                x_tilde = x_j - la.dot(X0, X0tX0_inv_X0t_xj)

                # Z̃2 = M_{X0} Z2
                X0tX0_inv_X0t_Z2 = la.solve(X0, Z2, method="qr")
                Z2_tilde = Z2 - la.dot(X0, X0tX0_inv_X0t_Z2)
            except Exception:  # noqa: BLE001
                Fj_tuples.append((nm, float("nan"), "Residualization failed"))
                continue
        else:
            x_tilde = x_j
            Z2_tilde = Z2

        # --- Step 2: First-stage regression on residualized instruments ---
        # x̃_j = Z̃2 π̃_j + ẽ_j
        try:
            pi_tilde = la.solve(Z2_tilde, x_tilde, method="qr").reshape(-1)  # (L2,)
            e_tilde = (x_tilde - la.dot(Z2_tilde, pi_tilde.reshape(-1, 1))).reshape(
                -1,
            )  # (n,)
        except Exception:  # noqa: BLE001
            Fj_tuples.append((nm, float("nan"), "First-stage regression failed"))
            continue

        # --- Step 3: Cluster-robust S = Σ_g (Z̃2_g' ẽ_g)(Z̃2_g' ẽ_g)' ---
        L2 = Z2_tilde.shape[1]
        S = np.zeros((L2, L2), dtype=np.float64)

        uniq_clusters = np.unique(clusters)
        for g in uniq_clusters:
            mask_g = clusters == g
            Z_g = Z2_tilde[mask_g, :]  # (n_g, L2)
            e_g = e_tilde[mask_g].reshape(-1, 1)  # (n_g, 1)

            # Cluster moment: Z_g' e_g  (L2, 1)
            moment_g = la.crossprod(Z_g, e_g).reshape(-1, 1)  # (L2, 1)

            # Outer product: (Z_g' e_g)(Z_g' e_g)'
            S += la.dot(moment_g, moment_g.T)  # (L2, L2)

        # Apply SSC to S (meat)
        if ssc is not None and n_original is not None:
             ssc_eval = _normalize_ssc(ssc)
             # k_eff for first stage? Typically same as model DF adjustment?
             # For weak IV, we care about structural model SSC?
             # Or first stage SSC?
             # Stata uses model SSC for robust VCV of IV.
             k_eff = k_params if k_params is not None else X.shape[1]
             factor = compute_ssc_correction(n_original, k_eff, clusters=clusters, ssc=ssc_eval)
             if abs(factor - 1.0) > 1e-9:
                 S *= factor

        # --- Step 4: Invert S safely via eigendecomposition (range-restricted pinv) ---
        try:
            evals, Q = la.eigh(S)
            tol = la.eig_tol(S)
            keep = evals > tol

            if not np.any(keep):
                Fj_tuples.append((nm, float("nan"), "Singular S matrix"))
                continue

            # S^{-1} = Q_r diag(1/λ_r) Q_r' (range-restricted)
            evals_inv = 1.0 / evals[keep]
            Q_keep = Q[:, keep]
            S_inv = la.dot(Q_keep, la.dot(np.diag(evals_inv), Q_keep.T))
        except Exception:  # noqa: BLE001
            Fj_tuples.append((nm, float("nan"), "S inversion failed"))
            continue

        # --- Step 5: Wald numerator W = π̃' [ (Z̃2'Z̃2) S^{-1} (Z̃2'Z̃2) ] π̃ ---
        try:
            Z2tZ2 = la.crossprod(Z2_tilde, Z2_tilde)  # (L2, L2)

            # Middle: (Z̃2'Z̃2) S^{-1} (Z̃2'Z̃2)
            middle = la.dot(Z2tZ2, la.dot(S_inv, Z2tZ2))  # (L2, L2)

            # Sandwich: π̃' middle π̃
            A = float(
                la.dot(
                    pi_tilde.reshape(1, -1), la.dot(middle, pi_tilde.reshape(-1, 1)),
                ).squeeze(),
            )
        except Exception:  # noqa: BLE001
            Fj_tuples.append((nm, float("nan"), "Numerator computation failed"))
            continue

        # --- Step 6: Effective F normalization only by q_eff (cluster-robust Wald form) ---

        # Determine effective number of excluded instruments
        try:
            qr_res = la.qr(la.to_dense(Z2_tilde), mode="economic", pivoting=True)
            if len(qr_res) == 3:
                _Q, R, _p = qr_res
            else:
                _Q, R = qr_res
            diagR = np.abs(np.diag(la.to_dense(R))) if R.size else np.array([])
            # Use caller-specified rank policy
            q_eff = (
                la.rank_from_diag(diagR, Z2_tilde.shape[1], mode=rank_policy)
                if diagR.size
                else 0
            )
        except Exception as exc:
            raise RuntimeError("Failed to compute q_eff under clustering.") from exc

        if q_eff <= 0:
            Fj_tuples.append((nm, float("nan"), "No effective excluded instruments"))
            continue

        # Effective F (cluster-robust SW): F = A / q_eff
        F_eff = A / float(q_eff)
        Fj_tuples.append((nm, float(F_eff), None))

    finite_F = [f for _, f, _ in Fj_tuples if np.isfinite(f)]
    minF = min(finite_F) if finite_F else float("nan")
    return minF, Fj_tuples


class IV2SLS(BaseEstimator):
    """Two-Stage Least Squares (2SLS) estimator.

    Estimates y = Xβ + u using instruments Z. Implements QR-based estimation
    with weak IV diagnostics (Cragg-Donald, Kleibergen-Paap, Sanderson-Windmeijer)
    and bootstrap inference.
    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        Dependent variable (outcome).
    X : array-like, shape (n, p)
        Full set of regressors including both exogenous and endogenous variables.
    Z : array-like, shape (n, q)
        Instrument matrix. Should include all exogenous regressors (X_exog) plus
        excluded instruments (Z_excluded). Total q ≥ p required for identification.
    endog_idx : Sequence[int]
        Indices of endogenous variables in X (0-indexed).
    z_excluded_idx : Sequence[int]
        Indices of excluded instruments in Z (0-indexed). These are instruments
        not in X.
    include_exog_in_Z : bool, default=True
        MUST be True for R/Stata parity. Exogenous regressors must appear in Z.
        Setting to False raises ValueError.
    add_const : bool, default=True
        If True, adds constant to X and Z. Constant is last column.
    var_names : Sequence[str], optional
        Names for X variables.
    instr_names : Sequence[str] | None, optional
        Names for Z instruments.

    Attributes
    ----------
    y_orig : ndarray, shape (n, 1)
        Original outcome variable.
    X_orig : ndarray, shape (n, p)
        Original design matrix with endogenous and exogenous variables.
    Z_orig : ndarray, shape (n, q)
        Original instrument matrix.
    _var_names : list of str
        Variable names for X.
    _instr_names : list of str
        Instrument names for Z.
    _endog_idx : list of int
        Indices of endogenous variables.
    _z_excluded_idx : list of int
        Indices of excluded instruments.

    Methods
    -------
    from_formula(formula, data, iv=None, id=None, time=None, options=None, W_dict=None)
        Create IV2SLS model from formula with IV clause.
    fit(boot=None, fe_codes_list=None, cluster_ids=None)
        Fit 2SLS model with optional bootstrap inference.

    Returns
    -------
    EstimationResult
        Object containing:
        - params : pd.Series - 2SLS coefficient estimates β̂
        - se : pd.Series - Bootstrap standard errors
        - extra : dict - Weak IV diagnostics (Cragg-Donald, Kleibergen-Paap,
                        Sanderson-Windmeijer F-stats, Hansen J-stat, etc.)

    Examples
    --------
    Basic IV/2SLS regression:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from lineareg.estimators.iv import IV2SLS
    >>> from lineareg.estimators.base import BootConfig
    >>>
    >>> # Generate IV data
    >>> np.random.seed(42)
    >>> n = 300
    >>> z1 = np.random.randn(n)
    >>> z2 = np.random.randn(n)
    >>> x_exog = np.random.randn(n)
    >>> x_endog = 0.8 * z1 + 0.6 * z2 + np.random.randn(n) * 0.3
    >>> y = 1.5 + 2.0 * x_endog + 1.0 * x_exog + np.random.randn(n) * 0.5
    >>>
    >>> df = pd.DataFrame({
    ...     'y': y, 'x_endog': x_endog, 'x_exog': x_exog, 'z1': z1, 'z2': z2
    ... })
    >>>
    >>> # Fit IV model using formula API
    >>> model = IV2SLS.from_formula(
    ...     "y ~ x_exog + x_endog",
    ...     df,
    ...     iv="(x_endog ~ z1 + z2)"
    ... )
    >>> result = model.fit(boot=BootConfig(n_boot=2000, seed=42))
    >>> print(result.params)
    >>> print(result.se)

    Check weak IV diagnostics:

    >>> from lineareg.output.summary import weakiv_table
    >>> print(weakiv_table(result))
    >>> # Shows: Cragg-Donald, Kleibergen-Paap, Sanderson-Windmeijer F-stats

    Overidentification test (Hansen J-statistic):

    >>> print("Hansen J-stat:", result.extra.get('J_stat'))
    >>> print("OverID df:", result.extra.get('OverID_df'))  # values-only; no p-values by policy

    Multiple endogenous variables:

    >>> # y ~ x1 + x2 + x_endog1 + x_endog2, with instruments z1, z2, z3, z4
    >>> model = IV2SLS.from_formula(
    ...     "y ~ x1 + x2 + x_endog1 + x_endog2",
    ...     df,
    ...     iv="(x_endog1 + x_endog2 ~ z1 + z2 + z3 + z4)"
    ... )
    >>> result = model.fit(boot=BootConfig(n_boot=2000))

    Notes
    -----
    - First stage: X_endog = Z π + v (project endogenous vars onto instruments)
    - Second stage: y = X̂ β + u (use predicted X̂_endog from first stage)
    - Weak IV diagnostics:
        * Cragg-Donald: Minimum eigenvalue statistic
        * Kleibergen-Paap: Robust to heteroskedasticity
        * Sanderson-Windmeijer: Partial F-stat for each endogenous variable
        * Stock-Yogo critical values embedded (5%, 10%, 15%, 20%, 25%, 30% bias)
    - Hansen J-test: Overidentification test (valid when q > p)
    - All standard errors from wild bootstrap (no analytical SE)

    References
    ----------
    .. [1] Stock, J. H., & Yogo, M. (2005). "Testing for Weak Instruments in Linear
           IV Regression." In Identification and Inference for Econometric Models:
           Essays in Honor of Thomas Rothenberg (pp. 80-108). Cambridge University Press.
    .. [2] Sanderson, E., & Windmeijer, F. (2016). "A Weak Instrument F-Test in Linear
           IV Models with Multiple Endogenous Variables." Journal of Econometrics,
           190(2), 212-221.
    .. [3] Kleibergen, F., & Paap, R. (2006). "Generalized Reduced Rank Tests Using
           the Singular Value Decomposition." Journal of Econometrics, 133(1), 97-126.
    .. [4] Hansen, L. P. (1982). "Large Sample Properties of Generalized Method of
           Moments Estimators." Econometrica, 50(4), 1029-1054.
    .. [5] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel
           Data (2nd ed.). MIT Press, Chapter 8.

    See Also
    --------
    GMM : Generalized method of moments (efficient IV with optimal weighting)
    OLS : Ordinary least squares (no endogeneity)

    """

    def __init__(  # noqa: PLR0913
        self,
        y: ArrayLike,
        X: MatrixLike,
        Z: MatrixLike,
        *,
        endog_idx: Sequence[int],
        z_excluded_idx: Sequence[int],
        include_exog_in_Z: bool = True,
        add_const: bool = True,
        var_names: Sequence[str] | None = None,
        instr_names: Sequence[str] | None = None,
    ) -> None:
        """Initialize IV2SLS estimator."""
        super().__init__()
        # Store excluded-instrument indices early so downstream bookkeeping does
        # not reference an unset attribute (runtime safety).
        self.z_excluded_idx = [int(i) for i in z_excluded_idx]
        # Strict R/Stata compliance: exogenous regressors MUST be in Z (always).
        # The user argument `include_exog_in_Z=False` is no longer accepted to
        # avoid non-standard behavior; raise an explicit error to prevent silent
        # API surprises.
        if not include_exog_in_Z:
            raise ValueError(
                "R/Stata convention: exogenous X must always be included in Z (include_exog_in_Z=False is not allowed).",
            )
        # Force to True for strict R/Stata parity
        include_exog_in_Z = True

        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if var_names is None and isinstance(X, pd.DataFrame):
            var_names = list(X.columns)
        if instr_names is None and isinstance(Z, pd.DataFrame):
            instr_names = list(Z.columns)
        X_arr = np.asarray(X, dtype=np.float64)
        Z_arr = np.asarray(Z, dtype=np.float64)
        if add_const:
            X_aug, x_names_out, const_name = add_constant(X_arr, var_names)
            # Avoid forcing addition of an intercept to Z when Z already
            # contains any common intercept alias (e.g. _cons, (Intercept), Intercept, const).
            # For Stata/R dialect (default), DO NOT use force_name parameter
            # Just call add_constant normally; it will use dialect default (_cons for Stata)
            Z_aug, z_names_out, _ = add_constant(Z_arr, instr_names)
            self._const_name = const_name
            self._var_names = list(x_names_out)
            self._instr_names = list(z_names_out)
        else:
            X_aug, Z_aug = X_arr, Z_arr
            self._const_name = None
            self._var_names = (
                list(X.columns)
                if isinstance(X, pd.DataFrame)
                else (
                    list(var_names)
                    if var_names is not None
                    else [f"x{i}" for i in range(X_arr.shape[1])]
                )
            )
            self._instr_names = (
                list(Z.columns)
                if isinstance(Z, pd.DataFrame)
                else (
                    list(instr_names)
                    if instr_names is not None
                    else [f"z{i}" for i in range(Z_arr.shape[1])]
                )
            )
        if len(self._var_names) != X_aug.shape[1]:
            self._var_names = [f"x{i}" for i in range(X_aug.shape[1])]
        if len(self._instr_names) != Z_aug.shape[1]:
            self._instr_names = [f"z{i}" for i in range(Z_aug.shape[1])]

        orig_var_names = (
            list(var_names) if var_names is not None
            else [f"x{i}" for i in range(X_arr.shape[1])]
        )
        if add_const:
            endog_names = [orig_var_names[i] for i in endog_idx if 0 <= i < len(orig_var_names)]
            self.endog_idx = [
                self._var_names.index(nm) for nm in endog_names
                if nm in self._var_names
            ]
            if len(self.endog_idx) != len(endog_idx):
                raise ValueError(
                    f"endog_idx references columns not found after add_constant. "
                    f"Original indices: {list(endog_idx)}, mapped names: {endog_names}, "
                    f"found in X_aug: {self.endog_idx}"
                )
        else:
            self.endog_idx = [int(i) for i in endog_idx]

        # Integrate included exogenous regressors into Z *only if* they are not
        # already present in the provided instrument set. The formula parser's
        # `iv_instr_full` already includes included exogenous terms, so blindly
        # prepending would duplicate columns.
        exog_idx = [j for j in range(X_aug.shape[1]) if j not in self.endog_idx]
        const_j = (
            None
            if self._const_name is None
            else (self._var_names.index(self._const_name) if self._const_name in self._var_names else None)
        )
        exog_names = [self._var_names[j] for j in exog_idx if (const_j is None or j != const_j)]
        instr_name_set = set(self._instr_names)
        exog_to_add = [
            self._var_names.index(nm) for nm in exog_names
            if nm in self._var_names and nm not in instr_name_set
        ]

        Z_full = (
            np.column_stack([la.to_dense(X_aug)[:, exog_to_add], la.to_dense(Z_aug)])
            if exog_to_add
            else la.to_dense(Z_aug)
        )
        instr_names_full = (
            [self._var_names[j] for j in exog_to_add] + list(self._instr_names)
            if exog_to_add
            else list(self._instr_names)
        )

        # Excluded indices refer to the original Z columns; when we prepend
        # missing included-exog columns, excluded indices shift.
        excluded0 = [int(i) for i in self.z_excluded_idx]
        if exog_to_add:
            excluded0 = [int(i) + len(exog_to_add) for i in excluded0]

        # Use stable order-preserving rank drop helper from core.linalg.
        Z_full_d, keep = la.drop_rank_deficient_cols_stable(Z_full, mode="stata")
        if Z_full_d.shape[1] < Z_full.shape[1]:
            self._dropped_instr_names = [
                nm for nm, k in zip(instr_names_full, keep) if not k
            ]
            self._instr_names = [nm for nm, k in zip(instr_names_full, keep) if k]
        else:
            self._dropped_instr_names = []
            self._instr_names = list(instr_names_full)

        Z_aug = Z_full_d
        # Re-index excluded instruments transparently after stable dropping.
        excluded_mask = np.zeros(Z_full.shape[1], dtype=bool)
        for j in excluded0:
            if 0 <= int(j) < excluded_mask.size:
                excluded_mask[int(j)] = True
        self.z_excluded_idx = np.flatnonzero(excluded_mask[keep]).tolist()

        # ---- Column-space safety: endogenous regressor must not be duplicated in instruments ----
        try:
            if getattr(self, "endog_idx", None) and la is not None and la.to_dense(Z_aug).size:
                Zd = la.to_dense(Z_aug)
                Xd = la.to_dense(X_aug)
                for j in self.endog_idx:
                    xj = Xd[:, int(j)]
                    xnorm = la.norm(xj)
                    if not np.isfinite(xnorm) or xnorm == 0.0:
                        continue
                    xunit = xj / float(xnorm)
                    # check maximum absolute correlation with any instrument column
                    maxcorr = 0.0
                    for ii in range(Zd.shape[1]):
                        zi = Zd[:, ii]
                        zn = la.norm(zi)
                        if not np.isfinite(zn) or zn == 0.0:
                            continue
                        dot_result = la.dot(zi.reshape(-1, 1).T, xunit.reshape(-1, 1))
                        # cosine similarity: |(z/||z||)' (x/||x||)| in [0,1]
                        corr = abs(float(np.asarray(dot_result).ravel()[0]) / float(zn))
                        if corr > maxcorr:
                            maxcorr = corr
                    if maxcorr > 1.0 - 1e-12:
                        raise ValueError(
                            "Invalid IV specification: an endogenous regressor column is duplicated in instruments (detected by column-space check)."
                        )
        except ValueError:
            raise
        except Exception:
            # If any unexpected failure occurs (e.g., backend issues), do not block
            # construction here; fit-time rank checks still enforce identification.
            pass

        self.y_orig = y_arr
        self.X_orig = X_aug
        self.Z_orig = Z_aug
        self._n_obs_init = y_arr.shape[0]
        self._n_features_init = X_aug.shape[1]
        # default rank policy (Stata-compatible by default); can be overridden at fit time
        self._rank_policy = "stata"
        # optional HAC lags for MOP effective-F moment covariance (None => no HAC)
        self._hac_lags = None

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
    ) -> IV2SLS:
        """2SLS from formula + IV clause.
        Example: IV2SLS.from_formula("y ~ x1 + x2", df, iv="(x2 ~ z1 + z2)")
        """
        parser = FormulaParser(data, id_name=id_name, t_name=time, W_dict=W_dict)
        parsed = parser.parse(formula, iv=iv, options=options)
        if not parsed.get("iv_endog"):
            raise ValueError(
                "IV2SLS.from_formula requires an IV clause specifying endogenous regressors.",
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
        # Use rows accepted by Patsy/parser to build Z (keeps alignment)
        Z_full, z_names, _ = parser.patsy_matrix(
            df_use, parsed.get("iv_instr_full") or "0",
        )
        if parsed.get("iv_instr_user"):
            _Z_user, z_user_names, _ = parser.patsy_matrix(
                df_use, " + ".join(parsed["iv_instr_user"]),
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
            z_excluded_idx=z_excluded_idx,
            add_const=bool(parsed.get("include_intercept", True)),
            var_names=parsed["var_names"],
            instr_names=z_names,
        )
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
                list(model._var_names),
                const_aliases=const_aliases,
            )
            model._constraints_from_formula = (R, q)
        return model

    def fit(  # noqa: PLR0913
        self,
        *,
        absorb_fe: pd.DataFrame | np.ndarray | None = None,
        fe_backend: str = "reghdfe",
        fe_tol: float | None = None,
        fe_max_iter: int | None = None,
        device: str | None = None,
        constraints: np.ndarray | None = None,
        constraint_vals: np.ndarray | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        multiway_ids: Sequence[Sequence] | None = None,
        method: str = "qr",
        ssc: dict[str, str | int] | None = None,
        boot: BootConfig | None = None,
        weights: Sequence[float] | None = None,
        rank_policy: str = "stata",
        hac_lags: int | None = None,
    ) -> EstimationResult:
        # Enforce weight policy for IV using centralized BaseEstimator helper.
        # This ensures consistent messaging and behavior across estimators.
        self._enforce_weight_policy("iv", weights)

        absorb_fe = self._absorb_fe_from_formula(absorb_fe)

        # FE absorption
        dropped_stats: dict[str, Any] = {}
        if absorb_fe is not None:
            be = str(fe_backend).lower()
            if be not in {"reghdfe", "fixest"}:
                raise ValueError("fe_backend must be one of {'reghdfe','fixest'}.")
            tol_eff = (
                (1e-8 if be == "reghdfe" else 1e-6) if fe_tol is None else float(fe_tol)
            )
            it_eff = (
                (16000 if be == "reghdfe" else 2000)
                if fe_max_iter is None
                else int(fe_max_iter)
            )
            Xw, Zw, yw, mask, dropped_stats = fe_core.demean_xyz(
                self.X_orig,
                self.Z_orig,
                self.y_orig,
                absorb_fe,
                na_action="drop",
                drop_na_fe_ids=True,
                drop_singletons=True,
                backend=be,
                tol=tol_eff,
                max_iter=it_eff,
                return_mask=True,
                return_dropped_stats=True,
            )
            try:
                import pandas as pd_module  # type: ignore[import]  # noqa: ICN001
            except ImportError:
                pd_module = None
            try:
                if pd_module is not None and isinstance(absorb_fe, pd_module.DataFrame):
                    fe_ids_masked = [
                        absorb_fe[col].to_numpy()[mask] for col in absorb_fe.columns
                    ]
                elif pd_module is not None and isinstance(absorb_fe, pd_module.Series):
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
            except Exception:  # noqa: BLE001
                fe_dof_info = {}
        else:
            Xw, Zw, yw = self.X_orig, self.Z_orig, self.y_orig
            mask = np.ones(self.X_orig.shape[0], dtype=bool)
            fe_dof_info = {}

        # Bootstrap policy for IV:
        # - Default to wild multipliers (Rademacher) like boottest/fwildclusterboot.
        if boot is None:
            boot = BootConfig()

        boot, cluster_spec = self._coerce_bootstrap(
            boot=boot,
            n_obs_original=self._n_obs_init,
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
        n_eff = Xw.shape[0]

        if int(n_eff) <= 0:
            raise ValueError(
                "No observations remain after NA/FE/singleton filtering; IV estimation is undefined.",
            )

        # apply and store rank policy and hac_lags on the instance for use in
        # QR-based rank decisions and solver calls. Accept values 'stata' or 'R'.
        rp = str(rank_policy).lower() if rank_policy is not None else "stata"
        if rp not in {"stata", "r"}:
            raise ValueError("rank_policy must be 'stata' or 'r'")
        self._rank_policy = rp
        self._hac_lags = None if hac_lags is None else int(hac_lags)

        # Remove zero-variance columns using a scale-aware tolerance from core.linalg
        # IMPORTANT: Protect constant columns from removal (they have zero variance by design)
        # Detect constant columns directly: max(col) - min(col) < tol
        def _is_constant_col(col: np.ndarray, tol: float = 1e-10) -> bool:
            col_arr = np.asarray(col).ravel()
            return float(np.max(col_arr) - np.min(col_arr)) < tol

        const_col_X = None
        if self._const_name is not None and self._const_name in self._var_names:
            const_col_X = self._var_names.index(self._const_name)
        const_col_Z = None
        if self._const_name is not None and self._const_name in self._instr_names:
            const_col_Z = self._instr_names.index(self._const_name)

        if Xw.size:
            n_rows = float(Xw.shape[0])
            sumX = la.dot(np.ones((1, Xw.shape[0])), Xw).ravel()
            diag_XX = np.diag(la.to_dense(la.crossprod(Xw, Xw))) / n_rows
            varX = diag_XX - (sumX / n_rows) ** 2
        else:
            varX = np.array([])
        if Zw.size:
            n_rows_z = float(Zw.shape[0])
            sumZ = la.dot(np.ones((1, Zw.shape[0])), Zw).ravel()
            diag_ZZ = np.diag(la.to_dense(la.crossprod(Zw, Zw))) / n_rows_z
            varZ = diag_ZZ - (sumZ / n_rows_z) ** 2
        else:
            varZ = np.array([])
        tolX = la.eig_tol(np.diag(varX)) if varX.size else np.finfo(float).eps
        tolZ = la.eig_tol(np.diag(varZ)) if varZ.size else np.finfo(float).eps

        # Remove zero-variance columns, but KEEP any constant column (user-supplied or added)
        # Create keep masks that protect constant columns
        keepX = np.array([True] * Xw.shape[1])
        for i in range(Xw.shape[1]):
            is_const = (i == const_col_X) or _is_constant_col(Xw[:, i])
            if not is_const and varX[i] < tolX:
                keepX[i] = False
        Xw = Xw[:, keepX] if not np.all(keepX) else Xw

        keepZ = np.array([True] * Zw.shape[1])
        for i in range(Zw.shape[1]):
            is_const_z = (i == const_col_Z) or _is_constant_col(Zw[:, i])
            if not is_const_z and varZ[i] < tolZ:
                keepZ[i] = False
        Zw = Zw[:, keepZ] if not np.all(keepZ) else Zw
        var_names_work = [name for name, keep in zip(self._var_names, keepX) if keep]
        instr_names_work = [
            name for name, keep in zip(self._instr_names, keepZ) if keep
        ]

        # --- New: strict Stata-style QRCP collinearity screening (post zero-variance) ---
        def _keep_columns_strict_local(A: np.ndarray) -> np.ndarray:
            _Qm, Rm, piv = la.qr(A, mode="economic", pivoting=True)
            Rm_dense = la.to_dense(Rm) if getattr(Rm, "size", None) else np.array([])
            diagR = np.abs(np.diag(Rm_dense)) if Rm_dense.size else np.array([])
            if diagR.size == 0:
                return np.zeros(A.shape[1], dtype=bool)
            # Stata/Mata qrsolve rank threshold: eta = 1e-13 * trace(|R|) / rows(R)
            eta = 1e-13 * float(np.sum(diagR)) / float(diagR.size)
            rkeep = int(np.sum(diagR > eta))
            keep_local = np.zeros(A.shape[1], dtype=bool)
            if rkeep > 0:
                keep_local[np.asarray(piv[:rkeep], dtype=int)] = True
            return keep_local

        # Compute local keep masks on the already-trimmed Xw and Zw (these map
        # to the positions in the current Xw/Zw arrays, which correspond to the
        # True positions in the prior keepX/keepZ masks). We must map back to
        # original column indices for consistent downstream bookkeeping.
        if Xw.size:
            keep_local_X = _keep_columns_strict_local(Xw)
        else:
            keep_local_X = np.array([], dtype=bool)
        if Zw.size:
            keep_local_Z = _keep_columns_strict_local(Zw)
        else:
            keep_local_Z = np.array([], dtype=bool)

        # Map local keeps back to original full-length masks
        if keep_local_X.size:
            orig_kept_idx_X = np.flatnonzero(keepX)
            new_keepX = np.zeros_like(keepX, dtype=bool)
            new_keepX[orig_kept_idx_X[keep_local_X]] = True
            dropped_X = [name for name, k in zip(self._var_names, new_keepX) if not k]
        else:
            new_keepX = keepX
            dropped_X = []
        if keep_local_Z.size:
            orig_kept_idx_Z = np.flatnonzero(keepZ)
            new_keepZ = np.zeros_like(keepZ, dtype=bool)
            new_keepZ[orig_kept_idx_Z[keep_local_Z]] = True
            dropped_Z = [name for name, k in zip(self._instr_names, new_keepZ) if not k]
        else:
            new_keepZ = keepZ
            dropped_Z = []

        # If any columns were dropped by strict QRCP screening, apply them and record
        if not np.all(new_keepX == keepX):
            Xw = (
                Xw[:, keep_local_X]
                if keep_local_X.size
                else np.empty((Xw.shape[0], 0), dtype=float)
            )
            keepX = new_keepX
            var_names_work = [
                name for name, keep in zip(self._var_names, keepX) if keep
            ]
        if not np.all(new_keepZ == keepZ):
            Zw = (
                Zw[:, keep_local_Z]
                if keep_local_Z.size
                else np.empty((Zw.shape[0], 0), dtype=float)
            )
            keepZ = new_keepZ
            instr_names_work = [
                name for name, keep in zip(self._instr_names, keepZ) if keep
            ]

        if dropped_stats is None:
            dropped_stats = {}
        dropped_stats.setdefault("dropped_collinear_X", []).extend(dropped_X)
        dropped_stats.setdefault("dropped_collinear_Z", []).extend(dropped_Z)

        # remap indices
        old2new_X = {old: new for new, old in enumerate(np.flatnonzero(keepX))}
        old2new_Z = {old: new for new, old in enumerate(np.flatnonzero(keepZ))}
        endog_idx_new = [old2new_X[i] for i in self.endog_idx if i in old2new_X]
        # If any endogenous regressor was removed during zero-variance/collinearity
        # screening, this is a fatal condition for IV estimation: raise an error
        # rather than silently proceeding with a reduced endogenous set.
        if len(endog_idx_new) != len(self.endog_idx):
            raise ValueError(
                "One or more endogenous regressors were removed by zero-variance or collinearity screening; cannot proceed.",
            )
        z_excluded_idx_new = [
            old2new_Z[i] for i in self.z_excluded_idx if i in old2new_Z
        ]

        # --- Column-space instrument integration + order-preserving rank screening ---
        # If Zw already contains the included exogenous regressors (as it does
        # for formula-based `iv_instr_full`), do not prepend them again.
        exog_idx_work = [j for j in range(Xw.shape[1]) if j not in endog_idx_new]
        exog_names_work = [var_names_work[j] for j in exog_idx_work]
        instr_set_work = set(instr_names_work)
        prepend_exog = bool(exog_idx_work) and (not set(exog_names_work).issubset(instr_set_work))

        if prepend_exog:
            Z_full = (
                la.hstack([la.to_dense(Xw)[:, exog_idx_work], la.to_dense(Zw)])
                if Zw.size
                else la.to_dense(Xw)[:, exog_idx_work]
            )
            z_names_aug = exog_names_work + instr_names_work
        else:
            Z_full = la.to_dense(Zw) if Zw.size else np.empty((Xw.shape[0], 0), dtype=float)
            z_names_aug = instr_names_work

        # Order-preserving rank screening via core.linalg helper
        Z_full_d, keep_Zfull = la.drop_rank_deficient_cols_stable(Z_full, mode=self._rank_policy)
        # If a prepended included exogenous regressor is dropped here, it means it is
        # redundant in the instrument column space under the current numerical rank
        # rule. This is not fatal: identification relies on span(Z), not literal
        # column inclusion.
        if exog_idx_work and not np.all(keep_Zfull[: len(exog_idx_work)]):
            dropped_stats.setdefault(
                "rank_screening_dropped_included_exog_as_instrument",
                [z_names_aug[i] for i in range(len(exog_idx_work)) if not keep_Zfull[i]],
            )

        if Z_full_d.shape[1] < Z_full.shape[1]:
            dropped = [nm for nm, k in zip(z_names_aug, keep_Zfull) if not k]
            self._dropped_instr_names = dropped
        else:
            self._dropped_instr_names = []

        # Update working instrument matrix and names to the screened set
        Zw = Z_full_d
        instr_names_work = [nm for nm, k in zip(z_names_aug, keep_Zfull) if k]

        # Re-index excluded instruments: original z_excluded_idx_new refer to
        # columns in the pre-augmented Zw (i.e., excluded instruments after
        # zero-var/collinearity/dedup). Shift indices by offset of prepended
        # exogenous columns and then filter by kept mask.
        try:
            offset = len(exog_idx_work) if prepend_exog else 0
            excluded_mask = np.zeros(Z_full.shape[1], dtype=bool)
            for j in z_excluded_idx_new:
                jj = offset + int(j)
                if 0 <= jj < excluded_mask.size:
                    excluded_mask[jj] = True
            self.z_excluded_idx = np.flatnonzero(excluded_mask[keep_Zfull]).tolist()
        except Exception:
            # fallback: keep previous indexing if something unexpected happens
            self.z_excluded_idx = [int(i) for i in z_excluded_idx_new]

        # --- Column-space safety: endogenous regressor must not be duplicated in instruments ---
        if endog_idx_new and Zw.size:
            Z_norm = la.norm(Zw, axis=0)
            Z_norm = np.where(Z_norm == 0.0, 1.0, Z_norm)
            Z_unit = Zw / Z_norm
            for j in endog_idx_new:
                xj = Xw[:, int(j)]
                xnorm = float(la.norm(xj))
                if xnorm == 0.0:
                    continue
                xunit = xj / xnorm
                corr = float(np.max(np.abs(la.dot(Z_unit.T, xunit))))
                if corr > 1.0 - 1e-12:
                    raise ValueError(
                        "Invalid IV specification: an endogenous regressor column is duplicated in instruments "
                        "(detected by column-space check)."
                    )

        # --- Drop exact duplicate columns AFTER FE absorption and zero-var removal
        # We need to return a keep mask so that var/instrument names and any
        # index mappings (endog_idx_new, z_excluded_idx_new) remain consistent.
        def _dedup_with_mask(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return (A_dedup, keep_mask) by exact column equality in double precision.

            This performs an exact equality check (np.array_equal) for each
            column and keeps only the first occurrence of duplicates. No
            tolerance-based matching is used to preserve strict reproducibility.
            """
            if A.size == 0:
                return A, np.zeros((0,), dtype=bool)
            ncols = A.shape[1]

            # Pre-normalize near-intercept columns: if a column is within the
            # rank_policy tolerance of being all-ones, round it to exact ones
            # to avoid duplicate-intercept additions caused by FE absorption
            # numeric noise. Tolerance mirrors rank rules (R vs Stata).
            def _normalize_intercept(col: np.ndarray) -> np.ndarray:
                v = col.reshape(-1)
                tol = 1e-7 if self._rank_policy == "r" else 1e-10
                if np.all(np.isfinite(v)) and float(np.max(np.abs(v - 1.0))) <= tol:
                    return np.ones_like(col)
                return col

            keep = np.zeros(ncols, dtype=bool)
            reprs: list[np.ndarray] = []
            for j in range(ncols):
                col = _normalize_intercept(A[:, j].reshape(-1, 1))
                if not reprs:
                    reprs.append(col)
                    keep[j] = True
                    continue
                dup = any(np.array_equal(col, r) for r in reprs)
                if not dup:
                    reprs.append(col)
                    keep[j] = True
            return (A[:, keep], keep)

        if Zw.size:
            Zw, keepZ_dedup = _dedup_with_mask(Zw)
            instr_names_work = [nm for nm, k in zip(instr_names_work, keepZ_dedup) if k]
            # remap excluded Z indices after dedup (only keep those that survived)
            old2newZ_dedup = {
                old: new for new, old in enumerate(np.flatnonzero(keepZ_dedup))
            }
            z_excluded_idx_new = [
                old2newZ_dedup[i] for i in z_excluded_idx_new if i in old2newZ_dedup
            ]
        if Xw.size:
            Xw, keepX_dedup = _dedup_with_mask(Xw)
            var_names_work = [nm for nm, k in zip(var_names_work, keepX_dedup) if k]
            # Update the *original-column* keep mask to reflect post-dedup column drops.
            # Constraints are specified on the full (post-constant) X design, so we
            # must reduce them using the final keep mask that matches Xw.
            try:
                orig_kept_idx_X = np.flatnonzero(keepX)
                keepX_updated = np.zeros_like(keepX, dtype=bool)
                keepX_updated[orig_kept_idx_X[np.asarray(keepX_dedup, dtype=bool)]] = True
                keepX = keepX_updated
            except Exception:  # noqa: BLE001
                pass
            old2newX_dedup = {
                old: new for new, old in enumerate(np.flatnonzero(keepX_dedup))
            }
            endog_idx_new = [
                old2newX_dedup[i] for i in endog_idx_new if i in old2newX_dedup
            ]

        endog_names_current = [var_names_work[i] for i in endog_idx_new]
        for ename in endog_names_current:
            if ename in instr_names_work:
                raise ValueError(
                    f"Endogenous regressor '{ename}' is included in Z. "
                    "This causes 2SLS to collapse to OLS. Remove it from instruments or fix the IV specification."
                )

        # --- Reduce constraints consistently with kept X columns (strict) ---
        # Auto-inject formula-derived constraints when absent
        if (
            constraints is None
            and getattr(self, "_constraints_from_formula", None) is not None
        ):
            constraints, constraint_vals = self._constraints_from_formula

        if (constraints is None) != (constraint_vals is None):
            raise ValueError(
                "constraints and constraint_vals must be provided together (or both omitted).",
            )

        if constraints is not None:
            C = np.asarray(constraints, dtype=np.float64)
            # Constraints are specified for the FULL X matrix (after add_constant),
            # so check against self.X_orig.shape[1], not keepX.size
            if C.shape[1] != self.X_orig.shape[1]:
                raise ValueError(
                    f"Constraints width ({C.shape[1]}) must match # of regressors ({self.X_orig.shape[1]}) after add_constant.",
                )
            C_red = C[:, keepX]
            r_vec = np.asarray(constraint_vals, dtype=np.float64).reshape(-1, 1)
            nonzero = np.any(np.abs(C_red) > 0, axis=1)
            # 0 = nonzero is infeasible
            bad = (~nonzero) & (np.abs(r_vec.reshape(-1)) > 0)
            if np.any(bad):
                raise ValueError(
                    "A constraint became infeasible after dropping zero-variance/collinear columns.",
                )
            # drop redundant 0=0 rows
            C_red, r_vec = C_red[nonzero, :], r_vec[nonzero, :]
            constraints, constraint_vals = C_red, r_vec

        # --- Identification check (constraint-aware, R/Stata aligned) ---
        # Primary: rank(Qz' X) = rank(P_Z X) (ensures instruments have relevance for X)
        try:
            qrZ = la.qr(la.to_dense(Zw), mode="economic", pivoting=True)
            if len(qrZ) == 3:
                Qz, Rz, _pz = qrZ
            else:
                Qz, Rz = qrZ
            diagRz = np.abs(np.diag(la.to_dense(Rz))) if Rz.size else np.array([])
            rZ = (
                la.rank_from_diag(diagRz, Zw.shape[1], mode=self._rank_policy)
                if diagRz.size
                else 0
            )
        except Exception:  # noqa: BLE001
            rank_ZX = 0
        else:
            if rZ == 0:
                raise RuntimeError(
                    "Z has rank 0 (all columns collinear). IV estimation is not feasible.",
                )
            Qz = Qz[:, :rZ]
            Xt = la.dot(Qz.T, Xw)
            qrXt = la.qr(la.to_dense(Xt), mode="economic", pivoting=True)
            if len(qrXt) == 3:
                _Qx, Rx, _px = qrXt
            else:
                _Qx, Rx = qrXt
            diagRx = np.abs(np.diag(la.to_dense(Rx))) if Rx.size else np.array([])
            rank_ZX = (
                la.rank_from_diag(diagRx, Xt.shape[1], mode=self._rank_policy)
                if diagRx.size
                else 0
            )

        # Secondary: rank(Z) metadata
        try:
            Rz_init = la.qr(la.to_dense(Zw), mode="economic", pivoting=True)
            if len(Rz_init) == 3:
                _Qz0, Rz0, _pz0 = Rz_init
            else:
                _Qz0, Rz0 = Rz_init
            diagR0 = np.abs(np.diag(la.to_dense(Rz0))) if Rz0.size else np.array([])
            L_c = (
                la.rank_from_diag(diagR0, Zw.shape[1], mode=self._rank_policy)
                if diagR0.size
                else 0
            )
        except Exception:  # noqa: BLE001
            L_c = 0

        # compute rank of linear constraints R (if any) to get effective parameter count
        rank_R = 0
        if (
            constraints is not None
            and constraint_vals is not None
            and getattr(constraints, "size", 0) > 0
        ):
            try:
                R_dense = np.asarray(constraints, dtype=np.float64)
                R_qr = la.qr(R_dense, mode="economic", pivoting=True)
                if len(R_qr) == 3:
                    _Q_R, R_R, _pR = R_qr
                else:
                    _Q_R, R_R = R_qr
                diagR_R = (
                    np.abs(np.diag(la.to_dense(R_R))) if R_R.size else np.array([])
                )
                rank_R = (
                    la.rank_from_diag(diagR_R, R_dense.shape[1], mode=self._rank_policy)
                    if diagR_R.size
                    else 0
                )
            except Exception:  # noqa: BLE001
                rank_R = 0

        K_eff = int(Xw.shape[1] - rank_R)
        # enforce identification rule: rank(Z'X) must be >= free params
        if rank_ZX == 0:
            raise ValueError(
                "rank(Z'X) is zero; the instrument set carries no identifying variation.",
            )
        if rank_ZX < K_eff:
            raise ValueError(
                f"Underidentified after collinearity/constraints: rank(Z'X) {rank_ZX} < effective parameters {K_eff}.",
            )

        # keep rank metadata for later storage
        _ident_rank_info = {
            "rank_ZX": rank_ZX,
            "rank_instruments": L_c,
            "rank_constraints": rank_R,
            "K_eff": K_eff,
        }

        # Enforce centralized weight policy for IV/2SLS
        # weights were already forbidden and checked at the start of fit(); no-op here

        # Propagate fixed effects DoF to SSC (moved up for diagnostics use)
        ssc_local = dict(ssc) if ssc is not None else {}
        if (
            ("fe_dof" not in ssc_local)
            and isinstance(fe_dof_info, dict)
            and ("fe_dof" in fe_dof_info)
        ):
            ssc_local["fe_dof"] = int(fe_dof_info["fe_dof"])

        # Point estimate (constrained or unconstrained)
        with self._device_context(device):
            if constraints is None or constraint_vals is None:
                beta_hat = _two_sls_qr(
                    Zw, Xw, yw, method=method, rank_policy=self._rank_policy,
                )
            else:
                Qz_res = la.qr(la.to_dense(Zw), mode="economic", pivoting=True)
                if len(Qz_res) == 3:
                    Qz, _Rz, _p = Qz_res
                else:
                    Qz, _Rz = Qz_res
                X_t = la.dot(Qz.T, Xw)
                y_t = la.dot(Qz.T, yw)
                beta_hat = solve_constrained(X_t, y_t, constraints, constraint_vals)

            yhat = la.dot(Xw, beta_hat)
            uhat = yw.reshape(-1) - yhat.reshape(-1)

        # Diagnostics: SW F-statistic (cluster-robust when clusters present)
        # Use Montiel-Olea & Pflueger (2013) effective F when clustering, else standard SW F
        if cluster_ids_proc is not None:
            try:
                minF, Fj_list = _cluster_robust_sw_f(
                    Xw,
                    Zw,
                    endog_idx=endog_idx_new,
                    z_excluded_idx=z_excluded_idx_new,
                    var_names=var_names_work,
                    clusters=cluster_ids_proc,
                    rank_policy=self._rank_policy,
                    ssc=ssc_local,
                    n_original=Xw.shape[0],
                    k_params=self.X_orig.shape[1],
                )
            except Exception:  # noqa: BLE001
                # Fallback to homoskedastic if cluster-robust fails
                minF, Fj_list = _partial_f_sw_qr(
                    Xw,
                    Zw,
                    endog_idx=endog_idx_new,
                    z_excluded_idx=z_excluded_idx_new,
                    var_names=var_names_work,
                )
        else:
            minF, Fj_list = _partial_f_sw_qr(
                Xw,
                Zw,
                endog_idx=endog_idx_new,
                z_excluded_idx=z_excluded_idx_new,
                var_names=var_names_work,
            )
        # Diagnostics: compute KP/CD and robust MOP once (cluster-aware)
        # unify diagnostics and first_stage_stats.
        try:
            cd_stats = _cd_kp_stats(
                Xw,
                Zw,
                endog_idx_new,
                z_excluded_idx_new,
                u=uhat,
                clusters=cluster_ids_proc,
                multiway_ids=(list(multiway_ids_proc) if multiway_ids_proc is not None else None),
                space_ids=(np.asarray(space_ids_proc) if space_ids_proc is not None else None),
                time_ids=(np.asarray(time_ids_proc) if time_ids_proc is not None else None),
                ssc=ssc_local,
                k_params=self.X_orig.shape[1], # Use original K (after constant added)
            )
        except Exception:  # noqa: BLE001
            cd_stats = {
                "cd_min_eig": float("nan"),
                "kp_min_eig": float("nan"),
                "cd_wald_F": float("nan"),
                "kp_rk_LM": float("nan"),
                "kp_rk_Wald_F": float("nan"),
            }

        # --- Montiel-Olea & Pflueger (2013) effective F (1 endogenous only): F_eff = pi' Qzz pi / tr(Sigma_pi_pi Qzz) ---
        cd_stats["MOP_effective_F"] = float("nan")
        if len(endog_idx_new) == 1:
            try:
                j = endog_idx_new[0]
                # Partial out exogenous controls X0 from Z2 and x1
                X0 = (
                    Xw[:, [i for i in range(Xw.shape[1]) if i != j]]
                    if Xw.shape[1] > 1
                    else np.zeros((Xw.shape[0], 0))
                )
                Z2 = (
                    Zw[:, z_excluded_idx_new]
                    if z_excluded_idx_new
                    else np.zeros((Zw.shape[0], 0))
                )
                if Z2.shape[1] > 0:
                    # Residualize Z2 and x against X0 if X0 present
                    if X0.shape[1] > 0:
                        projZ = la.solve(X0, Z2, method="qr")
                        Z2_t = Z2 - la.dot(X0, projZ)
                        projx = la.solve(X0, Xw[:, [j]], method="qr")
                        x_t = Xw[:, [j]] - la.dot(X0, projx)
                    else:
                        Z2_t = Z2
                        x_t = Xw[:, [j]]

                    # Use sample-mean normalization consistent with MOP2013: scale = n
                    # We disallow analytic weighting in IV MOP computations (weights are forbidden for IV).
                    scale = float(Z2_t.shape[0])
                    Z2l = Z2_t

                    pi_hat = la.solve(Z2_t, x_t, method="qr")
                    e = (x_t - la.dot(Z2_t, pi_hat)).reshape(-1, 1)

                    # Construct meat M (robust) and scale by /scale.
                    # Multiway uses CGM inclusion-exclusion; (space,time) uses 2-way intersection.
                    ze = la.hadamard(la.to_dense(Z2_t), e)
                    if multiway_ids_proc is not None:
                        code_dims = [np.asarray(c).reshape(-1) for c in multiway_ids_proc]
                        R = len(code_dims)
                        S_ze = np.zeros((Z2_t.shape[1], Z2_t.shape[1]), dtype=np.float64)
                        for r in range(1, R + 1):
                            sign = +1.0 if (r % 2 == 1) else -1.0
                            for idxs in itertools.combinations(range(R), r):
                                if r == 1:
                                    sums = la.group_sum(ze, code_dims[idxs[0]])
                                else:
                                    codes = np.column_stack([code_dims[j] for j in idxs])
                                    sums = la.group_sum_multi(ze, codes)
                                S_ze += sign * la.tdot(sums)
                    elif space_ids_proc is not None and time_ids_proc is not None:
                        code_space = np.asarray(space_ids_proc).reshape(-1)
                        code_time = np.asarray(time_ids_proc).reshape(-1)
                        S_space = la.tdot(la.group_sum(ze, code_space))
                        S_time = la.tdot(la.group_sum(ze, code_time))
                        codes_inter = np.column_stack([code_space, code_time])
                        S_inter = la.tdot(la.group_sum_multi(ze, codes_inter))
                        S_ze = S_space + S_time - S_inter
                    elif cluster_ids_proc is not None:
                        S_ze = la.tdot(la.group_sum(ze, np.asarray(cluster_ids_proc).reshape(-1)))
                    else:
                        S_ze = la.tdot(ze)
                    M_loc = la.to_dense(S_ze) / scale

                    # Apply SSC to M_loc (Variance of moments)
                    # Factor = (N-1)/(N-K) * G/(G-1)
                    if ssc_local:
                         ssc_eval = _normalize_ssc(ssc_local)
                         k_eff = self.X_orig.shape[1]
                         clusters_for_ssc = (
                             list(multiway_ids_proc)
                             if multiway_ids_proc is not None
                             else (
                                 [space_ids_proc, time_ids_proc]
                                 if (space_ids_proc is not None and time_ids_proc is not None)
                                 else cluster_ids_proc
                             )
                         )
                         factor = compute_ssc_correction(Xw.shape[0], k_eff, clusters=clusters_for_ssc, ssc=ssc_eval)
                         if abs(factor - 1.0) > 1e-9:
                             M_loc *= factor

                    # Build Qzz and Var(pi) on the sample-mean scale and delegate
                    # to the centralized MOP helper. This keeps the algebra consistent
                    # with the GMM implementation.
                    # Here pi_hat are coefficients on Z2 only (X0 already partialled out),
                    # therefore the entire vector constitutes π.
                    pi_vec = pi_hat.reshape(-1, 1)
                    Qzz_raw = la.crossprod(Z2l, Z2l)
                    Qzz_raw_inv = la.pinv(Qzz_raw)
                    M_raw = M_loc * scale
                    Sig_pipi = la.dot(Qzz_raw_inv, la.dot(M_raw, Qzz_raw_inv))
                    try:
                        cd_stats["MOP_effective_F"] = float(
                            la.effective_f_from_first_stage(pi_vec, Sig_pipi, Z2l),
                        )
                    except Exception:  # noqa: BLE001
                        Qzz = Qzz_raw / scale
                        num = float(la.dot(pi_vec.T, la.dot(Qzz, pi_vec)))
                        den = float(np.trace(la.to_dense(la.dot(Sig_pipi, Qzz))))
                        if den > 0:
                            cd_stats["MOP_effective_F"] = num / den * scale
            except Exception:  # noqa: BLE001
                cd_stats["MOP_effective_F"] = float("nan")

        # ---- Overidentification statistic (value only) ----
        J_stat = None
        OverID_label = None
        L, k = Zw.shape[1], Xw.shape[1]
        k_eff_overid_check = int(locals().get("K_eff", k))
        if k_eff_overid_check < L:
            # i.i.d. / homoskedastic with no clustering: Sargan
            if (
                cluster_ids_proc is None
                and multiway_ids_proc is None
                and space_ids_proc is None
                and time_ids_proc is None
            ):
                try:
                    n = Xw.shape[0]
                    scale = float(n)
                    # Standard Sargan (value-only):
                    #   J = (n / sigma^2) * gbar' Qzz^{-1} gbar,
                    # where gbar = (Z'u)/n and Qzz = (Z'Z)/n.
                    # Use constraint-aware effective parameter count when available.
                    k_eff = int(locals().get("K_eff", k))
                    denom = max(1, int(n - k_eff))
                    sigma2 = float(la.dot(uhat.T, uhat) / denom)
                    gbar = la.crossprod(Zw, uhat) / scale
                    Qzz = la.tdot(Zw) / scale
                    evals, Q = la.eigh(Qzz)
                    keep = evals > la.eig_tol(Qzz)
                    if not np.any(keep):
                        J_stat = float("nan")
                    else:
                        Qk = Q[:, keep]
                        Qzz_k = la.dot(Qk.T, la.dot(Qzz, Qk))
                        Lk = la.safe_cholesky(Qzz_k)
                        Qzz_k_inv = la.chol_solve(Lk, la.eye(Qzz_k.shape[0]))
                        gk = la.dot(Qk.T, gbar)
                        J = la.crossprod(gk, la.dot(Qzz_k_inv, gk))
                        J_stat = float(la.to_dense(J).squeeze()) * scale / sigma2
                    OverID_label = "Sargan"
                except Exception:  # noqa: BLE001
                    J_stat = float("nan")
            else:
                # Clustered/multiway/space-time: construct S using the shared moment-covariance
                # helper to keep conventions identical across IV and GMM.
                try:
                    from .gmm import _moment_covariance_zu  # local import to avoid circular dependency

                    S = _moment_covariance_zu(
                        la.to_dense(Zw),
                        la.to_dense(uhat),
                        n_eff=Xw.shape[0],
                        cluster_ids=cluster_ids_proc,
                        multiway_ids=multiway_ids_proc,
                        space_ids=space_ids_proc,
                        time_ids=time_ids_proc,
                        obs_weights=None,
                        adj=False,
                        n_features=int(Xw.shape[1]),
                        fixefK="nested",
                        cluster_df="conventional",
                        fe_count=0,
                        fe_nested_mask=None,
                    )
                    scale = float(Xw.shape[0])
                    S_bar = S / scale
                    evals, Q = la.eigh(S_bar)
                    keep = evals > la.eig_tol(S_bar)
                    if np.any(keep):
                        Sk = la.dot(Q[:, keep].T, la.dot(S_bar, Q[:, keep]))
                        Lk = la.safe_cholesky(Sk)
                        Sk_inv = la.chol_solve(Lk, la.eye(Sk.shape[0]))
                        gbar = la.crossprod(Zw, uhat) / scale
                        gk = la.dot(Q[:, keep].T, gbar)
                        J = la.crossprod(gk, la.dot(Sk_inv, gk))
                        J_stat = float(la.to_dense(J).squeeze()) * scale
                    else:
                        J_stat = float("nan")
                    OverID_label = "HansenJ"
                except Exception:  # noqa: BLE001
                    J_stat = float("nan")
                    OverID_label = "HansenJ"

        # ---- prepare bootstrap config and run multipliers ----
        # fit() SEs are always WCU (unrestricted). Reject other residual types here to
        # prevent non-standard mixing of WCR/WCU_score into fit-time SE construction.
        rtype = str(getattr(boot, "residual_type", "WCU")).upper()
        if rtype != "WCU":
            raise ValueError(
                f"fit() uses residual_type='WCU' only; got '{rtype}'. "
                "Use wald_test()/ar_test() for {'WCR','WCU_score'} if needed.",
            )

        Wmult, boot_log = self._bootstrap_multipliers(Xw.shape[0], boot=boot)
        W_arr = Wmult.to_numpy()
        B = W_arr.shape[1]

        # For residual SSC scaling in apply_wild_bootstrap: forward the same
        # clustering structure used for inference (single-way, multiway, or
        # space-time). `core.bootstrap.compute_ssc_correction` supports list-like
        # multiway inputs.
        clusters_for_ssc = (
            list(multiway_ids_proc)
            if multiway_ids_proc is not None
            else (
                [space_ids_proc, time_ids_proc]
                if (space_ids_proc is not None and time_ids_proc is not None)
                else cluster_ids_proc
            )
        )

        # --- Bootstrap Robust First Stage F (User Request) ---
        boot_f_stat = float("nan")
        # Only meaningful if 1 endogenous variable (Standard First Stage F context)
        # and there are excluded instruments to test.
        if len(endog_idx_new) == 1 and len(z_excluded_idx_new) > 0:
            try:
                # Target: First stage regression x_endog ~ Z_all (Zw).
                # Null hypothesis: Coefficients of excluded instruments (Z_excluded) are zero.

                # Regressand: endogenous variable vector
                x_endog_col = Xw[:, endog_idx_new].reshape(-1, 1)

                # Restriction Matrix R: Select coefficients of excluded instruments.
                # Zw columns corresponding to z_excluded_idx_new should be tested.
                L_total = Zw.shape[1]
                L_excluded = len(z_excluded_idx_new)
                R_fs = np.zeros((L_excluded, L_total), dtype=np.float64)
                for i, idx in enumerate(z_excluded_idx_new):
                    R_fs[i, int(idx)] = 1.0

                # Zero vector for null hypothesis R beta = 0
                r_fs = np.zeros(L_excluded, dtype=np.float64).reshape(-1, 1)

                # Run Wald test using the SAME multipliers for consistency
                # resid_type="WCU" ensures we use unrestricted residuals from the first stage fit.
                wt_res = wald_test_wild_bootstrap(
                    Zw, x_endog_col,
                    R=R_fs,
                    r=r_fs,
                    multipliers=W_arr,
                    ssc=ssc_local,
                    residual_type="WCU",
                    clusters=cluster_ids_proc,
                    multiway_ids=multiway_ids_proc,
                    space_ids=space_ids_proc,
                    time_ids=time_ids_proc,
                )
                # wald_test returns a list of tests for each critical_level?
                # No, wald_test returns a list of result dicts, but they contain "bootstrap_quantile".
                # Wait, wald_test_wild_bootstrap returns a DICT with "stat", "df", etc?
                # Let's check wald_test_wild_bootstrap signature/return in bootstrap.py.
                # It returns `dict` (line 1935).
                # Documentation says: "Returns stat, the bootstrap distribution (wald_star), df, and bootstrap details."
                # So it returns a dict.
                # Key is "wald_stat" (checked implementation)
                boot_f_stat = float(wt_res["wald_stat"])
                # Note: This statistic is W_obs. For F-stat, we usually divide by q (number of restrictions).
                # The user asks for "Robust First Stage F". Standard F is Wald/q.
                # KP rk Wald F is also scaled by params?
                # wald_test_wild_bootstrap "stat" is the Wald statistic (chi-sq form usually).
                # We should divide by L_excluded (numerator degrees of freedom).
                boot_f_stat /= float(L_excluded)
            except Exception as e:
                # print(f"DEBUG: Bootstrap First Stage F failed: {e}")
                boot_f_stat = float("nan")

        self.first_stage_boot_f = boot_f_stat
        boot_betas = np.empty((Xw.shape[1], B), dtype=np.float64)

        qr_res_z = la.qr(la.to_dense(Zw), mode="economic", pivoting=True)
        if len(qr_res_z) == 3:
            Qz_rf, Rz_rf, _piv_rf = qr_res_z
        else:
            Qz_rf, Rz_rf = qr_res_z
        diagR_rf = (
            np.abs(np.diag(la.to_dense(Rz_rf))) if getattr(Rz_rf, "size", 0) else np.array([])
        )
        r_rf = (
            la.rank_from_diag(diagR_rf, Zw.shape[1], mode=self._rank_policy)
            if diagR_rf.size
            else 0
        )
        if r_rf == 0:
            raise RuntimeError("Instrument matrix Z is rank-deficient (rank=0) in reduced-form bootstrap.")
        Qz_rf = Qz_rf[:, :r_rf]

        y_rf_hat = la.dot(Qz_rf, la.dot(Qz_rf.T, yw))
        e_y = yw - y_rf_hat

        # Use the *working* endogenous indices after all screening/dedup.
        # `self.endog_idx` refers to the original X design and may be misaligned
        # after FE absorption and column dropping.
        endog_idx_local = list(endog_idx_new)
        if len(endog_idx_local) == 0:
            Ystar, _ = bt.apply_wild_bootstrap(
                yhat,
                uhat,
                W_arr,
                ssc=ssc_local,
                x_dof=Xw.shape[1],
                clusters=clusters_for_ssc,
            )
        else:
            X_endog = Xw[:, endog_idx_local]
            X_rf_hat = la.dot(Qz_rf, la.dot(Qz_rf.T, X_endog))
            e_x = X_endog - X_rf_hat

            # Use apply_wild_bootstrap for y (Reduced Form)
            # This handles SSC internally via ssc_local
            Y_star_matrix, _ = bt.apply_wild_bootstrap(
                y_rf_hat,
                e_y,
                W_arr,
                ssc=ssc_local,
                x_dof=Xw.shape[1],
                clusters=clusters_for_ssc,
            )

            # Use apply_wild_bootstrap for X_endog (Reduced Form)
            # This handles SSC internally via ssc_local
            # Returns (N, K_endog, B) tensor
            X_star_tensor, _ = bt.apply_wild_bootstrap(
                X_rf_hat,
                e_x,
                W_arr,
                ssc=ssc_local,
                x_dof=Xw.shape[1],
                clusters=clusters_for_ssc,
            )
            # Ensure 3D shape (N, K, B) for consistent indexing even if K=1
            if X_star_tensor.ndim == 2:
                X_star_tensor = X_star_tensor[:, np.newaxis, :]  # (N, B) -> (N, 1, B)

        with self._device_context(device):
            if len(endog_idx_local) > 0:
                for b in range(B):
                    y_star_b = Y_star_matrix[:, b:b+1]
                    X_star_b = Xw.copy() if isinstance(Xw, np.ndarray) else la.to_dense(Xw).copy()
                    # Replace endogenous columns with bootstrapped versions
                    # X_star_tensor is (N, K_endog, B)
                    X_star_b[:, endog_idx_local] = X_star_tensor[:, :, b]


                    if constraints is None or constraint_vals is None:
                        qr_res_b = la.qr(la.to_dense(Zw), mode="economic", pivoting=True)
                        if len(qr_res_b) == 3:
                            Qz_b, Rz_b, _piv_b = qr_res_b
                        else:
                            Qz_b, Rz_b = qr_res_b
                        diagR_b = (
                            np.abs(np.diag(la.to_dense(Rz_b))) if getattr(Rz_b, "size", 0) else np.array([])
                        )
                        r_b = (
                            la.rank_from_diag(diagR_b, Zw.shape[1], mode=self._rank_policy)
                            if diagR_b.size
                            else 0
                        )
                        if r_b == 0:
                            boot_betas[:, b] = np.nan
                            continue
                        Qz_b_eff = Qz_b[:, :r_b]
                        X_t_b = la.dot(Qz_b_eff.T, X_star_b)
                        y_t_b = la.dot(Qz_b_eff.T, y_star_b)
                        beta_b = la.solve(X_t_b, y_t_b, method=method, rank_policy=self._rank_policy)
                        boot_betas[:, b] = np.asarray(beta_b).reshape(-1)
                    else:
                        if "Qz" not in locals():
                            qr_res = la.qr(la.to_dense(Zw), mode="economic", pivoting=True)
                            if len(qr_res) == 3:
                                Qz_tmp, Rz_tmp, _p = qr_res
                            else:
                                Qz_tmp, Rz_tmp = qr_res
                            diagR_tmp = (
                                np.abs(np.diag(la.to_dense(Rz_tmp)))
                                if Rz_tmp.size
                                else np.array([])
                            )
                            r_tmp = (
                                la.rank_from_diag(diagR_tmp, Zw.shape[1], mode=self._rank_policy)
                                if diagR_tmp.size
                                else 0
                            )
                            if r_tmp == 0:
                                raise RuntimeError("Instrument matrix Z is rank-deficient (rank=0).")
                            Qz = Qz_tmp[:, :r_tmp]
                        X_t_b = la.dot(Qz.T, X_star_b)
                        y_t_b = la.dot(Qz.T, y_star_b)
                        beta_b = solve_constrained(X_t_b, y_t_b, constraints, constraint_vals)
                        boot_betas[:, b] = np.asarray(beta_b).reshape(-1)
            else:
                Ystar, _ = bt.apply_wild_bootstrap(
                    yhat,
                    uhat,
                    W_arr,
                    ssc=ssc_local,
                    x_dof=Xw.shape[1],
                    clusters=clusters_for_ssc,
                )
                if constraints is None or constraint_vals is None:
                    qr_res = la.qr(la.to_dense(Zw), mode="economic", pivoting=True)
                    if len(qr_res) == 3:
                        Qz_loc, Rz_loc, _piv_loc = qr_res
                    else:
                        Qz_loc, Rz_loc = qr_res
                    diagR_loc = (
                        np.abs(np.diag(la.to_dense(Rz_loc))) if getattr(Rz_loc, "size", 0) else np.array([])
                    )
                    r_loc = (
                        la.rank_from_diag(diagR_loc, Zw.shape[1], mode=self._rank_policy)
                        if diagR_loc.size
                        else 0
                    )
                    if r_loc == 0:
                        raise RuntimeError("Instrument matrix Z is rank-deficient (rank=0).")
                    Qz_eff = Qz_loc[:, :r_loc]
                    X_t = la.dot(Qz_eff.T, Xw)
                    Y_t = la.dot(Qz_eff.T, Ystar)
                    beta_all = la.solve(
                        X_t, Y_t, method=method, rank_policy=self._rank_policy,
                    )
                    boot_betas[:, :] = np.asarray(beta_all)
                else:
                    if "Qz" not in locals():
                        qr_res = la.qr(la.to_dense(Zw), mode="economic", pivoting=True)
                        if len(qr_res) == 3:
                            Qz_tmp, Rz_tmp, _p = qr_res
                        else:
                            Qz_tmp, Rz_tmp = qr_res
                        diagR_tmp = (
                            np.abs(np.diag(la.to_dense(Rz_tmp)))
                            if Rz_tmp.size
                            else np.array([])
                        )
                        r_tmp = (
                            la.rank_from_diag(diagR_tmp, Zw.shape[1], mode=self._rank_policy)
                            if diagR_tmp.size
                            else 0
                        )
                        if r_tmp == 0:
                            raise RuntimeError("Instrument matrix Z is rank-deficient (rank=0).")
                        Qz = Qz_tmp[:, :r_tmp]
                    X_t = la.dot(Qz.T, Xw)
                    workers_env = os.getenv("LINEAREG_IV_BOOTSTRAP_WORKERS", "").strip()
                    try:
                        workers = int(workers_env) if workers_env else 0
                    except ValueError:
                        workers = 0
                    do_parallel = (workers and workers > 1) and (
                        device is None or str(device).lower() == "cpu"
                    )

                    def _solve_one(b: int) -> tuple[int, np.ndarray]:
                        yb = Ystar[:, b : b + 1]
                        yb_t = la.dot(Qz.T, yb)
                        beta_b = solve_constrained(X_t, yb_t, constraints, constraint_vals)
                        return b, np.asarray(beta_b).reshape(-1)

                    if do_parallel:
                        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                            for b_idx, beta_vec in ex.map(_solve_one, range(B)):
                                boot_betas[:, b_idx] = beta_vec
                    else:
                        for b in range(B):
                            _, beta_vec = _solve_one(b)
                            boot_betas[:, b] = beta_vec
        se_vals = bt.bootstrap_se(boot_betas)
        se = pd.Series(se_vals, index=var_names_work, name="se")

        params = pd.Series(beta_hat.reshape(-1), index=var_names_work, name="coef")
        ssc_local = dict(ssc) if ssc is not None else {}
        if (
            "fe_dof" not in ssc_local
            and isinstance(fe_dof_info, dict)
            and "fe_dof" in fe_dof_info
        ):
            ssc_local["fe_dof"] = int(fe_dof_info.get("fe_dof", 0))

        # QR rank information for reproducibility
        # Use the identification-stage instrument rank (L_c) computed earlier
        # so stored metadata exactly matches the identification check performed
        # above (collinearity/constraint-aware). This avoids recomputing a
        # possibly inconsistent numeric rank at the end of fit.
        try:
            rank_Z = int(L_c)
        except Exception:  # noqa: BLE001
            rank_Z = None

        model_info = {
            "Estimator": "IV (2SLS)",
            "WeakIV_SW_uses_within": absorb_fe is not None,
            "Constraints": constraints is not None and constraint_vals is not None,
            "FixedEffects": absorb_fe is not None,
            "Dropped": dropped_stats,
            "n_eff": n_eff,
            # report effective number of free parameters (K minus constraint rank)
            "n_params": int(K_eff),
            "ssc": ssc,
            "ssc_effective": ssc_local,
            "boot_config": boot,
            # Record effective bootstrap reps and provenance for summary display
            "B": int(getattr(Wmult, "shape", (0, 0))[1])
            if "Wmult" in locals()
            else (int(getattr(boot, "n_boot", 0))),
            "SE_Origin": "bootstrap",
            "NoAnalyticSE": True,
            "rank_PZ_X": int(rank_ZX),  # rank of Qz'X used for identification
        }

        # Compute and stash weak-IV diagnostics (values only) for user inspection.
        # These diagnostics are computed via numerically-stable QR-first helpers
        # and generalized-eigenvalue routines in core.linalg. No p-values or
        # automatic decisions are made here (project policy).
        # diagnostics were already computed above (cd_stats). Ensure keys exist
        cd_stats.setdefault("cd_min_eig", float("nan"))
        cd_stats.setdefault("kp_min_eig", float("nan"))
        cd_stats.setdefault("cd_wald_F", float("nan"))
        cd_stats.setdefault("kp_rk_LM", float("nan"))
    # Store only canonical lowercase keys (legacy uppercase aliases removed)

        minF_val = float(minF) if np.isfinite(minF) else float("nan")

        extra = {
            "SW_list": Fj_list,
            "sw_F": {
                "min_F": minF_val,
                "per_regressor": Fj_list,
            },
            "cd_kp_stats": cd_stats,
            "first_stage_stats": {
                "bootstrap_robust_f": (self.first_stage_boot_f if hasattr(self, "first_stage_boot_f") else float("nan")),
                "min_partial_F": minF_val,
                "mean_partial_F": (
                    float(np.nanmean(vals))
                    if (vals := [v for _, v, _ in Fj_list if np.isfinite(v)]) and len(vals) > 0
                    else float("nan")
                ),
                # per-regressor partial F entries (no p-values)
                **{
                    f"F_SW_partial_{name}": float(val)
                    if np.isfinite(val)
                    else float("nan")
                    for name, val, _ in Fj_list
                },
                "F_SW_min": minF_val,
                # canonical lowercase key for MOP effective F (value-only)
                "mop_F_effective": float(cd_stats.get("MOP_effective_F", float("nan"))),
                # expose KP/CD and derived metrics
                **cd_stats,
            },
            "mask_used": mask,
            "yhat": yhat,
            "y_inference": yw,
            "X_inference": Xw,
            "Z_inference": Zw,
            "u_inference": uhat,
            "OverID_stat": J_stat,
            "J_stat": J_stat,
            "OverID_label": OverID_label,
            "clusters_inference": cluster_spec["clusters_inference"],
            "multiway_ids_inference": multiway_ids_proc,
            "space_ids_inference": (
                np.asarray(space_ids_proc) if space_ids_proc is not None else None
            ),
            "time_ids_inference": (
                np.asarray(time_ids_proc) if time_ids_proc is not None else None
            ),
            "beta0_inference": beta_hat,
            "boot_betas": boot_betas,
            "W_multipliers_inference": Wmult,
            "multipliers_log": boot_log,
            "boot_variant_used": (
                "33"
                if (
                    multiway_ids_proc is not None
                    or (space_ids_proc is not None and time_ids_proc is not None)
                    or (cluster_ids_proc is not None)
                )
                else "11"
            ),
            "boot_residual_default": "WCU",
            "boot_policy_used": getattr(boot, "policy", None),
            "ssc_effective": ssc_local,
            "fe_dof_info": fe_dof_info,
            "rank_Z_used": rank_Z,
            "ident_rank_info": _ident_rank_info,
            "instr_names_used": instr_names_work,
        }

        # add OverID df to model_info and extra for diagnostics display (values only)
        k_eff_overid = int(locals().get("K_eff", Xw.shape[1]))
        overid_df_val = int(Zw.shape[1] - k_eff_overid) if Zw.shape[1] > k_eff_overid else 0
        model_info["OverID_df"] = overid_df_val
        extra["OverID_df"] = overid_df_val

        # store results on the instance for later inference
        self.params_ = params
        # Store bootstrap SE in result.se for consistency
        extra_local = dict(extra)
        extra_local["se_source"] = "bootstrap"  # enforce: SE are bootstrap-derived
        self.se_ = se
        self.extra = extra_local

        return EstimationResult(
            params=params,
            se=se,
            bands=None,
            n_obs=n_eff,
            model_info=model_info,
            extra=extra_local,
        )

    def wald_test(  # noqa: PLR0913
        self,
        R: np.ndarray,
        r: np.ndarray,
        *,
        boot: BootConfig | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        multiway_ids: Sequence[Sequence] | None = None,
        method: str = "qr",
        B: int | None = None,
        critical_level: Sequence[float] = (0.05,),
    ) -> list[dict[str, float | bool]]:
        """Bootstrap Wald-style test on R beta = r using empirical bootstrap covariance of R beta.

        This preserves the original behavior of the prior ar_test (Wald-style). No
        analytic p-values or critical values are computed; callers receive the
        observed statistic and bootstrap draws.
        """
        # Delegate to previous implementation (preserve semantics)
        return self._wald_test_impl(
            R=R,
            r=r,
            boot=boot,
            cluster_ids=cluster_ids,
            space_ids=space_ids,
            time_ids=time_ids,
            multiway_ids=multiway_ids,
            method=method,
            B=B,
            critical_level=critical_level,
        )

    def _wald_test_impl(  # noqa: PLR0913
        self,
        R: np.ndarray,
        r: np.ndarray,
        *,
        boot: BootConfig | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        multiway_ids: Sequence[Sequence] | None = None,
        method: str = "qr",
        B: int | None = None,
        critical_level: Sequence[float] = (0.05,),
    ) -> list[dict[str, float | bool]]:
        if B is None:
            B = bt.DEFAULT_BOOTSTRAP_ITERATIONS
        R = np.asarray(R, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64).reshape(-1, 1)

        if not hasattr(self, "X_orig"):
            raise RuntimeError("Fit the model before calling wald_test().")

        X = (
            self.extra.get("X_inference", self.X_orig)
            if hasattr(self, "extra")
            else self.X_orig
        )
        y = (
            self.extra.get("y_inference", self.y_orig)
            if hasattr(self, "extra")
            else self.y_orig
        )
        Z = (
            self.extra.get("Z_inference", self.Z_orig)
            if hasattr(self, "extra")
            else self.Z_orig
        )

        # Restricted fit under H0
        try:
            yhat_R, resid_R = bt.wls_fit_restricted(
                y=y, X=X, R=R, r=r, weights=None, method="qr",
            )
        except Exception:  # noqa: BLE001
            beta_R = solve_constrained(la.to_dense(X), la.to_dense(y), R, r)
            yhat_R = la.dot(X, beta_R)
            resid_R = y.reshape(-1) - yhat_R.reshape(-1)

        # Observed unrestricted estimate and residuals
        beta_hat = _two_sls_qr(Z, X, y, method=method, rank_policy=self._rank_policy)
        uhat = y.reshape(-1) - la.dot(X, beta_hat).reshape(-1)

        # Bootstrap multipliers: reconstruct clustering metadata from estimation
        # when the caller omits them so bootstrap flows remain reproducible.
        ex = getattr(self, "extra", {}) if hasattr(self, "extra") else {}
        default_mask = np.ones(self._n_obs_init, dtype=bool)
        mask_used = np.asarray(ex.get("mask_used", default_mask), dtype=bool)
        if mask_used.shape[0] != self._n_obs_init:
            raise ValueError(
                "Stored mask length inconsistent with original sample size.",
            )

        stored_multiway = ex.get("multiway_ids_inference", None)
        stored_space = ex.get("space_ids_inference", None)
        stored_time = ex.get("time_ids_inference", None)
        stored_cluster = ex.get("clusters_inference", None)

        base_multiway = multiway_ids if multiway_ids is not None else stored_multiway
        base_cluster = (
            cluster_ids
            if cluster_ids is not None
            else (None if base_multiway is not None else stored_cluster)
        )
        base_space = space_ids if space_ids is not None else stored_space
        base_time = time_ids if time_ids is not None else stored_time

        boot, cluster_spec = self._coerce_bootstrap(
            boot=boot,
            n_obs_original=self._n_obs_init,
            row_mask=mask_used,
            cluster_ids=base_cluster,
            space_ids=base_space,
            time_ids=base_time,
            multiway_ids=base_multiway,
        )
        cluster_ids_use = cluster_spec.get("cluster_ids")
        if getattr(boot, "n_boot", None) is None:
            boot = replace(boot, n_boot=B)
        Wmult_df, log = boot.make_multipliers(X.shape[0])
        Wmat = Wmult_df.to_numpy()
        B_actual = Wmat.shape[1]
        # IMPORTANT: Do NOT column-wise recenter/renormalize multipliers.
        # `core.bootstrap` follows boottest/fwildclusterboot semantics: multipliers
        # are mean-zero / unit-variance in expectation (distributional), and
        # per-draw recentering would change the finite-sample reference.
        Wmat = Wmat.astype(float, copy=False)

        rtype_raw = str(getattr(boot, "residual_type", "WCU_score"))
        rtype_u = rtype_raw.strip().upper()
        if rtype_u == "WCR":
            rtype = "WCR"
        elif rtype_u in {"WCU_SCORE", "WCU-SCORE", "WCU SCORE"}:
            rtype = "WCU_score"
        else:
            # Default to unrestricted residual bootstrap for any other spelling.
            rtype = "WCU"
        yhat_U = la.dot(X, beta_hat)
        if rtype == "WCR":
            base_y = yhat_R.reshape(-1, 1)
            base_u = resid_R.reshape(-1, 1)
        else:
            base_y = yhat_U.reshape(-1, 1)
            base_u = uhat.reshape(-1, 1)
            if rtype == "WCU_score":
                try:
                    base_u = bt.score_recentering_iv(base_u, Z, cluster_ids_use).reshape(-1, 1)
                except Exception:
                    base_u = uhat.reshape(-1, 1)

        betas_star = np.empty((X.shape[1], B_actual), dtype=np.float64)
        # Optional threading for bootstrap draws (independent across b)
        workers_env_wald = os.getenv("LINEAREG_IV_WALD_WORKERS", "").strip()
        try:
            workers_wald = int(workers_env_wald) if workers_env_wald else 0
        except ValueError:
            workers_wald = 0
        do_parallel_wald = (workers_wald and workers_wald > 1)

        def _wald_one(b: int) -> tuple[int, np.ndarray]:
            v = Wmat[:, b].reshape(-1, 1)
            yb = base_y + la.hadamard(base_u, v)
            beta_b = _two_sls_qr(
                Z, X, yb, method=method, rank_policy=self._rank_policy,
            )
            return b, np.asarray(beta_b).reshape(-1)

        if do_parallel_wald:
            with cf.ThreadPoolExecutor(max_workers=workers_wald) as ex:
                for b_idx, beta_vec in ex.map(_wald_one, range(B_actual)):
                    betas_star[:, b_idx] = beta_vec
        else:
            for b in range(B_actual):
                _, beta_vec = _wald_one(b)
                betas_star[:, b] = beta_vec

        Rb_hat = la.dot(R, beta_hat) - r
        Rb_star = la.dot(R, betas_star) - r

        # Empirical covariance of Rb_star. Rb_star shape: (q, B_actual)
        if B_actual <= 1:
            Rb_cov = (
                la.zeros((Rb_star.shape[0], Rb_star.shape[0]))
                if hasattr(la, "zeros")
                else np.zeros((Rb_star.shape[0], Rb_star.shape[0]))
            )
        else:
            mean_rb = np.mean(Rb_star, axis=1, keepdims=True)
            M_rb = Rb_star - mean_rb
            # Denominator policy: boottest-style uses B, otherwise B-1
            denom = float(
                B_actual
                if getattr(boot, "policy", None) == "boottest"
                else max(1, B_actual - 1),
            )
            Rb_cov = la.dot(M_rb, M_rb.T) / denom

        evals, Q = la.eigh(Rb_cov)
        tol = la.eig_tol(Rb_cov)
        keep = evals > tol
        if not np.any(keep):
            Rb_cov_inv = la.pinv(Rb_cov)
        else:
            Qk = Q[:, keep]
            Rk = la.dot(Qk.T, la.dot(Rb_cov, Qk))
            Lk = la.safe_cholesky(Rk)
            Rk_inv = la.chol_solve(Lk, la.eye(Rk.shape[0]))
            Rb_cov_inv = la.dot(Qk, la.dot(Rk_inv, Qk.T))


        val_obs = la.dot(la.dot(Rb_hat.T, Rb_cov_inv), Rb_hat)
        W_obs = float(val_obs.item()) if hasattr(val_obs, "item") else float(val_obs)
        W_star = np.empty(B_actual, dtype=float)
        for b in range(B_actual):
            rb = Rb_star[:, b : b + 1]
            val_star = la.dot(la.dot(rb.T, Rb_cov_inv), rb)
            W_star[b] = float(val_star.item()) if hasattr(val_star, "item") else float(val_star)

        finite = W_star[np.isfinite(W_star)]
        B_eff = finite.size

        tests: list[dict[str, float | bool]] = []
        for alpha in critical_level:
            a = float(alpha)
            if not (0.0 < a < 1.0):
                raise ValueError("critical_level entries must be in (0,1).")
            # finite-sample B+1 quantile for bootstrap order statistics
            cv = float(la.finite_sample_quantile_bplus1(W_star, 1.0 - a))
            tests.append({
                "stat": float(W_obs),
                "bootstrap_quantile": cv,
                "reject": bool(float(W_obs) > cv)
            })
        return tests

    def ar_test(  # noqa: PLR0913
        self,
        R: np.ndarray,
        r: np.ndarray,
        *,
        boot: BootConfig | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        multiway_ids: Sequence[Sequence] | None = None,
        B: int | None = None,
        critical_level: Sequence[float] = (0.05,),
    ) -> list[dict[str, float | bool]]:
        """Anderson-Rubin (AR) style statistic using wild bootstrap draws.

        The AR statistic here is Q_AR = u_R' P_Z u_R where u_R are residuals
        from the restricted fit under H0: R beta = r. No p-values/criticals are
        returned; the bootstrap draws of Q_AR are returned for external use.
        """
        if B is None:
            B = bt.DEFAULT_BOOTSTRAP_ITERATIONS
        R = np.asarray(R, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64).reshape(-1, 1)
        if not hasattr(self, "X_orig"):
            raise RuntimeError("Fit the model before calling ar_test().")
        X = (
            self.extra.get("X_inference", self.X_orig)
            if hasattr(self, "extra")
            else self.X_orig
        )
        y = (
            self.extra.get("y_inference", self.y_orig)
            if hasattr(self, "extra")
            else self.y_orig
        )
        Z = (
            self.extra.get("Z_inference", self.Z_orig)
            if hasattr(self, "extra")
            else self.Z_orig
        )

        # Restricted LS fit under H0
        beta_R = solve_constrained(la.to_dense(X), la.to_dense(y), R, r)
        u_R = y - la.dot(X, beta_R)
        ZtZ = la.crossprod(Z, Z)
        try:
            K = la.solve(ZtZ, la.eye(ZtZ.shape[0]), sym_pos=True)
        except Exception:  # noqa: BLE001
            K = la.pinv(ZtZ)
        PZu = la.dot(Z, la.dot(K, la.crossprod(Z, u_R)))
        Q_obs = float(la.dot(u_R.T, PZu))

        ex = getattr(self, "extra", {}) if hasattr(self, "extra") else {}
        default_mask = np.ones(self._n_obs_init, dtype=bool)
        mask_used = np.asarray(ex.get("mask_used", default_mask), dtype=bool)
        if mask_used.shape[0] != self._n_obs_init:
            raise ValueError(
                "Stored mask length inconsistent with original sample size.",
            )

        stored_multiway = ex.get("multiway_ids_inference", None)
        stored_space = ex.get("space_ids_inference", None)
        stored_time = ex.get("time_ids_inference", None)
        stored_cluster = ex.get("clusters_inference", None)

        base_multiway = multiway_ids if multiway_ids is not None else stored_multiway
        base_cluster = (
            cluster_ids
            if cluster_ids is not None
            else (None if base_multiway is not None else stored_cluster)
        )
        base_space = space_ids if space_ids is not None else stored_space
        base_time = time_ids if time_ids is not None else stored_time

        boot, _cluster_spec = self._coerce_bootstrap(
            boot=boot,
            n_obs_original=self._n_obs_init,
            row_mask=mask_used,
            cluster_ids=base_cluster,
            space_ids=base_space,
            time_ids=base_time,
            multiway_ids=base_multiway,
        )
        if getattr(boot, "n_boot", None) is None:
            boot = replace(boot, n_boot=B)
        Wmult_df, log = boot.make_multipliers(X.shape[0])
        W = Wmult_df.to_numpy()
        # IMPORTANT: Do NOT column-wise recenter/renormalize multipliers.
        # See `core.bootstrap.wild_bootstrap_betas` docstring for rationale.
        W = W.astype(float, copy=False)
        Q_star = np.empty(W.shape[1], dtype=float)
        yhat_R = la.dot(X, beta_R)
        # Optional threading for AR test bootstrap draws
        workers_env_ar = os.getenv("LINEAREG_IV_AR_WORKERS", "").strip()
        try:
            workers_ar = int(workers_env_ar) if workers_env_ar else 0
        except ValueError:
            workers_ar = 0
        do_parallel_ar = workers_ar and workers_ar > 1

        def _ar_one(b: int) -> tuple[int, float]:
            yb = yhat_R + la.hadamard(u_R, W[:, b : b + 1])
            beta_Rb = solve_constrained(la.to_dense(X), la.to_dense(yb), R, r)
            ub = yb - la.dot(X, beta_Rb)
            PZub = la.dot(Z, la.dot(K, la.crossprod(Z, ub)))
            q = float(la.dot(ub.T, PZub))
            return b, q

        if do_parallel_ar:
            with cf.ThreadPoolExecutor(max_workers=workers_ar) as ex:
                for b_idx, qv in ex.map(_ar_one, range(W.shape[1])):
                    Q_star[b_idx] = qv
        else:
            for b in range(W.shape[1]):
                _, qv = _ar_one(b)
                Q_star[b] = qv
        tests: list[dict[str, float | bool]] = []
        for alpha in critical_level:
            a = float(alpha)
            if not (0.0 < a < 1.0):
                raise ValueError("critical_level entries must be in (0,1).")
            cv = float(la.finite_sample_quantile_bplus1(Q_star, 1.0 - a))
            tests.append({"bootstrap_quantile": cv, "reject": bool(Q_obs > cv)})
        return tests
