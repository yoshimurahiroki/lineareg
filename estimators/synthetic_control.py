"""Synthetic Control estimator.

This module implements the Abadie-Diamond-Hainmueller SCM with Frank-Wolfe simplex
weights and bootstrap inference.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import replace

import numpy as np
import pandas as pd

from lineareg.core import bootstrap as bt
from lineareg.core import linalg as la
from lineareg.estimators.base import (
    BootConfig,
    EstimationResult,
    attach_formula_metadata,
    prepare_formula_environment,
)
from lineareg.utils.formula import (
    FormulaParser,
)

__all__ = ["SyntheticControl"]

LOGGER = logging.getLogger(__name__)


from lineareg.utils.helpers import event_tau, time_to_pos






def _frank_wolfe_simplex(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    init: np.ndarray | None = None,
    V: np.ndarray | None = None,
) -> np.ndarray:
    """Solve min_w 0.5||V^{1/2}(X w - y)||^2  s.t. w in simplex (w>=0, sum w=1) via Frank-Wolfe.

    Returns weights w on the simplex. Deterministic (no randomness).
    If V is provided (diagonal weights), applies V^{1/2} weighting to the objective.
    """
    T, J = X.shape
    if V is not None:
        sqrt_V = np.sqrt(np.asarray(V, dtype=np.float64).reshape(-1))
        X = X * sqrt_V.reshape(-1, 1)
        y = y * sqrt_V
    if init is None:
        errs = np.sum((X - y.reshape(T, 1)) ** 2, axis=0)
        j0 = int(np.argmin(errs))
        w = np.zeros(J, dtype=np.float64)
        w[j0] = 1.0
    else:
        w = np.asarray(init, dtype=np.float64).reshape(J)
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        w = (w / s) if s > 0 else np.full(J, 1.0 / J, dtype=np.float64)

    Xw = la.dot(X, w)
    for _ in range(int(max_iter)):
        g = la.dot(X.T, (Xw - y))
        j = int(np.argmin(g))
        e = np.zeros(J, dtype=np.float64)
        e[j] = 1.0
        d = e - w
        gap = float(-la.dot(d, g))
        if gap < tol:
            break
        Xd = la.dot(X, d)
        num = float(la.dot(Xd.T, (Xw - y)))
        den = float(la.dot(Xd.T, Xd))
        gamma = 0.0 if den <= 0.0 else -num / den
        if not np.isfinite(gamma):
            gamma = 0.0
        gamma = max(0.0, min(1.0, gamma))
        if gamma <= 0.0:
            break
        w_new = w + gamma * d
        Xw_new = Xw + gamma * Xd
        w, Xw = w_new, Xw_new
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        w[:] = 0.0
        w[0] = 1.0
    else:
        w /= s
    return w


def _optimize_V_nested(
    X_pred_treat: np.ndarray,
    X_pred_donors: np.ndarray,
    Y_pre_treat: np.ndarray,
    Y_pre_donors: np.ndarray,
    *,
    max_iter_fw: int = 10_000,
    tol_fw: float = 1e-10,
    max_iter_v: int = 500,
    newton_refine: int = 20,
    newton_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.optimize import minimize
    K = X_pred_treat.shape[0]
    T_pre = Y_pre_treat.shape[0]
    J = X_pred_donors.shape[1]

    def _softmax(u: np.ndarray) -> np.ndarray:
        z = u - np.max(u)
        ez = np.exp(z)
        return ez / np.sum(ez)

    def _softmax_jac(v: np.ndarray) -> np.ndarray:
        return np.diag(v) - np.outer(v, v)

    def _inner_w(v_diag: np.ndarray) -> np.ndarray:
        return _frank_wolfe_simplex(
            X_pred_donors.T, X_pred_treat, max_iter=max_iter_fw, tol=tol_fw, V=v_diag,
        )

    def _mspe_pre(u: np.ndarray) -> float:
        v_diag = _softmax(u)
        w = _inner_w(v_diag)
        synth_pre = la.dot(Y_pre_donors.T, w)
        return float(np.mean((Y_pre_treat - synth_pre) ** 2))

    def _kkt_solve_local(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        k = A.shape[0]
        ones = np.ones((k, 1))
        M = np.block([[A, ones], [ones.T, np.zeros((1, 1))]])
        rhs = np.vstack([b.reshape(-1, 1), np.ones((1, 1))])
        U, s, Vt = la.svd(M, full_matrices=False)
        s_inv = np.where(s > 1e-14, 1.0 / s, 0.0)
        sol = la.dot((Vt.T * s_inv), la.dot(U.T, rhs))
        return sol[:-1, 0]

    def _kkt_dw_local(Xs, y, v, wS, dX_s, dy, dv):
        if Xs.shape[1] <= 1:
            return np.zeros(Xs.shape[1])
        D = np.diag(v)
        dD = np.diag(dv) if dv is not None else np.zeros_like(D)
        A = la.dot(Xs.T, la.dot(D, Xs))
        b = la.dot(Xs.T, la.dot(D, y))
        dA = la.dot(dX_s.T, la.dot(D, Xs)) + la.dot(Xs.T, la.dot(dD, Xs)) + la.dot(Xs.T, la.dot(D, dX_s))
        db = la.dot(dX_s.T, la.dot(D, y)) + la.dot(Xs.T, la.dot(dD, y)) + la.dot(Xs.T, la.dot(D, dy))
        k = Xs.shape[1]
        ones = np.ones((k, 1))
        M = np.block([[A, ones], [ones.T, np.zeros((1, 1))]])
        rhs = np.vstack([(db - la.dot(dA, wS)).reshape(-1, 1), np.zeros((1, 1))])
        U, s, Vt = la.svd(M, full_matrices=False)
        s_inv = np.where(s > 1e-14, 1.0 / s, 0.0)
        sol = la.dot((Vt.T * s_inv), la.dot(U.T, rhs))
        return sol[:-1, 0]

    def _w_from_u_fixedS(X, y, u, S):
        v = _softmax(u)
        Xs = X[:, S]
        # Xs.T @ diag(v) @ Xs == Xs.T @ (Xs * v[:, None])
        A = la.dot(Xs.T, la.hadamard(Xs, v.reshape(-1, 1)))
        b = la.dot(Xs.T, la.hadamard(y.reshape(-1, 1), v.reshape(-1, 1)))
        wS = _kkt_solve_local(A, b)
        return v, wS

    def _grad_u_fixedS(X, y, u, S):
        Kp = X.shape[0]
        v, wS = _w_from_u_fixedS(X, y, u, S)
        w = np.zeros(X.shape[1], dtype=float)
        w[S] = wS
        r = (y - la.dot(X, w)).reshape(-1)
        Jv = _softmax_jac(v)
        g = np.zeros(Kp, dtype=float)
        Xs = X[:, S]
        for q in range(Kp):
            dv = Jv[:, q]
            dwS = _kkt_dw_local(Xs, y, v, wS, np.zeros_like(Xs), np.zeros_like(y), dv)
            dr = -la.dot(Xs, dwS)
            g[q] = (2.0 / Kp) * float(la.dot(r.reshape(1, -1), dr.reshape(-1, 1)))
        return g

    def _jac_u_num(X, y, u, S, eps=1e-6):
        Kp = X.shape[0]
        J = np.zeros((Kp, Kp), dtype=float)
        for q in range(Kp):
            du = np.zeros(Kp, dtype=float)
            du[q] = eps
            gp = _grad_u_fixedS(X, y, u + du, S)
            gm = _grad_u_fixedS(X, y, u - du, S)
            J[:, q] = (gp - gm) / (2.0 * eps)
        return J

    u0 = np.zeros(K, dtype=np.float64)
    res = minimize(_mspe_pre, u0, method="Nelder-Mead", options={"maxiter": max_iter_v})
    u_opt = res.x

    v_opt = _softmax(u_opt)
    w_opt = _inner_w(v_opt)
    S = w_opt > tol_fw
    if S.sum() == 0:
        S[np.argmax(w_opt)] = True
    X_pre = X_pred_donors.T
    y_pre = X_pred_treat

    for _ in range(newton_refine):
        g0 = _grad_u_fixedS(X_pre, y_pre, u_opt, S)
        if np.max(np.abs(g0)) < newton_tol:
            break
        if not np.isfinite(g0).all():
            break
        Jg = _jac_u_num(X_pre, y_pre, u_opt, S)
        U_jg, s_jg, Vt_jg = la.svd(Jg, full_matrices=False)
        s_jg_inv = np.where(s_jg > 1e-14, 1.0 / s_jg, 0.0)
        step = la.dot((Vt_jg.T * s_jg_inv), la.dot(U_jg.T, (-g0).reshape(-1, 1)))
        step = step[:, 0]
        if not np.isfinite(step).all():
            break
        u_opt = u_opt + step
        v_opt = _softmax(u_opt)
        w_opt = _inner_w(v_opt)
        S_new = w_opt > tol_fw
        if S_new.sum() == 0:
            S_new[np.argmax(w_opt)] = True
        S = S_new

    v_opt = _softmax(u_opt)
    w_opt = _inner_w(v_opt)
    return u_opt, v_opt, w_opt


@dataclass
class _Spec:
    id_name: str
    t_name: str
    y_name: str
    treat_name: str
    cohort_name: str | None = None
    center_at: int = -1
    alpha: float = 0.05


class SyntheticControl:
    """Synthetic Control estimator.

    Estimates treatment effects using a synthetic counterfactual constructed
    from donor units via convex weights (simplex). Supports "nested" optimization
    (Abadie et al. 2010).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        id_name: str,
        t_name: str,
        y_name: str,
        treat_name: str | None = None,
        cohort_name: str | None = None,
        center_at: int = -1,
        alpha: float = 0.05,
        max_iter: int = 10_000,
        tol: float = 1e-10,
        v_mode: str = "identity",
        anticipation: int = 0,
        base_period: str = "varying",
        control_group: str = "never",
        tau_weight: str = "treated_t",
        boot_cluster: str = "twoway",
    ) -> None:
        if treat_name is None and cohort_name is None:
            raise TypeError(
                "Provide either treat_name or cohort_name for SyntheticControl.",
            )
        if v_mode not in {"identity", "nested"}:
            raise ValueError("v_mode must be 'identity' or 'nested'")
        if base_period not in {"varying", "universal"}:
            raise ValueError("base_period must be 'varying' or 'universal'")
        cg_norm = str(control_group).lower().replace("_", "").replace("-", "").replace("treated", "")
        if cg_norm != "never":
            raise ValueError("SyntheticControl requires control_group='never' (donor pool must be never-treated).")
        self.control_group = cg_norm
        tau_weight_norm = str(tau_weight).lower()
        if tau_weight_norm not in {"equal", "group", "treated_t"}:
            raise ValueError("tau_weight must be one of {'equal','group','treated_t'}.")
        self.tau_weight = tau_weight_norm
        boot_cluster_norm = str(boot_cluster).lower()
        if boot_cluster_norm not in {"unit", "twoway", "time"}:
            raise ValueError("boot_cluster must be 'unit', 'twoway', or 'time'")
        self.boot_cluster = boot_cluster_norm
        self.v_mode = v_mode
        self.anticipation = int(anticipation)
        self.base_period = str(base_period)
        treat_col = "treat" if treat_name is None else str(treat_name)
        self.spec = _Spec(
            id_name=str(id_name),
            t_name=str(t_name),
            y_name=str(y_name),
            treat_name=treat_col,
            cohort_name=(None if cohort_name is None else str(cohort_name)),
            center_at=int(center_at),
            alpha=float(alpha),
        )
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._formula = None
        self._formula_df: pd.DataFrame | None = None

    # --------------------------------------------------------------
    def _wide_from_long(self, df: pd.DataFrame) -> tuple[np.ndarray, list, list]:
        i_name, t_name, y_name = self.spec.id_name, self.spec.t_name, self.spec.y_name
        dfx = df[[i_name, t_name, y_name]].copy()
        if dfx[y_name].isna().any():
            raise ValueError(
                "Missing outcomes are not allowed; please pass a balanced, complete panel.",
            )
        ids = sorted(dfx[i_name].unique())
        times = sorted(dfx[t_name].unique())
        id_index = {i: k for k, i in enumerate(ids)}
        t_index = {t: k for k, t in enumerate(times)}
        Y = np.empty((len(ids), len(times)), dtype=np.float64)
        Y.fill(np.nan)
        for _, r in dfx.iterrows():
            Y[id_index[r[i_name]], t_index[r[t_name]]] = float(r[y_name])
        if np.isnan(Y).any():
            raise ValueError("Panel must be balanced across ids and times (no gaps).")
        return Y, ids, times

    def _validate_treatment_path(self, df: pd.DataFrame) -> None:
        i_name, t_name, tr = self.spec.id_name, self.spec.t_name, self.spec.treat_name
        if tr not in df.columns:
            raise KeyError(f"Treatment column '{tr}' not found.")
        vals = pd.unique(df[tr].dropna())
        if not set(vals).issubset({0, 1}):
            raise ValueError("SyntheticControl requires binary treatment in {0,1}.")

        dfx = df[[i_name, t_name, tr]].copy()
        dfx = dfx.sort_values([i_name, t_name], kind="mergesort")
        d = dfx[tr].astype(int)
        d_cummax = dfx.groupby(i_name, sort=False)[tr].cummax().astype(int)
        if not np.array_equal(d.to_numpy(), d_cummax.to_numpy()):
            raise ValueError(
                "Treatment must be weakly increasing within unit over time (no 1->0 switches).",
            )

        if self.spec.cohort_name is not None:
            g_name = self.spec.cohort_name
            if g_name not in df.columns:
                raise KeyError(f"Cohort column '{g_name}' not found.")
            # Cohort must be constant within unit.
            g_nunique = df.groupby(i_name, sort=False)[g_name].nunique(dropna=False)
            if int(g_nunique.max()) > 1:
                raise ValueError("Cohort must be constant within unit.")
            g_by_id = df.groupby(i_name, sort=False)[g_name].first()
            expected = ((df[t_name] >= df[g_name]) & (df[g_name] > 0)).astype(int)
            if not np.array_equal(expected.to_numpy(), df[tr].astype(int).to_numpy()):
                raise ValueError(
                    "treat_name is inconsistent with cohort_name. Expected D_it = 1[t>=g_i] for g_i>0.",
                )

    def _treated_info(self, df: pd.DataFrame) -> tuple[dict[int, list[int]], list[int]]:
        """Extract cohort structure and donor pool (never-treated units).

        Returns
        -------
        cohorts : dict[int, list[int]]
            Maps cohort (adoption time) to list of treated unit IDs
        donors : list[int]
            List of never-treated unit IDs (donor pool)

        """
        i_name, t_name, tr = self.spec.id_name, self.spec.t_name, self.spec.treat_name
        g_by_id = (
            df.loc[df[tr] == 1, [i_name, t_name]]
            .groupby(i_name, sort=False)[t_name]
            .min()
            .to_dict()
        )
        # Group treated units by their adoption time (cohort)
        cohorts: dict[int, list[int]] = {}
        for unit_id, adoption_time in g_by_id.items():
            g = int(adoption_time)
            if g not in cohorts:
                cohorts[g] = []
            cohorts[g].append(unit_id)

        # Donor pool: never-treated units (those not in g_by_id)
        treated_ids = set(g_by_id.keys())
        all_ids = set(df[i_name].unique())
        donors = sorted(all_ids - treated_ids)

        return cohorts, donors

    def _sc_if_unit_scores(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        times: np.ndarray,
        cohorts: dict,
        results_g: dict,
        U_hat: np.ndarray,
        tau_grid: np.ndarray,
        *,
        tol: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        N, T = Y.shape
        t2pos = {t: idx for idx, t in enumerate(times)}
        tau2k = {int(tau): k for k, tau in enumerate(tau_grid)}

        tau_weight = getattr(self, "tau_weight", "treated_t")
        den_tau = {int(tau): 0.0 for tau in tau_grid}
        for g, meta in results_g.items():
            g_eff = meta.get("g_eff", g)
            treated_rows = np.asarray(meta.get("treated_row_ids", []), dtype=int)
            ntr = float(treated_rows.size)
            if ntr <= 0:
                continue
            weight_g = 1.0 if tau_weight == "equal" else ntr
            for t_idx in range(T):
                tau = event_tau(times[t_idx], g_eff, t2pos)
                if int(tau) in den_tau:
                    den_tau[int(tau)] += weight_g

        base_tau = int(self.spec.center_at)
        post_den = sum(v for tau, v in den_tau.items() if int(tau) > base_tau)
        if post_den <= 0:
            raise ValueError("No post-treatment periods to form post ATT.")

        psi_tau = np.zeros((N, len(tau_grid)))
        psi_post = np.zeros(N)

        def _softmax(u: np.ndarray) -> np.ndarray:
            z = u - np.max(u)
            ez = np.exp(z)
            return ez / np.sum(ez)

        def _softmax_jac(v: np.ndarray) -> np.ndarray:
            return np.diag(v) - np.outer(v, v)

        def _kkt_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
            k = A.shape[0]
            ones = np.ones((k, 1))
            M = np.block([[A, ones], [ones.T, np.zeros((1, 1))]])
            rhs = np.vstack([b.reshape(-1, 1), np.ones((1, 1))])
            U, s, Vt = la.svd(M, full_matrices=False)
            s_inv = np.where(s > 1e-14, 1.0 / s, 0.0)
            sol = la.dot((Vt.T * s_inv), la.dot(U.T, rhs))
            return sol[:-1, 0]

        def _kkt_dw(Xs, y, v, wS, dX_s, dy, dv):
            if Xs.shape[1] <= 1:
                return np.zeros(Xs.shape[1])
            D = np.diag(v)
            dD = np.diag(dv) if dv is not None else np.zeros_like(D)
            A = la.dot(Xs.T, la.hadamard(Xs, v.reshape(-1, 1)))
            b = la.dot(Xs.T, la.hadamard(y.reshape(-1, 1), v.reshape(-1, 1)))
            dA = la.dot(dX_s.T, la.dot(D, Xs)) + la.dot(Xs.T, la.dot(dD, Xs)) + la.dot(Xs.T, la.dot(D, dX_s))
            db = la.dot(dX_s.T, la.dot(D, y)) + la.dot(Xs.T, la.dot(dD, y)) + la.dot(Xs.T, la.dot(D, dy))
            k = Xs.shape[1]
            ones = np.ones((k, 1))
            M = np.block([[A, ones], [ones.T, np.zeros((1, 1))]])
            rhs = np.vstack([(db - la.dot(dA, wS)).reshape(-1, 1), np.zeros((1, 1))])
            U, s, Vt = la.svd(M, full_matrices=False)
            s_inv = np.where(s > 1e-14, 1.0 / s, 0.0)
            sol = la.dot((Vt.T * s_inv), la.dot(U.T, rhs))
            return sol[:-1, 0]

        def _w_from_u_fixedS(X, y, u, S):
            v = _softmax(u)
            Xs = X[:, S]
            A = la.dot(Xs.T, la.hadamard(Xs, v.reshape(-1, 1)))
            b = la.dot(Xs.T, la.hadamard(y.reshape(-1, 1), v.reshape(-1, 1)))
            wS = _kkt_solve(A, b)
            return v, wS

        def _grad_u_fixedS(X, y, u, S):
            Kp = X.shape[0]
            v, wS = _w_from_u_fixedS(X, y, u, S)
            w = np.zeros(X.shape[1], dtype=float)
            w[S] = wS
            r = (y - la.dot(X, w)).reshape(-1)
            Jv = _softmax_jac(v)
            g = np.zeros(Kp, dtype=float)
            Xs = X[:, S]
            for q in range(Kp):
                dv = Jv[:, q]
                dwS = _kkt_dw(Xs, y, v, wS, np.zeros_like(Xs), np.zeros_like(y), dv)
                dr = -la.dot(Xs, dwS)
                g[q] = (2.0 / Kp) * float(la.dot(r.reshape(1, -1), dr.reshape(-1, 1)))
            return g

        def _jac_u_num(X, y, u, S, eps=1e-6):
            Kp = X.shape[0]
            J = np.zeros((Kp, Kp), dtype=float)
            for q in range(Kp):
                du = np.zeros(Kp, dtype=float)
                du[q] = eps
                gp = _grad_u_fixedS(X, y, u + du, S)
                gm = _grad_u_fixedS(X, y, u - du, S)
                J[:, q] = (gp - gm) / (2.0 * eps)
            return J

        for g, meta in results_g.items():
            g_eff = meta.get("g_eff", g)
            pre_mask = times < g_eff
            idx_tr = meta["idx_tr"]
            tr_rows = np.asarray(meta.get("treated_row_ids", []), dtype=int)
            ntr = int(tr_rows.size)
            if ntr <= 0:
                continue

            weight_g = 1.0 if tau_weight == "equal" else float(ntr)

            donors_idx = np.asarray(meta["donors_idx"], dtype=int)
            N0 = donors_idx.size
            if N0 == 0:
                continue

            pre_idx = np.where(pre_mask)[0]
            Kp = len(pre_idx)
            if Kp == 0:
                continue

            X = Y[donors_idx][:, pre_mask].T

            weights_list = meta.get("weights", [])
            u_list = meta.get("u_list", [])

            treated_struct = []
            for k_tr, tr in enumerate(tr_rows):
                y = Y[tr, pre_mask].astype(float)
                w0 = np.asarray(weights_list[k_tr], dtype=float).reshape(-1)
                S = (w0 > tol)
                if S.sum() == 0:
                    S[np.argmax(w0)] = True

                if self.v_mode == "identity":
                    u = None
                    v = np.ones(Kp, dtype=float) / Kp
                    treated_struct.append({"tr": tr, "S": S, "u": u, "v": v, "w": w0, "y": y, "X": X})
                else:
                    u0 = np.asarray(u_list[k_tr], dtype=float).reshape(-1)
                    if u0.shape[0] != Kp:
                        raise ValueError("nested SC: stored u has wrong length vs pre periods")
                    u = u0.copy()
                    v, _wS = _w_from_u_fixedS(X, y, u, S)
                    Jg = _jac_u_num(X, y, u, S)
                    Jv = _softmax_jac(v)
                    treated_struct.append({"tr": tr, "S": S, "u": u, "v": v, "w": w0, "y": y, "X": X, "Jg": Jg, "Jv": Jv})

            for i in range(N):
                ui = U_hat[i, :]
                d_att = np.zeros(T, dtype=float)
                for st in treated_struct:
                    tr = st["tr"]
                    w = st["w"]
                    S = st["S"]
                    dtau = np.zeros(T, dtype=float)
                    if i == tr:
                        dtau += ui
                    if idx_tr[i] == False and i in donors_idx:
                        m = np.where(donors_idx == i)[0][0]
                        dtau -= w[m] * ui
                    affects_weights = (i == tr) or (idx_tr[i] == False and i in donors_idx)
                    if affects_weights:
                        dX = np.zeros_like(X)
                        dy = np.zeros(Kp, dtype=float)
                        u_pre = ui[pre_mask].astype(float)
                        if i == tr:
                            dy = u_pre
                        if idx_tr[i] == False and i in donors_idx:
                            m = np.where(donors_idx == i)[0][0]
                            dX[:, m] = u_pre
                        if self.v_mode == "identity":
                            v = st["v"]
                            Xs = X[:, S]
                            wS = st["w"][S]
                            dwS = _kkt_dw(Xs, st["y"], v, wS, dX[:, S], dy, None)
                            dw = np.zeros_like(w)
                            dw[S] = dwS
                        else:
                            u = st["u"]
                            Jg = st["Jg"]
                            Jv = st["Jv"]
                            y = st["y"]
                            g_base = _grad_u_fixedS(X, y, u, S)
                            eps = 1e-6
                            g_pert = _grad_u_fixedS(X + eps * dX, y + eps * dy, u, S)
                            gY = (g_pert - g_base) / eps
                            U_jg, s_jg, Vt_jg = la.svd(Jg, full_matrices=False)
                            s_jg_inv = np.where(s_jg > 1e-14, 1.0 / s_jg, 0.0)
                            du = la.dot((Vt_jg.T * s_jg_inv), la.dot(U_jg.T, (-gY).reshape(-1, 1)))
                            du = du[:, 0]
                            dv = la.dot(Jv, du)
                            v = st["v"]
                            Xs = X[:, S]
                            wS = st["w"][S]
                            dwS = _kkt_dw(Xs, y, v, wS, dX[:, S], dy, dv)
                            dw = np.zeros_like(w)
                            dw[S] = dwS
                        dtau -= la.dot(Y[donors_idx][:, :].T, dw)
                    d_att += dtau / float(ntr)

                # Bias-corrected SC: subtract pre-period mean gap.
                # delta_t = att_t - mean_pre(att_pre)
                if pre_idx.size > 0:
                    d_att = d_att - float(np.mean(d_att[pre_mask]))
                for t_idx in range(T):
                    tau = int(event_tau(times[t_idx], g_eff, t2pos))
                    if tau in tau2k and den_tau[tau] > 0:
                        k = tau2k[tau]
                        psi_tau[i, k] += (weight_g * d_att[t_idx]) / den_tau[tau]
                    if tau > base_tau:
                        psi_post[i] += (weight_g * d_att[t_idx]) / post_den

        return psi_tau, psi_post

    # --------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame | None = None,
        *,
        boot: BootConfig | None = None,
        _boot: object | None = None,
    ) -> EstimationResult:

        """Fit Synthetic Control model.

        Estimates ATT path, aggregated effects, and bootstrap confidence bands.
        """
        if _boot is None and boot is not None:
            _boot = boot
        if _boot is not None:
            if isinstance(_boot, dict):
                _boot = BootConfig(**_boot)
            elif not isinstance(_boot, BootConfig):
                raise TypeError("boot must be BootConfig, dict, or None")
        if df is None:
            df = getattr(self, "_formula_df", None)
            if df is None:
                raise ValueError(
                    "Provide df explicitly or instantiate the estimator via from_formula().",
                )

        t_name = self.spec.t_name
        tr = self.spec.treat_name
        if tr not in df.columns and self.spec.cohort_name is not None:
            coh = df[self.spec.cohort_name]
            df = df.copy()
            df[tr] = ((df[t_name] >= coh) & (coh > 0)).astype(int)

        self._validate_treatment_path(df)

        cohorts, donors = self._treated_info(df)
        if len(donors) == 0:
            raise ValueError(
                "Donor pool is empty; need at least one never-treated unit.",
            )
        if len(cohorts) == 0:
            raise ValueError("No treated units found.")

        # --- build wide Y (ids x times) ---
        Y, ids, times = self._wide_from_long(df)
        times_arr = np.array(times)
        t2pos = time_to_pos(times_arr)
        id_to_row = {i: k for k, i in enumerate(ids)}
        t_to_col = {t: k for k, t in enumerate(times)}
        j_donors = np.array([id_to_row[j] for j in donors], dtype=int)

        # --- Process each cohort and aggregate in event-time ---
        results_g: dict[int, dict[str, object]] = {}
        tau_union: set[int] = set()

        for g, treated_units in cohorts.items():
            t0_col = t_to_col[g]
            t0_eff_col = int(t0_col) - int(self.anticipation)
            if t0_eff_col < 0:
                continue
            pre = np.arange(0, t0_eff_col)
            post = np.arange(t0_eff_col, len(times))
            g_eff = times[t0_eff_col]

            if pre.size == 0:
                continue

            att_paths_g = []
            for treated_id in treated_units:
                i_treated = id_to_row[treated_id]

                y_pre = Y[i_treated, pre].astype(np.float64)
                X_pre = Y[j_donors][:, pre].T.astype(np.float64)
                Y_pre_donors = Y[j_donors][:, pre].astype(np.float64)

                if self.v_mode == "nested":
                    u, v, w = _optimize_V_nested(
                        y_pre, X_pre.T,
                        y_pre, Y_pre_donors,
                        max_iter_fw=self.max_iter,
                        tol_fw=self.tol,
                    )
                else:
                    w = _frank_wolfe_simplex(
                        X_pre, y_pre, max_iter=self.max_iter, tol=self.tol,
                    )
                    u = None

                y_treated = Y[i_treated, :].astype(np.float64)
                X_all = Y[j_donors, :].T.astype(np.float64)
                y_synth = la.dot(X_all, w)
                # Bias-corrected SC (SC with intercept via DID-style correction):
                # delta_t = (Y1t - Y0w,t) - mean_{pre}(Y1t - Y0w,t)
                gap = y_treated - y_synth
                bias = float(np.mean(gap[pre])) if pre.size > 0 else 0.0
                att_path = gap - bias
                att_paths_g.append(att_path)
                if "weights" not in results_g.get(g, {}):
                    results_g[g] = {"weights": [], "y_synth": [], "treated_row_ids": [], "u_list": []}
                results_g[g]["weights"].append(w)
                results_g[g]["y_synth"].append(y_synth)
                results_g[g]["treated_row_ids"].append(i_treated)
                results_g[g].setdefault("bias", []).append(bias)
                if self.v_mode == "nested":
                    results_g[g]["u_list"].append(u)

            att_path_g = np.mean(att_paths_g, axis=0)

            results_g[g]["att_path"] = att_path_g
            results_g[g]["n_treated"] = len(treated_units)
            results_g[g]["treated_ids"] = treated_units
            results_g[g]["g_eff"] = g_eff
            results_g[g]["g_eff_col"] = int(t0_eff_col)
            idx_tr = np.isin(ids, treated_units)
            results_g[g]["idx_tr"] = idx_tr
            omega_avg = np.mean([np.asarray(w, dtype=float) for w in results_g[g]["weights"]], axis=0)
            results_g[g]["omega"] = omega_avg
            results_g[g]["donors_idx"] = j_donors

            for t_val in times:
                tau_val = event_tau(t_val, g_eff, t2pos)
                tau_union.add(tau_val)

        if len(results_g) == 0:
            raise ValueError(
                "No treated cohorts have usable pre-treatment periods. "
                "Check that the panel has pre-treatment data (and that anticipation is not too large).",
            )

        # Aggregate across cohorts in event-time τ = t - g
        tau_union_sorted = sorted(tau_union)
        att_tau_num: dict[int, float] = {int(tau): 0.0 for tau in tau_union_sorted}
        att_tau_den: dict[int, float] = {int(tau): 0.0 for tau in tau_union_sorted}

        for g, meta in results_g.items():
            att_path = meta["att_path"]
            n_tr = meta["n_treated"]
            weight_g = 1.0 if self.tau_weight == "equal" else float(n_tr)
            g_eff = meta.get("g_eff", g)
            for t_idx, t_val in enumerate(times):
                tau_val = event_tau(t_val, g_eff, t2pos)
                att_tau_num[tau_val] += weight_g * float(att_path[t_idx])
                att_tau_den[tau_val] += weight_g

        # Compute weighted average ATT for each τ
        att_tau_list = []
        for tau in tau_union_sorted:
            if att_tau_den[tau] > 0:
                att_tau_list.append(att_tau_num[tau] / att_tau_den[tau])
            else:
                att_tau_list.append(np.nan)

        # Build Series indexed by tau
        tau_array = np.array(tau_union_sorted, dtype=int)
        att_series = pd.Series(
            att_tau_list, index=pd.Index(tau_array, name="tau"), name="params",
        )

        # Enforce event-time normalization at the base period (tau=center_at).
        # This matches regression-style omitted-base normalization and keeps
        # bootstrap draws comparable across taus.
        center_at = int(self.spec.center_at)
        base_locs = np.flatnonzero(tau_array == center_at)
        if base_locs.size != 1:
            raise ValueError(
                f"Expected exactly one base tau={center_at} in tau grid; got {base_locs.size}.",
            )
        base_idx = int(base_locs[0])
        theta_base = float(att_series.iloc[base_idx])
        if not np.isfinite(theta_base):
            raise ValueError(
                f"Non-finite base ATT at tau={center_at} (att={theta_base}).",
            )
        att_series = att_series.astype(float) - theta_base
        att_series.iloc[base_idx] = 0.0

        mask_post = tau_array > center_at
        post_den = float(sum(att_tau_den.get(int(t), 0.0) for t in tau_array[mask_post]))
        if post_den <= 0:
            post_agg = float("nan")
        else:
            post_num = float(sum(att_tau_num.get(int(t), 0.0) for t in tau_array[mask_post]))
            post_agg = float((post_num / post_den) - theta_base)

        tau_grid = np.array(tau_union_sorted, dtype=int)
        n_treated_total = sum(len(meta["treated_ids"]) for meta in results_g.values())

        # Default inference: permutation for treated=1, unit bootstrap for treated>=2
        if _boot is None:
            if n_treated_total >= 2:
                _boot = BootConfig(n_boot=bt.DEFAULT_BOOTSTRAP_ITERATIONS, mode="wild_if")
            else:
                _boot = BootConfig(n_boot=bt.DEFAULT_BOOTSTRAP_ITERATIONS, mode="permutation")
            boot = _boot  # Sync aliases

        B = 0
        band_level = round(100.0 * (1.0 - float(self.spec.alpha)))
        bands = None
        se_series = None
        att_tau_star = np.full((tau_grid.size, 0), np.nan, dtype=float)
        post_ci_df = pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
        post_att_se = None
        boot_info = None

        # Reproducibility artifacts (populated for wild_if)
        boot_cluster_req = self.boot_cluster
        boot_cluster_used: str | None = None
        W_multipliers_inference: pd.DataFrame | None = None
        multipliers_log: dict[str, object] | None = None
        boot_config_used: BootConfig | None = None

        if _boot is not None and int(getattr(_boot, "n_boot", 0)) > 1:
            B = int(_boot.n_boot)
            alpha_level = float(self.spec.alpha)

            # (variables already initialized above)

            mode = (getattr(_boot, "mode", None) or "auto").lower().strip()
            if mode == "auto":
                mode = "wild_if" if n_treated_total >= 2 else "permutation"
            # Backwards-compat aliases
            if mode == "if":
                mode = "wild_if"
            if mode == "placebo":
                mode = "permutation"
            if mode not in {"permutation", "wild_if"}:
                raise ValueError(
                    "SyntheticControl boot mode must be one of {'auto','permutation','placebo','wild_if','if'}.",
                )

            # Strict rule: treated_total==1 -> permutation; treated_total>=2 -> wild IF.
            if mode == "wild_if" and n_treated_total < 2:
                raise ValueError(
                    "SyntheticControl IF inference requires at least 2 treated units/clusters. Use mode='permutation' (or 'placebo') when treated_total==1.",
                )
            if mode == "permutation" and n_treated_total >= 2:
                raise ValueError(
                    "SyntheticControl permutation/placebo inference is only supported when treated_total==1. Use mode='if' when treated_total>=2.",
                )

            theta_hat = att_series.reindex(tau_grid).to_numpy(dtype=float)

            if mode == "permutation":
                j_donors = np.array([id_to_row[j] for j in donors], dtype=int)
                g0 = list(results_g.keys())[0]
                meta0 = results_g[g0]
                g0_eff = meta0.get("g_eff", g0)
                t0_eff_col = int(meta0.get("g_eff_col", t_to_col[g0]))
                pre = np.arange(0, t0_eff_col)
                att_placebo_list = []
                for j_idx in j_donors:
                    donors_for_j = np.array([d for d in j_donors if d != j_idx], dtype=int)
                    if donors_for_j.size == 0:
                        continue
                    y_pre_j = Y[j_idx, pre].astype(np.float64)
                    X_pre_j = Y[donors_for_j][:, pre].T.astype(np.float64)
                    try:
                        if self.v_mode == "nested":
                            _, _, w_j = _optimize_V_nested(
                                y_pre_j,
                                X_pre_j.T,
                                y_pre_j,
                                Y[donors_for_j][:, pre].astype(np.float64),
                                max_iter_fw=self.max_iter,
                                tol_fw=self.tol,
                            )
                        else:
                            w_j = _frank_wolfe_simplex(X_pre_j, y_pre_j, max_iter=self.max_iter, tol=self.tol)
                    except Exception:
                        continue
                    y_j = Y[j_idx, :].astype(np.float64)
                    X_all_j = Y[donors_for_j, :].T.astype(np.float64)
                    y_synth_j = la.dot(X_all_j, w_j)
                    gap_j = y_j - y_synth_j
                    bias_j = float(np.mean(gap_j[pre])) if pre.size > 0 else 0.0
                    att_path_j = gap_j - bias_j
                    att_tau_j = np.full(tau_grid.size, np.nan, dtype=float)
                    for t_idx, t_val in enumerate(times):
                        tau_val = event_tau(t_val, g0_eff, t2pos)
                        j_tau = np.searchsorted(tau_grid, tau_val)
                        if j_tau < tau_grid.size and tau_grid[j_tau] == tau_val:
                            att_tau_j[j_tau] = att_path_j[t_idx]
                    att_placebo_list.append(att_tau_j)
                n_placebos = len(att_placebo_list)
                if n_placebos > 0:
                    att_placebo = np.column_stack(att_placebo_list)
                    # Enforce base-period normalization within each placebo draw
                    base_draw = att_placebo[base_idx, :].copy()
                    att_placebo = att_placebo - base_draw[None, :]
                    att_placebo[base_idx, :] = 0.0

                    # SEs use placebo distribution centered at its own mean
                    mu = np.mean(att_placebo, axis=1)
                    diffs_all = att_placebo - mu[:, None]
                    se_vals = bt.bootstrap_se(diffs_all)
                    se_series = pd.Series(se_vals, index=tau_grid, dtype=float)
                    se_series.loc[center_at] = 0.0

                    def _sup_t_band_placebo(mask: np.ndarray) -> pd.DataFrame:
                        idx = np.flatnonzero(mask)
                        if idx.size == 0 or n_placebos < 2:
                            return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
                        th = theta_hat[idx]
                        thb = att_placebo[idx, :]
                        mu_b = thb.mean(axis=1)
                        diffs = thb - mu_b[:, None]
                        se = np.std(diffs, axis=1, ddof=1)
                        ok = np.isfinite(se) & (se > 0)
                        if not np.any(ok):
                            return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
                        tdraw = diffs[ok, :] / se[ok, None]
                        sup_abs = np.max(np.abs(tdraw), axis=0)
                        c = float(bt.finite_sample_quantile(sup_abs, 1.0 - alpha_level))
                        lo = th.copy(); hi = th.copy()
                        lo[ok] = th[ok] - c * se[ok]
                        hi[ok] = th[ok] + c * se[ok]
                        return pd.DataFrame({"lower": pd.Series(lo, index=tau_grid[idx]), "upper": pd.Series(hi, index=tau_grid[idx])}).sort_index()

                    pre_mask = tau_array < center_at
                    band_pre = _sup_t_band_placebo(pre_mask)
                    band_post = _sup_t_band_placebo(mask_post)
                    band_full = _sup_t_band_placebo(tau_array != center_at)
                    # Post scalar uses den-weighted average over post taus.
                    den_vec = np.array([float(att_tau_den.get(int(t), 0.0)) for t in tau_grid], dtype=float)
                    idx_post = np.flatnonzero(mask_post)
                    if idx_post.size > 0:
                        w = den_vec[idx_post]
                        if (not np.isfinite(w).all()) or np.any(w <= 0):
                            raise ValueError("Non-finite or non-positive weights for post_ATT aggregation.")
                        w = w / float(w.sum())
                        post_star = np.sum(att_placebo[idx_post, :] * w[:, None], axis=0)
                        post_att_se = float(np.std(post_star, ddof=1))
                        mu_post = float(post_star.mean())
                        tdraw_post = (post_star - mu_post) / post_att_se
                        c_post = float(bt.finite_sample_quantile(np.abs(tdraw_post), 1.0 - alpha_level))
                        post_ci_df = pd.DataFrame({"lower": [post_agg - c_post * post_att_se], "upper": [post_agg + c_post * post_att_se]})
                    bands = {
                        "pre": band_pre,
                        "post": band_post,
                        "full": band_full,
                        "post_scalar": post_ci_df,
                        "__meta__": {
                            "origin": "permutation",
                            "mode": mode,
                            "kind": "uniform",
                            "level": int(100 * (1.0 - alpha_level)),
                            "n_placebos": n_placebos,
                            "estimator": "sc",
                        },
                    }
                else:
                    bands = {
                        "pre": pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)}),
                        "post": pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)}),
                        "full": pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)}),
                        "post_scalar": post_ci_df,
                        "__meta__": {"origin": "permutation", "mode": mode, "kind": "uniform", "level": int(100 * (1.0 - alpha_level)), "n_placebos": 0, "estimator": "sc"},
                    }
                boot_info = {
                    "method": "permutation",
                    "B": int(n_placebos),
                    "n_placebos": int(n_placebos),
                    "ATT_tau_draws": att_placebo if n_placebos > 0 else np.full((tau_grid.size, 0), np.nan, dtype=float),
                }
                B = int(n_placebos)

            elif mode == "wild_if":
                # This implementation uses unit-level influence scores.
                # Time or two-way clustering would require an observation-level IF.
                if boot_cluster_req == "time":
                    raise ValueError(
                        "SyntheticControl wild_if currently supports unit-level multipliers only; "
                        "boot_cluster='time' is not compatible with unit-score IF.",
                    )
                boot_cluster_used = "unit" if boot_cluster_req in {"unit", "twoway"} else "unit"

                # Influence Function Bootstrap (Wild Multipliers on IF)
                U_hat = Y.astype(float, copy=False)

                # Check args for _sc_if_unit_scores
                # Needs W matrix (assignment).
                W_assign = np.zeros((len(ids), len(times)), dtype=float)
                for g, meta in results_g.items():
                    t0_col = t_to_col[g]
                    for tid in meta["treated_ids"]:
                        i_tr = id_to_row[tid]
                        W_assign[i_tr, t0_col:] = 1.0

                psi_tau, psi_post = self._sc_if_unit_scores(
                     Y=Y, W=W_assign, times=times_arr, cohorts=cohorts,
                     results_g=results_g, U_hat=U_hat, tau_grid=tau_grid,
                     tol=self.tol
                )
                # Setup multipliers (mean 0, var 1 by construction)
                bc = replace(
                    _boot,
                    n_boot=int(_boot.n_boot),
                    dist=getattr(_boot, "dist", "rademacher"),
                    seed=getattr(_boot, "seed", None),
                )
                W_df, mlog = bc.make_multipliers(n_obs=Y.shape[0])  # (N, B)
                W_mat = W_df.to_numpy()
                B_actual = W_mat.shape[1]

                perturbation = (psi_tau.T @ W_mat) / Y.shape[0]
                att_tau_star = theta_hat[:, None] + perturbation

                # Enforce base normalization draw-by-draw
                base_draw = att_tau_star[base_idx, :].copy()
                att_tau_star = att_tau_star - base_draw[None, :]
                att_tau_star[base_idx, :] = 0.0

                filled = B_actual
                se_vals = bt.bootstrap_se(att_tau_star)
                se_series = pd.Series(se_vals, index=tau_grid, dtype=float)
                se_series.loc[center_at] = 0.0

                # Uniform Bands (Placebo/Wild style)
                # ... reuse band logic ...
                # Re-using the logic below requires fitting into the loop structure or copying it.
                # The existing loop handles "unit" and "unit-like" bootstraps.
                # IF is vectorized.
                # I'll create `bands` here directly to avoid complex flow merge.

                def _sup_t_band_if(mask: np.ndarray) -> pd.DataFrame:
                    idx = np.flatnonzero(mask)
                    if idx.size == 0 or filled < 2:
                        return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
                    th = theta_hat[idx]
                    thb = att_tau_star[idx, :]
                    diffs = thb - th[:, None]
                    # SE from bootstrap distribution
                    se = np.std(diffs, axis=1, ddof=1)
                    ok = np.isfinite(se) & (se > 0)
                    if not np.any(ok):
                        return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
                    tdraw = diffs[ok, :] / se[ok, None]
                    sup_abs = np.nanmax(np.abs(tdraw), axis=0)
                    c = float(bt.finite_sample_quantile(sup_abs, 1.0 - alpha_level))
                    lo = th.copy(); hi = th.copy()
                    lo[ok] = th[ok] - c * se[ok]
                    hi[ok] = th[ok] + c * se[ok]
                    return pd.DataFrame({"lower": pd.Series(lo, index=tau_grid[idx]), "upper": pd.Series(hi, index=tau_grid[idx])}).sort_index()

                band_pre = _sup_t_band_if(tau_array < center_at)
                band_post = _sup_t_band_if(mask_post)
                band_full = _sup_t_band_if(tau_array != center_at)

                den_vec = np.array([float(att_tau_den.get(int(t), 0.0)) for t in tau_grid], dtype=float)
                idx_post = np.flatnonzero(mask_post)
                if filled > 1 and idx_post.size > 0:
                    w = den_vec[idx_post]
                    if (not np.isfinite(w).all()) or np.any(w <= 0):
                        raise ValueError("Non-finite or non-positive weights for post_ATT aggregation.")
                    w = w / float(w.sum())
                    post_star_arr = np.sum(att_tau_star[idx_post, :] * w[:, None], axis=0)
                    post_att_se = float(np.std(post_star_arr, ddof=1))
                    tdraw = (post_star_arr - post_agg) / post_att_se
                    c_post = float(bt.finite_sample_quantile(np.abs(tdraw), 1.0 - alpha_level))
                    post_ci_df = pd.DataFrame({"lower": [post_agg - c_post * post_att_se], "upper": [post_agg + c_post * post_att_se]})

                bands = {
                    "pre": band_pre, "post": band_post, "full": band_full, "post_scalar": post_ci_df,
                    "__meta__": {"origin": "wild-if", "mode": mode, "kind": "uniform", "level": int(100*(1-alpha_level)), "B": filled, "estimator": "synthetic"}
                }
                boot_info = {
                    "method": "wild_if",
                    "B": int(filled),
                    "dist": getattr(_boot, "dist", "rademacher"),
                    "ATT_tau_draws": att_tau_star,
                }
                B = int(filled)
                W_multipliers_inference = W_df
                multipliers_log = mlog
                boot_config_used = bc

            # Attach common draws where available
            if boot_info is not None and "ATT_tau_draws" in boot_info:
                den_vec = np.array([float(att_tau_den.get(int(t), 0.0)) for t in tau_grid], dtype=float)
                idx_post = np.flatnonzero(mask_post)
                if idx_post.size > 0:
                    w = den_vec[idx_post]
                    if (not np.isfinite(w).all()) or np.any(w <= 0):
                        raise ValueError("Non-finite or non-positive weights for post_ATT aggregation.")
                    w = w / float(w.sum())
                    draws = np.asarray(boot_info["ATT_tau_draws"], dtype=float)
                    post_star_arr = np.sum(draws[idx_post, :] * w[:, None], axis=0)
                    boot_info["post_ATT_draws"] = post_star_arr

        att_df = pd.DataFrame({"tau": att_series.index, "att": att_series.to_numpy()}).set_index("tau")

        rmspe_pre_list = []
        rmspe_post_list = []
        for g, meta in results_g.items():
            t0_col = t_to_col[g]
            pre = np.arange(0, t0_col)
            post = np.arange(t0_col, len(times))
            for treated_id in meta["treated_ids"]:
                i_treated = id_to_row[treated_id]
                y_pre_unit = Y[i_treated, pre].astype(np.float64)
                X_pre_unit = Y[j_donors][:, pre].T.astype(np.float64)
                w_unit = _frank_wolfe_simplex(X_pre_unit, y_pre_unit, max_iter=self.max_iter, tol=self.tol)
                y_treated_unit = Y[i_treated, :].astype(np.float64)
                X_all_unit = Y[j_donors, :].T.astype(np.float64)
                y_synth_unit = la.dot(X_all_unit, w_unit)
                rmspe_pre_list.append(float(np.sqrt(np.mean((y_pre_unit - la.dot(X_pre_unit, w_unit)) ** 2))))
                if post.size > 0:
                    rmspe_post_list.append(float(np.sqrt(np.mean((y_treated_unit[post] - y_synth_unit[post]) ** 2))))

        rmspe_pre = float(np.mean(rmspe_pre_list)) if len(rmspe_pre_list) > 0 else float("nan")
        rmspe_post = float(np.mean(rmspe_post_list)) if len(rmspe_post_list) > 0 else float("nan")

        model_info = {
            "Estimator": "Synthetic Control",
            "BandType": "uniform" if bands is not None else "none",
            "BandLevel": band_level,
            "Alpha": float(self.spec.alpha),
            "B": int(B),
            "CenterAt": center_at,
            "Cohorts": list(cohorts.keys()),
            "TreatedUnits": n_treated_total,
            "Donors": len(donors),
            "RMSPE_pre": rmspe_pre,
            "RMSPE_post": rmspe_post,
            "PostATT": post_agg,
        }
        if boot_info is not None and "post_ATT_draws" in boot_info:
            post_draws = np.asarray(boot_info["post_ATT_draws"], dtype=float).reshape(-1)
            model_info["PostATT_se"] = float(np.std(post_draws, ddof=1)) if post_draws.size > 1 else 0.0
        extra = {
            "cohorts": cohorts,
            "results_by_cohort": results_g,
            "times": times,
            "tau": tau_array,
            "att_tau": att_df.reset_index(),
            "post_scalar": post_ci_df,
            "donors": donors,
            "boot_meta": bands.get("__meta__") if bands else None,
            "boot": boot_info,
            "se_source": "bootstrap" if se_series is not None else None,
            "boot_config": boot_config_used,
            "W_multipliers_inference": W_multipliers_inference,
            "multipliers_log": multipliers_log,
            "boot_cluster_requested": boot_cluster_req if _boot is not None else None,
            "boot_cluster_used": boot_cluster_used,
        }

        res = EstimationResult(
            params=att_series,
            se=se_series,
            bands=bands,
            n_obs=int(df.shape[0]),
            model_info=model_info,
            extra=extra,
        )
        with suppress(Exception):
            attach_formula_metadata(res, getattr(self, "_formula_metadata", None))
        return res

    # --------------- formula constructor (R/Stata semantics) ----------------
    @classmethod
    def from_formula(  # noqa: PLR0913
        cls,
        formula: str,
        df: pd.DataFrame,
        *,
        id_name: str,
        t_name: str,
        treat_name: str | None = None,
        cohort_name: str | None = None,
        center_at: int = -1,
        alpha: float = 0.05,
        max_iter: int = 10_000,
        tol: float = 1e-10,
        options: str | None = None,
    ) -> SyntheticControl:
        """Construct from a minimal formula like 'y ~ 1' (features unused by SC).

        - LHS may include lag/lead/diff tokens in either Stata or R/fixest style.
        - RHS is ignored for SC weights (kept only so that formula parser can
          materialize specials consistently and we inherit row selection rules).
        """
        # Parse with FormulaParser to materialize LHS specials and enforce NA policy
        parser = FormulaParser(df, id_name=id_name, t_name=t_name, warn_if_no_id=True)
        parsed = parser.parse(formula, options=options or "")
        # Normalize working DataFrame and attach formula metadata for downstream
        treat_name = treat_name or "treat"

        df_use, _, meta = prepare_formula_environment(
            formula=formula,
            data=df,
            parsed=parsed,
            boot=None,
            default_boot_kwargs=None,
            # Attach nothing special; SC does not use X/FE/IV here.
            attr_keys=None,
            extra_attrs={
                "id_name": id_name,
                "t_name": t_name,
                "treat_name": treat_name,
                "cohort_name": cohort_name,
            },
        )
        # Extract LHS name from formula (parser already materialized the column)
        y_name = formula.split("~", 1)[0].strip()
        if y_name not in df_use.columns:
            # If parser rewrote LHS to a Q("...") name, keep original LHS fallback:
            # after prepare_formula_environment, df_use already holds the materialized column
            # under the original name used in the formula pipeline.
            raise KeyError(
                f"Response variable '{y_name}' not found after formula materialization.",
            )
        obj = cls(
            id_name=id_name,
            t_name=t_name,
            y_name=y_name,
            treat_name=treat_name,
            cohort_name=cohort_name,
            center_at=center_at,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
        # persist formula working frame for fit()
        obj._formula_df = df_use
        # store meta for attach on result
        obj._formula_metadata = meta
        obj._formula = formula
        return obj
