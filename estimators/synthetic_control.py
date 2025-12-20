# lineareg/estimators/syntheticcontorl.py
"""Synthetic Control estimator with uniform (sup-t) bands and post-period aggregation.

Scope and policy
----------------
- Inference uses unit bootstrap (when treated>=2) or IFE parametric wild bootstrap
  (when treated=1) following gsynth/fect design.
- Provides per-event-time effects with uniform sup-t bands and aggregated post-period
  ATT with bootstrap CI.
- All matrix ops route through :mod:`lineareg.core.linalg` to maintain consistency
  with the rest of the library. Results are returned in :class:`EstimationResult`.
- Bands are attached in `res.bands` under keys:
        * "pre" | "post" | "full": DataFrame(index=tau, columns={'lower','upper'})
        * "post_scalar": DataFrame with a single row {'lower','upper'}
        * "__meta__": {'origin':'bootstrap','kind':'uniform','estimator':'synthetic','level':95,'B':...}

Notes
-----
- Weight fitting solves a least-squares problem on the pre-period under the
  simplex constraint (w >= 0, sum w = 1) via Frank-Wolfe (conditional gradient).

"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass

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


def _time_to_pos(times: np.ndarray) -> dict:
    return {t: i for i, t in enumerate(times)}


def _event_tau(t, g, t2pos: dict) -> int:
    return t2pos[t] - t2pos[g]


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
) -> tuple[np.ndarray, np.ndarray]:
    """Optimize V (predictor importance weights) via nested MSPE minimization.

    Outer: minimize pre-period outcome MSPE over V (diagonal, sum-to-one via softmax)
    Inner: for each V, solve simplex LS for w using Frank-Wolfe
    """
    from scipy.optimize import minimize
    K = X_pred_treat.shape[0]
    T_pre = Y_pre_treat.shape[0]
    J = X_pred_donors.shape[1]

    def _inner_w(v_diag: np.ndarray) -> np.ndarray:
        return _frank_wolfe_simplex(
            X_pred_donors.T, X_pred_treat, max_iter=max_iter_fw, tol=tol_fw, V=v_diag,
        )

    def _mspe_pre(u: np.ndarray) -> float:
        v_diag = np.exp(u) / np.sum(np.exp(u))
        w = _inner_w(v_diag)
        synth_pre = la.dot(Y_pre_donors.T, w)
        return float(np.mean((Y_pre_treat - synth_pre) ** 2))

    u0 = np.zeros(K, dtype=np.float64)
    res = minimize(_mspe_pre, u0, method="Nelder-Mead", options={"maxiter": max_iter_v})
    u_opt = res.x
    v_opt = np.exp(u_opt) / np.sum(np.exp(u_opt))
    w_opt = _inner_w(v_opt)
    return v_opt, w_opt


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
    """Synthetic Control with placebo (uniform sup-t) bands and post aggregation.

    Parameters
    ----------
    id_name, t_name, y_name, treat_name : str
        Column names. `treat_name` must be 1 for treated unit *after* its adoption time
        and 0 otherwise. Exactly **one** treated id is required; donors must be never-treated.
    center_at : int, default -1
        Event-time baseline used for post aggregation (exclude tau == center_at).
    alpha : float, default 0.05
        Two-sided level for bands (e.g., 0.05 -> 95% bands).
    max_iter : int, default 10000
        Maximum Frank-Wolfe iterations for simplex LS on pre-period.
    tol : float, default 1e-10
        Tolerance for Frank-Wolfe line-search / convergence.
    v_mode : str, default "identity"
        V matrix mode for predictor weighting: "identity" (no weighting, current default),
        "nested" (optimize V via pre-MSPE minimization, Abadie-compatible).

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
    ) -> None:
        # Allow either treat_name or cohort_name for API parity with event-study estimators
        if treat_name is None and cohort_name is None:
            raise TypeError(
                "Provide either treat_name or cohort_name for SyntheticControl.",
            )
        if v_mode not in {"identity", "nested"}:
            raise ValueError("v_mode must be 'identity' or 'nested'")
        if base_period not in {"varying", "universal"}:
            raise ValueError("base_period must be 'varying' or 'universal'")
        self.v_mode = v_mode
        self.anticipation = int(anticipation)
        self.base_period = str(base_period)
        # Default treat column name when deriving from cohort
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
        # populated by from_formula()
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

    # --------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame | None = None,
        *,
        boot: BootConfig | None = None,
        _boot: object | None = None,
    ) -> EstimationResult:
        """Estimate ATT path and placebo uniform (sup-t) bands.

        Now supports multiple treated units with staggered adoption (cohorts),
        similar to SDID. Results are aggregated in event-time τ = t - g.

        If `from_formula()` was used, the working DataFrame (post-materialization
        and row selection) is used by default when `df` is None. This mirrors the
        formula pipeline used across estimators.
        """
        # For API parity only: Synthetic Control uses placebo inference; `boot` is accepted but ignored.
        # allow from_formula() defaulting
        # Accept either `boot` (public) or `_boot` (internal) for compatibility
        if _boot is None and boot is not None:
            _boot = boot
        if df is None:
            df = getattr(self, "_formula_df", None)
            if df is None:
                raise ValueError(
                    "Provide df explicitly or instantiate the estimator via from_formula().",
                )

        t_name = self.spec.t_name
        tr = self.spec.treat_name
        # If treat column is not present but cohort_name is available, derive treat = 1{t >= cohort & cohort > 0}
        if tr not in df.columns and self.spec.cohort_name is not None:
            coh = df[self.spec.cohort_name]
            df = df.copy()
            df[tr] = ((df[t_name] >= coh) & (coh > 0)).astype(int)

        # --- Extract cohort structure and donor pool ---
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
        t2pos = _time_to_pos(times_arr)
        id_to_row = {i: k for k, i in enumerate(ids)}
        t_to_col = {t: k for k, t in enumerate(times)}
        j_donors = np.array([id_to_row[j] for j in donors], dtype=int)

        # --- Process each cohort and aggregate in event-time ---
        results_g: dict[int, dict[str, object]] = {}
        tau_union: set[int] = set()

        for g, treated_units in cohorts.items():
            t0_col = t_to_col[g]
            pre = np.arange(0, t0_col)
            post = np.arange(t0_col, len(times))

            if pre.size == 0:
                continue

            att_paths_g = []
            for treated_id in treated_units:
                i_treated = id_to_row[treated_id]

                y_pre = Y[i_treated, pre].astype(np.float64)
                X_pre = Y[j_donors][:, pre].T.astype(np.float64)
                Y_pre_donors = Y[j_donors][:, pre].astype(np.float64)

                if self.v_mode == "nested":
                    _, w = _optimize_V_nested(
                        y_pre, X_pre.T,
                        y_pre, Y_pre_donors,
                        max_iter_fw=self.max_iter,
                        tol_fw=self.tol,
                    )
                else:
                    w = _frank_wolfe_simplex(
                        X_pre, y_pre, max_iter=self.max_iter, tol=self.tol,
                    )

                y_treated = Y[i_treated, :].astype(np.float64)
                X_all = Y[j_donors, :].T.astype(np.float64)
                y_synth = la.dot(X_all, w)
                att_path = y_treated - y_synth
                att_paths_g.append(att_path)
                if "weights" not in results_g.get(g, {}):
                    results_g[g] = {"weights": [], "y_synth": [], "treated_row_ids": []}
                results_g[g]["weights"].append(w)
                results_g[g]["y_synth"].append(y_synth)
                results_g[g]["treated_row_ids"].append(i_treated)

            att_path_g = np.mean(att_paths_g, axis=0)

            results_g[g]["att_path"] = att_path_g
            results_g[g]["n_treated"] = len(treated_units)
            results_g[g]["treated_ids"] = treated_units

            for t_val in times:
                tau_val = _event_tau(t_val, g, t2pos)
                tau_union.add(tau_val)

        # Aggregate across cohorts in event-time τ = t - g
        tau_union_sorted = sorted(tau_union)
        att_tau_num: dict[int, float] = {int(tau): 0.0 for tau in tau_union_sorted}
        att_tau_den: dict[int, float] = {int(tau): 0.0 for tau in tau_union_sorted}

        for g, meta in results_g.items():
            att_path = meta["att_path"]
            n_tr = meta["n_treated"]
            for t_idx, t_val in enumerate(times):
                tau_val = _event_tau(t_val, g, t2pos)
                att_tau_num[tau_val] += n_tr * float(att_path[t_idx])
                att_tau_den[tau_val] += n_tr

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

        center_at = int(self.spec.center_at)
        if center_at in att_series.index:
            with suppress(Exception):
                att_series.loc[center_at] = 0.0

        mask_post = tau_array > center_at
        post_agg = (
            float(np.nanmean(att_series.loc[mask_post].to_numpy(dtype=np.float64)))
            if np.any(mask_post)
            else float("nan")
        )

        tau_grid = np.array(tau_union_sorted, dtype=int)
        n_treated_total = sum(len(meta["treated_ids"]) for meta in results_g.values())
        B = 0
        band_level = round(100.0 * (1.0 - float(self.spec.alpha)))
        bands = None
        se_series = None
        att_tau_star = np.full((tau_grid.size, 0), np.nan, dtype=float)
        post_ci_df = pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
        post_att_se = None  # Will be set by bootstrap if successful
        boot_info = None  # Will be set by bootstrap if run

        if boot is not None and int(getattr(boot, "n_boot", 0)) > 1:
            B = int(boot.n_boot)
            alpha_level = float(self.spec.alpha)
            rng = np.random.default_rng(getattr(boot, "seed", None))

            mode = (getattr(boot, "mode", None) or "auto").lower()
            if mode == "auto":
                mode = "unit" if n_treated_total >= 2 else "placebo"
            if mode not in {"unit", "placebo", "permutation"}:
                raise ValueError("SC boot mode must be one of {'unit', 'placebo', 'permutation'}.")
            if mode == "unit" and n_treated_total < 2:
                raise ValueError("SC unit bootstrap needs >=2 treated units. Use mode='placebo' or mode='permutation'.")
            if n_treated_total >= 2 and mode in {"placebo", "permutation"}:
                raise ValueError("Policy: when treated units >=2, inference must use bootstrap (mode='unit'), not placebo/permutation.")

            theta_hat = att_series.reindex(tau_grid).to_numpy(dtype=float)

            if mode == "permutation":
                j_donors = np.array([id_to_row[j] for j in donors], dtype=int)
                g0 = list(results_g.keys())[0]
                t0_col = t_to_col[g0]
                pre = np.arange(0, t0_col)
                att_placebo_list = []
                for j_idx in j_donors:
                    donors_for_j = np.array([d for d in j_donors if d != j_idx], dtype=int)
                    if donors_for_j.size == 0:
                        continue
                    y_pre_j = Y[j_idx, pre].astype(np.float64)
                    X_pre_j = Y[donors_for_j][:, pre].T.astype(np.float64)
                    try:
                        w_j = _frank_wolfe_simplex(X_pre_j, y_pre_j, max_iter=self.max_iter, tol=self.tol)
                    except Exception:
                        continue
                    y_j = Y[j_idx, :].astype(np.float64)
                    X_all_j = Y[donors_for_j, :].T.astype(np.float64)
                    y_synth_j = la.dot(X_all_j, w_j)
                    att_path_j = y_j - y_synth_j
                    att_tau_j = np.full(tau_grid.size, np.nan, dtype=float)
                    for t_idx, t_val in enumerate(times):
                        tau_val = _event_tau(t_val, g0, t2pos)
                        j_tau = np.searchsorted(tau_grid, tau_val)
                        if j_tau < tau_grid.size and tau_grid[j_tau] == tau_val:
                            att_tau_j[j_tau] = att_path_j[t_idx]
                    att_placebo_list.append(att_tau_j)
                n_placebos = len(att_placebo_list)
                if n_placebos > 0:
                    att_placebo = np.column_stack(att_placebo_list)
                    se_vals = bt.bootstrap_se(att_placebo)
                    se_series = pd.Series(se_vals, index=tau_grid)
                    if center_at in se_series.index:
                        se_series.loc[center_at] = 0.0

                    def _sup_t_band_placebo(mask):
                        idx = np.flatnonzero(mask)
                        if idx.size == 0 or n_placebos < 2:
                            return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
                        th = theta_hat[idx]
                        thb = att_placebo[idx, :]
                        diffs = thb
                        se = np.nanstd(diffs, axis=1, ddof=1)
                        ok = np.isfinite(se) & (se > 0)
                        if not np.any(ok):
                            return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
                        tdraw = diffs[ok, :] / se[ok, None]
                        sup_abs = np.nanmax(np.abs(tdraw), axis=0)
                        sup_abs_valid = sup_abs[np.isfinite(sup_abs)]
                        if sup_abs_valid.size == 0:
                            return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
                        c = float(bt.finite_sample_quantile(sup_abs_valid, 1.0 - alpha_level))
                        lo = th.copy(); hi = th.copy()
                        lo[ok] = th[ok] - c * se[ok]
                        hi[ok] = th[ok] + c * se[ok]
                        return pd.DataFrame({"lower": pd.Series(lo, index=tau_grid[idx]), "upper": pd.Series(hi, index=tau_grid[idx])}).sort_index()

                    pre_mask = tau_array < center_at
                    band_pre = _sup_t_band_placebo(pre_mask)
                    band_post = _sup_t_band_placebo(mask_post)
                    band_full = _sup_t_band_placebo(tau_array != center_at)
                    if np.any(mask_post):
                        post_placebo = np.nanmean(att_placebo[np.flatnonzero(mask_post), :], axis=0)
                        post_placebo_valid = post_placebo[np.isfinite(post_placebo)]
                        post_att_se = float(np.nanstd(post_placebo_valid, ddof=1)) if post_placebo_valid.size > 1 else 0.0
                        if np.isfinite(post_att_se) and post_att_se > 0 and post_placebo_valid.size > 1:
                            tdraw_post = post_placebo_valid / post_att_se
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
                boot_info = {"method": "permutation", "n_placebos": n_placebos}
            else:
                # ----- Bootstrap variance estimation -----
                att_tau_star = np.full((tau_grid.size, B), np.nan, dtype=float)
                filled = 0
                attempts = 0
                max_attempts = max(10_000, 10 * B)

                W = np.zeros((len(ids), len(times)), dtype=int)
                for g, meta in results_g.items():
                    t0_col = t_to_col[g]
                    for tid in meta["treated_ids"]:
                        i_tr = id_to_row[tid]
                        W[i_tr, t0_col:] = 1
                control_mask_boot = np.all(W == 0, axis=1)

                while filled < B and attempts < max_attempts:
                    attempts += 1
                    try:
                        if mode == "unit":
                            n_units = Y.shape[0]
                            mult = rng.choice(np.array([-1.0, 1.0]), size=n_units, replace=True)
                            Y_cf = np.copy(Y)
                            for g, meta in results_g.items():
                                y_synth_list = meta.get("y_synth", [])
                                row_ids = meta.get("treated_row_ids", [])
                                t0_col = t_to_col.get(g, 0)
                                for i_tr, y_syn in zip(row_ids, y_synth_list):
                                    Y_cf[i_tr, :] = y_syn
                            U_hat = Y - Y_cf
                            Y_b = Y_cf + U_hat * mult[:, None]
                            ids_b, times_b = ids, times
                            cohorts_b = {g: meta["treated_ids"] for g, meta in results_g.items()}
                            donors_b = donors
                        else:  # mode == "placebo"
                            control_units = np.flatnonzero(control_mask_boot)
                            n_control = control_units.size
                            if n_control < n_treated_total:
                                raise ValueError("Placebo bootstrap requires at least as many control units as treated units.")
                            placebo_units = rng.choice(control_units, size=n_treated_total, replace=False)
                            cohorts_b = {list(results_g.keys())[i % len(results_g)]: [ids[pu]] for i, pu in enumerate(placebo_units)}
                            donors_b = [ids[i] for i in control_units if i not in placebo_units]
                            if len(donors_b) == 0:
                                continue
                            Y_b = Y.copy()
                            ids_b, times_b = ids, times

                        id_to_row_b = {i: k for k, i in enumerate(ids_b)}
                        t_to_col_b = {t: k for k, t in enumerate(times_b)}
                        j_donors_b = np.array([id_to_row_b[j] for j in donors_b], dtype=int)

                        results_g_b = {}
                        tau_union_b = set()
                        for g, treated_units in cohorts_b.items():
                            t0_col_b = t_to_col_b.get(g)
                            if t0_col_b is None:
                                continue
                            pre_b = np.arange(0, t0_col_b)
                            if pre_b.size == 0:
                                continue
                            att_paths_g_b = []
                            for tid in treated_units:
                                i_tr_b = id_to_row_b[tid]
                                y_pre_b = Y_b[i_tr_b, pre_b].astype(np.float64)
                                X_pre_b = Y_b[j_donors_b][:, pre_b].T.astype(np.float64)
                                w_b = _frank_wolfe_simplex(X_pre_b, y_pre_b, max_iter=self.max_iter, tol=self.tol)
                                y_treated_b = Y_b[i_tr_b, :].astype(np.float64)
                                X_all_b = Y_b[j_donors_b, :].T.astype(np.float64)
                                y_synth_b = la.dot(X_all_b, w_b)
                                att_paths_g_b.append(y_treated_b - y_synth_b)
                            att_path_g_b = np.mean(att_paths_g_b, axis=0)
                            results_g_b[g] = {"att_path": att_path_g_b, "n_treated": len(treated_units)}
                            for t_val in times_b:
                                tau_union_b.add(_event_tau(t_val, g, t2pos))

                        tau_union_sorted_b = sorted(tau_union_b)
                        att_tau_num_b = {int(tau): 0.0 for tau in tau_union_sorted_b}
                        att_tau_den_b = {int(tau): 0.0 for tau in tau_union_sorted_b}
                        for g, meta_b in results_g_b.items():
                            att_path_b = meta_b["att_path"]
                            n_tr_b = meta_b["n_treated"]
                            for t_idx, t_val in enumerate(times_b):
                                tau_val = _event_tau(t_val, g, t2pos)
                                att_tau_num_b[tau_val] += n_tr_b * float(att_path_b[t_idx])
                                att_tau_den_b[tau_val] += n_tr_b

                        vec_b = np.full(tau_grid.size, np.nan, dtype=float)
                        for j, tau in enumerate(tau_grid):
                            if att_tau_den_b.get(int(tau), 0) > 0:
                                vec_b[j] = att_tau_num_b[int(tau)] / att_tau_den_b[int(tau)]

                        if np.any(~np.isfinite(vec_b[mask_post])):
                            continue
                        att_tau_star[:, filled] = vec_b
                        filled += 1
                    except Exception:
                        continue

                if filled < B:
                    import warnings as _w
                    _w.warn(f"SC bootstrap: only {filled}/{B} draws succeeded.", RuntimeWarning, stacklevel=2)
                    att_tau_star = att_tau_star[:, :filled]

                if filled == 0:
                    raise RuntimeError("SC bootstrap failed: 0 successful draws. Check data/resampling settings.")

                if filled > 1:
                    se_vals = bt.bootstrap_se(att_tau_star)
                    se_series = pd.Series(se_vals, index=tau_grid)
                    if center_at in se_series.index:
                        se_series.loc[center_at] = 0.0

                    def _sup_t_band(mask):
                        idx = np.flatnonzero(mask)
                        if idx.size == 0 or filled < 2:
                            return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
                        th = theta_hat[idx]
                        thb = att_tau_star[idx, :]
                        if mode == "placebo":
                            diffs = thb
                        else:
                            diffs = thb - th[:, None]
                        se = np.nanstd(diffs, axis=1, ddof=1)
                        ok = np.isfinite(se) & (se > 0)
                        if not np.any(ok):
                            return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
                        tdraw = diffs[ok, :] / se[ok, None]
                        sup_abs = np.nanmax(np.abs(tdraw), axis=0)
                        sup_abs_valid = sup_abs[np.isfinite(sup_abs)]
                        if sup_abs_valid.size == 0:
                            return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
                        c = float(bt.finite_sample_quantile(sup_abs_valid, 1.0 - alpha_level))
                        lo = th.copy(); hi = th.copy()
                        lo[ok] = th[ok] - c * se[ok]
                        hi[ok] = th[ok] + c * se[ok]
                        return pd.DataFrame({"lower": pd.Series(lo, index=tau_grid[idx]), "upper": pd.Series(hi, index=tau_grid[idx])}).sort_index()

                    pre_mask = tau_array < center_at
                    band_pre = _sup_t_band(pre_mask)
                    band_post = _sup_t_band(mask_post)
                    band_full = _sup_t_band(tau_array != center_at)

                    if np.any(mask_post):
                        post_star = np.nanmean(att_tau_star[np.flatnonzero(mask_post), :], axis=0)
                        post_star_valid = post_star[np.isfinite(post_star)]
                        se_hat = float(np.nanstd(post_star_valid, ddof=1)) if post_star_valid.size > 1 else 0.0
                        if np.isfinite(se_hat) and se_hat > 0 and post_star_valid.size > 1:
                            if mode == "placebo":
                                tdraw_post = post_star_valid / se_hat
                            else:
                                tdraw_post = (post_star_valid - post_agg) / se_hat
                            c_post = float(bt.finite_sample_quantile(np.abs(tdraw_post), 1.0 - alpha_level))
                            post_ci_df = pd.DataFrame({"lower": [post_agg - c_post * se_hat], "upper": [post_agg + c_post * se_hat]})

                    bands = {
                        "pre": band_pre,
                        "post": band_post,
                        "full": band_full,
                        "post_scalar": post_ci_df,
                        "__meta__": {
                            "origin": "bootstrap",
                            "mode": mode,
                            "kind": "uniform",
                            "estimator": "synthetic",
                            "level": band_level,
                            "B": int(filled),
                            "donors": list(donors),
                            "cohorts": sorted(list(cohorts.keys())),
                        },
                    }
                boot_info = {"B": filled, "post_ATT_draws": np.full(filled, np.nan), "ATT_tau_draws": att_tau_star}

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
        # Compute PostATT_se from bootstrap draws if available
        if se_series is not None and np.any(mask_post):
            post_star_draws = np.nanmean(att_tau_star[np.flatnonzero(mask_post), :], axis=0)
            model_info["PostATT_se"] = float(np.std(post_star_draws, ddof=1))
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
