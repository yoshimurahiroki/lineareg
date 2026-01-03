"""Synthetic Difference-in-Differences.

This module implements the Arkhangelsky et al. (2021) estimator with Frank-Wolfe
weights and uniform bootstrap bands.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, cast

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
from lineareg.utils.formula import FormulaParser

if TYPE_CHECKING:
    from numpy.typing import NDArray

_EPS = 1e-12

LOGGER = logging.getLogger(__name__)


from lineareg.utils.helpers import event_tau, time_to_pos






class SDID:
    """Synthetic Difference-in-Differences estimator.

    Uses unit and time weights via Frank-Wolfe regularization to construct
    counterfactuals. Inference via placebo bootstrap.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        id_name: str,
        t_name: str,
        treat_name: str,
        y_name: str,
        cohort_name: str = "g",
        control_group: str = "never",
        anticipation: int = 0,
        base_period: str = "varying",
        center_at: int = -1,
        tau_weight: str = "treated_t",
        eta_omega: float | None = None,
        eta_lambda: float = 1e-6,
        boot: BootConfig | None = None,
        alpha: float = 0.05,
        fw_max_iter: int = 2000,
        fw_tol: float = 1e-8,
        omega_intercept: bool = True,
        lambda_intercept: bool = True,
        boot_cluster: str = "twoway",
    ) -> None:
        self.id_name = str(id_name)
        self.t_name = str(t_name)
        self.treat_name = str(treat_name)
        self.y_name = str(y_name)
        self.cohort_name = str(cohort_name)
        self.control_group = str(control_group)
        self.anticipation = int(anticipation)
        if self.anticipation < 0:
            raise ValueError("anticipation must be >= 0.")
        base_period_norm = str(base_period).lower()
        if base_period_norm not in {"varying", "universal"}:
            raise ValueError("base_period must be 'varying' or 'universal'")
        self.base_period = base_period_norm
        self.center_at = int(center_at)
        tau_weight_norm = str(tau_weight).lower()
        if tau_weight_norm not in {"equal", "group", "treated_t"}:
            raise ValueError("tau_weight must be one of {'equal','group','treated_t'}.")
        self.tau_weight = tau_weight_norm
        self.eta_omega = None if eta_omega is None else float(eta_omega)
        self.eta_lambda = float(eta_lambda)
        self.boot = boot
        self.alpha = float(alpha)
        self.fw_max_iter = int(fw_max_iter)
        self.fw_tol = float(fw_tol)
        self.omega_intercept = bool(omega_intercept)
        self.lambda_intercept = bool(lambda_intercept)
        boot_cluster_norm = str(boot_cluster).lower()
        if boot_cluster_norm not in {"unit", "twoway", "time"}:
            raise ValueError("boot_cluster must be 'unit', 'twoway', or 'time'")
        self.boot_cluster = boot_cluster_norm

    def fit(self, df: pd.DataFrame | None = None, boot: BootConfig | None = None) -> EstimationResult:
        """Fit SDID model.

        Yields event-time parameters, bootstrap confidence bands, and
        estimation diagnostics.
        """
        if df is None:
            df = getattr(self, "_formula_df", None)
            if df is None:
                raise ValueError(
                    "Provide df explicitly or instantiate via from_formula().",
                )
        else:
            self._formula_df = df

        # Build matrices and validate panel
        Y, W, _units, times = self._panel_matrices(df)
        t2pos = time_to_pos(times)

        if self.base_period != "varying":
            raise ValueError(
                "SDID currently supports only base_period='varying'. "
                "base_period='universal' is not implemented for the SDID estimator.",
            )

        # Identify cohorts (adoption times) and treated indexes per cohort
        cohorts = self._cohorts_from_w(W, times)

        results_g: dict[int, dict[str, object]] = {}
        used_units: set[int] = set()
        # R/Stata parity: base period is τ = base_tau (typically -1) from center_at
        base_tau = int(self.center_at)

        T = int(times.size)
        time_pos = np.arange(T, dtype=int)

        # First-treatment positions for robust not-yet-treated logic (works for non-numeric time).
        first_pos = (W > 0).argmax(axis=1)
        ever = W.sum(axis=1) > 0
        first_pos_full = np.full(W.shape[0], fill_value=np.inf, dtype=float)
        first_pos_full[ever] = first_pos[ever].astype(float)

        for g, idx_tr in cohorts.items():
            g_pos = t2pos.get(g, None)
            if g_pos is None:
                raise ValueError("Internal error: cohort time not found in time index.")
            g_pos = int(g_pos)
            g_start_pos = g_pos - int(self.anticipation)
            if g_start_pos < 0:
                continue
            g_start = times[g_start_pos]

            pre_mask = time_pos < g_start_pos
            post_mask = time_pos >= g_start_pos

            if int(pre_mask.sum()) < 2:
                msg = "SDID requires at least two pre-treatment periods (T0>=2) for each cohort."
                raise ValueError(msg)
            if int(post_mask.sum()) == 0:
                continue

            cg = self.control_group.lower().replace("_", "")
            if cg == "notyet":
                donors = first_pos_full > float(g_start_pos)
            else:
                donors = (~ever).astype(bool)

            N0 = int(donors.sum())
            if N0 == 0:
                continue

            if cg == "notyet":
                cutoff_pos = float(np.min(first_pos_full[donors]))
                if np.isfinite(cutoff_pos):
                    post_mask = (time_pos >= g_start_pos) & (time_pos < int(cutoff_pos))
                    if int(post_mask.sum()) == 0:
                        continue

            N1 = int(idx_tr.sum())
            T0 = int(pre_mask.sum())
            T1 = int(post_mask.sum())

            valid_t_mask = pre_mask | post_mask

            # Assemble Yg with donors first then treated units
            donors_idx = np.where(donors)[0]
            treated_idx = np.where(idx_tr)[0]
            Yg = Y[np.r_[donors_idx, treated_idx], :]

            # Collapsed form identical to R synthdid::collapsed.form
            Yc = self._collapsed_form(Yg[:N0, :], Yg[N0:, :], pre_mask, post_mask)

            # Estimate noise level and regularization scaling (strict: no fallback)
            noise = np.std(np.diff(Yg[:N0, pre_mask], axis=1), ddof=1)
            if not np.isfinite(noise) or noise < _EPS:
                msg = "Insufficient pre-treatment variation to calibrate noise.level for SDID (T0>=2 required)."
                raise ValueError(msg)
            # min_decrease used by _solve_weights_fw stopping rule: 1e-5 * noise (R default)
            self._min_dec = float(noise) * 1e-5
            # R synthdid default: set eta_omega = (N_tr * T_post)^(1/4) when None
            # This matches the synthdid package default regularization scaling.
            eta_omega = (
                ((float(N1) * float(T1)) ** 0.25)
                if (self.eta_omega is None)
                else float(self.eta_omega)
            )
            zeta_omega = eta_omega * noise
            zeta_lambda = float(self.eta_lambda) * noise

            # track units actually used in estimation/evaluation
            for u in np.r_[donors_idx, treated_idx]:
                used_units.add(int(u))

            # Solve for weights via Frank-Wolfe (omega, lambda) and capture FW diagnostics
            omega, lam, fw_meta = self._solve_weights_fw(
                Yc,
                zeta_omega=zeta_omega,
                zeta_lambda=zeta_lambda,
                omega_intercept=self.omega_intercept,
                lambda_intercept=self.lambda_intercept,
                max_iter=self.fw_max_iter,
                tol=self.fw_tol,
            )

            # Construct per-time synthetic control and treatment means
            y_tr_mean = Yg[N0:, :].mean(axis=0)
            y_co_omega = np.average(Yg[:N0, :], axis=0, weights=omega)
            diff = y_tr_mean - y_co_omega
            bias = float(la.dot(lam, diff[pre_mask]))
            delta_gt = diff - bias
            att_post_g = float(delta_gt[post_mask].mean())

            results_g[g] = {
                "omega": omega,
                "lambda": lam,
                "bias": bias,
                "fw_meta": fw_meta,
                "post_ATT": att_post_g,
                "n_tr": N1,
                "T0": T0,
                "T1": T1,
                "delta_t": delta_gt,
                "y_tr": y_tr_mean,
                "y_sc": y_co_omega,
                "donors_idx": donors_idx,
                "zeta_omega": zeta_omega,
                "zeta_lambda": zeta_lambda,
                "pre_mask": pre_mask,
                "post_mask": post_mask,
                "valid_t_mask": valid_t_mask,
                "g_start": g_start,
            }

        # Aggregate across cohorts in event time τ = t - g
        tau_union: set[int] = set()
        for g, meta in results_g.items():
            valid_t_mask = np.asarray(meta.get("valid_t_mask"), dtype=bool)
            for t_val in times[valid_t_mask]:
                tau_union.add(event_tau(t_val, g, t2pos))
        tau_union.add(int(base_tau))
        tau_union_sorted = sorted(tau_union)

        att_tau_num: dict[int, float] = {int(tau): 0.0 for tau in tau_union_sorted}
        att_tau_den: dict[int, float] = {int(tau): 0.0 for tau in tau_union_sorted}
        for g, meta in results_g.items():
            delta_gt = np.asarray(meta["delta_t"], dtype=float)
            ntr_g = float(cast(int, meta["n_tr"]))
            valid_t_mask = np.asarray(meta.get("valid_t_mask"), dtype=bool)
            for t_idx, t_val in enumerate(times):
                if not bool(valid_t_mask[t_idx]):
                    continue
                tau_val = event_tau(t_val, g, t2pos)
                if self.tau_weight == "equal":
                    w_cell = 1.0
                else:
                    # For balanced SDID panels, group and treated_t coincide.
                    w_cell = ntr_g
                att_tau_num[int(tau_val)] += float(w_cell) * float(delta_gt[t_idx])
                att_tau_den[int(tau_val)] += float(w_cell)
            meta["delta_tau"] = {
                event_tau(times[t_idx], g, t2pos): float(delta_gt[t_idx])
                for t_idx in range(times.size)
                if bool(valid_t_mask[t_idx])
            }

        att_tau = pd.DataFrame({"tau": pd.Series(tau_union_sorted, dtype=int)})
        att_tau["den"] = att_tau["tau"].map(
            lambda tau: float(att_tau_den.get(int(tau), 0.0)),
        )
        att_tau["att"] = att_tau["tau"].map(
            lambda tau: (
                att_tau_num.get(int(tau), 0.0) / att_tau_den.get(int(tau), 0.0)
                if att_tau_den.get(int(tau), 0.0) > 0
                else np.nan
            ),
        )
        att_tau.loc[att_tau["den"] <= 0.0, "att"] = np.nan
        valid_post = att_tau[(att_tau["den"] > 0) & (att_tau["tau"] > base_tau)]
        post_num = sum(att_tau_num.get(int(tau), 0.0) for tau in valid_post["tau"])
        post_den = sum(att_tau_den.get(int(tau), 0.0) for tau in valid_post["tau"])
        att_post = float(post_num / post_den) if post_den > 0 else float("nan")

        # Normalize the event-time path at the base period (tau = base_tau) so ATT(base_tau)=0.
        base_rows = att_tau.index[att_tau["tau"].astype(int) == base_tau]
        if base_rows.size != 1:
            raise ValueError(
                f"Expected exactly one base_tau={base_tau} row in att_tau; got {base_rows.size}.",
            )
        base_row = int(base_rows[0])
        theta_base = float(att_tau.loc[base_row, "att"])
        if not np.isfinite(theta_base):
            raise ValueError(f"Non-finite base ATT at tau={base_tau} (att={theta_base}).")

        att_tau["att"] = att_tau["att"].astype(float) - theta_base
        att_tau.loc[base_row, "att"] = 0.0
        if not np.isfinite(att_tau["att"]).all():
            bad = (
                att_tau.loc[~np.isfinite(att_tau["att"]), "tau"]
                .astype(int)
                .tolist()
            )
            raise ValueError(
                f"Non-finite ATT path entries after centering for tau={bad}.",
            )

        if not np.isfinite(att_post):
            raise ValueError("Non-finite post_ATT (no post support or numerical failure).")
        att_post = float(att_post - theta_base)

        tau_grid = np.array(tau_union_sorted, dtype=int)
        n_treated_units_total = sum(int(idx_tr.sum()) for idx_tr in cohorts.values())

        # Treated count used for inference mode selection.
        # If cluster_ids are supplied at the unit level (length == N_units),
        # treat "treated_total" as the number of treated clusters.
        # Otherwise fall back to treated units.
        treated_mask = np.zeros(Y.shape[0], dtype=bool)
        for idx_tr in cohorts.values():
            treated_mask |= np.asarray(idx_tr, dtype=bool)
        n_treated_total = int(n_treated_units_total)
        if boot is not None and getattr(boot, "cluster_ids", None) is not None:
            cids = np.asarray(getattr(boot, "cluster_ids"))
            if cids.shape[0] == Y.shape[0]:
                n_treated_total = int(np.unique(cids[treated_mask]).size)

        # Default inference: placebo for treated_total==1, wild IF for treated_total>=2
        boot_eff = boot or self.boot
        bands = None
        boot_info: dict[str, object] = {"B": 0}
        se_series = None
        att_tau_star: NDArray[np.float64] = np.full((tau_grid.size, 0), np.nan, dtype=float)

        boot_cluster_req = self.boot_cluster
        boot_cluster_used: str | None = None
        W_multipliers_inference: pd.DataFrame | None = None
        multipliers_log: dict[str, object] | None = None
        boot_config_used: BootConfig | None = None

        if boot_eff is not None and getattr(boot_eff, "n_boot", 0) > 1:
            B = int(boot_eff.n_boot)
            alpha_level = float(self.alpha)

            mode = (getattr(boot_eff, "mode", None) or "auto").lower().strip()
            if mode == "auto":
                mode = "wild_if" if n_treated_total >= 2 else "placebo"
            # Backwards-compat alias
            if mode == "if":
                mode = "wild_if"
            if mode not in {"wild_if", "placebo"}:
                raise ValueError(
                    "SDID boot mode must be one of {'auto','wild_if','if','placebo'}.",
                )

            filled = 0
            # Strict policy: treated_total==1 -> placebo only; treated_total>=2 -> wild_if only.
            if mode == "wild_if" and n_treated_total < 2:
                raise ValueError(
                    "SDID wild_if inference requires treated_total>=2. Use placebo when treated_total==1.",
                )
            if mode == "placebo" and n_treated_total != 1:
                raise ValueError("Placebo inference is allowed only when treated_total == 1.")

            if mode == "wild_if":
                # SDID IF implementation is unit-score based. Time/two-way clustering would
                # require an observation-level linearization.
                if boot_cluster_req == "time":
                    raise ValueError(
                        "SDID wild_if currently supports unit-level multipliers only; "
                        "boot_cluster='time' is not compatible with unit-score IF. "
                        "Use boot_cluster='unit' or placebo inference when treated_total==1.",
                    )
                # Backward-compatible behavior: treat 'twoway' request as unit-level for IF.
                boot_cluster_used = "unit" if boot_cluster_req in {"unit", "twoway"} else "unit"

                # Strict rule: treated_total >= 2 only.
                if n_treated_total < 2:
                    raise ValueError(
                        "SDID wild_if inference requires at least 2 treated units/clusters. Use placebo when treated_total==1.",
                    )

                # Implement IF Wild Bootstrap
                # 1. Compute U_hat (perturbation directions)
                U_hat = Y.astype(float, copy=False)

                # 2. Compute Influence Functions
                psi_tau, psi_post = self._sdid_if_unit_scores(
                    Y=Y,
                    W=W,
                    times=times,
                    cohorts=cohorts,
                    results_g=results_g,
                    U_hat=U_hat,
                    tau_grid=tau_grid,
                    omega_intercept=self.omega_intercept,
                    lambda_intercept=self.lambda_intercept,
                    zeta_omega=0.0,
                    zeta_lambda=0.0,
                )

                # 3. Wild Bootstrap Replicates
                N_units = Y.shape[0]
                seed_val = getattr(boot_eff, "seed", None)
                bc = replace(
                    boot_eff,
                    n_boot=B,
                    dist=getattr(boot_eff, "dist", "rademacher"),
                    seed=seed_val,
                )
                W_boot_df, mlog = bc.make_multipliers(n_obs=N_units)
                W_boot = W_boot_df.to_numpy()
                W_multipliers_inference = W_boot_df
                multipliers_log = mlog
                boot_config_used = bc

                # theta_hat* = theta_hat + (1/N) sum_i W_i psi_i
                att_b = att_post + (psi_post @ W_boot) / float(N_units)

                # Event-time path (levels before centering).
                att_map = dict(
                    zip(
                        att_tau["tau"].astype(int).tolist(),
                        att_tau["att"].astype(float).tolist(),
                    )
                )
                att_tau_vec = np.full(tau_grid.size, np.nan, dtype=float)
                for j, t in enumerate(tau_grid):
                    tj = int(t)
                    if tj in att_map:
                        att_tau_vec[j] = att_map[tj]

                if not np.isfinite(att_tau_vec).all():
                    missing = [int(tau_grid[i]) for i, v in enumerate(att_tau_vec) if not np.isfinite(v)]
                    raise ValueError(
                        "Non-finite ATT path entries (missing support or bug) for tau="
                        f"{missing}.",
                    )

                att_tau_star = att_tau_vec[:, None] + (psi_tau.T @ W_boot) / float(N_units)

                if not np.isfinite(att_tau_star).all():
                    raise ValueError("Non-finite values in IF bootstrap att_tau_star; numerical failure.")
                if not np.isfinite(att_b).all():
                    raise ValueError("Non-finite values in IF bootstrap att_b; numerical failure.")

                filled = B
                boot_info = {
                    "method": "wild_if",
                    "B": B,
                    "dist": getattr(boot_eff, "dist", "rademacher"),
                    "post_ATT_draws": att_b,
                    "ATT_tau_draws": att_tau_star,
                }

            elif mode == "placebo":
                if n_treated_total != 1:
                    raise ValueError("Placebo inference is allowed only when treated_total == 1.")

                boot_cluster_used = None
                boot_config_used = boot_eff

                control_unit_mask = W.sum(axis=1) == 0
                control_units = np.flatnonzero(control_unit_mask)
                n_control = control_units.size
                if n_control < 1:
                    raise ValueError("Placebo mode with treated=1 requires at least 1 control unit.")

                # Placebo B is limited by n_control
                B = n_control
                att_tau_star = np.full((tau_grid.size, B), np.nan, dtype=float)
                att_b = np.full(B, np.nan, dtype=float)
                filled = 0
                treated_unit_indices = np.flatnonzero(~control_unit_mask)
                true_treated_idx = treated_unit_indices[0]
                true_treated_pattern = W[true_treated_idx, :]

                for b, pu in enumerate(control_units):
                    # Placebo draw: treat control unit `pu` as if it had the treated adoption pattern.
                    W_b = np.zeros_like(W)
                    W_b[pu, :] = true_treated_pattern

                    att_post_b, att_tau_b = self._att_path_from_w(
                        Y,
                        W_b,
                        times,
                        zeta_omega=None,
                        zeta_lambda=None,
                        omega_intercept=self.omega_intercept,
                        lambda_intercept=self.lambda_intercept,
                        max_iter=self.fw_max_iter,
                        tol=self.fw_tol,
                    )

                    vec_b = np.full(tau_grid.size, np.nan, dtype=float)
                    for j, tau in enumerate(tau_grid):
                        tj = int(tau)
                        if tj not in att_tau_b:
                            raise ValueError(
                                "SDID placebo draw missing tau="
                                f"{tj}. This indicates inconsistent tau support or a bug."
                            )
                        vec_b[j] = float(att_tau_b[tj])

                    if not np.isfinite(vec_b).all():
                        bad = [int(tau_grid[i]) for i, v in enumerate(vec_b) if not np.isfinite(v)]
                        raise ValueError(
                            "SDID placebo draw produced non-finite ATT for tau="
                            f"{bad}. This indicates missing support or numerical failure."
                        )
                    if not np.isfinite(att_post_b):
                        raise ValueError("SDID placebo draw produced non-finite post ATT.")

                    att_tau_star[:, filled] = vec_b
                    att_b[filled] = float(att_post_b)
                    filled += 1

                if filled == 0:
                     raise RuntimeError("SDID placebo enumeration failed: 0 successful draws.")

                if filled < B:
                    att_tau_star = att_tau_star[:, :filled]
                    att_b = att_b[:filled]

                boot_info = {
                    "method": "placebo",
                    "n_placebos": filled,
                    "B": filled,
                    "post_ATT_draws": att_b,
                    "ATT_tau_draws": att_tau_star,
                    "enumerated": True
                }


            theta_hat = att_tau.set_index("tau")["att"].reindex(tau_grid).to_numpy(dtype=float)

            # Center event-time path at the base period (tau = base_tau) so that ATT(base_tau)=0.
            base_locs = np.flatnonzero(tau_grid == base_tau)
            if base_locs.size != 1:
                raise ValueError(
                    f"Expected exactly one base_tau={base_tau} in tau_grid; got {base_locs.size}.",
                )
            base_idx = int(base_locs[0])
            if not np.isfinite(theta_hat).all():
                bad = [
                    int(tau_grid[i])
                    for i, v in enumerate(theta_hat)
                    if not np.isfinite(v)
                ]
                raise ValueError(f"Non-finite theta_hat entries for tau={bad}.")
            theta_base = float(theta_hat[base_idx])
            theta_hat = theta_hat - theta_base
            theta_hat[base_idx] = 0.0

            # Apply identical centering to bootstrap draws. This enforces the
            # event-study normalization within *each* draw, mirroring an omitted
            # base category in regression-based event-study designs.
            if att_tau_star.size > 0:
                if not np.isfinite(att_tau_star).all():
                    raise ValueError(
                        "Non-finite bootstrap draws in att_tau_star (should not happen).",
                    )
                base_draw = att_tau_star[base_idx, :].copy()
                att_tau_star = att_tau_star - base_draw[None, :]
                att_tau_star[base_idx, :] = 0.0

            # Center scalar post-ATT accordingly.
            att_post = float(att_post - theta_base)
            if "att_b" in locals() and getattr(att_b, "size", 0) > 0:
                att_b = att_b - theta_base

            if filled > 1:
                # --- τ-wise SEs ---
                se_vals = bt.bootstrap_se(att_tau_star)
                se_series_tau = pd.Series(se_vals, index=tau_grid, dtype=float)
                se_series_tau.loc[base_tau] = 0.0

                # --- Uniform sup-t bands over τ ---
                def _sup_t_band(mask: NDArray[np.bool_]) -> pd.DataFrame:
                    idx = np.flatnonzero(mask)
                    if idx.size == 0:
                        return pd.DataFrame(
                            {
                                "lower": pd.Series(dtype=float),
                                "upper": pd.Series(dtype=float),
                            },
                        )
                    if filled < 2:
                        raise ValueError("Uniform band requires at least 2 bootstrap draws.")

                    th = theta_hat[idx]
                    thb = att_tau_star[idx, :]
                    if not np.isfinite(thb).all():
                        raise ValueError("Non-finite values in att_tau_star for selected taus.")

                    if mode == "wild_if":
                        diffs = thb - th[:, None]
                    else:
                        # Placebo/permutation: center at the bootstrap mean.
                        mu = thb.mean(axis=1)
                        diffs = thb - mu[:, None]

                    se = np.std(diffs, axis=1, ddof=1)
                    if (not np.isfinite(se).all()) or np.any(se <= 0):
                        bad = [
                            int(tau_grid[idx[i]])
                            for i, v in enumerate(se)
                            if (not np.isfinite(v)) or (v <= 0)
                        ]
                        raise ValueError(
                            f"Non-finite or non-positive bootstrap SE for tau={bad}.",
                        )

                    tdraw = diffs / se[:, None]
                    sup_abs = np.max(np.abs(tdraw), axis=0)
                    if not np.isfinite(sup_abs).all():
                        raise ValueError("Non-finite sup-t bootstrap draws encountered.")

                    c = float(bt.finite_sample_quantile(sup_abs, 1.0 - alpha_level))
                    lo = th - c * se
                    hi = th + c * se
                    return (
                        pd.DataFrame(
                            {
                                "lower": pd.Series(lo, index=tau_grid[idx]),
                                "upper": pd.Series(hi, index=tau_grid[idx]),
                            },
                        )
                        .sort_index()
                    )

                mask_full = np.isfinite(theta_hat) & (tau_grid != base_tau)
                mask_pre = mask_full & (tau_grid < base_tau)
                mask_post = mask_full & (tau_grid > base_tau)

                bands = {
                    "full": _sup_t_band(mask_full),
                    "pre": _sup_t_band(mask_pre),
                    "post": _sup_t_band(mask_post),
                }

                # --- Scalar post_ATT inference (consistent with att_post definition) ---
                den_vec = (
                    att_tau.set_index("tau")["den"]
                    .reindex(tau_grid)
                    .to_numpy(dtype=float)
                )
                idx_post = np.flatnonzero((tau_grid > base_tau) & np.isfinite(theta_hat))
                if idx_post.size == 0:
                    raise ValueError(
                        "No post-treatment event times available to compute post_ATT.",
                    )
                w = den_vec[idx_post]
                if (not np.isfinite(w).all()) or np.any(w <= 0):
                    raise ValueError("Non-finite or non-positive weights for post_ATT.")
                w_sum = float(w.sum())
                if w_sum <= 0:
                    raise ValueError("No positive weight mass for post_ATT.")
                w = w / w_sum

                post_star = np.sum(att_tau_star[idx_post, :] * w[:, None], axis=0)
                if not np.isfinite(post_star).all():
                    raise ValueError("Non-finite post_ATT draws encountered.")

                post_att_se = float(np.std(post_star, ddof=1))
                if (not np.isfinite(post_att_se)) or (post_att_se <= 0):
                    raise ValueError(
                        "Non-finite or non-positive bootstrap SE for post_ATT.",
                    )

                if mode == "wild_if":
                    tdraw_post = (post_star - att_post) / post_att_se
                else:
                    mu_post = float(post_star.mean())
                    tdraw_post = (post_star - mu_post) / post_att_se
                if not np.isfinite(tdraw_post).all():
                    raise ValueError("Non-finite post_ATT t-draws encountered.")

                c_post = float(
                    bt.finite_sample_quantile(np.abs(tdraw_post), 1.0 - alpha_level),
                )
                lo_post = att_post - c_post * post_att_se
                hi_post = att_post + c_post * post_att_se

                bands["post_ATT"] = pd.DataFrame({"lower": [lo_post], "upper": [hi_post]})
                bands["post_scalar"] = bands["post_ATT"]
                bands["__meta__"] = {
                    "origin": "bootstrap",
                    "mode": mode,
                    "kind": "uniform",
                    "level": int(100 * (1.0 - alpha_level)),
                    "B": int(filled),
                    "estimator": "sdid",
                }

                # Expose centered draws consistently.
                boot_info["ATT_tau_draws"] = att_tau_star
                boot_info["post_ATT_draws"] = post_star

                se_series = pd.concat(
                    [se_series_tau, pd.Series({"post_ATT": post_att_se})],
                )
        else:
            boot_info = {"B": 0, "post_ATT_draws": np.empty(0, float), "ATT_tau_draws": np.full((0, 0), np.nan, dtype=float)}

        params_tau = att_tau.set_index("tau")["att"].astype(float)
        params = params_tau.copy()
        params.loc["post_ATT"] = float(att_post) if np.isfinite(att_post) else att_post

        if se_series is not None:
            # Align and validate. No silent dropping is allowed.
            se_series = se_series.reindex(params.index)

            if params.isna().any():
                bad = [str(k) for k, v in params.items() if pd.isna(v)]
                raise ValueError(
                    f"Non-finite parameter estimates for: {bad}. This indicates missing support or a bug.",
                )
            if se_series.isna().any():
                bad = [str(k) for k, v in se_series.items() if pd.isna(v)]
                raise ValueError(
                    f"Missing bootstrap SE for: {bad}. This indicates failed inference construction.",
                )

            se_arr = se_series.to_numpy(dtype=float)
            if not np.isfinite(se_arr).all():
                bad = [str(se_series.index[i]) for i, v in enumerate(se_arr) if not np.isfinite(v)]
                raise ValueError(f"Non-finite bootstrap SE for: {bad}.")
            if np.any(se_arr < 0):
                bad = [str(se_series.index[i]) for i, v in enumerate(se_arr) if v < 0]
                raise ValueError(f"Negative bootstrap SE for: {bad}.")
            if float(se_series.loc["post_ATT"]) <= 0:
                raise ValueError("Non-positive bootstrap SE for post_ATT.")


        info: dict[str, object] = {
            "Estimator": "EventStudy: SDID",
            "ControlGroup": self.control_group,
            "EtaOmega": float(self.eta_omega) if self.eta_omega is not None else None,
            "EtaLambda": float(self.eta_lambda),
            "Alpha": float(self.alpha),
            "Cohorts": sorted(cohorts.keys()),
            "Times": times.tolist(),
            "Bootstrap": "placebo (treated_total==1) or wild IF multiplier bootstrap (treated_total>=2) with uniform sup-t bands over τ",

        }
        if boot_cluster_used is not None:
            info["BootCluster"] = str(boot_cluster_used)
            info["BootClusterRequested"] = str(boot_cluster_req)
        info["PostATT"] = float(att_post) if np.isfinite(att_post) else att_post
        info["CenterAt"] = int(base_tau)
        if isinstance(bands, dict):
            info["B"] = int(bands.get("__meta__", {}).get("B", 0))
        else:
            info["B"] = 0
        if isinstance(bands, dict) and any(k in bands for k in ("pre", "post", "full")):
            info["BandType"] = "uniform"
        bands_meta = bands.get("__meta__") if isinstance(bands, dict) else None
        if isinstance(bands_meta, dict):
            kind = str(bands_meta.get("kind", "")).lower()
            if kind:
                info["BandType"] = kind
        if "BandType" not in info:
            info["BandType"] = "none"

        extra = {
            "per_cohort": results_g,
            "att_tau": att_tau.set_index("tau")["att"],
            "att_tau_den": att_tau.set_index("tau")["den"],
            "boot": boot_info,
            "boot_meta": bands_meta if isinstance(bands_meta, dict) else None,
            "bands_source": ("bootstrap" if bands is not None else None),
            "se_source": "bootstrap" if se_series is not None else None,
            "boot_config": boot_config_used,
            "W_multipliers_inference": W_multipliers_inference,
            "multipliers_log": multipliers_log,
            "boot_cluster_requested": boot_cluster_req,
            "boot_cluster_used": boot_cluster_used,
        }

        if se_series is not None:
            post_att_draws_obj = boot_info.get("post_ATT_draws", np.empty(0, float))
            post_att_draws = np.asarray(post_att_draws_obj, dtype=float).reshape(-1)
            if post_att_draws.size > 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    info["PostATT_se"] = float(np.std(post_att_draws, ddof=1))
        elif boot_info.get("method") == "jackknife":
            jack_se_obj = boot_info.get("post_ATT_se")
            if jack_se_obj is not None and np.isfinite(float(jack_se_obj)):
                info["PostATT_se"] = float(jack_se_obj)

        return EstimationResult(
            params=params,
            se=se_series,
            bands=bands,
            n_obs=int(Y.size),
            model_info=info,
            extra=extra,
        )

    @classmethod
    def from_formula(  # noqa: PLR0913
        cls,
        *,
        data: pd.DataFrame,
        formula: str | None = None,
        id_col: str | None = None,
        time_col: str | None = None,
        options: str | None = None,
        W_dict: dict[str, object] | None = None,
        boot: BootConfig | None = None,
        **kwargs,
    ) -> SDID:
        """Construct an SDID estimator from a formula without fitting."""
        parsed = None
        if formula is not None:
            parser = FormulaParser(
                data,
                id_name=id_col,
                t_name=time_col,
                W_dict=W_dict,
            )
            parsed = parser.parse(formula, iv=None, options=options)

        df_use, boot_eff, meta = prepare_formula_environment(
            formula=formula,
            data=data,
            parsed=parsed,
            boot=boot,
        )
        meta.attrs["_formula_df"] = df_use

        kw = dict(kwargs) if kwargs else {}

        y_name = kw.pop("y_name", None)
        if y_name is None and formula is not None and "~" in formula:
            y_name = formula.split("~", 1)[0].strip()
        y_name = y_name or (parsed.get("y_name") if parsed is not None else None) or "y"

        id_name = (
            kw.pop("id_name", None)
            or id_col
            or (parsed.get("id_name") if parsed is not None else None)
        )
        t_name = (
            kw.pop("t_name", None)
            or time_col
            or (parsed.get("t_name") if parsed is not None else None)
        )
        if id_name is None or t_name is None:
            raise ValueError(
                "id/time column names must be provided via id/t and/or id_name/t_name.",
            )

        treat_name = kw.pop("treat_name", None)
        if treat_name is None:
            treat_name = "D" if "D" in df_use.columns else None
        if treat_name is None:
            raise ValueError("treat_name must be provided (e.g., 'D' in demo data).")

        ctor_keys = {
            "cohort_name",
            "control_group",
            "eta_omega",
            "eta_lambda",
            "alpha",
            "fw_max_iter",
            "fw_tol",
            "omega_intercept",
            "lambda_intercept",
        }
        ctor_opts = {k: kw.pop(k) for k in list(kw.keys()) if k in ctor_keys}

        boot_to_use = boot_eff if boot_eff is not None else boot

        inst = cls(
            id_name=id_name,
            t_name=t_name,
            treat_name=treat_name,
            y_name=y_name,
            boot=boot_to_use,
            **ctor_opts,
        )
        attach_formula_metadata(inst, meta)

        if kw:
            unknown = ", ".join(sorted(kw.keys()))
            raise TypeError(
                f"Unexpected keyword arguments for SDID.from_formula: {unknown}",
            )

        return inst

    @classmethod
    def fit_from_formula(  # noqa: PLR0913
        cls,
        *,
        data: pd.DataFrame,
        formula: str | None = None,
        id_col: str | None = None,
        time_col: str | None = None,
        options: str | None = None,
        W_dict: dict[str, object] | None = None,
        boot: BootConfig | None = None,
        fit_kwargs: dict[str, object] | None = None,
        **kwargs,
    ) -> EstimationResult:
        est = cls.from_formula(
            data=data,
            formula=formula,
            id_col=id_col,
            time_col=time_col,
            options=options,
            W_dict=W_dict,
            boot=boot,
            **kwargs,
        )
        extra = fit_kwargs or {}
        res = est.fit(**extra)
        if getattr(est, "_cluster_ids_from_formula", None) is not None:
            res.model_info["ClusterIDsProvided"] = True
            res.model_info["ClusterIDsUsedInPlacebo"] = False
        return res

    # -------------------- internals --------------------
    def _panel_matrices(
        self, df: pd.DataFrame,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray, NDArray]:
        """Construct balanced Y (N x T) and W (N x T) matrices.

        Raises on missing values, duplicate (id,time), or unbalanced panel.
        """
        keep = [self.id_name, self.t_name, self.y_name, self.treat_name]
        X = df[keep].copy()
        if X.isna().any().any():
            msg = "Missing values are not allowed in SDID."
            raise ValueError(msg)

        X = X.sort_values([self.id_name, self.t_name])

        # Duplicate (id,time) detection
        n_by = X.groupby([self.id_name, self.t_name]).size()
        if (n_by > 1).any():
            dup = n_by[n_by > 1].index[0]
            msg = f"Duplicate (id,time) found: {dup}"
            raise ValueError(msg)

        # Balanced panel check
        n_panel = X.groupby(self.id_name)[self.t_name].nunique().nunique()
        if n_panel != 1:
            msg = "Panel must be balanced (each id observed at all times)."
            raise ValueError(msg)

        units = X[self.id_name].unique()
        times = np.sort(X[self.t_name].unique())

        Y = (
            X.pivot_table(
                index=self.id_name,
                columns=self.t_name,
                values=self.y_name,
                aggfunc="mean",
            )
            .reindex(index=units, columns=times)
            .to_numpy(float)
        )
        W = (
            X.pivot_table(
                index=self.id_name,
                columns=self.t_name,
                values=self.treat_name,
                aggfunc="mean",
            )
            .reindex(index=units, columns=times)
            .fillna(0)
            .to_numpy(int)
        )

        # Monotone treatment check: once treated -> always treated
        if not np.all(np.diff(W, axis=1) >= -_EPS):
            msg = "Treatment must be monotone (once treated, always treated)."
            raise ValueError(msg)

        return Y, W, units, times

    def _cohorts_from_w(
        self, W: NDArray[np.int64], times: NDArray[np.int64],
    ) -> dict[int, NDArray[np.bool_]]:
        """Return a dict mapping cohort time g to boolean index of treated units."""
        first_treat = (W > 0).argmax(axis=1)
        ever = W.sum(axis=1) > 0
        g_vals = np.full(W.shape[0], fill_value=np.inf)
        g_vals[ever] = times[first_treat[ever]]
        cohorts: dict[int, NDArray[np.bool_]] = {}
        for g in np.unique(g_vals[ever]):
            cohorts[int(g)] = g_vals == g
        if len(cohorts) == 0:
            msg = "No treated units."
            raise ValueError(msg)
        return cohorts

    def _donor_mask(
        self, W: NDArray[np.int64], times: NDArray[np.int64], g: int,
    ) -> NDArray[np.bool_]:
        cg = self.control_group.lower().replace("_", "")
        if cg == "notyet":
            # not-yet at time g: first treatment > g (never also allowed)
            first = (W > 0).argmax(axis=1)
            ever = W.sum(axis=1) > 0
            g_first = np.full(W.shape[0], fill_value=np.inf)
            g_first[ever] = times[first[ever]]
            return g_first > g
        if cg == "never":
            return (W.sum(axis=1) == 0).astype(bool)
        msg = "control_group must be 'notyet' or 'never'."
        raise ValueError(msg)

    def _collapsed_form(
        self,
        Yco: NDArray[np.float64],
        Ytr: NDArray[np.float64],
        pre: NDArray[np.bool_],
        post: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Construct collapsed form matrix (N0+1 x T0+1) matching R synthdid."""
        top = np.c_[Yco[:, pre], Yco[:, post].mean(axis=1, keepdims=True)]
        bot = np.c_[Ytr[:, pre].mean(axis=0, keepdims=True), Ytr[:, post].mean()]
        return np.vstack([top, bot])

    def _fw_step(
        self,
        A: NDArray[np.float64],
        x: NDArray[np.float64],
        b: NDArray[np.float64],
        eta: float,
    ) -> NDArray[np.float64]:
        """Single Frank-Wolfe update on the probability simplex with exact line-search.

        Minimizes 0.5 * ||A x - b||^2 + 0.5 * eta * ||x||^2 over x in the simplex.
        This implementation mirrors R's fw.step: compute gradient, pick LMO vertex,
        compute exact stepsize (closed form) and re-normalize to the simplex.
        """
        Ax = la.dot(A, x.reshape(-1, 1)).reshape(-1)
        # gradient = A^T(Ax - b) + eta * x  (consistent with R's half.grad convention)
        grad = la.dot(A.T, (Ax - b).reshape(-1, 1)).reshape(-1) + eta * x

        # LMO: choose vertex with minimal gradient component
        j = int(np.argmin(grad))
        d = -x.copy()
        d[j] += 1.0
        d_err = A[:, j] - Ax

        num = -float(la.dot(grad, d))
        den = float(la.dot(d_err, d_err) + eta * la.dot(d, d))
        step = np.clip(num / den, 0.0, 1.0) if den > _EPS else 0.0

        x_new = x + step * d
        # numerical cleanup: zero negative entries then renormalize to the simplex
        x_new[x_new < 0] = 0.0
        s = float(x_new.sum())
        if s <= 0.0:
            x_new[:] = 1.0 / float(len(x_new))
        else:
            x_new /= s
        return x_new

    def _solve_weights_fw(  # noqa: PLR0913
        self,
        Yc: NDArray[np.float64],
        *,
        zeta_omega: float,
        zeta_lambda: float,
        omega_intercept: bool,
        lambda_intercept: bool,
        max_iter: int,
        tol: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
        # Dimensions: Yc is (N0+1) x (T0+1)
        N0 = Yc.shape[0] - 1
        T0 = Yc.shape[1] - 1

        Ylam = Yc[:N0, :].copy()
        if lambda_intercept:
            Ylam = Ylam - Ylam.mean(axis=1, keepdims=True)
        A_lam = Ylam[:, :T0]
        b_lam = Ylam[:, T0]
        lam = np.full(T0, 1.0 / max(T0, 1))

        Yomg = Yc[:, :T0].T.copy()  # T0 x (N0+1)
        if omega_intercept:
            Yomg = Yomg - Yomg.mean(axis=0, keepdims=True)
        A_omg = Yomg[:, :N0]
        b_omg = Yomg[:, N0]
        omg = np.full(N0, 1.0 / max(N0, 1))

        # Implement R's two-stage procedure: first solve for lambda (column weights),
        # then sparsify and re-optimize; then solve for omega (row weights) likewise.
        eta_l = N0 * (zeta_lambda**2)
        eta_o = T0 * (zeta_omega**2)

        # min_dec must be set by caller (fit) as 1e-5 * noise; else fallback to tol
        min_dec = getattr(self, "_min_dec", tol)

        # --- lambda (T0-length) estimation ---
        lam0 = lam.copy()
        lam_sol, lam_vals = self._sc_fw(A_lam, lam0, b_lam, eta_l, min_dec, max_iter)
        # sparsify per R default and re-optimize
        lam_sp = self._sparsify(lam_sol)
        lam_sol, lam_vals2 = self._sc_fw(A_lam, lam_sp, b_lam, eta_l, min_dec, max_iter)

        # --- omega (N0-length) estimation on transposed problem ---
        omg0 = omg.copy()
        omg_sol, omg_vals = self._sc_fw(A_omg, omg0, b_omg, eta_o, min_dec, max_iter)
        omg_sp = self._sparsify(omg_sol)
        omg_sol, omg_vals2 = self._sc_fw(A_omg, omg_sp, b_omg, eta_o, min_dec, max_iter)

        # Build FW diagnostics metadata for auditing (iters, last dual gap, sparsity)
        lam_iters = int(lam_vals.size + lam_vals2.size)
        lam_gap_last_val = (
            lam_vals2[-1]
            if lam_vals2.size > 0
            else (lam_vals[-1] if lam_vals.size > 0 else np.nan)
        )
        lam_gap_last = float(lam_gap_last_val)
        lam_sparsity = (
            float(np.mean(lam_sol <= (lam_sol.max() / 4.0)))
            if lam_sol.size > 0
            else 0.0
        )
        omg_iters = int(omg_vals.size + omg_vals2.size)
        omg_gap_last_val = (
            omg_vals2[-1]
            if omg_vals2.size > 0
            else (omg_vals[-1] if omg_vals.size > 0 else np.nan)
        )
        omg_gap_last = float(omg_gap_last_val)
        omg_sparsity = (
            float(np.mean(omg_sol <= (omg_sol.max() / 4.0)))
            if omg_sol.size > 0
            else 0.0
        )

        fw_meta: dict[str, object] = {
            "lam": {
                "iters": lam_iters,
                "gap_last": lam_gap_last,
                "sparsity": lam_sparsity,
            },
            "omg": {
                "iters": omg_iters,
                "gap_last": omg_gap_last,
                "sparsity": omg_sparsity,
            },
        }

        return omg_sol, lam_sol, fw_meta

    def _sc_fw(  # noqa: PLR0913
        self,
        A: NDArray[np.float64],
        x0: NDArray[np.float64],
        b: NDArray[np.float64],
        eta: float,
        min_dec: float,
        max_iter: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Frank-Wolfe on the simplex with exact line-search.

        Stopping condition (R parity): stop when objective decrease is ≤ min_dec^2.
        This matches R synthdid's sc.weight.fw which checks:
            vals[t-1] - vals[t] > min.decrease^2
        Objective trace records f(x) = eta*||x||^2 + ||A x - b||^2.
        """
        # Incremental implementation: maintain Ax and avoid repeated A@x multiplies.
        # Also precompute A' and A'b so grad = A'Ax - A'b + eta x.
        x = x0.copy()
        # Ensure on simplex (guard against tiny numeric drift in callers)
        sx = float(np.sum(x))
        x = np.full_like(x, 1.0 / max(1, x.size)) if sx <= 0.0 else (x / sx)
        At = A.T
        Atb = la.dot(At, b.reshape(-1, 1)).reshape(-1)
        Ax = la.dot(A, x.reshape(-1, 1)).reshape(-1)
        vals: list[float] = []
        # R parity: threshold is min_dec^2
        min_dec_sq = float(min_dec) ** 2
        prev_val = float("inf")
        for t in range(max_iter):
            # grad = A'(Ax - b) + eta x = (A'Ax) - (A'b) + eta x
            grad = la.dot(At, Ax.reshape(-1, 1)).reshape(-1) - Atb + eta * x
            j = int(np.argmin(grad))
            # objective (scaled consistently): eta*||x||^2 + ||A x - b||^2
            val = float(eta * la.dot(x, x) + la.dot((Ax - b), (Ax - b)))
            vals.append(val)
            # R parity: stop when objective decrease is too small (t >= 2 in R indexing)
            if t >= 1 and (prev_val - val) <= min_dec_sq:
                break
            prev_val = val
            # FW step with exact line-search; use d_err = A[:, j] - Ax to avoid A@d
            d = -x.copy()
            d[j] += 1.0
            d_err = A[:, j] - Ax
            num = -float(la.dot(grad, d))
            den = float(la.dot(d_err, d_err) + eta * la.dot(d, d))
            step = np.clip(num / den, 0.0, 1.0) if den > _EPS else 0.0
            # update x and Ax incrementally
            x = x + step * d
            Ax = Ax + step * d_err
            # Numerical stability: project back to simplex
            x[x < 0] = 0.0
            sx = float(x.sum())
            if sx <= 0.0 or not np.isfinite(sx):
                x = np.full_like(x, 1.0 / max(1, x.size))
                Ax = la.dot(A, x.reshape(-1, 1)).reshape(-1)
            elif abs(sx - 1.0) > 1e-12:
                x = x / sx
                Ax = Ax / sx
        return x, np.array(vals, float)

    def _sparsify(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """R default sparsify: set entries <= max(v)/4 to zero and renormalize to sum 1."""
        if v.size == 0:
            return v
        thr = float(v.max()) / 4.0
        w = v.copy()
        w[w <= thr] = 0.0
        s = float(w.sum())
        return w if s <= 0.0 else (w / s)

    def _sdid_if_unit_scores(
        self,
        Y: NDArray[np.float64],
        W: NDArray[np.int64],
        times: NDArray[np.int64],
        cohorts: dict[int, NDArray[np.bool_]],
        results_g: dict[int, dict[str, object]],
        U_hat: NDArray[np.float64],
        tau_grid: NDArray[np.int64],
        *,
        omega_intercept: bool,
        lambda_intercept: bool,
        zeta_omega: float,
        zeta_lambda: float,
        tol: float = 1e-12,
        jitter: float = 1e-12,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        N, T = Y.shape
        t2pos = {t: idx for idx, t in enumerate(times)}
        tau2k = {int(tau): k for k, tau in enumerate(tau_grid)}

        den_tau = {int(tau): 0.0 for tau in tau_grid}
        for g, meta in results_g.items():
            idx_tr = cohorts[g]
            ntr = float(idx_tr.sum())
            if ntr <= 0:
                continue
            valid_t_mask = np.asarray(meta.get("valid_t_mask"), dtype=bool)
            w_g = 1.0 if self.tau_weight == "equal" else ntr
            for t_idx in range(T):
                if not bool(valid_t_mask[t_idx]):
                    continue
                tau = event_tau(times[t_idx], g, t2pos)
                if int(tau) in den_tau:
                    den_tau[int(tau)] += float(w_g)

        # Post-period definition uses tau > base_tau (self.center_at, typically -1)
        base_tau = self.center_at
        post_den = sum(v for tau, v in den_tau.items() if tau > base_tau)
        if post_den <= 0:
            raise ValueError("No post-treatment periods to form post ATT.")

        psi_tau = np.zeros((N, len(tau_grid)))
        psi_post = np.zeros(N)

        def _kkt_dx_active(
            As: NDArray[np.float64],
            b: NDArray[np.float64],
            xs: NDArray[np.float64],
            eta: float,
            dAs: NDArray[np.float64],
            db: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            k = xs.size
            if k <= 1:
                return np.zeros_like(xs)
            K = la.crossprod(As, As) + float(eta) * la.eye(k)
            dg = la.dot(dAs.T, b.reshape(-1, 1)).reshape(-1) + la.dot(As.T, db.reshape(-1, 1)).reshape(-1)
            Ax = la.dot(As, xs.reshape(-1, 1)).reshape(-1)
            dAx = la.dot(dAs, xs.reshape(-1, 1)).reshape(-1)
            dKxs = la.dot(dAs.T, Ax.reshape(-1, 1)).reshape(-1) + la.dot(As.T, dAx.reshape(-1, 1)).reshape(-1)
            r = dg - dKxs

            ones = np.ones((k, 1))
            M = np.block([[K, ones], [ones.T, np.zeros((1, 1))]])
            rhs = np.concatenate([r, np.zeros(1)])
            U, s, Vt = la.svd(M, full_matrices=False)
            s_inv = np.where(s > 1e-14, 1.0 / s, 0.0)
            sol = la.dot(la.hadamard(Vt.T, s_inv.reshape(1, -1)), la.dot(U.T, rhs.reshape(-1, 1)))
            dxs = sol[:k, 0]
            return dxs

        for g, meta in results_g.items():
            idx_tr = cohorts[g]
            ntr = int(idx_tr.sum())
            if ntr <= 0:
                continue

            donors_idx = np.asarray(meta["donors_idx"], dtype=int)
            N0 = donors_idx.size
            if N0 == 0:
                continue

            omega = np.asarray(meta["omega"], dtype=float).reshape(-1)
            lam = np.asarray(meta["lambda"], dtype=float).reshape(-1)
            zeta_omega = float(meta.get("zeta_omega", 0.0))
            zeta_lambda = float(meta.get("zeta_lambda", 0.0))

            pre = np.asarray(meta.get("pre_mask"), dtype=bool)
            post = np.asarray(meta.get("post_mask"), dtype=bool)
            valid_t_mask = np.asarray(meta.get("valid_t_mask"), dtype=bool)
            T0 = int(pre.sum())
            if T0 <= 0:
                continue

            y_tr = np.asarray(meta["y_tr"], dtype=float).reshape(-1)
            y_sc = np.asarray(meta["y_sc"], dtype=float).reshape(-1)
            diff = y_tr - y_sc
            diff_pre = diff[pre]

            Y0_pre = Y[donors_idx][:, pre]
            y1_pre = y_tr[pre].copy()
            if omega_intercept:
                Y0_pre = Y0_pre - Y0_pre.mean(axis=1, keepdims=True)
                y1_pre = y1_pre - float(y1_pre.mean())
            A_omg = Y0_pre.T
            b_omg = y1_pre

            supp_omg = np.flatnonzero(omega > tol)
            As_omg = A_omg[:, supp_omg]
            xs_omg = omega[supp_omg]

            Y0_post_mean = Y[donors_idx][:, post].mean(axis=1)
            A_lam = Y[donors_idx][:, pre].copy()
            b_lam = Y0_post_mean.copy()
            if lambda_intercept:
                m = (A_lam.sum(axis=1) + b_lam) / float(T0 + 1)
                A_lam = A_lam - m[:, None]
                b_lam = b_lam - m
            supp_lam = np.flatnonzero(lam > tol)
            As_lam = A_lam[:, supp_lam]
            xs_lam = lam[supp_lam]

            Y_donors = Y[donors_idx]

            w_g = 1.0 if self.tau_weight == "equal" else float(ntr)

            involved_units = np.unique(np.concatenate([np.flatnonzero(idx_tr), donors_idx]))
            for i in involved_units:
                u = U_hat[i, :]

                d_y_tr = np.zeros(T)
                if idx_tr[i]:
                    d_y_tr = u / float(ntr)

                d_omega = np.zeros(N0)
                if supp_omg.size >= 2:
                    dA_s = np.zeros_like(As_omg)
                    db = np.zeros_like(b_omg)

                    if idx_tr[i]:
                        u_pre = u[pre] / float(ntr)
                        if omega_intercept:
                            u_pre = u_pre - float(u_pre.mean())
                        db = u_pre
                    else:
                        if i in donors_idx:
                            j_loc = int(np.where(donors_idx == i)[0][0])
                            u_pre = u[pre]
                            if omega_intercept:
                                u_pre = u_pre - float(u_pre.mean())
                            if j_loc in supp_omg:
                                col = int(np.where(supp_omg == j_loc)[0][0])
                                dA_s[:, col] = u_pre

                    if (dA_s != 0).any() or (db != 0).any():
                        dxs = _kkt_dx_active(As_omg, b_omg, xs_omg, zeta_omega, dA_s, db)
                        d_omega[supp_omg] = dxs.ravel()

                d_lam = np.zeros(T0)
                if supp_lam.size >= 2 and (not idx_tr[i]) and (i in donors_idx):
                    dA_s = np.zeros_like(As_lam)
                    db = np.zeros_like(b_lam)

                    j_loc = int(np.where(donors_idx == i)[0][0])
                    u_pre = u[pre]
                    u_post_mean = float(u[post].mean())

                    if lambda_intercept:
                        dm = (float(u_pre.sum()) + u_post_mean) / float(T0 + 1)
                        u_pre_row = u_pre - dm
                        u_post_row = u_post_mean - dm
                    else:
                        u_pre_row = u_pre
                        u_post_row = u_post_mean

                    for kk, s in enumerate(supp_lam):
                        dA_s[j_loc, kk] = u_pre_row[int(s)]
                    db[j_loc] = u_post_row

                    dxs = _kkt_dx_active(As_lam, b_lam, xs_lam, zeta_lambda, dA_s, db)
                    d_lam[supp_lam] = dxs.ravel()

                d_y_sc = np.zeros(T)
                if i in donors_idx:
                    j_loc = int(np.where(donors_idx == i)[0][0])
                    d_y_sc += omega[j_loc] * u
                if (d_omega != 0).any():
                    d_y_sc += la.dot(d_omega.reshape(1, -1), Y_donors).reshape(-1)

                d_diff = d_y_tr - d_y_sc
                d_diff_pre = d_diff[pre]

                dbias = float(la.dot(d_lam.reshape(1, -1), diff_pre.reshape(-1, 1)).item()) + float(la.dot(lam.reshape(1, -1), d_diff_pre.reshape(-1, 1)).item())
                d_delta = d_diff - dbias

                for t_idx in range(T):
                    if not bool(valid_t_mask[t_idx]):
                        continue
                    tau = int(event_tau(times[t_idx], g, t2pos))
                    if tau in tau2k and den_tau[tau] > 0:
                        k = tau2k[tau]
                        psi_tau[i, k] += float(w_g) * float(d_delta[t_idx]) / float(den_tau[tau])
                    if tau > base_tau:
                        psi_post[i] += float(w_g) * float(d_delta[t_idx]) / float(post_den)

        return psi_tau, psi_post

    def _att_from_w(
        self, Y: NDArray[np.float64], W: NDArray[np.int64], times: NDArray[np.int64],
    ) -> float:
        # Backwards-compatible scalar ATT using the implemented path helper.
        att_post, _ = self._att_path_from_w(Y, W, times)
        return float(att_post)

    def _att_path_from_w(  # noqa: PLR0913
        self,
        Y: NDArray[np.float64],
        W: NDArray[np.int64],
        times: NDArray,
        *,
        zeta_omega: float | None = None,
        zeta_lambda: float | None = None,
        omega_intercept: bool = True,
        lambda_intercept: bool = True,
        max_iter: int = 2000,
        tol: float = 1e-8,
    ) -> tuple[float, dict[int, float]]:
        """Compute event-time ATT path and scalar post-treatment ATT for a given adoption matrix."""
        t2pos = time_to_pos(times)
        first_treat = (W > 0).argmax(axis=1)
        ever = W.sum(axis=1) > 0
        N = Y.shape[0]
        g_vals = np.full(N, fill_value=np.nan, dtype=float)
        g_vals[ever] = times[first_treat[ever]]
        cohorts: dict[int, NDArray[np.bool_]] = {}
        for g in pd.unique(g_vals[ever]):
            cohorts[int(g)] = g_vals == g
        if not cohorts:
            return float("nan"), {}

        att_tau_num: dict[int, float] = {}
        att_tau_den: dict[int, float] = {}

        T = int(times.size)
        time_pos = np.arange(T, dtype=int)
        first_pos = (W > 0).argmax(axis=1)
        ever = W.sum(axis=1) > 0
        first_pos_full = np.full(W.shape[0], fill_value=np.inf, dtype=float)
        first_pos_full[ever] = first_pos[ever].astype(float)

        for g, idx_tr in cohorts.items():
            g_pos = t2pos.get(g, None)
            if g_pos is None:
                continue
            g_pos = int(g_pos)
            g_start_pos = g_pos - int(self.anticipation)
            if g_start_pos < 0:
                continue
            g_start = times[g_start_pos]

            pre = time_pos < g_start_pos
            post = time_pos >= g_start_pos
            if int(pre.sum()) < 2 or int(post.sum()) == 0:
                continue

            cg = self.control_group.lower().replace("_", "")
            if cg == "notyet":
                donors = first_pos_full > float(g_start_pos)
            else:
                donors = (~ever).astype(bool)
            N0 = int(donors.sum())
            if N0 == 0:
                continue

            cg = self.control_group.lower().replace("_", "")
            if cg == "notyet":
                cutoff_pos = float(np.min(first_pos_full[donors]))
                if np.isfinite(cutoff_pos):
                    post = (time_pos >= g_start_pos) & (time_pos < int(cutoff_pos))
                    if int(post.sum()) == 0:
                        continue

            valid_t_mask = pre | post

            donors_idx = np.where(donors)[0]
            treated_idx = np.where(idx_tr)[0]
            Yg = Y[np.r_[donors_idx, treated_idx], :]
            Yc = self._collapsed_form(Yg[:N0, :], Yg[N0:, :], pre, post)

            noise = np.std(np.diff(Yg[:N0, pre], axis=1), ddof=1)
            if not np.isfinite(noise) or noise <= _EPS:
                continue
            self._min_dec = float(noise) * 1e-5
            eta_omega = (
                ((float(idx_tr.sum()) * float(post.sum())) ** 0.25)
                if (zeta_omega is None)
                else (float(zeta_omega) / float(noise))
            )
            zeta_omega_eff = float(eta_omega) * float(noise)
            zeta_lambda_eff = (
                (float(self.eta_lambda) * float(noise))
                if (zeta_lambda is None)
                else float(zeta_lambda)
            )
            omg, lam, _ = self._solve_weights_fw(
                Yc,
                zeta_omega=zeta_omega_eff,
                zeta_lambda=zeta_lambda_eff,
                omega_intercept=omega_intercept,
                lambda_intercept=lambda_intercept,
                max_iter=int(max_iter),
                tol=float(tol),
            )
            y_tr = Yg[N0:, :].mean(axis=0)
            y_sc = np.average(Yg[:N0, :], axis=0, weights=omg)
            diff = y_tr - y_sc
            bias = float(la.dot(lam, diff[pre]))
            delta = diff - bias
            ntr = float(idx_tr.sum())
            for t_idx, t_val in enumerate(times):
                if not bool(valid_t_mask[t_idx]):
                    continue
                tau_val = event_tau(t_val, g, t2pos)
                w_cell = 1.0 if self.tau_weight == "equal" else ntr
                att_tau_num[tau_val] = att_tau_num.get(tau_val, 0.0) + float(w_cell) * float(
                    delta[t_idx],
                )
                att_tau_den[tau_val] = att_tau_den.get(tau_val, 0.0) + float(w_cell)

        att_tau: dict[int, float] = {}
        for tau_val, num in att_tau_num.items():
            den = float(att_tau_den.get(tau_val, 0.0))
            att_tau[int(tau_val)] = num / den if den > 0 else float("nan")

        # Post-period definition uses tau > base_tau (self.center_at, typically -1)
        base_tau = self.center_at
        post_num = sum(att_tau_num.get(tau, 0.0) for tau in att_tau_num if tau > base_tau)
        post_den = sum(att_tau_den.get(tau, 0.0) for tau in att_tau_den if tau > base_tau)
        att_post = float(post_num / post_den) if post_den > 0 else float("nan")
        return att_post, att_tau
