"""Synthetic Difference-in-Differences (SDID).

R `synthdid` / Stata `sdid` / Arkhangelsky et al. (2021, AER) reproducing
implementation. Provides collapsed-form construction, Frank-Wolfe solver for
weights (omega, lambda) with regularization defaults, staggered aggregation in
event time (tau = t - g), and bootstrap-only uniform sup-t confidence bands.
Analytical SEs and p-values remain intentionally unavailable (project policy).

PATCH (event-time grid & post shrinkage):
- Build the union τ-grid implied by all treated cohorts x observed calendar times.
- Preserve τ with zero treated denominator; keep the row and mark ``NaN``.
- Align placebo/bootstrap draws to the full τ-grid so summaries and plots retain
  every event time (mirrors R/stata behavior).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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


class SDID:
    """Synthetic Difference-in-Differences estimator.

    Parameters
    ----------
    id_name, t_name, treat_name, y_name: column names in long panel
    cohort_name: optional cohort column name (adoption time) if present
    control_group: 'notyet' (default) or 'never'
    eta_omega, eta_lambda: regularization defaults (None -> formula for eta_omega)
    boot: BootConfig for placebo bootstrap (default None -> no bootstrap)
    alpha: confidence level for bootstrap bands
    fw_max_iter, fw_tol: Frank-Wolfe solver controls
    omega_intercept/lambda_intercept: whether to demean rows/cols before solving

    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        id_name: str,
        t_name: str,
        treat_name: str,
        y_name: str,
        cohort_name: str = "g",
        control_group: str = "notyet",
        eta_omega: float | None = None,
        eta_lambda: float = 1e-6,
        boot: BootConfig | None = None,
        alpha: float = 0.05,
        fw_max_iter: int = 2000,
        fw_tol: float = 1e-8,
        omega_intercept: bool = True,
        lambda_intercept: bool = True,
    ) -> None:
        self.id_name = str(id_name)
        self.t_name = str(t_name)
        self.treat_name = str(treat_name)
        self.y_name = str(y_name)
        self.cohort_name = str(cohort_name)
        self.control_group = str(control_group)
        self.eta_omega = None if eta_omega is None else float(eta_omega)
        self.eta_lambda = float(eta_lambda)
        self.boot = boot
        self.alpha = float(alpha)
        self.fw_max_iter = int(fw_max_iter)
        self.fw_tol = float(fw_tol)
        self.omega_intercept = bool(omega_intercept)
        self.lambda_intercept = bool(lambda_intercept)

    def fit(self, df: pd.DataFrame | None = None) -> EstimationResult:
        """Fit SDID on a long balanced panel DataFrame.

        Returns an EstimationResult with params (event-time series indexed by τ), bands (placebo CI and uniform sup-t),
        model_info and extra fields including per-cohort weights and time series.
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

        # Identify cohorts (adoption times) and treated indexes per cohort
        cohorts = self._cohorts_from_w(W, times)

        results_g: dict[int, dict[str, object]] = {}
        used_units: set[int] = set()
        # ---- R/Stata 準拠の基準点: τ = -1（最後のプレ期）----
        base_tau = -1

        for g, idx_tr in cohorts.items():
            pre_mask = times < g
            post_mask = times >= g
            if pre_mask.sum() < 2:
                msg = "SDID requires at least two pre-treatment periods (T0>=2) for each cohort."
                raise ValueError(msg)
            if post_mask.sum() == 0:
                continue

            donors = self._donor_mask(W, times, g)
            N0 = int(donors.sum())
            if N0 == 0:
                continue
            N1 = int(idx_tr.sum())
            T0 = int(pre_mask.sum())
            T1 = int(post_mask.sum())

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
                "fw_meta": fw_meta,  # diagnostics: iterations, last gap, sparsity
                "post_ATT": att_post_g,
                "n_tr": N1,
                "T0": T0,
                "T1": T1,
                "delta_t": delta_gt,
                "y_tr": y_tr_mean,
                "y_sc": y_co_omega,
            }

        # Aggregate across cohorts in event time τ = t - g
        tau_union: set[int] = set()
        for g in results_g:
            for t_val in times:
                tau_union.add(int(t_val) - int(g))
        tau_union_sorted = sorted(tau_union)

        att_tau_num: dict[int, float] = {int(tau): 0.0 for tau in tau_union_sorted}
        att_tau_den: dict[int, float] = {int(tau): 0.0 for tau in tau_union_sorted}
        for g, meta in results_g.items():
            delta_gt = np.asarray(meta["delta_t"], dtype=float)
            ntr_g = float(meta["n_tr"])
            for t_idx, t_val in enumerate(times):
                tau_val = int(t_val) - int(g)
                att_tau_num[int(tau_val)] += ntr_g * float(delta_gt[t_idx])
                att_tau_den[int(tau_val)] += ntr_g
            # Store per-cohort event-time path for diagnostics
            meta["delta_tau"] = {
                int(times[t_idx]) - int(g): float(delta_gt[t_idx])
                for t_idx in range(times.size)
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
        valid_post = att_tau[(att_tau["den"] > 0) & (att_tau["tau"] >= 0)]
        att_post = (
            float(valid_post["att"].mean()) if not valid_post.empty else float("nan")
        )

        # Placebo bootstrap for post_ATT if requested
        bands = None
        boot_info: dict[str, object] = {"B": 0}
        info_bands_meta: dict[str, object] = {}
        if self.boot is not None and getattr(self.boot, "n_boot", 0) > 0:
            boot = self.boot or BootConfig(n_boot=bt.DEFAULT_BOOTSTRAP_ITERATIONS)
            mode_raw = getattr(boot, "mode", None)
            mode = "placebo" if not mode_raw else str(mode_raw).lower()
            if mode not in {"placebo"}:
                msg = "SDID bootstrap currently supports mode='placebo' only."
                raise ValueError(msg)

            B = int(boot.n_boot)
            rng = np.random.default_rng(getattr(boot, "seed", None))
            att_b = np.full(B, np.nan, dtype=float)
            att_tau_star: np.ndarray | None = None
            att_tau_grid: np.ndarray | None = None

            controls_all = W.sum(axis=1) == 0
            co_ids = np.where(controls_all)[0]
            if len(co_ids) <= 1:
                import warnings as _w

                _w.warn(
                    "SDID placebo bootstrap requires at least 2 control units; skipping bootstrap.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                B = 0
            else:
                cohort_items = list(cohorts.items())
                cohort_sizes = [int(idx_tr.sum()) for _, idx_tr in cohort_items]
                for b in range(B):
                    Wb = np.zeros_like(W)
                    assigned_units: list[np.ndarray] = []
                    for (g, _idx_tr), sz in zip(cohort_items, cohort_sizes):
                        if sz == 0:
                            assigned_units.append(np.empty(0, dtype=np.int64))
                            continue
                        draw = rng.choice(co_ids, size=sz, replace=(len(co_ids) < sz))
                        for u in np.asarray(draw, dtype=int):
                            Wb[u, times >= g] = 1
                        assigned_units.append(np.asarray(draw, dtype=np.int64))
                    boot_info.setdefault("assignments", []).append(assigned_units)
                    att_post_b, att_tau_b = self._att_path_from_w(
                        Y,
                        Wb,
                        times,
                        zeta_omega=None,
                        zeta_lambda=None,
                        omega_intercept=self.omega_intercept,
                        lambda_intercept=self.lambda_intercept,
                        max_iter=self.fw_max_iter,
                        tol=self.fw_tol,
                    )
                    att_b[b] = float(att_post_b)
                    taus_b = np.array(sorted(att_tau_b.keys()), dtype=int)
                    if att_tau_star is None:
                        att_tau_grid = taus_b
                        att_tau_star = np.full((taus_b.size, B), np.nan, dtype=float)
                    else:
                        new_grid = np.union1d(att_tau_grid, taus_b).astype(int)
                        if new_grid.size != att_tau_grid.size or np.any(
                            new_grid != att_tau_grid,
                        ):
                            old = att_tau_star
                            att_tau_star = np.full(
                                (new_grid.size, B), np.nan, dtype=float,
                            )
                            idx_old = {int(t): i for i, t in enumerate(att_tau_grid)}
                            for i_t, t in enumerate(new_grid):
                                j = idx_old.get(int(t))
                                if j is not None:
                                    att_tau_star[i_t, :b] = old[j, :b]
                            att_tau_grid = new_grid
                    assert att_tau_star is not None  # for type checkers
                    assert att_tau_grid is not None
                    pos = {int(t): i for i, t in enumerate(att_tau_grid)}
                    for t, v in att_tau_b.items():
                        att_tau_star[pos[int(t)], b] = float(v)

            if B > 0 and att_tau_star is not None and att_tau_grid is not None:
                # R synthdid parity: placebo matrix column-centering then unit-variance scaling
                # Center each bootstrap column to mean zero over the available τ rows, then scale to unit variance.
                # Rows with NaN are ignored in the per-column mean/std.
                with np.errstate(all="ignore"):
                    col_means = np.nanmean(att_tau_star, axis=0, keepdims=True)
                    att_tau_star = att_tau_star - col_means
                    col_std = np.nanstd(att_tau_star, axis=0, ddof=1, keepdims=True)
                    # Avoid division by zero; leave columns with zero std unscaled
                    safe_std = np.where(col_std > 0, col_std, 1.0)
                    att_tau_star = att_tau_star / safe_std
                q_lo = float(self.alpha / 2)
                q_hi = float(1 - self.alpha / 2)
                lo, hi = np.quantile(att_b, [q_lo, q_hi])
                bands = {
                    "post_ATT": pd.DataFrame(
                        {"lower": [float(lo)], "upper": [float(hi)]},
                    ),
                }
                bands.setdefault("__meta__", {})
                bands["__meta__"].update(
                    {
                        "origin": "placebo",
                        "kind": "uniform",
                        "policy": "placebo-reassignment",
                        "level": int(100 * (1.0 - self.alpha)),
                        "B": int(B),
                        "estimator": "sdid",
                    },
                )
                placebo_assignments = boot_info.get("assignments")
                if placebo_assignments is not None:
                    formatted = []
                    for draw in placebo_assignments:
                        draw_fmt = []
                        for g_key, arr in zip(sorted(cohorts.keys()), draw):
                            draw_fmt.append(
                                {
                                    "g": int(g_key),
                                    "units": [
                                        int(u) for u in np.asarray(arr, dtype=int)
                                    ],
                                },
                            )
                        formatted.append(draw_fmt)
                    bands["__meta__"]["assignments"] = formatted

                att_hat = att_tau.set_index("tau")["att"].astype(float)
                grid_hat = att_hat.index.to_numpy(dtype=int)
                tau_union_boot = np.union1d(att_tau_grid, grid_hat).astype(int)
                if tau_union_boot.size != grid_hat.size or np.any(
                    tau_union_boot != grid_hat,
                ):
                    att_hat = att_hat.reindex(tau_union_boot)
                    grid_hat = tau_union_boot
                if tau_union_boot.size != att_tau_grid.size or np.any(
                    tau_union_boot != att_tau_grid,
                ):
                    old = att_tau_star
                    new_star = np.full((tau_union_boot.size, B), np.nan, dtype=float)
                    idx_old = {int(t): i for i, t in enumerate(att_tau_grid)}
                    for i_t, t in enumerate(tau_union_boot):
                        j = idx_old.get(int(t))
                        if j is not None:
                            new_star[i_t, :] = old[j, :]
                    att_tau_star = new_star
                    att_tau_grid = tau_union_boot

                # ---- CenterAt=-1 に合わせて帯域を分割（uniform sup-t のみ） ----
                mask_full = np.isfinite(att_hat.to_numpy())
                mask_pre = mask_full & (grid_hat < base_tau)  # τ < -1
                mask_post = mask_full & (grid_hat > base_tau)  # τ >= 0

                def _sup_t_band(mask: np.ndarray) -> pd.DataFrame | None:
                    idx = np.where(mask)[0]
                    if idx.size == 0:
                        return None
                    diffs = att_tau_star[idx, :] - att_hat.to_numpy()[idx, None]
                    se = np.nanstd(diffs, axis=1, ddof=1)
                    ok = np.isfinite(se) & (se > 0)
                    if not np.any(ok):
                        return None
                    tdraw = np.full_like(diffs, np.nan, dtype=float)
                    tdraw[ok, :] = diffs[ok, :] / se[ok, None]
                    sup_abs = np.nanmax(np.abs(tdraw), axis=0)
                    c = float(np.nanquantile(sup_abs, 1 - self.alpha))
                    lo_vec = att_hat.to_numpy()[idx] - c * se
                    hi_vec = att_hat.to_numpy()[idx] + c * se
                    return pd.DataFrame(
                        {
                            "lower": pd.Series(lo_vec, index=grid_hat[idx]),
                            "upper": pd.Series(hi_vec, index=grid_hat[idx]),
                        },
                    ).sort_index()

                bands["full"] = _sup_t_band(mask_full)
                bands["pre"] = _sup_t_band(mask_pre)
                bands["post"] = _sup_t_band(mask_post)
                bands_meta = bands.setdefault("__meta__", {})
                bands_meta.update(
                    {
                        "kind": "uniform",
                        "estimator": "eventstudy",
                        "level": float(100.0 * (1.0 - self.alpha)),
                        "B": int(B),
                        # Document regularization defaults for reproducibility
                        "eta_omega_default": (
                            "(N_tr*T_post)^(1/4)"
                            if self.eta_omega is None
                            else float(self.eta_omega)
                        ),
                        "eta_lambda_default": float(self.eta_lambda),
                    },
                )

                # --- PostATT scalar: bootstrap studentized（uniform for scalar）---
                try:
                    post_vals = att_hat.loc[grid_hat[mask_post]].to_numpy()
                    if post_vals.size > 0 and np.isfinite(post_vals).any():
                        theta_hat = float(np.nanmean(post_vals))
                        se_hat = float(np.nanstd(att_b, ddof=1))
                        if np.isfinite(se_hat) and se_hat > 0:
                            with np.errstate(all="ignore"):
                                t_abs = np.abs((att_b - theta_hat) / se_hat)
                                c = float(np.nanquantile(t_abs, 1.0 - self.alpha))
                            lo_s = theta_hat - c * se_hat
                            hi_s = theta_hat + c * se_hat
                            bands["post_scalar"] = pd.DataFrame(
                                {"lower": [float(lo_s)], "upper": [float(hi_s)]},
                            )
                        else:
                            bands["post_scalar"] = pd.DataFrame(
                                {"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)}
                            )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Failed to compute post ATT scalar band (uniform): %s", exc)

                boot_info = {
                    "B": B,
                    "post_ATT_draws": att_b,
                    "ATT_tau_draws": att_tau_star,
                }
            else:
                bands = None
                boot_info = {
                    "B": 0,
                    "post_ATT_draws": np.empty(0, float),
                    "ATT_tau_draws": np.full((0, 0), np.nan, dtype=float),
                }

        # ==== Expose event-time series as params indexed by τ ====
        params_tau = att_tau.set_index("tau")["att"].astype(float)
        params = params_tau.copy()
        params.loc["post_ATT"] = float(att_post) if np.isfinite(att_post) else att_post
        # ---- R/Stata 準拠: 基準点は τ = -1 （最後のプレ期）----
        info: dict[str, object] = {
            "Estimator": "EventStudy: SDID",
            "ControlGroup": self.control_group,
            "EtaOmega": float(self.eta_omega) if self.eta_omega is not None else None,
            "EtaLambda": float(self.eta_lambda),
            "Alpha": float(self.alpha),
            "Cohorts": sorted(cohorts.keys()),
            "Times": times.tolist(),
            "Bootstrap": "in-space placebo (never-treated reassignment; NOT pairs bootstrap), uniform bands over τ",
            "NoAnalyticPValues": True,
        }
        # Unified aggregate reporting
        info["PostATT"] = float(att_post) if np.isfinite(att_post) else att_post
        # ---- 重要: CenterAt = -1（R/Stata と同じ）----
        info["CenterAt"] = int(base_tau)
        # bootstrap 実行回数を明示（summary の厳格ゲートを通す）
        if isinstance(bands, dict):
            info["B"] = int(bands.get("__meta__", {}).get("B", 0))
        else:
            info["B"] = 0
        # uniform を出すことを明示（あれば）
        if isinstance(bands, dict) and any(k in bands for k in ("pre", "post", "full")):
            info["BandType"] = "uniform"
        bands_meta = bands.get("__meta__") if isinstance(bands, dict) else None
        if isinstance(bands_meta, dict):
            kind = str(bands_meta.get("kind", "")).lower()
            if kind:
                info["BandType"] = kind
        if "BandType" not in info:
            info["BandType"] = "none"
        # No legacy percentile metadata or aggregation rule attached.

        extra = {
            "per_cohort": results_g,
            "att_tau": att_tau.set_index("tau")["att"],
            "att_tau_den": att_tau.set_index("tau")["den"],
            "boot": boot_info,
            "boot_meta": bands_meta if isinstance(bands_meta, dict) else None,
            "bands_source": ("placebo" if bands is not None else None),
            "se_source": "bootstrap",
        }
        att_tau_star_arr = boot_info.get("ATT_tau_draws")
        tau_idx = att_tau.set_index("tau").index.to_numpy(dtype=int)
        if (
            att_tau_star_arr is not None
            and isinstance(att_tau_star_arr, np.ndarray)
            and att_tau_star_arr.ndim == 2
            and att_tau_star_arr.shape[1] > 1
        ):
            se_vals = bt.bootstrap_se(att_tau_star_arr)
            se_series_tau = pd.Series(se_vals, index=tau_idx)
            if int(base_tau) in se_series_tau.index:
                se_series_tau.loc[int(base_tau)] = 0.0
            post_att_draws = boot_info.get("post_ATT_draws", [])
            post_att_se = float(np.std(post_att_draws, ddof=1)) if len(post_att_draws) > 1 else np.nan
            info["PostATT_se"] = post_att_se
            se_series = pd.concat([se_series_tau, pd.Series({"post_ATT": post_att_se})])
        else:
            se_series = None
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
        W_dict: dict | None = None,
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
        W_dict: dict | None = None,
        boot: BootConfig | None = None,
        fit_kwargs: dict | None = None,
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
            return W.sum(axis=1) == 0
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

        fw_meta = {
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
        Stop when the dual gap g(x)=<grad, x - e_j> <= min_dec, where j = argmin(grad).
        Objective trace records f(x) = eta*||x||^2 + ||A x - b||^2 (for monitoring only).
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
        for _ in range(max_iter):
            # grad = A'(Ax - b) + eta x = (A'Ax) - (A'b) + eta x
            grad = la.dot(At, Ax.reshape(-1, 1)).reshape(-1) - Atb + eta * x
            j = int(np.argmin(grad))
            # dual gap on the simplex: g(x) = <grad, x - e_j> = grad·x - grad_j
            gap = float(la.dot(grad, x) - grad[j])
            # objective (scaled consistently): eta*||x||^2 + ||A x - b||^2
            val = float(eta * la.dot(x, x) + la.dot((Ax - b), (Ax - b)))
            vals.append(val)
            if gap <= float(min_dec):
                break
            # FW step with exact line-search; use d_err = A[:, j] - Ax to avoid A@d
            d = -x.copy()
            d[j] += 1.0
            d_err = A[:, j] - Ax
            num = -float(la.dot(grad, d))
            den = float(la.dot(d_err, d_err) + eta * la.dot(d, d))
            step = np.clip(num / den, 0.0, 1.0) if den > _EPS else 0.0
            # update x and Ax incrementally
            x = x + step * d
            x[x < 0] = 0.0
            sx = float(x.sum())
            if sx <= 0.0:
                x = np.full_like(x, 1.0 / max(1, x.size))
                Ax = la.dot(A, x.reshape(-1, 1)).reshape(-1)
            else:
                x = x / sx
                Ax = Ax + step * d_err
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

        for g, idx_tr in cohorts.items():
            pre = times < g
            post = times >= g
            if pre.sum() < 2 or post.sum() == 0:
                continue
            donors = self._donor_mask(W, times, g)
            N0 = int(donors.sum())
            if N0 == 0:
                continue
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
                tau_val = int(t_val) - int(g)
                att_tau_num[tau_val] = att_tau_num.get(tau_val, 0.0) + ntr * float(
                    delta[t_idx],
                )
                att_tau_den[tau_val] = att_tau_den.get(tau_val, 0.0) + ntr

        att_tau: dict[int, float] = {}
        for tau_val, num in att_tau_num.items():
            den = float(att_tau_den.get(tau_val, 0.0))
            att_tau[int(tau_val)] = num / den if den > 0 else float("nan")

        post_vals = [v for tau, v in att_tau.items() if tau >= 0 and np.isfinite(v)]
        att_post = float(np.mean(post_vals)) if post_vals else float("nan")
        return att_post, att_tau
