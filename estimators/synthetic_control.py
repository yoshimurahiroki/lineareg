# lineareg/estimators/syntheticcontorl.py
"""Synthetic Control estimator with uniform (sup-t) bands and post-period aggregation.

Scope and policy
----------------
- Single treated unit, never-treated donor pool.
- Inference uses in-space placebos with uniform sup-t bands (no analytic SE/p-values).
- Provides per-event-time effects with uniform bands and aggregated post-period ATT
    with a studentized bootstrap CI.
- All matrix ops route through :mod:`lineareg.core.linalg` to maintain consistency
  with the rest of the library. Results are returned in :class:`EstimationResult`.
- Bands are attached in `res.bands` under keys:
        * "pre" | "post" | "full": DataFrame(index=tau, columns={'lower','upper'})
        * "post_scalar": DataFrame with a single row {'lower','upper'}
        * "__meta__": {'origin':'placebo','kind':'uniform','estimator':'synthetic','level':95,'B':J}
    This aligns with the uniform-only policy shared with ES/SDID.

Notes
-----
- Weight fitting solves a least-squares problem on the pre-period under the
  simplex constraint (w >= 0, sum w = 1) via Frank-Wolfe (conditional gradient).
  This mirrors the original SC simplex restriction without external QP deps.
- Event time is defined as tau = t - T0 where T0 is the first treated period
  for the treated unit. The baseline (center_at=-1) is excluded from post aggregation.

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
    FormulaParser,  # integrate with formula.py for LHS materialization
)

__all__ = ["SyntheticControl"]

LOGGER = logging.getLogger(__name__)


def _frank_wolfe_simplex(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Solve min_w 0.5||X w - y||^2  s.t. w in simplex (w>=0, sum w=1) via Frank-Wolfe.

    Returns weights w on the simplex. Deterministic (no randomness).
    """
    T, J = X.shape
    # start at best single donor
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
        g = la.dot(X.T, (Xw - y))  # gradient
        j = int(np.argmin(g))  # FW vertex
        e = np.zeros(J, dtype=np.float64)
        e[j] = 1.0
        d = e - w
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
        if float(np.linalg.norm(Xw_new - Xw)) < float(tol):
            w, Xw = w_new, Xw_new
            break
        w, Xw = w_new, Xw_new
        if gamma < tol:
            break
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        w[:] = 0.0
        w[0] = 1.0
    else:
        w /= s
    return w


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
    ) -> None:
        # Allow either treat_name or cohort_name for API parity with event-study estimators
        if treat_name is None and cohort_name is None:
            raise TypeError(
                "Provide either treat_name or cohort_name for SyntheticControl.",
            )
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
        id_to_row = {i: k for k, i in enumerate(ids)}
        t_to_col = {t: k for k, t in enumerate(times)}
        j_donors = np.array([id_to_row[j] for j in donors], dtype=int)

        # --- Process each cohort and aggregate in event-time ---
        results_g: dict[int, dict[str, object]] = {}
        tau_union: set[int] = set()

        for g, treated_units in cohorts.items():
            # Pre/post masks for this cohort
            t0_col = t_to_col[g]
            pre = np.arange(0, t0_col)  # strictly before g
            post = np.arange(t0_col, len(times))  # at/after g

            if pre.size == 0:
                # Skip cohorts with no pre-treatment periods
                continue

            # For each treated unit in this cohort, fit synthetic control
            att_paths_g = []
            for treated_id in treated_units:
                i_treated = id_to_row[treated_id]

                # Design matrices for this treated unit
                y_pre = Y[i_treated, pre].astype(np.float64)
                X_pre = Y[j_donors][:, pre].T.astype(np.float64)  # T_pre x J

                # Fit simplex weights by Frank-Wolfe on pre-period
                w = _frank_wolfe_simplex(
                    X_pre, y_pre, max_iter=self.max_iter, tol=self.tol,
                )

                # Treated and synthetic series across all times
                y_treated = Y[i_treated, :].astype(np.float64)
                X_all = Y[j_donors, :].T.astype(np.float64)
                y_synth = la.dot(X_all, w)
                att_path = y_treated - y_synth  # absolute-time path
                att_paths_g.append(att_path)

            # Average ATT path across all treated units in this cohort
            att_path_g = np.mean(att_paths_g, axis=0)

            # Store cohort results
            results_g[g] = {
                "att_path": att_path_g,
                "n_treated": len(treated_units),
                "treated_ids": treated_units,
            }

            # Build event-time union
            for t_val in times:
                tau_val = int(t_val) - int(g)
                tau_union.add(tau_val)

        # Aggregate across cohorts in event-time τ = t - g
        tau_union_sorted = sorted(tau_union)
        att_tau_num: dict[int, float] = {int(tau): 0.0 for tau in tau_union_sorted}
        att_tau_den: dict[int, float] = {int(tau): 0.0 for tau in tau_union_sorted}

        for g, meta in results_g.items():
            att_path = meta["att_path"]
            n_tr = meta["n_treated"]
            for t_idx, t_val in enumerate(times):
                tau_val = int(t_val) - int(g)
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

        # Enforce unified baseline convention: set baseline event time to exactly 0
        # so that summary tables align with other event-study estimators.
        center_at = int(self.spec.center_at)
        if center_at in att_series.index:
            with suppress(Exception):
                # Defensive: ignore if index assignment fails (non-unique index, etc.)
                att_series.loc[center_at] = 0.0

        # Post-period aggregated ATT (exclude baseline tau == center_at)
        mask_post = tau_array > center_at
        post_agg = (
            float(np.nanmean(att_series.loc[mask_post].to_numpy(dtype=np.float64)))
            if np.any(mask_post)
            else float("nan")
        )

        # --------------------- Placebo (in-space) bands ---------------------
        # For each donor unit, create a placebo by treating it as treated
        # and using other donors to construct synthetic control. We record
        # a complete placebo path per (donor, cohort) over the unified τ-grid.
        tau_grid = np.array(tau_union_sorted, dtype=int)
        placebo_cols: list[np.ndarray] = []  # each element is shape (K,), K=len(tau_grid)

        for jj in donors:
            pool = [k for k in donors if k != jj]
            if len(pool) == 0:
                continue

            jp = id_to_row[jj]
            pool_indices = np.array([id_to_row[p] for p in pool], dtype=int)

            # For each cohort, compute placebo effect
            for g in results_g:
                t0_col = t_to_col[g]
                pre = np.arange(0, t0_col)

                if pre.size == 0:
                    continue

                # Fit synthetic control for this placebo unit
                Xp_pre = Y[pool_indices][:, pre].T.astype(np.float64)
                yp_pre = Y[jp, pre].astype(np.float64)
                wp = _frank_wolfe_simplex(
                    Xp_pre, yp_pre, max_iter=self.max_iter, tol=self.tol,
                )

                Xp_all = Y[pool_indices, :].T.astype(np.float64)
                yp_all = Y[jp, :].astype(np.float64)
                yp_synth = la.dot(Xp_all, wp)
                placebo_path = yp_all - yp_synth

                # Map to event-time vector on unified grid
                col = np.full(tau_grid.size, np.nan, dtype=float)
                for t_idx, t_val in enumerate(times):
                    tau_val = int(t_val) - int(g)
                    try:
                        j = int(np.where(tau_grid == int(tau_val))[0][0])
                        col[j] = float(placebo_path[t_idx])
                    except Exception:  # noqa: BLE001
                        continue
                placebo_cols.append(col)

        # Stack placebo columns into matrix K x B
        B = len(placebo_cols)
        band_level = round(100.0 * (1.0 - float(self.spec.alpha)))
        if B == 0:
            att_tau_star = np.full((tau_grid.size, 0), np.nan, dtype=float)
        else:
            att_tau_star = np.column_stack(placebo_cols).astype(float)

        # Construct uniform sup-t bands (pre/post/full)
        pre_mask = tau_array < center_at
        post_mask = tau_array > center_at

        def _uniform_band(side: str) -> pd.DataFrame:
            if B == 0:
                return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
            if side == "pre":
                mask = pre_mask
            elif side == "post":
                mask = post_mask
            else:
                mask = tau_array != center_at
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
            theta = att_series.reindex(tau_grid).to_numpy(dtype=float)[idx]
            theta_star = att_tau_star[idx, :]
            diffs = theta_star - theta[:, None]
            se = np.nanstd(diffs, axis=1, ddof=1)
            ok = np.isfinite(se) & (se > 0)
            if not np.any(ok):
                return pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
            with np.errstate(all="ignore"):
                tdraw = diffs[ok, :] / se[ok, None]
                sup_abs = np.nanmax(np.abs(tdraw), axis=0)
            c = float(np.nanquantile(sup_abs, 1.0 - self.spec.alpha))
            lo = theta.copy(); hi = theta.copy()
            lo[ok] = theta[ok] - c * se[ok]
            hi[ok] = theta[ok] + c * se[ok]
            lab = pd.Index(tau_grid[idx])
            return pd.DataFrame({"lower": pd.Series(lo, index=lab), "upper": pd.Series(hi, index=lab)}).sort_index()

        band_pre = _uniform_band("pre")
        band_post = _uniform_band("post")
        band_full = _uniform_band("full")

        # PostATT scalar: studentized from placebo column means over post-period
        if np.any(post_mask) and B > 0:
            theta_hat = float(np.nanmean(att_series.reindex(tau_grid)[post_mask].to_numpy()))
            post_star = np.nanmean(att_tau_star[np.flatnonzero(post_mask), :], axis=0)
            se_hat = float(np.nanstd(post_star, ddof=1))
            if np.isfinite(se_hat) and se_hat > 0:
                with np.errstate(all="ignore"):
                    t_abs = np.abs((post_star - theta_hat) / se_hat)
                    c = float(np.nanquantile(t_abs, 1.0 - self.spec.alpha))
                post_ci_df = pd.DataFrame({
                    "lower": [theta_hat - c * se_hat],
                    "upper": [theta_hat + c * se_hat],
                })
            else:
                post_ci_df = pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})
        else:
            post_ci_df = pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)})

        bands = {
            "pre": band_pre,
            "post": band_post,
            "full": band_full,
            "post_scalar": post_ci_df,
            "__meta__": {
                "origin": "placebo",
                "kind": "uniform",
                "estimator": "synthetic",
                "level": band_level,
                "B": int(B),
                # Reproducibility/audit: document donor pool and cohort grid
                "donors": list(donors),
                "cohorts": sorted(list(cohorts.keys())),
                # SC FW solver here is deterministic (no RNG); include note for parity
                "rng": "none (deterministic Frank–Wolfe)",
            },
        }

        # Basic per-tau output for summaries/exports
        att_df = pd.DataFrame({"tau": att_series.index, "att": att_series.to_numpy()}).set_index("tau")

        # RMSPE diagnostics (compute average across all cohorts)
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
                w_unit = _frank_wolfe_simplex(
                    X_pre_unit, y_pre_unit, max_iter=self.max_iter, tol=self.tol,
                )

                y_treated_unit = Y[i_treated, :].astype(np.float64)
                X_all_unit = Y[j_donors, :].T.astype(np.float64)
                y_synth_unit = la.dot(X_all_unit, w_unit)

                rmspe_pre_list.append(
                    float(
                        np.sqrt(np.mean((y_pre_unit - la.dot(X_pre_unit, w_unit)) ** 2)),
                    ),
                )
                if post.size > 0:
                    rmspe_post_list.append(
                        float(
                            np.sqrt(
                                np.mean(
                                    (y_treated_unit[post] - y_synth_unit[post]) ** 2,
                                ),
                            ),
                        ),
                    )

        rmspe_pre = (
            float(np.mean(rmspe_pre_list)) if len(rmspe_pre_list) > 0 else float("nan")
        )
        rmspe_post = (
            float(np.mean(rmspe_post_list))
            if len(rmspe_post_list) > 0
            else float("nan")
        )

        # Count total treated units across all cohorts
        n_treated_total = sum(len(meta["treated_ids"]) for meta in results_g.values())

        model_info = {
            "Estimator": "Synthetic Control",
            "BandType": "uniform",
            "BandLevel": band_level,
            "Alpha": float(self.spec.alpha),
            "B": int(B),  # placebo paths used
            "CenterAt": center_at,
            "Cohorts": list(cohorts.keys()),
            "TreatedUnits": n_treated_total,
            "Donors": len(donors),
            "RMSPE_pre": rmspe_pre,
            "RMSPE_post": rmspe_post,
            "PostATT": post_agg,
        }
        extra = {
            "cohorts": cohorts,
            "results_by_cohort": results_g,
            "times": times,
            "tau": tau_array,
            "att_tau": att_df.reset_index(),
            "post_scalar": post_ci_df,
            "donors": donors,
            "boot_meta": {
                "origin": "placebo",
                "kind": "uniform",
                "level": band_level,
                "B": int(B),
            },
        }

        if B > 1 and att_tau_star.shape[1] > 1:
            se_vals = bt.bootstrap_se(att_tau_star)
            se_series = pd.Series(se_vals, index=tau_grid)
            if center_at in se_series.index:
                se_series.loc[center_at] = 0.0
            if np.any(post_mask):
                post_star_draws = np.nanmean(att_tau_star[np.flatnonzero(post_mask), :], axis=0)
                model_info["PostATT_se"] = float(np.std(post_star_draws, ddof=1))
            extra["se_source"] = "bootstrap"
        else:
            se_series = None

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
