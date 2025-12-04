"""Difference-in-Difference-in-Differences (DDD) event-study wrapper.

Scope and policy
----------------
- Computes DDD by subtracting two event-study series built using the
    Callaway-Sant'Anna long-differences construction, aligning on common support.
- Bootstrap-only inference (wild/multiplier). Default B=2000 with B+1 rule.
- Provides uniform sup-t bands for pre, post, and full windows; staggered-
    robust handling mirrors the underlying ES runs.
- All matrix operations route through :mod:`lineareg.core.linalg`; no explicit
    inverses in estimator code. Formula/FE/NA alignment follow project defaults.

Comments/docstrings are English-only by policy.
"""

from __future__ import annotations

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
from lineareg.estimators.eventstudy_cs import (
    CallawaySantAnnaES,
    compute_uniform_bands,
)
from lineareg.utils.formula import FormulaParser

if TYPE_CHECKING:
    from collections.abc import Sequence
else:
    Sequence = tuple  # type: ignore[assignment]

__all__ = ["DDDEventStudy"]


# DDDResult dataclass intentionally removed; EstimationResult is used for outputs


class DDDEventStudy:
    """Difference-in-Difference-in-Differences (DDD) by subtracting two ES series.

    We reuse the same BootConfig for both groups so both ES runs share the same W.
    The τ grid is **aligned on the intersection (common support)** only.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        group_name: str,
        group_A_value: object,
        group_B_value: object,
        id_name: str,
        t_name: str,
        cohort_name: str,
        y_name: str,
        event_time_name: str | None = None,
        control_group: str = "nevertreated",
        center_at: int = -1,
        boot: BootConfig | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        include_base_in_post: bool = False,
        tau_weight: str = "treated_t",
        alpha: float = 0.05,
    ) -> None:
        self.group_name = str(group_name)
        self.group_A_value = group_A_value
        self.group_B_value = group_B_value
        self.id_name = str(id_name)
        self.t_name = str(t_name)
        self.cohort_name = str(cohort_name)
        self.y_name = str(y_name)
        self.event_time_name = None if event_time_name is None else str(event_time_name)

        # Normalize control_group to canonical forms accepted by CallawaySantAnnaES
        _cg = str(control_group).lower().replace("-", "")
        if _cg in {"nevertreated", "never"}:
            _cg = "never"
        elif _cg in {"notyettreated", "notyet"}:
            _cg = "notyet"
        else:
            raise ValueError(
                "control_group must be 'never' or 'notyet' (R/Stata compatible).",
            )

        # only pass arguments strictly accepted by CallawaySantAnnaES
        self.common_kwargs = {
            "id_name": self.id_name,
            "t_name": self.t_name,
            "cohort_name": self.cohort_name,
            "y_name": self.y_name,
            "event_time_name": self.event_time_name,
            "control_group": _cg,
        }
        self.center_at = int(center_at)
        self.include_base_in_post = bool(include_base_in_post)
        # Tau aggregation weight choice: 'equal', 'group', or 'treated_t'
        self.tau_weight = str(tau_weight).lower()
        if self.tau_weight not in {"equal", "group", "treated_t"}:
            raise ValueError("tau_weight must be one of {'equal','group','treated_t'}.")

        self.boot = boot
        self.cluster_ids = cluster_ids
        self.space_ids = space_ids
        self.time_ids = time_ids
        self.alpha = float(alpha)

    def fit(self, df: pd.DataFrame | None = None) -> EstimationResult:
        """Run ES for A/B, form exact bootstrap difference, and compute uniform bands."""
        if df is None:
            df = getattr(self, "_formula_df", None)
            if df is None:
                raise ValueError(
                    "Provide df explicitly or instantiate via from_formula().",
                )
        else:
            self._formula_df = df
        dfA = df[df[self.group_name] == self.group_A_value].copy()
        dfB = df[df[self.group_name] == self.group_B_value].copy()
        if dfA.empty or dfB.empty:
            msg = "Both groups must have non-empty data."
            raise ValueError(msg)

        # Shared BootConfig for constructing shared multipliers (default B).
        # Do not override user-specified boot policy or enumeration here;
        # inject only aligned ID arrays when creating a default BootConfig.
        boot_shared = self.boot or BootConfig(
            n_boot=bt.DEFAULT_BOOTSTRAP_ITERATIONS,
            # Default IF multiplier distribution: standard normal for IF multiplier bootstrap
            dist="standard_normal",
            cluster_ids=(
                np.asarray(self.cluster_ids) if self.cluster_ids is not None else None
            ),
            space_ids=(
                np.asarray(self.space_ids) if self.space_ids is not None else None
            ),
            time_ids=(np.asarray(self.time_ids) if self.time_ids is not None else None),
        )

        # Build observation-aligned multipliers for the UNION of A and B observations,
        # then slice rows for A/B. Delegate enumeration and Webb promotion to
        # bootstrap.cluster_multipliers / bootstrap.wild_multipliers.
        # Enforce paired (space_ids,time_ids); and include space in multiplier generation
        use_cluster = boot_shared.cluster_ids is not None
        use_space = boot_shared.space_ids is not None
        use_time = boot_shared.time_ids is not None
        if use_space ^ use_time:
            raise ValueError("space_ids and time_ids must be provided jointly.")
        if use_cluster and len(boot_shared.cluster_ids) != len(df.index):
            msg = f"BootConfig.cluster_ids length {len(boot_shared.cluster_ids)} != len(df) {len(df.index)}."
            raise ValueError(
                msg,
            )
        if use_time and len(boot_shared.time_ids) != len(df.index):
            msg = f"BootConfig.time_ids length {len(boot_shared.time_ids)} != len(df) {len(df.index)}."
            raise ValueError(
                msg,
            )
        if use_space and len(boot_shared.space_ids) != len(df.index):
            msg = f"BootConfig.space_ids length {len(boot_shared.space_ids)} != len(df) {len(df.index)}."
            raise ValueError(msg)

        if use_cluster or use_time or use_space:
            # Multiway support:
            # - include cluster if provided
            # - include both space and time when both provided
            clusters_all: list = []
            if use_cluster:
                clusters_all.append(np.asarray(boot_shared.cluster_ids))
            if use_space and use_time:
                clusters_all.append(np.asarray(boot_shared.space_ids))
                clusters_all.append(np.asarray(boot_shared.time_ids))
            bootcluster = "intersection" if len(clusters_all) > 1 else "first"
            # Delegate enumeration and policy decisions to BootConfig defaults
            W_all, wlog = bt.cluster_multipliers(
                clusters_all,
                n_boot=boot_shared.n_boot,
                dist=getattr(boot_shared, "dist", "standard_normal"),
                use_enumeration=(
                    "disabled"
                    if len(clusters_all) > 1
                    else getattr(boot_shared, "use_enumeration", None)
                ),
                enumeration_mode=(
                    "disabled"
                    if len(clusters_all) > 1
                    else getattr(boot_shared, "enumeration_mode", None)
                ),
                enum_max_g=None,
                policy=getattr(boot_shared, "policy", None),
                bootcluster=bootcluster,
            )
        else:
            # IID wild multipliers at the observation level (Rademacher by default)
            W_all = bt.wild_multipliers(
                len(df.index), n_boot=boot_shared.n_boot, dist="standard_normal",
            )
            wlog = {"method": "iid", "effective_B": int(W_all.shape[1])}

        # Enforce zero column means on the UNION once (shared-W principle), then slice
        W_all = W_all.astype(np.float64, copy=True)
        # Centre to mean zero and scale to unit variance (E[W]=0, Var[W]=1) per column
        W_all -= W_all.mean(axis=0, keepdims=True)
        v = W_all.var(axis=0, ddof=0)
        if not np.all(v > 0.0):
            raise ValueError(
                "external W (union) has zero-variance column(s) after recentering.",
            )
        W_all /= np.sqrt(v.reshape(1, -1))
        W_A = W_all[dfA.index, :]
        W_B = W_all[dfB.index, :]

        # Normalize dfA/dfB to RangeIndex so that external_W rows align exactly with
        # the passed data rows (CallawaySantAnnaES expects external_W row-order == df row-order).
        dfA = dfA.reset_index(drop=True)
        dfB = dfB.reset_index(drop=True)

        # Pass a minimal BootConfig into the ES estimator; the actual multipliers
        # used by the ES fit are provided via external_W (row-ordered to match dfA/dfB).
        boot_es_min = BootConfig(n_boot=boot_shared.n_boot, dist=boot_shared.dist)
        esA = CallawaySantAnnaES(
            **self.common_kwargs, center_at=self.center_at, boot=boot_es_min,
        )
        esB = CallawaySantAnnaES(
            **self.common_kwargs, center_at=self.center_at, boot=boot_es_min,
        )

        resA = esA.fit(dfA, external_W=W_A)
        resB = esB.fit(dfB, external_W=W_B)

        # Extract att_tau from extra (EstimationResult compatibility)
        att_tau_A = resA.extra.get("att_tau")
        att_tau_B = resB.extra.get("att_tau")
        att_gt_A = resA.extra.get("att_gt")
        att_gt_B = resB.extra.get("att_gt")
        att_tau_star_A = resA.extra.get("att_tau_star")
        att_tau_star_B = resB.extra.get("att_tau_star")
        missing_payload = []
        if att_tau_A is None or att_tau_B is None:
            missing_payload.append("att_tau")
        if att_gt_A is None or att_gt_B is None:
            missing_payload.append("att_gt")
        if att_tau_star_A is None or att_tau_star_B is None:
            missing_payload.append("att_tau_star")
        if missing_payload:
            payload = ", ".join(sorted(set(missing_payload)))
            raise ValueError(
                f"CallawaySantAnnaES results missing required extra payload: {payload}.",
            )

        # τ alignment: intersection only (no zero-filling)
        tauA = set(att_tau_A["tau"].tolist())
        tauB = set(att_tau_B["tau"].tolist())
        tau_common = sorted(tauA & tauB)
        if len(tau_common) == 0:
            missing_A = sorted(tauB - tauA)
            missing_B = sorted(tauA - tauB)
            msg = f"No common τ support. Missing in A: {missing_A}; missing in B: {missing_B}."
            raise RuntimeError(msg)

        # Source columns can be either 'att' (legacy) or 'params' (standardized). Prefer 'params'.
        colA = "params" if "params" in att_tau_A.columns else "att"
        colB = "params" if "params" in att_tau_B.columns else "att"
        a_map = att_tau_A.set_index("tau")[colA]
        b_map = att_tau_B.set_index("tau")[colB]
        diff_tau = pd.DataFrame({"tau": tau_common})
        diff_tau["params"] = (
            a_map.reindex(tau_common) - b_map.reindex(tau_common)
        ).to_numpy()
        # >>> PATCH: expose per-τ treated counts for A/B (useful for strict numerical comparisons)
        if "n_treat" in att_gt_A.columns and "n_treat" in att_gt_B.columns:
            wA_tau = att_gt_A.groupby("tau")["n_treat"].sum()
            wB_tau = att_gt_B.groupby("tau")["n_treat"].sum()
            diff_tau["n_treat_A"] = [float(wA_tau.get(t, np.nan)) for t in tau_common]
            diff_tau["n_treat_B"] = [float(wB_tau.get(t, np.nan)) for t in tau_common]

        # Exact bootstrap difference using τ-replicates (aligned to common grid)
        if att_tau_star_A.shape[1] != att_tau_star_B.shape[1]:
            msg = "Bootstrap draws B differ between groups; ensure shared BootConfig."
            raise RuntimeError(msg)
        B = int(att_tau_star_A.shape[1])

        def _align_star(
            att_tau: pd.DataFrame, att_tau_star: np.ndarray, grid: list[int],
        ) -> np.ndarray:
            pos = {tau: i for i, tau in enumerate(att_tau["tau"].tolist())}
            if len(pos) != len(set(pos.keys())):
                msg = "Duplicate τ in ES result; cannot align bootstrap replicates uniquely."
                raise RuntimeError(msg)
            K = len(grid)
            out = np.zeros((K, B), dtype=np.float64)
            for j, tau in enumerate(grid):
                out[j, :] = att_tau_star[pos[tau], :]
            return out

        Astar = _align_star(att_tau_A, att_tau_star_A, tau_common)
        Bstar = _align_star(att_tau_B, att_tau_star_B, tau_common)
        diff_star = Astar - Bstar  # (K x B)

        # --- Overall post-period ATE of the difference (dynamic aggregation)
        # Weighting aligned with did::aggte(type='dynamic'): use treated counts per tau
        def _tau_weights(att_gt: pd.DataFrame, att_tau: pd.DataFrame) -> pd.Series:
            if "n_treat" in att_gt.columns:
                return att_gt.groupby("tau")["n_treat"].sum()
            return pd.Series(1.0, index=att_tau["tau"].tolist())

        wA = (
            _tau_weights(att_gt_A, att_tau_A)
            .reindex(tau_common)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        wB = (
            _tau_weights(att_gt_B, att_tau_B)
            .reindex(tau_common)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        taus_arr = np.asarray(tau_common, dtype=int)
        if self.include_base_in_post:
            post_sel = taus_arr >= int(self.center_at)
        else:
            post_sel = taus_arr > int(self.center_at)

        # Compute weighted average post-ATT for reporting and produce a scalar bootstrap CI
        post_att_diff = float("nan")
        post_scalar = pd.DataFrame(
            {"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)},
        )
        post_weights = None
        if np.any(post_sel):
            # Normalize A and B post weights separately (R-style aggregation per-series then differencing)
            if float(wA[post_sel].sum()) <= 0.0 or float(wB[post_sel].sum()) <= 0.0:
                post_att_diff = float("nan")
                post_star = np.full((1, B), np.nan)
            else:
                vA = wA[post_sel] / float(wA[post_sel].sum())
                vB = wB[post_sel] / float(wB[post_sel].sum())
                # point estimates per series (align to att_tau 'params' or 'att')
                colA_pts = "params" if "params" in att_tau_A.columns else "att"
                colB_pts = "params" if "params" in att_tau_B.columns else "att"
                valsA = (
                    att_tau_A.set_index("tau")[colA_pts]
                    .reindex(taus_arr[post_sel])
                    .to_numpy(float)
                )
                valsB = (
                    att_tau_B.set_index("tau")[colB_pts]
                    .reindex(taus_arr[post_sel])
                    .to_numpy(float)
                )
                post_att_A = float(la.dot(vA, valsA))
                post_att_B = float(la.dot(vB, valsB))
                post_att_diff = post_att_A - post_att_B
                # bootstrap replicates aggregated per-series and differenced
                Astar_post = la.dot(
                    vA.reshape(1, -1), Astar[np.array(post_sel, dtype=bool), :],
                )
                Bstar_post = la.dot(
                    vB.reshape(1, -1), Bstar[np.array(post_sel, dtype=bool), :],
                )
                post_star = Astar_post - Bstar_post
                # --- scalar studentized CI (bootstrap-based; no analytic SE) ---
                theta_star = np.asarray(post_star, dtype=np.float64).reshape(-1)
                sd_boot = (
                    float(np.std(theta_star, ddof=1)) if theta_star.size > 1 else np.nan
                )
                if np.isfinite(sd_boot) and sd_boot > 0.0:
                    t_abs = np.abs((theta_star - post_att_diff) / sd_boot)
                    # finite-sample friendly rank (B+1) for the upper quantile
                    alpha = self.alpha
                    B = int(theta_star.size)
                    k = int(np.ceil((1.0 - alpha) * (B + 1)))
                    k = min(max(k, 1), B)
                    c = float(np.partition(t_abs, k - 1)[k - 1])
                    post_scalar = pd.DataFrame(
                        {
                            "lower": [post_att_diff - c * sd_boot],
                            "upper": [post_att_diff + c * sd_boot],
                        },
                    )
                else:
                    post_scalar = pd.DataFrame({"lower": [np.nan], "upper": [np.nan]})
            # audit weights stored per series (A/B normalized weights over post taus)
            if (
                np.any(post_sel)
                and float(wA[post_sel].sum()) > 0
                and float(wB[post_sel].sum()) > 0
            ):
                post_weights = pd.DataFrame(
                    {
                        "tau": np.asarray(taus_arr[post_sel], dtype=int),
                        "wA_post": vA,
                        "wB_post": vB,
                    },
                ).set_index("tau")

        # Pre/post uniform bands for the difference (bootstrap-studentized)
        # Delegate sup-t band computation to centralized utility to ensure
        # consistent studentization and baseline exclusion rules.
        att_tau_df = pd.DataFrame(
            {
                "tau": diff_tau["tau"].to_numpy(dtype=int),
                "att": diff_tau["params"].to_numpy(dtype=float),
            },
        )
        _att_tau_plot, (lo_pre, hi_pre), (lo_post, hi_post), (lo_full, hi_full) = (
            compute_uniform_bands(
                att_tau_df,
                diff_star,
                base_tau=self.center_at,
                alpha=self.alpha,
            )
        )

        # Populate standardized band columns for plotting/audit convenience
        diff_tau["pre_lower_95"] = np.nan
        diff_tau["pre_upper_95"] = np.nan
        diff_tau["post_lower_95"] = np.nan
        diff_tau["post_upper_95"] = np.nan
        diff_tau["full_lower_95"] = np.nan
        diff_tau["full_upper_95"] = np.nan
        # band series (lo_pre/hi_pre etc.) are indexed by tau values; map into diff_tau rows
        for tau_val, v in lo_pre.items():
            row_idx = diff_tau.index[diff_tau["tau"] == int(tau_val)].tolist()
            if row_idx:
                i = row_idx[0]
                diff_tau.loc[i, "pre_lower_95"] = v
                diff_tau.loc[i, "pre_upper_95"] = hi_pre.loc[tau_val]
        for tau_val, v in lo_post.items():
            row_idx = diff_tau.index[diff_tau["tau"] == int(tau_val)].tolist()
            if row_idx:
                i = row_idx[0]
                diff_tau.loc[i, "post_lower_95"] = v
                diff_tau.loc[i, "post_upper_95"] = hi_post.loc[tau_val]

        # PATCH: post_att_diff is a single scalar (not a band) for reporting summary
        info: dict[str, object] = {
            "Estimator": "EventStudy: DDD",
            "ControlGroup": self.common_kwargs["control_group"],
            "Base": None,
            "Method": None,
            "Bootstrap": f"shared-W (standard normal multipliers), policy={getattr(boot_shared, 'policy', None)}; no analytic SE/p-values",
            "NoAnalyticPValues": True,
            "CenterAt": self.center_at,
            "B": B,
            "BootLog": wlog,
            "TauSupport": "intersection-only (no zero-filling outside common support)",
            # Unified aggregate name: expose as PostATT (diff series aggregate)
            "PostATT": post_att_diff,
            "PostWeights": (None if post_weights is None else post_weights.to_dict()),
            "n_obs_A": int(resA.n_obs),
            "n_obs_B": int(resB.n_obs),
            # >>> PATCH: weights table for auditing (drop NaN rows for clarity)
            "WeightsByTau": (
                diff_tau.loc[
                    diff_tau[["n_treat_A", "n_treat_B"]].notna().all(axis=1),
                    ["tau", "n_treat_A", "n_treat_B"],
                ]
                .set_index("tau")
                .to_dict()
                if {"n_treat_A", "n_treat_B"}.issubset(diff_tau.columns)
                else None
            ),
            "PostATT_alpha": 0.05,
        }

        bands = {
            "pre": pd.DataFrame({"lower": lo_pre, "upper": hi_pre}),
            "post": pd.DataFrame({"lower": lo_post, "upper": hi_post}),
            "full": pd.DataFrame({"lower": lo_full, "upper": hi_full}),
            "post_scalar": post_scalar,
            "__meta__": {
                "origin": "bootstrap",
                "policy": str(getattr(boot_shared, "policy", "bootstrap")),
                "dist": getattr(boot_shared, "dist", None),
                "kind": "uniform",
                "level": 95,
                "B": int(B),
                "estimator": "eventstudy",
            },
        }

        return EstimationResult(
            params=diff_tau.set_index("tau")["params"],
            bands=bands,
            # >>> PATCH: total observations actually used by ES (sum of A and B)
            n_obs=int(resA.n_obs + resB.n_obs),
            model_info=info,
            extra={
                "series_A": resA,
                "series_B": resB,
                "diff_tau": diff_tau,
                "W_multipliers_union": W_all,
                "W_multipliers_A": W_A,
                "W_multipliers_B": W_B,
                "bands_source": "bootstrap",
                "se_source": "bootstrap",
                "boot_config": boot_shared,
            },
        )

    @classmethod
    def from_formula(  # noqa: PLR0913
        cls,
        *,
        data: pd.DataFrame,
        group_name: str,
        group_A_value: object,
        group_B_value: object,
        formula: str | None = None,
        id_name: str | None = None,
        time: str | None = None,
        options: str | None = None,
        W_dict: dict | None = None,
        boot: BootConfig | None = None,
        **kwargs,
    ) -> DDDEventStudy:
        """Instantiate the estimator from a common formula without fitting."""
        parsed = None
        if formula is not None:
            parser = FormulaParser(data, id_name=id_name, t_name=time, W_dict=W_dict)
            parsed = parser.parse(formula, iv=None, options=options)

        df_use, boot_eff, meta = prepare_formula_environment(
            formula=formula,
            data=data,
            parsed=parsed,
            boot=boot,
            default_boot_kwargs={
                "dist": "standard_normal",
            },
        )
        meta.attrs["_formula_df"] = df_use

        inst_kwargs = dict(kwargs)
        if id_name is not None and "id_name" not in inst_kwargs:
            inst_kwargs["id_name"] = id_name
        if time is not None and ("t_name" not in inst_kwargs):
            inst_kwargs["t_name"] = time
        if (
            ("y_name" not in inst_kwargs)
            and isinstance(formula, str)
            and ("~" in formula)
        ):
            lhs = str(formula).split("~", 1)[0].strip()
            if lhs:
                inst_kwargs["y_name"] = lhs
        if "cohort_name" not in inst_kwargs:
            if "g" in data.columns:
                inst_kwargs["cohort_name"] = "g"
            else:
                raise TypeError(
                    "DDDEventStudy.from_formula requires 'cohort_name'; couldn't infer a default.",
                )

        boot_to_use = boot_eff if boot_eff is not None else boot
        if boot_to_use is not None:
            inst_kwargs["boot"] = boot_to_use

        inst = cls(
            group_name=group_name,
            group_A_value=group_A_value,
            group_B_value=group_B_value,
            **inst_kwargs,
        )
        attach_formula_metadata(inst, meta)
        inst._formula_df = df_use
        return inst

    @classmethod
    def fit_from_formula(  # noqa: PLR0913
        cls,
        *,
        data: pd.DataFrame,
        group_name: str,
        group_A_value: object,
        group_B_value: object,
        formula: str | None = None,
        id_name: str | None = None,
        time: str | None = None,
        options: str | None = None,
        W_dict: dict | None = None,
        boot: BootConfig | None = None,
        fit_kwargs: dict | None = None,
        **kwargs,
    ) -> EstimationResult:
        est = cls.from_formula(
            data=data,
            group_name=group_name,
            group_A_value=group_A_value,
            group_B_value=group_B_value,
            formula=formula,
            id_name=id_name,
            time=time,
            options=options,
            W_dict=W_dict,
            boot=boot,
            **kwargs,
        )
        extra = fit_kwargs or {}
        return est.fit(**extra)
