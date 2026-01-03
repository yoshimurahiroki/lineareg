"""Difference-in-Difference-in-Differences (DDD).

This module computes DDD parameters and simultaneous confidence bands.
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
)
from lineareg.core.inference import compute_uniform_bands
from lineareg.utils.formula import FormulaParser

if TYPE_CHECKING:
    from collections.abc import Sequence
else:
    Sequence = tuple  # type: ignore[assignment]

__all__ = ["DDDEventStudy"]


# DDDResult dataclass intentionally removed; EstimationResult is used for outputs


class DDDEventStudy:
    """Difference-in-Difference-in-Differences (DDD) estimator.

    Computes parameters by differencing two Callaway-Sant'Anna estimators.
    Inference via shared bootstrap samples.

    Parameters
    ----------
    mode : str
        "diff_of_cs" (difference) or "pooled".
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
        anticipation: int = 0,
        base_period: str = "varying",
        boot: BootConfig | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        include_base_in_post: bool = False,
        tau_weight: str = "treated_t",
        alpha: float = 0.05,
        mode: str = "diff_of_cs",
    ) -> None:
        if mode not in {"diff_of_cs", "pooled"}:
            raise ValueError("mode must be 'diff_of_cs' or 'pooled'")
        self.mode = mode
        self.group_name = str(group_name)
        self.group_A_value = group_A_value
        self.group_B_value = group_B_value
        self.id_name = str(id_name)
        self.t_name = str(t_name)
        self.cohort_name = str(cohort_name)
        self.y_name = str(y_name)
        self.event_time_name = None if event_time_name is None else str(event_time_name)
        self.anticipation = int(anticipation)
        base_period_norm = str(base_period).lower()
        if base_period_norm not in {"varying", "universal"}:
            raise ValueError("base_period must be 'varying' or 'universal'")
        self.base_period = base_period_norm

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

        self.common_kwargs = {
            "id_name": self.id_name,
            "t_name": self.t_name,
            "cohort_name": self.cohort_name,
            "y_name": self.y_name,
            "event_time_name": self.event_time_name,
            "control_group": _cg,
            "anticipation": self.anticipation,
            "base_period": self.base_period,
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

    def fit(
        self,
        df: pd.DataFrame | None = None,
        *,
        ssc: dict[str, str | int | float | bool] | None = None,
    ) -> EstimationResult:
        """Fit DDD estimator.

        Estimates coefficients and bands.
        """
        if df is None:
            df = getattr(self, "_formula_df", None)
            if df is None:
                raise ValueError(
                    "Provide df explicitly or instantiate via from_formula().",
                )
        else:
            self._formula_df = df

        # DID compatibility checks: panel only (no repeated cross-sections)
        if df.duplicated([self.id_name, self.t_name]).any():
            raise ValueError("DID requires unique (id,time) rows. Duplicates detected.")
        t_per_id = df.groupby(self.id_name)[self.t_name].nunique()
        if t_per_id.max() <= 1:
            raise ValueError("Repeated cross-section (RCS) is not supported. Panel data required.")

        if self.mode == "pooled":
            return self._fit_pooled(df, ssc=ssc)
        return self._fit_diff_of_cs(df, ssc=ssc)

    def _fit_pooled(
        self,
        df: pd.DataFrame,
        ssc: dict[str, str | int | float | bool] | None = None,
    ) -> EstimationResult:
        """Pooled mode: run single C&S with unified control pool, then compute A-B diff."""
        # Keep pooled rows in the original df order to preserve alignment with any
        # external ID arrays provided by the user (cluster/space/time).
        maskA_df = df[self.group_name] == self.group_A_value
        maskB_df = df[self.group_name] == self.group_B_value
        if not bool(maskA_df.any()) or not bool(maskB_df.any()):
            msg = "Both groups must have non-empty data."
            raise ValueError(msg)
        df_pooled = df.loc[maskA_df | maskB_df].copy()
        df_pooled["_ddd_group"] = np.where(
            df_pooled[self.group_name] == self.group_A_value,
            "A",
            "B",
        )

        def _ids_from_df_or_seq(val, df_ref: pd.DataFrame) -> np.ndarray | None:
            if val is None:
                return None
            if isinstance(val, str):
                if val not in df_ref.columns:
                    raise ValueError(f"ID column '{val}' not found in df.")
                return df_ref[val].to_numpy()
            arr = np.asarray(val)
            if arr.shape[0] != df.shape[0]:
                raise ValueError("ID arrays must have the same length as df passed to fit().")
            # Subset to df_ref rows while preserving df_ref order.
            return arr[df_ref.index.to_numpy()]

        boot_shared = self.boot or BootConfig(
            n_boot=bt.DEFAULT_BOOTSTRAP_ITERATIONS,
            dist="standard_normal",
        )
        # Ensure the shared bootstrap IDs are aligned to df_pooled.
        boot_shared = BootConfig(
            n_boot=int(getattr(boot_shared, "n_boot", bt.DEFAULT_BOOTSTRAP_ITERATIONS)),
            dist=getattr(boot_shared, "dist", "standard_normal"),
            seed=getattr(boot_shared, "seed", None),
            policy=getattr(boot_shared, "policy", None),
            enumeration_mode=getattr(boot_shared, "enumeration_mode", None),
            use_enumeration=getattr(boot_shared, "use_enumeration", True),
            cluster_ids=_ids_from_df_or_seq(getattr(boot_shared, "cluster_ids", None) or self.cluster_ids, df_pooled),
            space_ids=_ids_from_df_or_seq(getattr(boot_shared, "space_ids", None) or self.space_ids, df_pooled),
            time_ids=_ids_from_df_or_seq(getattr(boot_shared, "time_ids", None) or self.time_ids, df_pooled),
        )
        if (boot_shared.space_ids is None) ^ (boot_shared.time_ids is None):
            raise ValueError("space_ids and time_ids must be provided jointly.")

        # Keep C&S bootstrap metadata consistent with the DDD bootstrap design. When external_W is
        # supplied, C&S does not need to generate multipliers, but it may still use boot IDs for SSC.
        boot_es = BootConfig(
            n_boot=boot_shared.n_boot,
            dist=boot_shared.dist,
            cluster_ids=(
                boot_shared.cluster_ids if not isinstance(self.cluster_ids, str) else None
            ),
            space_ids=(
                boot_shared.space_ids if not isinstance(self.space_ids, str) else None
            ),
            time_ids=(
                boot_shared.time_ids if not isinstance(self.time_ids, str) else None
            ),
        )
        esA = CallawaySantAnnaES(
            **self.common_kwargs, center_at=self.center_at, boot=boot_es,
            cluster_ids=self.cluster_ids,
            space_ids=self.space_ids,
            time_ids=self.time_ids,
        )
        esB = CallawaySantAnnaES(
            **self.common_kwargs, center_at=self.center_at, boot=boot_es,
            cluster_ids=self.cluster_ids,
            space_ids=self.space_ids,
            time_ids=self.time_ids,
        )

        W_full_df, wlog = boot_shared.make_multipliers(n_obs=df_pooled.shape[0])
        W_full = W_full_df.to_numpy(dtype=np.float64)
        # Standardize (mean 0, var 1) explicitly to match multiplier bootstrap contracts.
        W_full = W_full - W_full.mean(axis=0, keepdims=True)
        v = W_full.var(axis=0, ddof=0)
        if not np.all(v > 0.0):
            raise ValueError("zero-variance multipliers")
        W_full /= np.sqrt(v.reshape(1, -1))
        W_full_df = pd.DataFrame(W_full, columns=list(W_full_df.columns))

        maskA = df_pooled["_ddd_group"] == "A"
        maskB = df_pooled["_ddd_group"] == "B"
        dfA_use = df_pooled.loc[maskA].reset_index(drop=True)
        dfB_use = df_pooled.loc[maskB].reset_index(drop=True)
        W_A = W_full[maskA.to_numpy(), :]
        W_B = W_full[maskB.to_numpy(), :]

        resA = esA.fit(dfA_use, external_W=W_A, ssc=ssc)
        resB = esB.fit(dfB_use, external_W=W_B, ssc=ssc)

        wlog = {"method": "pooled", "effective_B": int(W_full.shape[1]), "inner": wlog}
        return self._combine_results(resA, resB, boot_shared, W_full_df, W_A, W_B, wlog)

    def _fit_diff_of_cs(
        self,
        df: pd.DataFrame,
        ssc: dict[str, str | int | float | bool] | None = None,
    ) -> EstimationResult:
        """Original diff_of_cs mode: separate C&S runs, then difference."""
        dfA = df[df[self.group_name] == self.group_A_value].copy()
        dfB = df[df[self.group_name] == self.group_B_value].copy()
        if dfA.empty or dfB.empty:
            msg = "Both groups must have non-empty data."
            raise ValueError(msg)

        cohorts_A = dfA[self.cohort_name].unique()
        cohorts_B = dfB[self.cohort_name].unique()
        has_treatment_A = any(c > 0 for c in cohorts_A if pd.notna(c) and c != 0)
        has_treatment_B = any(c > 0 for c in cohorts_B if pd.notna(c) and c != 0)

        if not has_treatment_A:
            msg = "Group A must have at least one treated cohort (cohort > 0)."
            raise ValueError(msg)

        def _ids_full(val, df_ref: pd.DataFrame) -> np.ndarray | None:
            if val is None:
                return None
            if isinstance(val, str):
                if val not in df_ref.columns:
                    raise ValueError(f"ID column '{val}' not found in df.")
                return df_ref[val].to_numpy()
            arr = np.asarray(val)
            if arr.shape[0] != df_ref.shape[0]:
                raise ValueError("ID arrays must have the same length as df passed to fit().")
            return arr

        boot_in = self.boot
        boot_shared = boot_in or BootConfig(
            n_boot=bt.DEFAULT_BOOTSTRAP_ITERATIONS,
            dist="standard_normal",
        )
        # Ensure IDs are actual vectors aligned to df (not a column-name string).
        boot_shared = BootConfig(
            n_boot=int(getattr(boot_shared, "n_boot", bt.DEFAULT_BOOTSTRAP_ITERATIONS)),
            dist=getattr(boot_shared, "dist", "standard_normal"),
            seed=getattr(boot_shared, "seed", None),
            policy=getattr(boot_shared, "policy", None),
            enumeration_mode=getattr(boot_shared, "enumeration_mode", None),
            use_enumeration=getattr(boot_shared, "use_enumeration", True),
            cluster_ids=_ids_full(getattr(boot_shared, "cluster_ids", None) or self.cluster_ids, df),
            space_ids=_ids_full(getattr(boot_shared, "space_ids", None) or self.space_ids, df),
            time_ids=_ids_full(getattr(boot_shared, "time_ids", None) or self.time_ids, df),
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
            if use_cluster and (use_space or use_time):
                multiway_list = [np.asarray(boot_shared.cluster_ids)]
                if use_space:
                    multiway_list.append(np.asarray(boot_shared.space_ids))
                if use_time:
                    multiway_list.append(np.asarray(boot_shared.time_ids))
                boot_all = BootConfig(
                    n_boot=boot_shared.n_boot,
                    dist=getattr(boot_shared, "dist", "standard_normal"),
                    multiway_ids=multiway_list,
                    bootcluster="product",
                    policy=getattr(boot_shared, "policy", "boottest"),
                    use_enumeration=False,
                    seed=getattr(boot_shared, "seed", None),
                )
            elif use_space and use_time:
                boot_all = BootConfig(
                    n_boot=boot_shared.n_boot,
                    dist=getattr(boot_shared, "dist", "standard_normal"),
                    space_ids=boot_shared.space_ids,
                    time_ids=boot_shared.time_ids,
                    bootcluster="product",
                    policy=getattr(boot_shared, "policy", "boottest"),
                    use_enumeration=False,
                    seed=getattr(boot_shared, "seed", None),
                )
            elif use_cluster:
                boot_all = BootConfig(
                    n_boot=boot_shared.n_boot,
                    dist=getattr(boot_shared, "dist", "standard_normal"),
                    cluster_ids=boot_shared.cluster_ids,
                    policy=getattr(boot_shared, "policy", "boottest"),
                    use_enumeration=getattr(boot_shared, "use_enumeration", True),
                    seed=getattr(boot_shared, "seed", None),
                )
            elif use_space:
                boot_all = BootConfig(
                    n_boot=boot_shared.n_boot,
                    dist=getattr(boot_shared, "dist", "standard_normal"),
                    cluster_ids=boot_shared.space_ids,
                    policy=getattr(boot_shared, "policy", "boottest"),
                    use_enumeration=getattr(boot_shared, "use_enumeration", True),
                    seed=getattr(boot_shared, "seed", None),
                )
            else:
                boot_all = BootConfig(
                    n_boot=boot_shared.n_boot,
                    dist=getattr(boot_shared, "dist", "standard_normal"),
                    cluster_ids=boot_shared.time_ids,
                    policy=getattr(boot_shared, "policy", "boottest"),
                    use_enumeration=getattr(boot_shared, "use_enumeration", True),
                    seed=getattr(boot_shared, "seed", None),
                )
            W_all_df, wlog = boot_all.make_multipliers(n_obs=len(df.index))
            W_all = W_all_df.to_numpy(dtype=np.float64)
        else:
            W_all = bt.wild_multipliers(
                len(df.index), n_boot=boot_shared.n_boot, dist="standard_normal",
            )
            wlog = {"method": "iid", "effective_B": int(W_all.shape[1])}
            W_all_df = pd.DataFrame(W_all, columns=[f"b{i}" for i in range(W_all.shape[1])])

        W_all = W_all.astype(np.float64, copy=True)
        # Standardize (mean 0, var 1) defensively to ensure a strict multiplier contract.
        W_all = W_all - W_all.mean(axis=0, keepdims=True)
        v = W_all.var(axis=0, ddof=0)
        if not np.all(v > 0.0):
            raise ValueError("zero-variance multipliers")
        W_all = W_all / np.sqrt(v.reshape(1, -1))
        W_all_df = pd.DataFrame(W_all, columns=list(W_all_df.columns))

        df_with_pos = df.copy()
        df_with_pos["__orig_pos__"] = np.arange(len(df))
        pos_A = df_with_pos.loc[df_with_pos[self.group_name] == self.group_A_value, "__orig_pos__"].to_numpy()
        pos_B = df_with_pos.loc[df_with_pos[self.group_name] == self.group_B_value, "__orig_pos__"].to_numpy()
        W_A = W_all[pos_A, :]
        W_B = W_all[pos_B, :]

        dfA = dfA.reset_index(drop=True)
        dfB = dfB.reset_index(drop=True)

        boot_es_min = BootConfig(n_boot=boot_shared.n_boot, dist=boot_shared.dist)
        esA = CallawaySantAnnaES(
            **self.common_kwargs, center_at=self.center_at, boot=boot_es_min,
        )
        resA = esA.fit(dfA, external_W=W_A, ssc=ssc)

        if has_treatment_B:
            esB = CallawaySantAnnaES(
                **self.common_kwargs, center_at=self.center_at, boot=boot_es_min,
            )
            resB = esB.fit(dfB, external_W=W_B, ssc=ssc)
        else:
            resB = self._null_result_from_A(resA, W_B)

        return self._combine_results(resA, resB, boot_shared, W_all_df, W_A, W_B, wlog)

    def _null_result_from_A(
        self, resA: EstimationResult, W_B: np.ndarray,
    ) -> EstimationResult:
        """Create a null result for group B when it has no treatment cohorts.

        In classic DDD, group B (comparison) may have no treatment. In this case,
        ATT(B,τ) = 0 for all τ, and bootstrap draws are also zero.
        """
        att_tau_A = resA.extra.get("att_tau")
        att_gt_A = resA.extra.get("att_gt")
        att_tau_star_A = resA.extra.get("att_tau_star")

        if att_tau_A is None or att_tau_star_A is None:
            raise ValueError("resA must contain att_tau and att_tau_star in extra.")

        tau_grid = att_tau_A["tau"].tolist()
        B = att_tau_star_A.shape[1]

        att_tau_B = pd.DataFrame({
            "tau": tau_grid,
            "params": [0.0] * len(tau_grid),
        })
        att_gt_B = pd.DataFrame({
            "g": [0] * len(tau_grid),
            "t": tau_grid,
            "tau": tau_grid,
            "att": [0.0] * len(tau_grid),
            "n_treat": [0] * len(tau_grid),
        })
        att_tau_star_B = np.zeros((len(tau_grid), B), dtype=np.float64)

        null_bands = {
            "pre": pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)}),
            "post": pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)}),
            "full": pd.DataFrame({"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)}),
            "__meta__": {"origin": "null_B", "kind": "uniform", "estimator": "ddd"},
        }

        return EstimationResult(
            params=pd.Series([0.0] * len(tau_grid), index=tau_grid, name="params"),
            se=pd.Series([0.0] * len(tau_grid), index=tau_grid),
            n_obs=0,
            bands=null_bands,
            model_info={"Estimator": "DDD_null_B", "B": B},
            extra={
                "att_tau": att_tau_B,
                "att_gt": att_gt_B,
                "att_tau_star": att_tau_star_B,
                "se_source": "multiplier",
            },
        )

    def _combine_results(
        self, resA: EstimationResult, resB: EstimationResult,
        boot_shared: BootConfig | None = None,
        W_all: pd.DataFrame | None = None,
        W_A: np.ndarray | None = None,
        W_B: np.ndarray | None = None,
        wlog: dict | None = None,
    ) -> EstimationResult:

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
        post_att_diff_se = float("nan")
        post_scalar = pd.DataFrame(
            {"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)},
        )
        post_weights = None
        if np.any(post_sel):
            # Normalize A and B post weights separately (R-style aggregation per-series then differencing)
            if float(wA[post_sel].sum()) <= 0.0:
                post_att_diff = float("nan")
                post_star = np.full((1, B), np.nan)
            else:
                vA = wA[post_sel] / float(wA[post_sel].sum())
                # If group B has no treated cohorts (null series), DDD reduces to series A.
                if float(wB[post_sel].sum()) <= 0.0:
                    vB = vA
                    valsB = np.zeros_like(vA, dtype=float)
                    Bstar_post = np.zeros((1, B), dtype=np.float64)
                else:
                    vB = wB[post_sel] / float(wB[post_sel].sum())
                    colB_pts = "params" if "params" in att_tau_B.columns else "att"
                    valsB = (
                        att_tau_B.set_index("tau")[colB_pts]
                        .reindex(taus_arr[post_sel])
                        .to_numpy(float)
                    )
                    Bstar_post = la.dot(
                        vB.reshape(1, -1), Bstar[np.array(post_sel, dtype=bool), :],
                    )
                # point estimates per series (align to att_tau 'params' or 'att')
                colA_pts = "params" if "params" in att_tau_A.columns else "att"
                valsA = (
                    att_tau_A.set_index("tau")[colA_pts]
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
                post_star = Astar_post - Bstar_post
                # --- scalar studentized CI (bootstrap-based; no analytic SE) ---
                theta_star = np.asarray(post_star, dtype=np.float64).reshape(-1)
                sd_boot = (
                    float(np.std(theta_star, ddof=1)) if theta_star.size > 1 else np.nan
                )
                post_att_diff_se = float(sd_boot) if np.isfinite(sd_boot) else float("nan")
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

            "CenterAt": self.center_at,
            "B": B,
            "BootLog": wlog,
            "TauSupport": "intersection-only (no zero-filling outside common support)",
            # Unified aggregate name: expose as PostATT (diff series aggregate)
            "PostATT": post_att_diff,
            "PostATT_se": post_att_diff_se,
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
            "PostATT_alpha": float(self.alpha),
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

        se_series = pd.Series(
            bt.bootstrap_se(diff_star),
            index=diff_tau["tau"].astype(int).to_numpy(),
        )
        if int(self.center_at) in se_series.index:
            se_series.loc[int(self.center_at)] = 0.0

        return EstimationResult(
            params=diff_tau.set_index("tau")["params"],
            se=se_series,
            bands=bands,
            # >>> PATCH: total observations actually used by ES (sum of A and B)
            n_obs=int(resA.n_obs + resB.n_obs),
            model_info=info,
            extra={
                "series_A": resA,
                "series_B": resB,
                "diff_tau": diff_tau,
                "att_tau": diff_tau,  # Alias for consistency
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
