"""Doubly-robust DiD event-study.

This module implements the Sant'Anna-Zhao (2020) DR-DID estimator with
cross-fitting and multiplier bootstrap inference.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

from lineareg.core import bootstrap as bt
from lineareg.core.bootstrap import compute_ssc_correction, _normalize_ssc
from lineareg.core import linalg as la
from lineareg.estimators.base import (
    BootConfig,
    EstimationResult,
    FormulaMetadata,
    attach_formula_metadata,
)
from lineareg.utils.formula import FormulaParser
from lineareg.utils.eventstudy_helpers import (
    ESCellSpec,
    aggregate_tau,
    build_cells,
)
from lineareg.utils.helpers import event_tau, time_to_pos

from lineareg.core.inference import (
    compute_uniform_bands,
    post_aggregate_uniform_band,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
else:
    Sequence = tuple  # type: ignore[assignment]

__all__ = ["DREventStudy"]





class DREventStudy:
    """Doubly-robust DR-DID estimator.

    Combines outcome regression and inverse probability weighting (doubly robust).
    Uses K-fold cross-fitting for nuisance parameters to mitigate overfitting bias.
    Inference via multiplier bootstrap.
    """

    # Cross-fitting and IF bootstrap strictly follow the theory-first design.

    def __init__(  # noqa: PLR0913
        self,
        *,
        id_name: str,
        t_name: str,
        cohort_name: str,
        treat_name: str,
        y_name: str,
        event_time_name: str | None = None,
        control_group: str = "notyet",
        base_tau: int = -1,
        anticipation: int = 0,
        base_period: str = "varying",
        method: str = "dr",
        ps_learner=None,
        or_learner=None,
        n_folds: int = 5,
        fold_seed: int | None = None,
        hajek: bool = True,
        hajek_eps: float = 1e-6,
        boot: BootConfig | None = None,
        alpha: float = 0.05,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        pihat: np.ndarray | None = None,
        tau_weight: str = "group",
        trim_ps: float = 0.0,
        trim_mode: str = "clip",
    ) -> None:
        self.id_name = str(id_name)
        self.t_name = str(t_name)
        self.cohort_name = str(cohort_name)
        self.treat_name = str(treat_name)
        self.y_name = str(y_name)
        self.event_time_name = None if event_time_name is None else str(event_time_name)
        control_group = str(control_group).lower().strip()
        if control_group not in {"notyet", "never"}:
            raise ValueError(
                "control_group must be one of {'notyet','never'} for DR-DID event-study.",
            )
        self.control_group = control_group
        self.base_tau = int(base_tau)
        self.anticipation = int(anticipation)
        base_period_norm = str(base_period).lower()
        if base_period_norm not in {"varying", "universal"}:
            raise ValueError("base_period must be 'varying' or 'universal'")
        self.base_period = base_period_norm
        self.alpha = float(alpha)
        method = method.lower().strip()
        if method not in {"dr", "ipw"}:
            msg = "method must be one of {'dr','ipw'}."
            raise ValueError(msg)
        self.method = method

        # Defaults aligned with R did / csdid: PS = plain logistic MLE (no penalty), OR = OLS
        # Both learners are configured with fit_intercept=False so that callers
        # provide an explicit constant column when desired; this matches R/Stata
        # behavior (no implicit intercept when design matrix already contains const).
        # Explicit solver for scikit-learn stability; penalty=None implements MLE
        self.ps_learner = (
            LogisticRegression(
                penalty=None,
                solver="lbfgs",
                fit_intercept=False,
                max_iter=10_000,
                tol=1e-10,
            )
            if ps_learner is None
            else ps_learner
        )
        self.or_learner = (
            LinearRegression(fit_intercept=False) if or_learner is None else or_learner
        )

        self.n_folds = int(n_folds)
        self.fold_seed = fold_seed
        # Strict policy: Hájek normalization only
        if hajek is False:
            raise ValueError("Strict policy: only Hájek normalization is permitted.")
        self.hajek = True
        self.hajek_eps = float(hajek_eps)
        self.trim_ps = float(trim_ps)
        if self.trim_ps < 0 or self.trim_ps >= 0.5:
            msg = "trim_ps must be in [0, 0.5)."
            raise ValueError(msg)
        self.trim_mode = str(trim_mode).lower().strip()
        if self.trim_mode not in {"clip", "drop"}:
            msg = "trim_mode must be 'clip' or 'drop'."
            raise ValueError(msg)

        self.boot = boot
        self.cluster_ids = cluster_ids
        self.space_ids = space_ids
        self.time_ids = time_ids
        self.tau_weight = str(tau_weight).lower()
        if self.tau_weight not in {"group", "equal", "treated_t"}:
            msg = "tau_weight must be one of {'group','equal','treated_t'}."
            raise ValueError(msg)
        # Optional external propensity scores aligned with original df rows
        # CRITICAL: external pihat is ALWAYS clipped to [hajek_eps, 1-hajek_eps] to prevent division issues
        self._pihat_external = None
        self._pihat_clipped_bounds = None  # Will store (eps, 1-eps) if clipping applied
        if pihat is not None:
            pihat = np.asarray(pihat, dtype=np.float64).reshape(-1)
            # Force clipping to [hajek_eps, 1 - hajek_eps] for stability
            pihat_clipped = np.clip(pihat, hajek_eps, 1.0 - hajek_eps)
            n_clipped = int(np.sum((pihat < hajek_eps) | (pihat > 1.0 - hajek_eps)))
            self._pihat_external = pihat_clipped
            self._pihat_clipped_bounds = (hajek_eps, 1.0 - hajek_eps, n_clipped)

    def fit(
        self,
        df: pd.DataFrame | None = None,
        x_ps: pd.DataFrame | None = None,
        x_or: pd.DataFrame | None = None,
        external_W: np.ndarray | None = None,
        ssc: dict[str, str | int | float | bool] | None = None,
    ) -> EstimationResult:

        """Fit DR-DID estimator.

        Estimates ATT(g,t) using cross-fitted doubly robust scores and
        aggregates results.
        """
        if df is None:
            df = getattr(self, "_formula_df", None)
            if df is None:
                raise ValueError(
                    "Provide df explicitly or instantiate via from_formula().",
                )
        else:
            self._formula_df = df

        if x_ps is None:
            x_ps = getattr(self, "_formula_x_ps", None)
        else:
            self._formula_x_ps = x_ps

        if x_or is None:
            x_or = getattr(self, "_formula_x_or", None)
        else:
            self._formula_x_or = x_or

        # operate on level data: keep original order -> sort; track permutation for strict realignment
        df0 = df.copy()
        df0["_origpos"] = np.arange(df0.shape[0], dtype=np.int64)
        # DID compatibility checks: panel only (no repeated cross-sections)
        if df0.duplicated([self.id_name, self.t_name]).any():
            raise ValueError("DID requires unique (id,time) rows. Duplicates detected.")
        t_per_id = df0.groupby(self.id_name)[self.t_name].nunique()
        if t_per_id.max() <= 1:
            raise ValueError("Repeated cross-section (RCS) is not supported. Panel data required.")
        # Upfront strict length checks for external IDs: fail early if lengths do not match original sample
        for nm, arr in [
            ("cluster_ids", self.cluster_ids),
            ("space_ids", self.space_ids),
            ("time_ids", self.time_ids),
        ]:
            if arr is not None and len(np.asarray(arr)) != df0.shape[0]:
                raise ValueError(
                    f"{nm} length must equal original number of rows before sorting.",
                )
        # Staggered adoption (monotone treatment) check: D_{i,t} must be non-decreasing in t
        # Evaluate on the original order grouped by id, then by time. This enforces
        # the standard staggered-adoption assumption (no re-treatment or reversal).
        _g = df0[[self.id_name, self.t_name, self.treat_name]].copy()
        _g[self.t_name] = _g[self.t_name].astype(int)
        _g = _g.sort_values([self.id_name, self.t_name])
        by_id = _g.groupby(self.id_name, sort=False)[self.treat_name].apply(
            lambda s: np.any(np.diff(s.to_numpy()) < 0),
        )
        if bool(by_id.any()):
            bad = int(by_id.sum())
            raise ValueError(
                f"Non-monotone treatment paths detected for {bad} unit(s); DR-DID event-study requires staggered adoption.",
            )
        df_aug = df0.sort_values([self.id_name, self.t_name]).reset_index(drop=True)
        spec = ESCellSpec(
            id_name=self.id_name,
            t_name=self.t_name,
            cohort_name=self.cohort_name,
            y_name=self.y_name,
            event_time_name=self.event_time_name,
            control_group=self.control_group,
            center_at=self.base_tau,
            anticipation=self.anticipation,
            base_period=self.base_period,
        )
        df_aug, cell_keys, cell_meta = build_cells(df_aug, spec)
        pret_map = cell_meta.get("pret_map", {})

        # Precompute base-period level mappings for constructing long-differences
        df_sorted = df_aug.sort_values([self.id_name, self.t_name])
        times_all = np.sort(df_sorted[self.t_name].astype(int).unique())
        t2pos = time_to_pos(times_all)
        base_map = {
            int(tt): df_sorted.loc[
                df_sorted[self.t_name].astype(int).to_numpy() == int(tt), :,
            ].set_index(self.id_name)[self.y_name]
            for tt in times_all
        }

        if len(cell_keys) != len(set(cell_keys)):
            msg = "Duplicate (g,t) cells detected; check cohort and time definitions."
            raise ValueError(msg)

        # Default feature sets
        def _ensure_const_df(X: pd.DataFrame) -> pd.DataFrame:
            """Ensure a constant column named 'const' exists on X (no duplicate insertion)."""
            Xc = X.copy()
            if "const" not in Xc.columns:
                Xc["const"] = 1.0
            return Xc

        # Allow covariates=None (constant-only model) for DR-DID
        # When x_ps or x_or is None, create a constant-only DataFrame
        if x_ps is None:
            x_ps = pd.DataFrame({"const": 1.0}, index=df_aug.index)
        if x_or is None:
            x_or = pd.DataFrame({"const": 1.0}, index=df_aug.index)
        x_ps = _ensure_const_df(x_ps.loc[df_aug.index, :])
        x_or = _ensure_const_df(x_or.loc[df_aug.index, :])

        # --- PATCH: BootConfig strict policy + realign IDs to the sorted order ---
        def _align_ids(arr_like: Sequence | None) -> np.ndarray | None:
            if arr_like is None:
                return None
            arr = np.asarray(arr_like)
            if arr.shape[0] != df0.shape[0]:
                msg = "Provided IDs must have length equal to original number of rows before sorting."
                raise ValueError(msg)
            perm = df_aug["_origpos"].to_numpy(dtype=np.int64)
            return np.asarray(arr, dtype=object)[perm]

        if self.boot is None:
            # Delegate policy/enumeration defaults to BootConfig; align only IDs to df order.
            boot = BootConfig(
                n_boot=bt.DEFAULT_BOOTSTRAP_ITERATIONS,
                # Default IF multiplier distribution for DR event-study uses standard normal
                # to align with theoretical multiplier IF bootstrap conventions.
                dist="standard_normal",
                cluster_ids=_align_ids(self.cluster_ids),
                space_ids=_align_ids(self.space_ids),
                time_ids=_align_ids(self.time_ids),
            )
        else:
            # Respect user BootConfig: only realign IDs to df order and preserve policy/enumeration.
            bc = self.boot
            bkw = {f: getattr(bc, f) for f in bc.__dataclass_fields__}
            bkw["cluster_ids"] = _align_ids(getattr(bc, "cluster_ids", None))
            bkw["space_ids"] = _align_ids(getattr(bc, "space_ids", None))
            bkw["time_ids"] = _align_ids(getattr(bc, "time_ids", None))
            boot = BootConfig(**bkw)
        # Paired rule for space×time (if either is given, both must be)
        if (boot.space_ids is None) ^ (boot.time_ids is None):
            raise ValueError("space_ids and time_ids must be provided jointly.")

        # Enforce: enumeration disabled for space×time (multiway) per boottest parity
        if (boot.space_ids is not None) and (boot.time_ids is not None):
            # make explicit in the BootConfig for reproducibility
            try:
                boot = dataclasses.replace(boot, enumeration_mode="disabled")
            except (TypeError, AttributeError):
                # fallback if BootConfig is not a dataclass instance
                boot.enumeration_mode = "disabled"

        if boot.cluster_ids is not None and len(boot.cluster_ids) != df_aug.shape[0]:
            msg = f"cluster_ids length {len(boot.cluster_ids)} must equal n_obs {df_aug.shape[0]} after build_cells"
            raise ValueError(msg)
        if boot.space_ids is not None and len(boot.space_ids) != df_aug.shape[0]:
            msg = f"space_ids length {len(boot.space_ids)} must equal n_obs {df_aug.shape[0]} after build_cells"
            raise ValueError(msg)
        if boot.time_ids is not None and len(boot.time_ids) != df_aug.shape[0]:
            msg = f"time_ids length {len(boot.time_ids)} must equal n_obs {df_aug.shape[0]} after build_cells"
            raise ValueError(msg)
        if boot.multiway_ids is not None:
            for i, mw in enumerate(boot.multiway_ids):
                if len(mw) != df_aug.shape[0]:
                    msg = f"multiway_ids[{i}] length {len(mw)} must equal n_obs {df_aug.shape[0]} after build_cells"
                    raise ValueError(msg)

        # multipliers: shape (n,B) - accept external_W in original df order and realign to sorted df
        if external_W is not None:
            W0 = np.asarray(external_W, dtype=np.float64, order="C")
            if W0.ndim != 2 or W0.shape[0] != df0.shape[0] or W0.shape[1] <= 0:
                msg = f"external_W must be (n={df0.shape[0]}) x B with B>0; got {W0.shape}."
                raise ValueError(msg)
            perm = df_aug["_origpos"].to_numpy(dtype=np.int64)
            W = W0[perm, :]
            W_df = pd.DataFrame(
                W,
                columns=[f"b{i}" for i in range(W.shape[1])],
            )
            boot_log = {
                "origin": "external_W",
                "note": "external multipliers were provided by the caller and then realigned to sorted df order",
            }
        else:
            W_df, boot_log = boot.make_multipliers(n_obs=df_aug.shape[0])
            W = W_df.to_numpy()

        # Prepare SSC configuration
        ssc_local = _normalize_ssc(ssc)
        fe_dof_val = int(ssc_local.get("fe_dof", 0)) if ssc_local else 0

        records: list[tuple[int, int, int, float, float]] = []
        phi_store: dict[tuple[int, int], np.ndarray] = {}
        idx_store: dict[tuple[int, int], np.ndarray] = {}
        balance_payloads: list[dict[str, Any]] = []
        skipped: list[tuple[int, int, str]] = []
        folds_used: list[int] = []

        gcol_all = pd.to_numeric(df_aug[self.cohort_name], errors="coerce").fillna(0).astype(int).to_numpy()
        tcol_all = df_aug[self.t_name].astype(int).to_numpy()

        for cell_key in cell_keys:
            if len(cell_key) == 3:
                g, t, base_time = cell_key
            else:
                g, t = cell_key
                base_time = int(g) - 1
            base_time = int(base_time)
            mask_t = tcol_all == int(t)
            ctrl_mask = spec.control_mask(df_aug, int(g), int(t), base_time, times_all)
            mask_cell = mask_t & ((gcol_all == int(g)) | ctrl_mask)
            sub = df_aug.loc[mask_cell, :].copy()
            if sub.shape[0] == 0:
                skipped.append((int(g), int(t), "empty cell"))
                continue

            if base_time not in base_map:
                skipped.append((int(g), int(t), "no base period"))
                continue
            sub["_Ybase"] = sub[self.id_name].map(base_map[base_time])
            sub["_dY"] = sub[self.y_name].to_numpy(dtype=np.float64) - sub[
                "_Ybase"
            ].to_numpy(dtype=np.float64)
            # Strict: drop rows with non-finite long-differences to avoid NaN propagation
            valid_dy = np.isfinite(sub["_dY"].to_numpy(dtype=np.float64))
            if not np.any(valid_dy):
                skipped.append((int(g), int(t), "no finite long-diff"))
                continue
            sub = sub.loc[valid_dy, :].copy()

            cohort_sub = pd.to_numeric(sub[self.cohort_name], errors="coerce").fillna(0).astype(int).to_numpy()
            Dg = ((cohort_sub > 0) & (cohort_sub == int(g))).astype(np.float64)
            n1 = int(Dg.sum())
            n0 = int(Dg.shape[0] - n1)
            if n1 == 0 or n0 == 0:
                skipped.append((int(g), int(t), "single class in Dg"))
                continue

            # Cross-fitting feasibility: allow full-sample fit (k_folds=1) or CV when enough data
            # Require at least one obs in each class (already ensured by n1,n0 checks above)
            # Adaptive fold count (at least 1 but never exceeding per-class counts)
            k_folds = min(max(1, self.n_folds), n1, n0)
            pihat, m0hat, m1hat = self._cross_fit(
                sub,
                x_ps.loc[sub.index, :],
                x_or.loc[sub.index, :],
                Dg=Dg,
                k_folds=k_folds,
                fold_seed=self.fold_seed,
                do_or=(self.method == "dr"),
            )
            folds_used.append(k_folds)

            dY_arr = sub["_dY"].to_numpy(dtype=np.float64).reshape(-1)
            if not np.any(np.isfinite(dY_arr)):
                skipped.append((int(g), int(t), "no finite dY"))
                continue

            # If external pihat supplied on original df order, realign once and use for this cell
            if self._pihat_external is not None:
                if self._pihat_external.shape[0] != df0.shape[0]:
                    msg = "External pihat length must equal original number of rows before sorting."
                    raise ValueError(msg)
                perm = df_aug["_origpos"].to_numpy(dtype=np.int64)
                pihat_sorted = self._pihat_external[perm]
                pext = pihat_sorted[sub.index.to_numpy()]
                if pext.shape[0] != sub.shape[0]:
                    msg = "Length mismatch: external pihat does not align with current cell rows."
                    raise ValueError(msg)
                pihat = pext

            orig_idx = sub.index.to_numpy(dtype=np.int64, copy=True)
            if self.method == "dr":
                psi, att, n_treat, keep_index, balance_info = self._dr_score(
                    Dg=Dg,
                    dY=dY_arr,
                    pihat=pihat,
                    m0hat=m0hat,
                    m1hat=m1hat,
                    hajek=self.hajek,
                    hajek_eps=self.hajek_eps,
                    orig_index=orig_idx,
                )
            else:
                psi, att, n_treat, keep_index, balance_info = self._ipw_score(
                    Dg=Dg,
                    dY=dY_arr,
                    pihat=pihat,
                    hajek=self.hajek,
                    hajek_eps=self.hajek_eps,
                    orig_index=orig_idx,
                )
            # If trimming removed a class or all rows, skip this cell
            if psi.size == 0 or n_treat == 0:
                skipped.append(
                    (int(g), int(t), "trim removed class or empty after trimming"),
                )
                continue
            if balance_info is not None:
                keep_mask = np.isin(sub.index.to_numpy(dtype=np.int64), keep_index)
                if keep_mask.sum() == balance_info["w_after"].shape[0]:
                    X_bal = x_ps.loc[sub.index, :].to_numpy(
                        dtype=np.float64, copy=True,
                    )[keep_mask, :]
                    treat_bal = Dg[keep_mask].astype(int, copy=False)
                    payload_name = f"g{int(g)}_t{int(t)}_{self.method.upper()}"
                    balance_payloads.append(
                        {
                            "name": payload_name,
                            "X": X_bal,
                            "group": treat_bal,
                            "w_before": np.asarray(
                                balance_info["w_before"], dtype=np.float64,
                            ),
                            "w_after": np.asarray(
                                balance_info["w_after"], dtype=np.float64,
                            ),
                            "covariate_names": list(x_ps.columns),
                            "metadata": {
                                "g": int(g),
                                "t": int(t),
                                "method": self.method,
                            },
                        },
                    )

            # No additional first-stage IF contributions beyond cross-fitting in strict mode

            # Apply SSC to influence function `psi`
            # For DR-DID with cross-fitting, the dimensionality of nuisance models does not
            # enter the asymptotic variance (double robustness/orthogonality).
            # We enforce k=0 (plus any FE degrees of freedom if provided) to avoid
            # ad-hoc penalization that lacks theoretical justification.
            # (User feedback: strict theory alignment required).

            # Local clusters: see eventstudy_cs.py logic
            local_clusters = None
            if boot.cluster_ids is not None:
                local_clusters = np.asarray(boot.cluster_ids)[sub.index]

            ssc_factor = compute_ssc_correction(
                n=sub.shape[0],
                k=0 + fe_dof_val,
                clusters=local_clusters,
                ssc=ssc_local
            )
            if ssc_factor != 1.0:
                psi *= ssc_factor

            tau_val = event_tau(t, g, t2pos)
            records.append((int(g), int(t), tau_val, float(att), float(n_treat)))
            phi_store[(int(g), int(t))] = psi.reshape(-1)
            # Store row labels that actually entered psi after trimming
            idx_store[(int(g), int(t))] = keep_index

        if not records:
            msg = "No valid (g,t) cells after screening."
            raise RuntimeError(msg)

        att_gt = (
            pd.DataFrame(records, columns=["g", "t", "tau", "att", "n_treat"])
            .sort_values(["g", "t"])
            .reset_index(drop=True)
        )
        B = W.shape[1]
        att_star = []
        for g, t in att_gt[["g", "t"]].itertuples(index=False, name=None):
            idx = idx_store[(g, t)]
            phi = phi_store[(g, t)]
            incr = la.dot(phi.reshape(1, -1), W[idx, :])
            point = float(
                att_gt.loc[(att_gt["g"] == g) & (att_gt["t"] == t), "att"].iloc[0],
            )
            att_star.append(point + incr.reshape(-1))

        att_star = (
            np.vstack(att_star)
            if len(att_star) > 0
            else np.empty((0, W.shape[1]), dtype=np.float64)
        )

        att_tau, rows_by_tau = aggregate_tau(
            att_gt[["g", "t", "tau", "att", "n_treat"]], self.base_tau,
        )
        # --- τ aggregation weights (group/equal/treated_t) ---
        # Align with R did::aggte 'group' weighting: count unique IDs per cohort (not conditional on base period)
        group_size: dict[int, int] = {}
        if att_gt.shape[0] > 0:
            for g in np.sort(att_gt["g"].unique()):
                ids_g = df_aug.loc[
                    df_aug[self.cohort_name].astype(int).to_numpy() == int(g),
                    self.id_name,
                ].unique()
                group_size[int(g)] = len(ids_g)
        tau_vals = sorted(att_tau["tau"].unique())
        att_vals: list[float] = []
        for tau in tau_vals:
            rows = rows_by_tau[tau]
            atts = att_gt.loc[rows, "att"].to_numpy(dtype=np.float64)
            gs = att_gt.loc[rows, "g"].to_numpy(dtype=int)
            if self.tau_weight == "equal":
                w = np.ones_like(atts, dtype=np.float64)
            elif self.tau_weight == "group":
                w = np.array([group_size.get(int(g), 0) for g in gs], dtype=np.float64)
            else:  # treated_t
                w = att_gt.loc[rows, "n_treat"].to_numpy(dtype=np.float64)
            w = w / np.sum(w) if np.sum(w) > 0 else np.ones_like(w) / max(len(w), 1)
            att_vals.append(float(np.sum(w * atts)))
        att_tau["att"] = att_vals

        att_tau_star = np.empty((att_tau.shape[0], B), dtype=np.float64)
        for j, tau in enumerate(att_tau["tau"].tolist()):
            rows = rows_by_tau[tau]
            if rows.size == 0:
                att_tau_star[j, :] = np.nan
                continue
            if self.tau_weight == "equal":
                w = np.ones((len(rows),), dtype=np.float64)
            elif self.tau_weight == "group":
                gs = att_gt.iloc[rows]["g"].to_numpy(dtype=int)
                w = np.array([group_size.get(int(g), 0) for g in gs], dtype=np.float64)
            else:  # treated_t
                w = att_gt.iloc[rows]["n_treat"].to_numpy(dtype=np.float64)
            if np.sum(w) <= 0:
                att_tau_star[j, :] = np.nan
            else:
                w = w / np.sum(w)
                att_tau_star[j, :] = (w.reshape(-1, 1) * att_star[rows, :]).sum(axis=0)

        # Compute uniform bands using centralized helper; ensure bootstrap studentization
        att_tau_plot, band_pre, band_post, band_full = compute_uniform_bands(
            att_tau,
            att_tau_star,
            base_tau=self.base_tau,
            alpha=self.alpha,
        )

        # --- PostATT (period-aggregated over post-τ) with bootstrap scalar CI ---
        # Post scalar aggregated ATT using the same group_size map and uniform-band helper
        post_mask_tau = att_tau_plot["tau"] > self.base_tau
        PostATT_se = float("nan")
        if np.any(post_mask_tau):
            post_att, (lo_s, hi_s), PostATT_se = post_aggregate_uniform_band(
                att_gt=att_gt,
                att_tau=att_tau_plot[["tau", "att"]],
                att_tau_star=att_tau_star,
                base_tau=self.base_tau,
                tau_weight=self.tau_weight,
                alpha=self.alpha,
                group_size_map=group_size,
            )
            PostATT = post_att
            post_scalar = pd.DataFrame(
                {"lower": pd.Series([lo_s]), "upper": pd.Series([hi_s])},
            )
        else:
            PostATT = float("nan")
            post_scalar = pd.DataFrame(
                {"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)},
            )

        from collections import Counter

        skipped_summary = dict(Counter(reason for _, _, reason in skipped))

        # Build model_info with PS trimming/clipping details
        model_info: dict[str, object] = {
            "Estimator": "EventStudy: DR-DID",
            "ControlGroup": self.control_group,
            "CrossFitting": f"Adaptive per-cell folds (min={min(folds_used) if folds_used else 'NA'}, max={max(folds_used) if folds_used else 'NA'})",
            "Normalization": "Hajek",
            "CenteringApplied": True,
            "Bootstrap": f"IF multiplier (shared W), policy={getattr(boot, 'policy', None)}, dist={getattr(boot, 'dist', None)}",
            "B": B,
            # Harmonize with summary expectations: it reads 'CenterAt'
            # to decide the base τ and side-specific bands.
            "CenterAt": self.base_tau,
            # Keep BaseTau as well for backward compatibility/diagnostics
            "BaseTau": self.base_tau,
            "PostATT": PostATT,
            "PostATT_se": PostATT_se,
            "SkippedCells": skipped,
            "SkippedSummary": skipped_summary,
            "FoldSeed": self.fold_seed,
            "TauWeight": self.tau_weight,
            "MultiplierSharedAcrossCells": True,
            # Be explicit about band type to satisfy strict gates when present
            "BandType": "uniform",
        }

        # Log PS trimming details
        if self.trim_ps > 0.0:
            model_info["PSTrimming"] = {
                "trim_ps": self.trim_ps,
                "trim_mode": self.trim_mode,
                "trim_bounds": (self.trim_ps, 1.0 - self.trim_ps),
            }

        # Log external pihat clipping if applied
        if self._pihat_clipped_bounds is not None:
            eps, one_minus_eps, n_clipped = self._pihat_clipped_bounds
            model_info["ExternalPihatClipping"] = {
                "applied": True,
                "bounds": (eps, one_minus_eps),
                "n_values_clipped": n_clipped,
                "reason": "External pihat forced to [hajek_eps, 1-hajek_eps] for numerical stability",
            }

        # unique rows used across all cells (no double counting)
        if idx_store:
            total_used = int(
                np.unique(np.concatenate([idx_store[k] for k in idx_store])).size,
            )
        else:
            total_used = 0
        bands = {
            "pre": pd.DataFrame({"lower": band_pre[0], "upper": band_pre[1]}),
            "post": pd.DataFrame({"lower": band_post[0], "upper": band_post[1]}),
            "full": pd.DataFrame({"lower": band_full[0], "upper": band_full[1]}),
            "post_scalar": post_scalar,
            "__meta__": {
                "origin": "bootstrap",
                "policy": str(getattr(boot, "policy", "bootstrap")),
                "dist": getattr(boot, "dist", None),
                "preset": getattr(boot, "preset", None),
                "kind": "uniform",
                "level": 95,
                "B": int(B),
                "estimator": "eventstudy",
                "context": (
                    "did" if str(getattr(boot, "preset", "")).lower() == "did" else "eventstudy"
                ),
            },
        }

        extra = {
            "att_gt": att_gt,
            "att_tau": att_tau_plot,  # For consistency with other ES estimators
            "att_tau_star": att_tau_star,
            "W_multipliers_inference": W_df,
            "multipliers_log": boot_log,
            "group_size_map": group_size,
            # provenance metadata per project policy
            "bands_source": "bootstrap",
            "se_source": "bootstrap",
            "boot_config": boot,
        }
        if balance_payloads:
            sanitized: list[dict[str, Any]] = [
                {
                    "name": payload.get("name"),
                    "X": np.asarray(payload.get("X"), dtype=np.float64),
                    "group": np.asarray(payload.get("group"), dtype=int),
                    "w_before": np.asarray(payload.get("w_before"), dtype=np.float64),
                    "w_after": np.asarray(payload.get("w_after"), dtype=np.float64),
                    "covariate_names": list(payload.get("covariate_names", [])),
                    "metadata": payload.get("metadata", {}),
                }
                for payload in balance_payloads
            ]
            extra["balance_plot"] = {
                "payloads": sanitized,
                "default": sanitized[0]["name"] if sanitized else None,
            }

        se_series = pd.Series(
            bt.bootstrap_se(att_tau_star),
            index=att_tau_plot["tau"].astype(int).to_numpy(),
        )
        if int(self.base_tau) in se_series.index:
            se_series.loc[int(self.base_tau)] = 0.0

        return EstimationResult(
            params=att_tau_plot.set_index("tau")["att"],
            se=se_series,
            bands=bands,
            n_obs=total_used,
            model_info=model_info,
            extra=extra,
        )

    def _cross_fit(  # noqa: PLR0913
        self,
        sub: pd.DataFrame,
        x_ps: pd.DataFrame,
        x_or: pd.DataFrame,
        *,
        Dg: np.ndarray,
        k_folds: int,
        fold_seed: int | None = None,
        do_or: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cross-fit propensity score and outcome regression with adaptive folds."""
        n = sub.shape[0]
        pihat = np.zeros(n, dtype=np.float64)
        m0hat = np.zeros(n, dtype=np.float64)
        m1hat = np.zeros(n, dtype=np.float64)
        yvals = sub["_dY"].to_numpy(dtype=np.float64)

        if k_folds == 1:
            # compat full-sample fit: train on whole sample and predict on whole sample
            ps = clone(self.ps_learner)
            # Promote ConvergenceWarning to error to enforce strict convergence
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("error", category=ConvergenceWarning)
                ps.fit(x_ps, Dg)
            probs = ps.predict_proba(x_ps)
            if probs.shape[1] < 2:
                msg = "PS learner did not produce two-column predict_proba; check learner configuration."
                raise RuntimeError(msg)
            pihat[:] = probs[:, 1]
            if do_or:
                # Fit OR separately on treated and untreated groups if available
                if np.any(Dg == 0):
                    or0 = clone(self.or_learner)
                    or0.fit(x_or.loc[Dg == 0, :], yvals[Dg == 0])
                    m0hat[:] = or0.predict(x_or)
                if np.any(Dg == 1):
                    or1 = clone(self.or_learner)
                    or1.fit(x_or.loc[Dg == 1, :], yvals[Dg == 1])
                    m1hat[:] = or1.predict(x_or)
            return pihat, m0hat, m1hat

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=fold_seed)
        for train_idx, test_idx in skf.split(np.arange(n), Dg):
            # PS: P(D=1 | X) -- clone learner per fold to avoid state overwrite
            ps = clone(self.ps_learner)
            if len(train_idx) == 0:
                msg = f"Empty training set in cross-fit fold (fold size train=0, test={len(test_idx)}); increase sample or reduce folds."
                raise RuntimeError(msg)
            # Promote ConvergenceWarning to error to enforce strict convergence
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("error", category=ConvergenceWarning)
                ps.fit(x_ps.iloc[train_idx, :], Dg[train_idx])
            # predict_proba shape handling
            probs = ps.predict_proba(x_ps.iloc[test_idx, :])
            if probs.shape[1] < 2:
                msg = "PS learner did not produce two-column predict_proba; check learner configuration."
                raise RuntimeError(msg)
            pihat[test_idx] = probs[:, 1]

            if do_or:
                # OR: E[ΔY | X, D=0] -- clone and fit only on untreated training rows
                untreated_mask = Dg[train_idx] == 0
                if np.any(untreated_mask):
                    or0 = clone(self.or_learner)
                    tr_idx0 = train_idx[untreated_mask]
                    if len(tr_idx0) == 0:
                        msg = f"No untreated observations in training fold for OR(0) (fold train size={len(train_idx)}, treated_in_train={int((Dg[train_idx] == 1).sum())}); increase sample or reduce folds."
                        raise RuntimeError(msg)
                    or0.fit(x_or.iloc[tr_idx0, :], yvals[tr_idx0])
                    m0hat[test_idx] = or0.predict(x_or.iloc[test_idx, :])

                # OR: E[ΔY | X, D=1] -- clone and fit only on treated training rows
                treated_mask = Dg[train_idx] == 1
                if np.any(treated_mask):
                    or1 = clone(self.or_learner)
                    tr_idx1 = train_idx[treated_mask]
                    if len(tr_idx1) == 0:
                        msg = f"No treated observations in training fold for OR(1) (fold train size={len(train_idx)}, treated_in_train={int((Dg[train_idx] == 1).sum())}); increase sample or reduce folds."
                        raise RuntimeError(msg)
                    or1.fit(x_or.iloc[tr_idx1, :], yvals[tr_idx1])
                    m1hat[test_idx] = or1.predict(x_or.iloc[test_idx, :])
        return pihat, m0hat, m1hat

    def _dr_score(  # noqa: PLR0913
        self,
        *,
        Dg: np.ndarray,
        dY: np.ndarray,
        pihat: np.ndarray,
        m0hat: np.ndarray,
        m1hat: np.ndarray,
        hajek: bool,
        hajek_eps: float,
        orig_index: np.ndarray,
    ) -> tuple[np.ndarray, float, int, np.ndarray, dict[str, np.ndarray] | None]:
        """Compute Hájek-normalized DR influence function and ATT for a cell.

        Returns (psi, att, n_treat) where psi is centered and suitable for multiplier bootstrap.
        """
        n_treat = int(Dg.sum())
        if n_treat == 0:
            return np.zeros(0, dtype=np.float64), 0.0, 0, orig_index[:0], None

        # ensure 1d arrays
        dY = dY.reshape(-1)
        pihat = pihat.reshape(-1)
        m0hat = m0hat.reshape(-1)
        m1hat = m1hat.reshape(-1)

        keep_index = orig_index
        balance_info: dict[str, np.ndarray] | None = None
        # PS trimming/clip handling
        ph = pihat.reshape(-1)
        if self.trim_ps > 0.0:
            if self.trim_mode == "clip":
                # Clip propensity scores to [trim_ps, 1 - trim_ps] before further processing
                ph = np.clip(ph, self.trim_ps, 1.0 - self.trim_ps)
            elif self.trim_mode == "drop":
                # Drop observations with PS outside [trim_ps, 1 - trim_ps]
                keep = (ph >= self.trim_ps) & (ph <= 1.0 - self.trim_ps)
                Dg = Dg[keep]
                dY = dY[keep]
                m0hat = m0hat[keep]
                m1hat = m1hat[keep]
                ph = ph[keep]
                keep_index = orig_index[keep]

        # After potential trim/drop, apply Hajek epsilon clipping
        p_safe = np.clip(ph, hajek_eps, 1.0 - hajek_eps)
        n = len(dY)  # Updated n after potential drop
        n_treat = int(Dg.sum())  # UPDATED after potential drop
        if n == 0 or n_treat in {0, n}:
            return np.zeros(0, dtype=np.float64), 0.0, 0, keep_index[:0], None

        w1 = Dg.reshape(-1)
        w0 = ((1.0 - Dg) * p_safe / (1.0 - p_safe)).reshape(-1)

        if hajek:
            den1 = float(w1.sum()) if w1.sum() > 0 else 1.0
            den0 = float(w0.sum()) if w0.sum() > 0 else 1.0
        else:
            den1 = max(float(n_treat), 1.0)
            den0 = max(float(n - n_treat), 1.0)

        # Hájek-weighted residual means
        R1 = float((w1 * (dY - m1hat)).sum() / den1)
        R0 = float((w0 * (dY - m0hat)).sum() / den0)

        # Augmentation term: E[m1 - m0 | D=1] estimated among treated (simple sample mean)
        mu1 = float((Dg * (m1hat - m0hat)).sum() / n_treat) if n_treat > 0 else 0.0

        att = mu1 + R1 - R0

        # Build centered influence function consistent with Hájek normalization
        psi_t = (w1 * (dY - m1hat)) / den1 - R1 * (w1 / den1)
        psi_c = -(w0 * (dY - m0hat)) / den0 + R0 * (w0 / den0)

        aug = np.zeros(n, dtype=np.float64)
        if n_treat > 0:
            aug_mask = Dg == 1
            aug_vals = (m1hat - m0hat)[aug_mask] - mu1
            aug[aug_mask] = aug_vals / float(n_treat)

        psi = psi_t + psi_c + aug
        # Centering: subtract mean (Hájek-style)
        psi = psi - float(la.col_mean(psi.reshape(-1, 1))[0])
        balance_info = {
            "w_before": np.ones_like(w1, dtype=np.float64),
            "w_after": np.where(Dg == 1, w1, w0),
        }

        return psi, float(att), n_treat, keep_index, balance_info

    def _ipw_score(  # noqa: PLR0913
        self,
        *,
        Dg: np.ndarray,
        dY: np.ndarray,
        pihat: np.ndarray,
        hajek: bool,
        hajek_eps: float,
        orig_index: np.ndarray,
    ) -> tuple[np.ndarray, float, int, np.ndarray, dict[str, np.ndarray] | None]:
        """Hájek-normalized IPW score (PS-only) and ATT."""
        n_treat = int(Dg.sum())
        if n_treat == 0:
            return np.zeros(0, dtype=np.float64), 0.0, 0, orig_index[:0], None

        dY = dY.reshape(-1)
        ph = pihat.reshape(-1)
        keep_index = orig_index
        if self.trim_ps > 0.0:
            if self.trim_mode == "clip":
                ph = np.clip(ph, self.trim_ps, 1.0 - self.trim_ps)
            elif self.trim_mode == "drop":
                keep = (ph >= self.trim_ps) & (ph <= 1.0 - self.trim_ps)
                Dg = Dg[keep]
                dY = dY[keep]
                ph = ph[keep]
                keep_index = orig_index[keep]
        p_safe = np.clip(ph, hajek_eps, 1.0 - hajek_eps)
        w1 = Dg.reshape(-1)
        w0 = ((1.0 - Dg) * p_safe / (1.0 - p_safe)).reshape(-1)
        if hajek:
            den1 = float(w1.sum()) if w1.sum() > 0 else 1.0
            den0 = float(w0.sum()) if w0.sum() > 0 else 1.0
        else:
            den1 = max(float(n_treat), 1.0)
            den0 = max(float(Dg.size - n_treat), 1.0)
        n_treat = int(Dg.sum())
        if dY.size == 0 or n_treat in {0, Dg.size}:
            return np.zeros(0, dtype=np.float64), 0.0, 0, keep_index[:0], None
        mu1 = float((w1 * dY).sum() / den1)
        mu0 = float((w0 * dY).sum() / den0)
        att = mu1 - mu0
        psi = (w1 * (dY - mu1)) / den1 - (w0 * (dY - mu0)) / den0
        # Centering: subtract mean (Hájek-style)
        psi = psi - float(la.col_mean(psi.reshape(-1, 1))[0])
        balance_info = {
            "w_after": np.where(Dg == 1, w1, w0),
            "w_before": np.ones_like(w1, dtype=np.float64),
        }
        return psi, float(att), n_treat, keep_index, balance_info

    def _ols_control_info(
        self,
        Dg: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Control-only OLS: return (G_c^{-1}, e_c, xbar_1)."""
        treated = Dg == 1
        Xc = X[~treated, :]
        yc = y[~treated].reshape(-1, 1)
        beta = la.solve(Xc, yc, method="qr")
        e_c = (yc - la.dot(Xc, beta)).reshape(-1)
        Gc = la.dot(Xc.T, Xc)
        Gc_inv = la.solve(Gc, la.eye(Gc.shape[0]), sym_pos=True)
        xbar1 = la.col_mean(X[treated, :]).reshape(-1)
        return Gc_inv, e_c, xbar1

    @classmethod
    def from_formula(  # noqa: PLR0913
        cls,
        *,
        data: pd.DataFrame,
        formula_ps: str | None = None,
        formula_or: str | None = None,
        id_name: str | None = None,
        time: str | None = None,
        options: str | None = None,
        W_dict: dict | None = None,
        boot: BootConfig | None = None,
        **kwargs,
    ) -> DREventStudy:
        """Instantiate the estimator from formula specifications without fitting."""
        parsed_ps = None
        parsed_or = None
        if formula_ps is not None:
            parser_ps = FormulaParser(data, id_name=id_name, t_name=time, W_dict=W_dict)
            parsed_ps = parser_ps.parse(formula_ps, iv=None, options=options)
        if formula_or is not None:
            parser_or = FormulaParser(data, id_name=id_name, t_name=time, W_dict=W_dict)
            parsed_or = parser_or.parse(formula_or, iv=None, options=options)

        if parsed_ps is not None and parsed_or is not None:
            idx_ps = list(parsed_ps.get("row_index_used", data.index))
            idx_or = list(parsed_or.get("row_index_used", data.index))
            idx_use = [i for i in idx_ps if i in idx_or]
        elif parsed_ps is not None:
            idx_use = list(parsed_ps.get("row_index_used", data.index))
        elif parsed_or is not None:
            idx_use = list(parsed_or.get("row_index_used", data.index))
        else:
            idx_use = list(data.index)

        df_use = data.loc[idx_use]

        def _design_from(parsed: dict | None) -> pd.DataFrame | None:
            if parsed is None:
                return None
            X = parsed.get("X")
            names = parsed.get("var_names")
            if X is None or names is None:
                return None
            dfX = pd.DataFrame(
                X, index=parsed.get("row_index_used", data.index), columns=names,
            )
            return dfX.loc[df_use.index, :]

        x_ps = _design_from(parsed_ps)
        x_or = _design_from(parsed_or)

        cluster_ids = None
        if parsed_ps is not None:
            cluster_ids = parsed_ps.get("cluster_ids_used")
        if cluster_ids is None and parsed_or is not None:
            cluster_ids = parsed_or.get("cluster_ids_used")

        boot_cfg = boot
        if boot_cfg is None and cluster_ids is not None:
            boot_cfg = BootConfig(
                n_boot=bt.DEFAULT_BOOTSTRAP_ITERATIONS,
                cluster_ids=cluster_ids,
                policy="boottest",
                enumeration_mode="boottest",
                dist="standard_normal",
            )

        init_kwargs = dict(kwargs)
        if id_name is not None:
            init_kwargs.setdefault("id_name", id_name)
        if time is not None:
            init_kwargs.setdefault("t_name", time)
        if boot_cfg is not None:
            init_kwargs["boot"] = boot_cfg

        est = cls(**init_kwargs)

        meta = FormulaMetadata(
            formula=None,
            row_index=idx_use,
            cluster_ids=cluster_ids,
            attrs={
                "_formula_ps": formula_ps,
                "_formula_or": formula_or,
                "_formula_df": df_use,
                "_formula_x_ps": x_ps,
                "_formula_x_or": x_or,
            },
        )
        if boot_cfg is not None:
            meta.attrs["_boot_from_formula"] = boot_cfg
        attach_formula_metadata(est, meta)
        return est

    @classmethod
    def fit_from_formula(  # noqa: PLR0913
        cls,
        *,
        data: pd.DataFrame,
        formula_ps: str | None = None,
        formula_or: str | None = None,
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
            formula_ps=formula_ps,
            formula_or=formula_or,
            id_name=id_name,
            time=time,
            options=options,
            W_dict=W_dict,
            boot=boot,
            **kwargs,
        )
        extra = fit_kwargs or {}
        return est.fit(**extra)
