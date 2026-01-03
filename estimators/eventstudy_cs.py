"""Callaway-Sant'Anna (2021) event-study estimator.

This module implements the long-difference DiD estimator with staggered adoption,
providing group-time ATTs, event-time aggregation, and simultaneous confidence bands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from lineareg.core import linalg as la

from lineareg.core import bootstrap as bt
from lineareg.core.bootstrap import compute_ssc_correction, _normalize_ssc
from lineareg.estimators.base import (
    BootConfig,
    EstimationResult,
    attach_formula_metadata,
    prepare_formula_environment,
)
from lineareg.utils.formula import FormulaParser
try:  # optional: used only when cov_method in {"ipw","dr"}
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
except Exception:  # pragma: no cover - allow absence but raise when actually needed
    LogisticRegression = None  # type: ignore[assignment]
    ConvergenceWarning = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Sequence
else:
    Sequence = tuple  # type: ignore[assignment]

# Long-differences (Y_t - Y_{g-1}) are constructed per-(g,t) cell.
# Units missing the base period g-1 are excluded from that cell.
# This implementation mirrors Callaway & Sant'Anna (2021) / did / csdid
# exactly for deterministic reproducibility (no approximations).


from lineareg.utils.eventstudy_helpers import (
    ESCellSpec,
    aggregate_tau,
    build_cells,
    pret_for_cohort,
    prev_time,
)
from lineareg.utils.helpers import event_tau, time_to_pos



from lineareg.core.inference import (
    compute_uniform_bands,
    post_aggregate_uniform_band,
)


class CallawaySantAnnaES:
    """Callaway-Sant'Anna (2021) estimator.

    Estimates group-time average treatment effects ATT(g,t) using long differences.
    Supports IPW, outcome regression, and doubly-robust covariate adjustment.
    Inference via multiplier bootstrap.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        id_name: str,
        t_name: str,
        cohort_name: str,
        y_name: str,
        event_time_name: str | None = None,
        control_group: str = "notyet",
        center_at: int = -1,
        anticipation: int = 0,
        base_period: str = "varying",
        boot: BootConfig | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        alpha: float = 0.05,
        tau_weight: str = "group",
        covariate_names: Sequence[str] | None = None,
        cov_method: str = "none",
        trim_ps: float = 0.0,
        trim_mode: str = "clip",
        balance_e: int | None = None,
        enforce_tau_intersection: bool = False,
    ) -> None:
        self.id_name = str(id_name)
        self.t_name = str(t_name)
        self.cohort_name = str(cohort_name)
        self.y_name = str(y_name)
        self.event_time_name = None if event_time_name is None else str(event_time_name)
        self.control_group = control_group
        self.center_at = int(center_at)
        self.anticipation = int(anticipation)
        base_period_norm = str(base_period).lower()
        if base_period_norm not in {"varying", "universal"}:
            raise ValueError("base_period must be 'varying' or 'universal'")
        self.base_period = base_period_norm
        self.alpha = float(alpha)
        self.tau_weight = str(tau_weight).lower()
        if self.tau_weight not in {"group", "equal", "treated_t"}:
            raise ValueError("tau_weight must be one of {'group','equal','treated_t'}.")
        self.covariate_names = (
            list(covariate_names) if covariate_names is not None else None
        )
        self.cov_method = str(cov_method).lower()
        self.trim_ps = float(trim_ps)
        self.trim_mode = str(trim_mode).lower()
        if self.trim_mode not in {"clip", "drop"}:
            raise ValueError("trim_mode must be 'clip' or 'drop'")
        self.balance_e = balance_e
        self.enforce_tau_intersection = bool(enforce_tau_intersection)

        self.boot = boot if boot is not None else BootConfig()
        self.cluster_ids = cluster_ids
        self.space_ids = space_ids
        self.time_ids = time_ids

    @classmethod
    def from_formula(  # noqa: PLR0913
        cls,
        *,
        data: pd.DataFrame,
        formula: str | None = None,
        id_name: str | None = None,
        time: str | None = None,
        options: str | None = None,
        W_dict: dict | None = None,
        boot: BootConfig | None = None,
        **kwargs,
    ) -> CallawaySantAnnaES:
        """Construct an estimator instance from a formula without fitting.

        The returned object retains parser metadata (formula string, retained rows,
        cluster identifiers) and a view of the filtered DataFrame so that
        :meth:`fit` can be called without arguments.
        """
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
                "policy": "boottest",
                "enumeration_mode": "boottest",
                "dist": "standard_normal",
            },
        )
        if boot_eff is not None:
            meta.attrs.setdefault("_boot_from_formula", boot_eff)
        meta.attrs["_formula_df"] = df_use

        init_kwargs = dict(kwargs)
        if id_name is not None and "id_name" not in init_kwargs:
            init_kwargs["id_name"] = id_name
        if time is not None and "t_name" not in init_kwargs:
            init_kwargs["t_name"] = time
        boot_to_use = boot_eff if boot_eff is not None else boot
        if boot_to_use is not None:
            init_kwargs["boot"] = boot_to_use

        est = cls(**init_kwargs)
        attach_formula_metadata(est, meta)
        est._formula_df = df_use
        est._formula_parser = parsed
        return est

    @classmethod
    def fit_from_formula(  # noqa: PLR0913
        cls,
        *,
        data: pd.DataFrame,
        formula: str | None = None,
        id_name: str | None = None,
        time: str | None = None,
        options: str | None = None,
        W_dict: dict | None = None,
        boot: BootConfig | None = None,
        fit_kwargs: dict | None = None,
        **kwargs,
    ) -> EstimationResult:
        """One-shot convenience wrapper mirroring the legacy API."""
        est = cls.from_formula(
            data=data,
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

    def fit(
        self,
        df: pd.DataFrame | None = None,
        *,
        external_W: np.ndarray | None = None,
        ssc: dict[str, str | int | float | bool] | None = None,
    ) -> EstimationResult:

        """Fit Callaway-Sant'Anna estimator.

        Computes ATT(g,t) for all valid group-time pairs and aggregates
        results.
        """
        if df is None:
            df = getattr(self, "_formula_df", None)
            if df is None:
                msg = "Provide df explicitly or instantiate the estimator via from_formula()."
                raise ValueError(msg)
        else:
            self._formula_df = df

        # Keep a stable mapping to the input row order so that any external
        # ID arrays (cluster/space/time) can be realigned after build_cells
        # potentially reorders and/or drops rows.
        df0 = df.copy()
        df0["_origpos"] = np.arange(df0.shape[0], dtype=np.int64)

        spec = ESCellSpec(
            id_name=self.id_name,
            t_name=self.t_name,
            cohort_name=self.cohort_name,
            y_name=self.y_name,
            event_time_name=self.event_time_name,
            control_group=self.control_group,
            center_at=self.center_at,
            covariate_names=self.covariate_names,
            cov_method=self.cov_method,
            anticipation=self.anticipation,
            base_period=self.base_period,
        )

        df_aug, cell_keys, cell_meta = build_cells(df0, spec)
        times_all = cell_meta.get("times", np.sort(df_aug[self.t_name].unique()))
        pret_map = cell_meta.get("pret_map", {})

        def _align_ids(arr_like) -> np.ndarray | None:
            if arr_like is None:
                return None
            arr = np.asarray(arr_like)
            if arr.shape[0] != df0.shape[0]:
                raise ValueError(
                    "Provided IDs must have length equal to the number of rows in df passed to fit().",
                )
            perm = df_aug["_origpos"].to_numpy(dtype=np.int64)
            return np.asarray(arr, dtype=object)[perm]

        from dataclasses import replace as dc_replace
        boot_eff = self.boot
        if boot_eff is not None:
            if isinstance(self.cluster_ids, str) and self.cluster_ids in df_aug.columns:
                boot_eff = dc_replace(boot_eff, cluster_ids=df_aug[self.cluster_ids].values)
            elif self.cluster_ids is not None and not isinstance(self.cluster_ids, str):
                boot_eff = dc_replace(boot_eff, cluster_ids=_align_ids(self.cluster_ids))
            elif isinstance(self.space_ids, str) and self.space_ids in df_aug.columns:
                space_vals = df_aug[self.space_ids].values
                time_vals = df_aug[self.time_ids].values if isinstance(self.time_ids, str) else None
                boot_eff = dc_replace(boot_eff, space_ids=space_vals, time_ids=time_vals)
            elif (self.space_ids is not None) and (self.time_ids is not None) and (not isinstance(self.space_ids, str)) and (not isinstance(self.time_ids, str)):
                boot_eff = dc_replace(
                    boot_eff,
                    space_ids=_align_ids(self.space_ids),
                    time_ids=_align_ids(self.time_ids),
                )

            # If boot already carries IDs aligned to df0, subset them to df_aug.
            # (If already aligned to df_aug, leave unchanged.)
            if getattr(boot_eff, "cluster_ids", None) is not None and (len(boot_eff.cluster_ids) == df0.shape[0]):
                boot_eff = dc_replace(boot_eff, cluster_ids=_align_ids(boot_eff.cluster_ids))
            if (getattr(boot_eff, "space_ids", None) is not None) and (len(boot_eff.space_ids) == df0.shape[0]):
                boot_eff = dc_replace(boot_eff, space_ids=_align_ids(boot_eff.space_ids))
            if (getattr(boot_eff, "time_ids", None) is not None) and (len(boot_eff.time_ids) == df0.shape[0]):
                boot_eff = dc_replace(boot_eff, time_ids=_align_ids(boot_eff.time_ids))

        # Paired rule for space×time clustering.
        if (getattr(boot_eff, "space_ids", None) is None) ^ (getattr(boot_eff, "time_ids", None) is None):
            raise ValueError("space_ids and time_ids must be provided jointly.")

        if external_W is not None:
            W = np.asarray(external_W, dtype=np.float64, order="C")
            if W.ndim != 2 or W.shape[1] <= 0:
                raise ValueError("external_W must be 2D with B>0.")
            if W.shape[0] != df_aug.shape[0]:
                msg = f"external_W has {W.shape[0]} rows but df_aug has {df_aug.shape[0]} rows"
                raise ValueError(msg)
            W_df = pd.DataFrame(W, columns=[f"b{i}" for i in range(W.shape[1])])
            _boot_log = {"source": "external"}
        else:
            if boot_eff is None:
                raise ValueError("boot configuration is required when external_W is not provided.")
            W_df, _boot_log = boot_eff.make_multipliers(n_obs=df_aug.shape[0])
            W = W_df.to_numpy()

        if W.shape[1] < 2:
            raise ValueError("Bootstrap multipliers must have B>=2 for inference.")

        # Prepare SSC configuration
        ssc_local = _normalize_ssc(ssc)
        # Check if ssc contains explicit fe_dof (not typical for CS, but supported conceptually)
        fe_dof_val = int(ssc_local.get("fe_dof", 0)) if ssc_local else 0

        att_rows: list[tuple[int, int, int, float, int]] = []
        att_star_cells: list[np.ndarray] = []
        used_index: list[np.ndarray] = []
        cells_order: list[tuple[int, int]] = []

        # ---- CS long-diff estimator per cell with optional covariates (reg/ipw/dr) ----
        # Build base maps Y_{g-1}(i) by calendar time once
        base_map: dict[int, pd.Series] = {}
        cov_base_map: dict[int, pd.DataFrame] = {}
        times_all = np.sort(df_aug[self.t_name].unique())
        t2pos = time_to_pos(times_all)
        for t0 in times_all:
            at_t0 = df_aug.loc[
                df_aug[self.t_name] == int(t0), [self.id_name, self.y_name],
            ]
            base_map[int(t0)] = at_t0.set_index(self.id_name)[self.y_name].astype(float)
            if self.covariate_names:
                cov_cols = [self.id_name] + list(self.covariate_names)
                cov_at_t0 = df_aug.loc[df_aug[self.t_name] == int(t0), cov_cols]
                cov_base_map[int(t0)] = cov_at_t0.drop_duplicates(subset=[self.id_name]).set_index(self.id_name)

        # group_size is computed later (line ~809) after exclusion filters;
        # prior accumulation code was removed as it was shadowed.

        for g, t, pret in cell_keys:
            mask_t = df_aug[self.t_name].to_numpy() == t
            ctrl_mask = spec.control_mask(df_aug, g, t, pret, times_all)
            cohort_arr = df_aug[self.cohort_name].to_numpy()
            mask_cell = mask_t & ((cohort_arr == g) | ctrl_mask)
            sub = df_aug.loc[mask_cell, :].copy()
            if sub.shape[0] == 0:
                continue

            base_time = pret
            base_key = None
            for k in base_map.keys():
                if k == base_time:
                    base_key = k
                    break
            if base_key is None:
                continue
            sub["_Ybase"] = sub[self.id_name].map(base_map[base_key])
            sub["_dY"] = (
                sub[self.y_name].astype(float).to_numpy()
                - sub["_Ybase"].astype(float).to_numpy()
            )
            ok = np.isfinite(sub["_dY"].to_numpy())
            sub = sub.loc[ok, :]
            if sub.shape[0] == 0:
                continue

            # Local row indices in df_aug for this cell (must stay aligned with dY/Dg/X after any trimming)
            idx_local = sub.index.to_numpy(dtype=int)

            Dg = (sub[self.cohort_name].to_numpy() == g).astype(float)
            n1 = int(Dg.sum())
            n0 = int(Dg.size - n1)
            if n1 == 0 or n0 == 0:
                continue
            # NOTE: group_size is computed later after cohort exclusion filters (line ~807)

            dY = sub["_dY"].to_numpy(dtype=float).reshape(-1)

            X = None
            if self.covariate_names and base_key in cov_base_map:
                cov_df = cov_base_map[base_key]
                unit_ids = sub[self.id_name].to_numpy()
                valid_ids = [uid for uid in unit_ids if uid in cov_df.index]
                if len(valid_ids) == len(unit_ids):
                    X = cov_df.loc[unit_ids, list(self.covariate_names)].to_numpy(dtype=float)
                else:
                    X = None
            cov_mode = self.cov_method

            ps = None
            if cov_mode in {"ipw", "dr"}:
                if LogisticRegression is None:
                    raise ImportError(
                        "scikit-learn is required for ipw/dr in CallawaySantAnnaES. Install scikit-learn.",
                    )
                X_ps = X if X is not None else np.empty((dY.shape[0], 0), dtype=np.float64)
                X_ps = np.column_stack([X_ps, np.ones((X_ps.shape[0], 1))])
                clf = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=10_000,
                    fit_intercept=False,
                    tol=1e-10,
                )
                # Promote ConvergenceWarning to error to enforce strict convergence
                import warnings as _warnings
                with _warnings.catch_warnings():
                    if ConvergenceWarning is not None:
                        _warnings.filterwarnings("error", category=ConvergenceWarning)
                    clf.fit(X_ps, Dg.astype(int))
                ps = np.clip(clf.predict_proba(X_ps)[:, 1],
                              self.trim_ps if self.trim_mode == "clip" else 0.0,
                              1.0 - (self.trim_ps if self.trim_mode == "clip" else 0.0))
                if self.trim_mode == "drop" and self.trim_ps > 0.0:
                    keep = (ps >= self.trim_ps) & (ps <= 1.0 - self.trim_ps)
                    dY = dY[keep]
                    Dg = Dg[keep]
                    if X is not None:
                        X = X[keep, :]
                    X_ps = X_ps[keep, :]
                    ps = ps[keep]
                    idx_local = idx_local[keep]

            # Compute ATT and influence function under selected mode
            if cov_mode == "reg" and X is not None and X.shape[0] == dY.shape[0]:
                # OLS within groups to estimate conditional means of dY
                # Treated
                Xt = X[Dg == 1.0, :]
                yt = dY[Dg == 1.0]
                Xc = X[Dg == 0.0, :]
                yc = dY[Dg == 0.0]
                # Use centralized linear solver to respect library numerical policy
                Xdt = np.c_[np.ones((Xt.shape[0], 1)), Xt]
                beta_t = la.solve_normal_eq(Xdt, yt.reshape(-1, 1), method="qr").reshape(-1)
                Xdc = np.c_[np.ones((Xc.shape[0], 1)), Xc]
                beta_c = la.solve_normal_eq(Xdc, yc.reshape(-1, 1), method="qr").reshape(-1)
                m1 = beta_t[0] + la.dot(beta_t[1:].reshape(1, -1), la.to_dense(X).T).reshape(-1)
                m0 = beta_c[0] + la.dot(beta_c[1:].reshape(1, -1), la.to_dense(X).T).reshape(-1)
                # Hájek weights
                w1 = Dg / max(float(Dg.sum()), 1.0)
                w0 = (1.0 - Dg) / max(float((1.0 - Dg).sum()), 1.0)
                att = float((w1 * (dY - m0)).sum() - (w0 * (dY - m0)).sum())
                mu1 = float((w1 * (dY - m0)).sum())
                mu0 = float((w0 * (dY - m0)).sum())
                psi = (Dg * ((dY - m0) - mu1) / max(float(Dg.sum()), 1.0)) - (
                    (1.0 - Dg) * ((dY - m0) - mu0) / max(float((1.0 - Dg).sum()), 1.0)
                )
            elif cov_mode == "ipw" and ps is not None:
                ps_safe = np.clip(ps, 1e-8, 1.0 - 1e-8)
                w1 = Dg
                w0 = (1.0 - Dg) * ps_safe / (1.0 - ps_safe)
                den1 = float(w1.sum()) if w1.sum() > 0 else 1.0
                den0 = float(w0.sum()) if w0.sum() > 0 else 1.0
                mu1 = float((w1 * dY).sum() / den1)
                mu0 = float((w0 * dY).sum() / den0)
                att = mu1 - mu0
                psi = (w1 * (dY - mu1)) / den1 - (w0 * (dY - mu0)) / den0
                psi_centered = psi - float(psi.mean())
                if X_ps is not None and X_ps.shape[0] == psi_centered.shape[0]:
                    score = (Dg - ps_safe).reshape(-1, 1) * X_ps
                    Ihat = la.dot((X_ps.T * (ps_safe * (1.0 - ps_safe)).reshape(1, -1)), X_ps)
                    try:
                        Iinv = la.solve(Ihat, la.eye(Ihat.shape[0]))
                    except Exception:
                        Iinv = la.pinv(Ihat)
                    sens = la.dot(la.to_dense(X_ps).T, (w0 * (dY - mu0)).reshape(-1, 1)).reshape(-1) / den0
                    IF_gamma = la.dot(score, Iinv)
                    psi_fs = la.dot(IF_gamma, sens.reshape(-1, 1)).reshape(-1)
                    psi_centered = psi_centered + psi_fs - float(psi_fs.mean())
                psi = psi_centered
            elif cov_mode == "dr" and X is not None and ps is not None:
                # Outcome regressions
                Xt = X[Dg == 1.0, :]
                yt = dY[Dg == 1.0]
                Xc = X[Dg == 0.0, :]
                yc = dY[Dg == 0.0]
                Xdt = np.c_[np.ones((Xt.shape[0], 1)), Xt]
                beta_t = la.solve_normal_eq(Xdt, yt.reshape(-1, 1), method="qr").reshape(-1)
                Xdc = np.c_[np.ones((Xc.shape[0], 1)), Xc]
                beta_c = la.solve_normal_eq(Xdc, yc.reshape(-1, 1), method="qr").reshape(-1)
                m1 = beta_t[0] + la.dot(beta_t[1:].reshape(1, -1), la.to_dense(X).T).reshape(-1)
                m0 = beta_c[0] + la.dot(beta_c[1:].reshape(1, -1), la.to_dense(X).T).reshape(-1)
                ps_safe = np.clip(ps, 1e-8, 1.0 - 1e-8)
                w1 = Dg
                w0 = (1.0 - Dg) * ps_safe / (1.0 - ps_safe)
                den1 = float(w1.sum()) if w1.sum() > 0 else 1.0
                den0 = float(w0.sum()) if w0.sum() > 0 else 1.0
                R1 = float((w1 * (dY - m1)).sum() / den1)
                R0 = float((w0 * (dY - m0)).sum() / den0)
                n_treat_local = int(Dg.sum())
                mu_aug = float((Dg * (m1 - m0)).sum() / n_treat_local) if n_treat_local > 0 else 0.0
                att = mu_aug + R1 - R0
                psi_t = (w1 * (dY - m1)) / den1 - R1 * (w1 / den1)
                psi_c = -(w0 * (dY - m0)) / den0 + R0 * (w0 / den0)
                aug = np.zeros_like(dY)
                if n_treat_local > 0:
                    aug_mask = Dg == 1
                    aug[aug_mask] = ((m1 - m0)[aug_mask] - mu_aug) / float(n_treat_local)
                psi = psi_t + psi_c + aug
                psi_centered = psi - float(psi.mean())
                Xc_aug = np.c_[np.ones(Xc.shape[0]), Xc]
                e_c = yc - la.dot(Xc_aug, beta_c)
                try:
                    Gc_inv = la.solve(la.crossprod(Xc_aug, Xc_aug), la.eye(Xc_aug.shape[1]))
                except Exception:
                    Gc_inv = la.pinv(la.crossprod(Xc_aug, Xc_aug))
                X_aug = np.c_[np.ones(X.shape[0]), X]
                xbar1 = np.mean(X_aug[Dg == 1, :], axis=0)
                w_ctrl = np.zeros_like(ps_safe)
                w_ctrl[Dg == 0] = ps_safe[Dg == 0] / np.maximum(1.0 - ps_safe[Dg == 0], 1e-12)
                sens_or = -(xbar1 + np.sum(w_ctrl[Dg == 0].reshape(-1, 1) * X_aug[Dg == 0, :], axis=0) / max(float(n_treat_local), 1.0))
                psi_or = np.zeros_like(psi_centered)
                inner = la.dot(Gc_inv, sens_or.reshape(-1, 1))
                tmp = la.dot(Xc_aug, inner).reshape(-1)
                psi_or[Dg == 0] = -tmp * e_c
                psi_centered = psi_centered + psi_or - float(psi_or.mean())
                if X_ps is not None and X_ps.shape[0] == psi_centered.shape[0]:
                    score = (Dg - ps_safe).reshape(-1, 1) * X_ps
                    Ihat = la.dot((X_ps.T * (ps_safe * (1.0 - ps_safe)).reshape(1, -1)), X_ps)
                    try:
                        Iinv = la.solve(Ihat, la.eye(Ihat.shape[0]))
                    except Exception:
                        Iinv = la.pinv(Ihat)
                    resid_dr_t = (dY - m1) * Dg
                    resid_dr_c = (dY - m0) * (1.0 - Dg)
                    sens_ps = np.mean(w0.reshape(-1, 1) * resid_dr_c.reshape(-1, 1) * X_ps, axis=0)
                    IF_gamma = la.dot(score, Iinv)
                    psi_ps = la.dot(IF_gamma, sens_ps.reshape(-1, 1)).reshape(-1)
                    psi_centered = psi_centered + psi_ps - float(psi_ps.mean())
                psi = psi_centered
            else:
                # No covariates (default): original Hájek difference-in-means on dY
                w1 = Dg / max(float(Dg.sum()), 1.0)
                w0 = (1.0 - Dg) / max(float((1.0 - Dg).sum()), 1.0)
                att = float((w1 * dY).sum() - (w0 * dY).sum())
                mu1 = float((w1 * dY).sum())
                mu0 = float((w0 * dY).sum())
                psi = (Dg * (dY - mu1) / max(float(Dg.sum()), 1.0)) - (
                    (1.0 - Dg) * (dY - mu0) / max(float((1.0 - Dg).sum()), 1.0)
                )
            # map indices back to df_aug to align with W rows
            used_idx = idx_local

            # Apply Small Sample Correction (SSC) to the influence function `psi`
            # The standard adjustment is (N-1)/(N-K) * G/(G-1)
            # For CS event study, "K" is typically not well-defined per global model,
            # but often fixed at 0 or effectively handled by influence function estimation.
            # Stata `csdid` often defaults to effectively no adjustment or simple DOF.
            # However, if user requests adjustment, we apply it based on local sample size n_sub.
            # Here n = sub.shape[0] (total in the (g,t) 2x2 difference).
            # k is usually 0 for simple DiD, or covariates count.
            # We base it on n=sub.shape[0] and k= (X.shape[1] if X else 0)
            k_params = int(X.shape[1] + 1) if X is not None else 1 # +1 for intercept implicitly

            # If clustering is used (self.cluster_ids is not None), verify group counts
            local_clusters = None
            if self.cluster_ids is not None:
                # Resolve cluster IDs for this local subset (used for SSC corrections).
                if isinstance(self.cluster_ids, str):
                    if self.cluster_ids not in sub.columns:
                        raise ValueError("cluster_ids column not found in local data subset.")
                    local_clusters = sub.loc[used_idx, self.cluster_ids].to_numpy()
                else:
                    # External array: fit() stores the resolved/filtered IDs on boot_eff.cluster_ids.
                    if boot_eff is None or getattr(boot_eff, "cluster_ids", None) is None:
                        raise ValueError(
                            "SSC correction requires cluster_ids; provide a column name or pass cluster_ids via BootConfig."
                        )
                    local_clusters = np.asarray(boot_eff.cluster_ids)[used_idx]

            # Calculate adjustment factor
            # We treat the (g,t) estimation as a local regression.
            ssc_factor = compute_ssc_correction(
                n=int(psi.size),
                k=k_params + fe_dof_val,
                clusters=local_clusters,
                ssc=ssc_local
            )

            if ssc_factor != 1.0:
                psi *= ssc_factor

            # Center influence function for multiplier bootstrap
            psi = psi - float(np.mean(psi))

            # multiplier bootstrap draws: att* = att + sum_i ψ_i * W_i (per-column b)
            att_star = att + (psi.reshape(-1, 1) * W[used_idx, :]).sum(axis=0, keepdims=False)
            att_star = att_star.reshape(1, -1)

            tau = event_tau(t, g, t2pos)
            att_rows.append((g, t, int(tau), float(att), int(n1)))
            att_star_cells.append(att_star)
            used_index.append(used_idx)
            cells_order.append((g, t))

        if not att_rows:
            raise RuntimeError("No valid (g,t) cells found.")
        att_gt = (
            pd.DataFrame(att_rows, columns=["g", "t", "tau", "att", "n_treat"])
            .sort_values(["g", "t"])
            .reset_index(drop=True)
        )

        ord_map = {key: i for i, key in enumerate(cells_order)}
        star_sorted = [
            att_star_cells[ord_map[(int(r.g), int(r.t))]]
            for r in att_gt.itertuples(index=False)
        ]
        att_star_all = np.vstack(star_sorted)
        B = att_star_all.shape[1]

        att_tau, rows_by_tau = aggregate_tau(att_gt, self.center_at)

        # STRICT balance_e implementation: verify all cohorts cover τ∈{0..E}, exclude non-compliant, reconstruct
        balance_e_excluded_cohorts = []
        if self.balance_e is not None:
            be = int(self.balance_e)
            if be < 0:
                raise ValueError("balance_e must be nonnegative.")

            # Strict verification: each cohort must cover EXACTLY τ∈{0, 1, ..., be}
            tau_by_g = (
                att_gt.groupby("g")["tau"]
                .apply(lambda s: {int(x) for x in s.tolist()})
                .to_dict()
            )
            required = set(range(be + 1))

            # Identify compliant and non-compliant cohorts
            good = []
            bad = []
            for g, S in tau_by_g.items():
                if required.issubset(S):
                    good.append(g)
                else:
                    missing = sorted(required - S)
                    bad.append((g, missing))

            if len(good) == 0:
                raise ValueError(
                    f"balance_e={be}: no cohorts have complete exposure set {sorted(required)}.",
                )

            # Log excluded cohorts for model_info
            balance_e_excluded_cohorts = bad

            # Filter att_gt to only compliant cohorts and τ≤be (post-period) or τ<0 (pre-period)
            keep_mask = att_gt["g"].isin(good)
            att_gt = att_gt[keep_mask].reset_index(drop=True)
            att_gt = att_gt[(att_gt["tau"] < 0) | (att_gt["tau"] <= be)].reset_index(
                drop=True,
            )

            # Reconstruct att_star_all: map back to original cell order, then filter
            ord_map = {key: i for i, key in enumerate(cells_order)}
            star_idx = [
                ord_map[(int(r.g), int(r.t))] for r in att_gt.itertuples(index=False)
            ]
            att_star_all = att_star_all[star_idx, :]

            # CRITICAL: Reconstruct att_tau and rows_by_tau after cohort exclusion
            att_tau, rows_by_tau = aggregate_tau(att_gt, self.center_at)

        # Compute group sizes (N_g) for auditing and for group-weight aggregation
        cohort_arr = df_aug[self.cohort_name].to_numpy(dtype=int)
        group_size: dict[int, int] = {}
        for g in sorted(np.unique(cohort_arr[cohort_arr > 0])):
            ids_g = df_aug.loc[cohort_arr == g, self.id_name].unique()
            group_size[int(g)] = len(ids_g)

        # Aggregate per-tau point estimates and weighted bootstrap draws
        tau_vals = sorted(att_tau["tau"].unique())
        att_agg_vals: list[float] = []
        for tau in tau_vals:
            rows = rows_by_tau[tau]
            atts = att_gt.loc[rows, "att"].to_numpy(dtype=np.float64)
            gs = att_gt.loc[rows, "g"].to_numpy(dtype=int)
            if self.tau_weight == "equal":
                w = np.ones_like(atts, dtype=np.float64)
            elif self.tau_weight == "group":
                w = np.array(
                    [group_size.get(int(gg), 0) for gg in gs], dtype=np.float64,
                )
            else:
                w = att_gt.loc[rows, "n_treat"].to_numpy(dtype=np.float64)
            w = w / np.sum(w)
            att_agg_vals.append(float(np.sum(w * atts)))
        att_tau["att"] = att_agg_vals

        tau_index = {tau: i for i, tau in enumerate(tau_vals)}
        att_tau_star = np.empty((len(tau_vals), B), dtype=np.float64)
        for tau, rows in rows_by_tau.items():
            if self.tau_weight == "equal":
                w = np.ones((len(rows),), dtype=np.float64)
            elif self.tau_weight == "group":
                gs = att_gt.loc[rows, "g"].to_numpy(dtype=int)
                w = np.array(
                    [group_size.get(int(gg), 0) for gg in gs], dtype=np.float64,
                )
            else:
                w = att_gt.loc[rows, "n_treat"].to_numpy(dtype=np.float64)
            w = w / np.sum(w)
            att_tau_star[tau_index[tau], :] = (
                w.reshape(-1, 1) * att_star_all[rows, :]
            ).sum(axis=0)

        def _intersection_support(att_gt_df: pd.DataFrame, base_tau: int) -> list[int]:
            g_groups = att_gt_df.groupby("g")["tau"].apply(lambda s: set(s.tolist()))
            inter = set.intersection(*list(g_groups)) if len(g_groups) > 0 else set()
            return sorted(int(t) for t in inter if int(t) != int(base_tau))

        if self.enforce_tau_intersection:
            tau_supported = _intersection_support(att_gt, base_tau=self.center_at)
            tau_order = list(att_tau["tau"].tolist())
            keep_rows = (
                [tau_order.index(t) for t in tau_supported] if tau_supported else []
            )
            if keep_rows:
                att_tau = att_tau.iloc[keep_rows, :].reset_index(drop=True)
                att_tau_star = att_tau_star[keep_rows, :]
                tau_flag = "intersection-only"
            else:
                tau_flag = "no-intersection-change"
        else:
            tau_flag = "unrestricted"

        plot_df, band_pre, band_post, band_full = compute_uniform_bands(
            att_tau, att_tau_star, self.center_at, self.alpha,
        )

        # PostATT aggregation: if group weighting is requested, prefer the
        # explicit group_size mapping; otherwise the helper will infer a
        # conservative mapping from att_gt when possible.
        post_mask_tau = plot_df["tau"] > self.center_at
        post_att_se = float("nan")
        if np.any(post_mask_tau):
            post_att, (lo_s, hi_s), post_att_se = post_aggregate_uniform_band(
                att_gt=att_gt,
                att_tau=plot_df[["tau", "att"]],
                att_tau_star=att_tau_star,
                base_tau=self.center_at,
                tau_weight=self.tau_weight,
                alpha=self.alpha,
                group_size_map=(group_size if self.tau_weight == "group" else None),
            )
            post_scalar = pd.DataFrame(
                {"lower": pd.Series([lo_s]), "upper": pd.Series([hi_s])},
            )
        else:
            post_att = float("nan")
            post_scalar = pd.DataFrame(
                {"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)},
            )

        n_unique = 0
        if used_index:
            n_unique = int(np.unique(np.concatenate(used_index, axis=0)).size)

        band_level = float(100.0 * (1.0 - self.alpha))

        model_info: dict[str, object] = {
            # Ensure summary recognises this as an event-study family. The
            # summary gate searches for tokens like 'eventstudy'/'did'.
            # Include an explicit 'eventstudy' token to avoid abbreviation
            # mismatches (e.g., 'ES').
            "Estimator": "EventStudy: CS-DID",
            "ControlGroup": self.control_group,
            "Alpha": float(self.alpha),
            "BandLevel": band_level,

            "Bootstrap": f"IF common-multiplier; policy={getattr(boot_eff, 'policy', None)}; dist={getattr(boot_eff, 'dist', None)}",
            "B": int(att_tau_star.shape[1]),
            # CenterAt key is used by the summary layer to determine base τ
            # and which side-specific band ('pre' vs 'post') to read from.
            "CenterAt": self.center_at,
            # Be explicit that we computed uniform sup-t bands
            "BandType": "uniform",
            "TauWeight": self.tau_weight,
            "BalanceE": self.balance_e,
            "TauSetPolicy": tau_flag,
            "PostATT": post_att,
            "PostATT_se": post_att_se,
            # audit metadata
            "GroupSizes": group_size,
            "NumCells": int(att_gt.shape[0]),
        }
        if not post_scalar.empty:
            model_info["PostATT_Band"] = (
                float(post_scalar["lower"].to_numpy()[0]),
                float(post_scalar["upper"].to_numpy()[0]),
            )

        # Log balance_e excluded cohorts if any
        if self.balance_e is not None and balance_e_excluded_cohorts:
            model_info["BalanceEExcludedCohorts"] = [
                {"cohort": int(g), "missing_taus": missing}
                for g, missing in balance_e_excluded_cohorts
            ]
            model_info["BalanceENumCohortsExcluded"] = len(balance_e_excluded_cohorts)

        # Build bands dict with provenance meta required by plotting API
        bands = {
            "pre": pd.DataFrame({"lower": band_pre[0], "upper": band_pre[1]}),
            "post": pd.DataFrame({"lower": band_post[0], "upper": band_post[1]}),
            "full": pd.DataFrame({"lower": band_full[0], "upper": band_full[1]}),
            "post_scalar": post_scalar,
            "__meta__": {
                "origin": "bootstrap",
                "policy": str(getattr(boot_eff, "policy", "bootstrap")),
                "dist": getattr(boot_eff, "dist", None),
                "preset": getattr(boot_eff, "preset", None),
                "kind": "uniform",
                "level": band_level,
                "B": int(att_tau_star.shape[1])
                if isinstance(att_tau_star, np.ndarray)
                else int(model_info.get("B", 0)),
                "estimator": "eventstudy",
                "context": (
                    "did" if str(getattr(boot_eff, "preset", "")).lower() == "did" else "eventstudy"
                ),
            },
        }

        se_series = pd.Series(
            bt.bootstrap_se(att_tau_star),
            index=plot_df["tau"].astype(int).to_numpy(),
        )
        if int(self.center_at) in se_series.index:
            se_series.loc[int(self.center_at)] = 0.0

        return EstimationResult(
            params=plot_df.set_index("tau")["att"],
            se=se_series,
            bands=bands,
            n_obs=n_unique,
            model_info=model_info,
            extra={
                "att_gt": att_gt,
                "att_tau": plot_df,  # For DDD compatibility
                "att_tau_star": att_tau_star,
                "boot_config": boot_eff,
                "W_multipliers_inference": W_df,
                "multipliers_log": _boot_log,
                "bands_source": "bootstrap",
                "se_source": "bootstrap",
                "group_size_map_inferred": group_size,
            },
        )


EventStudyCS = CallawaySantAnnaES
__all__ = ["CallawaySantAnnaES", "EventStudyCS"]
