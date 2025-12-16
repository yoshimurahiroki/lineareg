"""Callaway and Sant'Anna (2021) event-study estimator with uniform bands.

Scope and policy
----------------
- Implements long-differences per (g,t) cell following csdid for exact parity.
- Bootstrap-only inference (wild/multiplier). Default B=2000 with B+1 rule.
- Provides uniform sup-t bands for pre, post, and full windows; handles
    staggered adoption robustly. Aggregated post-period ATT is reported.
- Formula parsing, FE absorption, and NA/weights alignment follow
    :mod:`lineareg.utils.formula`; all matrix ops are via :mod:`lineareg.core.linalg`.
- No analytical SEs or p-values; no pairs bootstrap.

Comments/docstrings are English-only by policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

from lineareg.core import bootstrap as bt
from lineareg.estimators.base import (
    BootConfig,
    EstimationResult,
    attach_formula_metadata,
    prepare_formula_environment,
)
from lineareg.utils.formula import FormulaParser
try:  # optional: used only when cov_method in {"ipw","dr"}
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - allow absence but raise when actually needed
    LogisticRegression = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Sequence
else:
    Sequence = tuple  # type: ignore[assignment]

# Long-differences (Y_t - Y_{g-1}) are constructed per-(g,t) cell.
# Units missing the base period g-1 are excluded from that cell.
# This implementation mirrors Callaway & Sant'Anna (2021) / did / csdid
# exactly for deterministic reproducibility (no approximations).


@dataclass
class ESCellSpec:
    id_name: str
    t_name: str
    cohort_name: str
    y_name: str
    event_time_name: str | None = None
    control_group: str = "notyet"
    center_at: int = -1
    covariate_names: Sequence[str] | None = None
    cov_method: str = "none"  # {"none","reg","ipw","dr"}

    def control_mask(self, df: pd.DataFrame, t: int) -> np.ndarray:
        cg_normalized = self.control_group.lower().replace("treated", "")
        if cg_normalized == "never":
            return df[self.cohort_name] == 0
        if cg_normalized == "notyet":
            return (df[self.cohort_name] > t) | (df[self.cohort_name] == 0)
        raise ValueError(
            "control_group must be 'never'/'nevertreated' or 'notyet'/'notyettreated'",
        )


def build_cells(
    df: pd.DataFrame, spec: ESCellSpec,
) -> tuple[pd.DataFrame, list[tuple[int, int]], dict]:
    df_aug = df.copy()
    times = np.sort(df_aug[spec.t_name].astype(int).unique())
    cohorts = np.sort(df_aug[spec.cohort_name].astype(int).unique())
    times_set = {int(t) for t in times}
    cell_keys = [
        (int(g), int(t))
        for g in cohorts
        if g > 0 and int(g - 1) in times_set
        for t in times
    ]
    return df_aug, cell_keys, {}


def aggregate_tau(att_gt: pd.DataFrame, base_tau: int) -> tuple[pd.DataFrame, dict]:
    base_tau_int = int(base_tau)
    tau_values = {int(t) for t in att_gt["tau"].unique()}
    tau_values.add(base_tau_int)
    taus = sorted(tau_values)
    att_tau = pd.DataFrame({"tau": taus, "att": 0.0})
    rows_by_tau = {tau: att_gt[att_gt["tau"] == tau].index.to_numpy() for tau in taus}
    return att_tau, rows_by_tau


def compute_uniform_bands(
    att_tau: pd.DataFrame,
    att_tau_star: np.ndarray,
    base_tau: int,
    alpha: float,
):
    taus = att_tau["tau"].to_numpy(dtype=int)
    pre_mask = taus < int(base_tau)
    post_mask = taus > int(base_tau)
    full_mask = taus != int(base_tau)

    def _band(mask: np.ndarray):
        if not np.any(mask):
            return (pd.Series([], dtype=float), pd.Series([], dtype=float))
        idx = np.flatnonzero(mask)
        theta = att_tau.loc[mask, "att"].to_numpy(dtype=np.float64).reshape(1, -1)
        theta_star = att_tau_star[idx, :]
        lo, hi = bt.uniform_confidence_band(
            theta,
            theta_star,
            alpha=alpha,
            studentize="bootstrap",
            context="eventstudy",
        )
        series = att_tau.loc[mask, "tau"]
        return (
            pd.Series(lo.reshape(-1), index=series),
            pd.Series(hi.reshape(-1), index=series),
        )

    band_pre = _band(pre_mask)
    band_post = _band(post_mask)
    band_full = _band(full_mask)

    att_tau = att_tau.copy()
    att_tau["ci_lower"] = np.nan
    att_tau["ci_upper"] = np.nan
    att_tau["ci_lower_full"] = np.nan
    att_tau["ci_upper_full"] = np.nan
    if pre_mask.any():
        att_tau.loc[pre_mask, "ci_lower"] = (
            band_pre[0].reindex(att_tau.loc[pre_mask, "tau"]).to_numpy()
        )
        att_tau.loc[pre_mask, "ci_upper"] = (
            band_pre[1].reindex(att_tau.loc[pre_mask, "tau"]).to_numpy()
        )
    if post_mask.any():
        att_tau.loc[post_mask, "ci_lower"] = (
            band_post[0].reindex(att_tau.loc[post_mask, "tau"]).to_numpy()
        )
        att_tau.loc[post_mask, "ci_upper"] = (
            band_post[1].reindex(att_tau.loc[post_mask, "tau"]).to_numpy()
        )
    if full_mask.any():
        att_tau.loc[full_mask, "ci_lower_full"] = (
            band_full[0].reindex(att_tau.loc[full_mask, "tau"]).to_numpy()
        )
        att_tau.loc[full_mask, "ci_upper_full"] = (
            band_full[1].reindex(att_tau.loc[full_mask, "tau"]).to_numpy()
        )

    return att_tau, band_pre, band_post, band_full


def post_aggregate_uniform_band(  # noqa: PLR0913
    att_gt: pd.DataFrame,
    att_tau: pd.DataFrame,
    att_tau_star: np.ndarray,
    base_tau: int,
    tau_weight: str = "group",
    alpha: float = 0.05,
    group_size_map: dict[int, int] | None = None,
) -> tuple[float, tuple[float, float]]:
    taus = att_tau["tau"].to_numpy(dtype=int)
    sel = taus > int(base_tau)
    if not np.any(sel):
        return float("nan"), (float("nan"), float("nan"))
    if tau_weight not in {"equal", "group", "treated_t"}:
        raise ValueError("tau_weight must be one of {'equal','group','treated_t'}.")

    # When 'group' weighting is requested and no mapping is provided, attempt
    # to infer sensible group sizes from att_gt if possible. Prefer explicit
    # mapping supplied by caller for auditability; otherwise fall back to
    # cohort-level treated counts as a conservative proxy.
    if tau_weight == "equal":
        w_by_tau = pd.Series(1.0, index=att_tau.loc[sel, "tau"]).groupby(level=0).sum()
        w = w_by_tau.reindex(att_tau.loc[sel, "tau"]).to_numpy(dtype=np.float64)
    elif tau_weight == "group":
        if group_size_map is None:
            # Conservative fallback: use the maximum treated count observed
            # for each cohort across event-times in att_gt. This is not a
            # perfect substitute for unique-ID counts but provides a stable,
            # reproducible default when the caller omitted mapping.
            gs = att_gt.loc[att_gt["tau"] > base_tau, ["g", "n_treat"]].copy()
            if gs.shape[0] == 0:
                raise ValueError(
                    "No post-period cohort entries to infer group sizes from.",
                )
            inferred = gs.groupby("g")["n_treat"].max().to_dict()
            group_size_map = {int(k): int(v) for k, v in inferred.items()}
        w_map = (
            att_gt.loc[att_gt["tau"] > base_tau, ["tau", "g"]]
            .assign(
                Ng=att_gt.loc[att_gt["tau"] > base_tau, "g"].map(
                    lambda gg: int(group_size_map.get(int(gg), 0)),
                ),
            )
            .groupby("tau")["Ng"]
            .sum()
        )
        w = w_map.reindex(att_tau.loc[sel, "tau"]).to_numpy(dtype=np.float64)
    else:
        w_map = att_gt.loc[att_gt["tau"] > base_tau, :].groupby("tau")["n_treat"].sum()
        w = w_map.reindex(att_tau.loc[sel, "tau"]).to_numpy(dtype=np.float64)

    w = np.where(np.asarray(w, dtype=float) <= 0.0, 0.0, w)
    s = float(np.sum(w))
    if s <= 0.0:
        return float("nan"), (float("nan"), float("nan"))
    w_norm = (w / s).reshape(-1, 1)

    sel_idx = np.flatnonzero(sel)
    theta = float(
        np.sum(w_norm.reshape(-1) * att_tau.loc[sel, "att"].to_numpy(dtype=np.float64)),
    )
    post_star = (w_norm * att_tau_star[sel_idx, :]).sum(axis=0).reshape(1, -1)

    lo, hi = bt.uniform_confidence_band(
        np.array([theta], dtype=np.float64),
        post_star,
        alpha=alpha,
        studentize="bootstrap",
        context="eventstudy",
    )
    return float(theta), (float(lo[0]), float(hi[0]))


class CallawaySantAnnaES:
    """Callaway-Sant'Anna (2021) event-study via long-differences dY.

    Strict policy summary
    ---------------------
    - Analytic SE/p-values/critical values: not provided (bootstrap-only).
    - Pair bootstrap: disallowed. Use positive multiplier / wild bootstrap.
    - Baseline event-time `center_at` is excluded from uniform bands.
    - By default, intersection support is NOT enforced (R did default); use
      `enforce_tau_intersection=True` to require common support across cohorts.
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
    ) -> EstimationResult:
        """Fit Callaway-Sant'Anna event study estimator.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Panel data with id_name, t_name, cohort_name, y_name columns.
            When omitted, the DataFrame retained by :meth:`from_formula` is used.
        external_W : np.ndarray, optional
            External bootstrap multipliers (n_rows x n_boot). If provided, these
            multipliers are used instead of generating new ones. Useful for DDD
            when the same multipliers must be shared across estimators.

        Returns
        -------
        EstimationResult
            Estimation results with params, bands, and extra info.

        """
        if df is None:
            df = getattr(self, "_formula_df", None)
            if df is None:
                msg = "Provide df explicitly or instantiate the estimator via from_formula()."
                raise ValueError(msg)
        else:
            self._formula_df = df

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
        )

        df_aug, cell_keys, _ = build_cells(df, spec)

        # Shared multiplier draws W (common multiplier across cells)
        # Use external_W if provided (for DDD), otherwise generate fresh multipliers
        if external_W is not None:
            W = external_W
            if W.shape[0] != df_aug.shape[0]:
                msg = f"external_W has {W.shape[0]} rows but df_aug has {df_aug.shape[0]} rows"
                raise ValueError(msg)
            _boot_log = {"source": "external"}
        else:
            W_df, _boot_log = (
                self.boot.make_multipliers(df_aug.shape[0])
                if self.boot is not None
                else (None, {})
            )
            W = W_df.to_numpy() if W_df is not None else None

        att_rows: list[tuple[int, int, int, float, int]] = []
        att_star_cells: list[np.ndarray] = []
        used_index: list[np.ndarray] = []
        cells_order: list[tuple[int, int]] = []

        # ---- CS long-diff estimator per cell with optional covariates (reg/ipw/dr) ----
        # Build base maps Y_{g-1}(i) by calendar time once
        base_map: dict[int, pd.Series] = {}
        times_all = np.sort(df_aug[self.t_name].unique())
        for t0 in times_all:
            at_t0 = df_aug.loc[
                df_aug[self.t_name] == int(t0), [self.id_name, self.y_name],
            ]
            base_map[int(t0)] = at_t0.set_index(self.id_name)[self.y_name].astype(float)

        # group size map for tau aggregation (treated count per cohort)
        group_size: dict[int, int] = {}

        # W is aligned to df_aug row order (shared columns across cells)
        # center columns (finite-sample recentering) and validate variance
        if W is not None:
            W = W - W.mean(axis=0, keepdims=True)
            if not np.all(np.var(W, axis=0) > 0.0):
                raise ValueError(
                    "bootstrap multipliers have zero-variance column(s) after recentering.",
                )

        for g, t in cell_keys:
            # select rows for calendar time t and cohort g (treated) vs control (not-yet or never)
            mask_t = df_aug[self.t_name].astype(int).to_numpy() == int(t)
            if self.control_group.lower().replace("_", "") == "notyet":
                ctrl_mask = (df_aug[self.cohort_name].astype(int).to_numpy() == 0) | (
                    df_aug[self.cohort_name].astype(int).to_numpy() > int(t)
                )
            else:  # "never"
                ctrl_mask = df_aug[self.cohort_name].astype(int).to_numpy() == 0
            mask_cell = mask_t & (
                (df_aug[self.cohort_name].astype(int).to_numpy() == int(g)) | ctrl_mask
            )
            sub = df_aug.loc[mask_cell, :].copy()
            if sub.shape[0] == 0:
                continue

            # long-diff dY = Y_t - Y_{g-1}
            base_time = int(g) - 1
            if base_time not in base_map:
                continue
            sub["_Ybase"] = sub[self.id_name].map(base_map[base_time])
            sub["_dY"] = (
                sub[self.y_name].astype(float).to_numpy()
                - sub["_Ybase"].astype(float).to_numpy()
            )
            ok = np.isfinite(sub["_dY"].to_numpy())
            sub = sub.loc[ok, :]
            if sub.shape[0] == 0:
                continue

            Dg = (sub[self.cohort_name].astype(int).to_numpy() == int(g)).astype(float)
            n1 = int(Dg.sum())
            n0 = int(Dg.size - n1)
            if n1 == 0 or n0 == 0:
                continue
            group_size[int(g)] = group_size.get(int(g), 0) + n1

            dY = sub["_dY"].to_numpy(dtype=float).reshape(-1)

            # Optional covariates
            X = None
            if self.covariate_names:
                X = sub.loc[:, list(self.covariate_names)].to_numpy(dtype=float, copy=False)
            cov_mode = self.cov_method

            # Estimate propensity when requested
            ps = None
            if cov_mode in {"ipw", "dr"}:
                if LogisticRegression is None:
                    raise ImportError(
                        "scikit-learn is required for ipw/dr in CallawaySantAnnaES. Install scikit-learn.",
                    )
                # CRITICAL: Never use outcome dY as features for PS. If X is None, use a constant-only design.
                X_ps = X if X is not None else np.ones((dY.shape[0], 1), dtype=np.float64)
                clf = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=1000,
                )
                clf.fit(X_ps, Dg.astype(int))
                ps = np.clip(clf.predict_proba(X_ps)[:, 1],
                              self.trim_ps if self.trim_mode in {"clip","truncate"} else 0.0,
                              1.0 - (self.trim_ps if self.trim_mode in {"clip","truncate"} else 0.0))
                if self.trim_mode in {"drop","discard"} and self.trim_ps > 0.0:
                    keep = (ps >= self.trim_ps) & (ps <= 1.0 - self.trim_ps)
                    dY = dY[keep]
                    Dg = Dg[keep]
                    if X is not None:
                        X = X[keep, :]
                    ps = ps[keep]

            # Compute ATT and influence function under selected mode
            if cov_mode == "reg" and X is not None and X.shape[0] == dY.shape[0]:
                # OLS within groups to estimate conditional means of dY
                # Treated
                Xt = X[Dg == 1.0, :]
                yt = dY[Dg == 1.0]
                Xc = X[Dg == 0.0, :]
                yc = dY[Dg == 0.0]
                beta_t, *_ = lstsq(np.c_[np.ones(Xt.shape[0]), Xt], yt, rcond=None)
                beta_c, *_ = lstsq(np.c_[np.ones(Xc.shape[0]), Xc], yc, rcond=None)
                m1 = beta_t[0] + (beta_t[1:].reshape(1, -1) @ X.T).reshape(-1)
                m0 = beta_c[0] + (beta_c[1:].reshape(1, -1) @ X.T).reshape(-1)
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
                    Ihat = (X_ps.T * (ps_safe * (1.0 - ps_safe)).reshape(1, -1)) @ X_ps
                    try:
                        Iinv = np.linalg.solve(Ihat, np.eye(Ihat.shape[0]))
                    except np.linalg.LinAlgError:
                        Iinv = np.linalg.pinv(Ihat)
                    sens = (X_ps.T @ (w0 * (dY - mu0)).reshape(-1, 1)).reshape(-1) / den0
                    IF_gamma = score @ Iinv
                    psi_fs = (IF_gamma @ sens.reshape(-1, 1)).reshape(-1)
                    psi_centered = psi_centered + psi_fs - float(psi_fs.mean())
                psi = psi_centered
            elif cov_mode == "dr" and X is not None and ps is not None:
                # Outcome regressions
                Xt = X[Dg == 1.0, :]
                yt = dY[Dg == 1.0]
                Xc = X[Dg == 0.0, :]
                yc = dY[Dg == 0.0]
                beta_t, *_ = lstsq(np.c_[np.ones(Xt.shape[0]), Xt], yt, rcond=None)
                beta_c, *_ = lstsq(np.c_[np.ones(Xc.shape[0]), Xc], yc, rcond=None)
                m1 = beta_t[0] + (beta_t[1:].reshape(1, -1) @ X.T).reshape(-1)
                m0 = beta_c[0] + (beta_c[1:].reshape(1, -1) @ X.T).reshape(-1)
                ps_safe = np.clip(ps, 1e-8, 1.0 - 1e-8)
                w1 = Dg
                w0 = (1.0 - Dg) * ps_safe / (1.0 - ps_safe)
                den1 = float(w1.sum()) if w1.sum() > 0 else 1.0
                den0 = float(w0.sum()) if w0.sum() > 0 else 1.0
                R1 = float((w1 * (dY - m0)).sum() / den1)
                R0 = float((w0 * (dY - m0)).sum() / den0)
                n_treat_local = int(Dg.sum())
                mu_aug = float((Dg * (m1 - m0)).sum() / n_treat_local) if n_treat_local > 0 else 0.0
                att = mu_aug + R1 - R0
                psi_t = (w1 * (dY - m0)) / den1 - R1 * (w1 / den1)
                psi_c = -(w0 * (dY - m0)) / den0 + R0 * (w0 / den0)
                aug = np.zeros_like(dY)
                if n_treat_local > 0:
                    aug_mask = Dg == 1
                    aug[aug_mask] = ((m1 - m0)[aug_mask] - mu_aug) / float(n_treat_local)
                psi = psi_t + psi_c + aug
                psi_centered = psi - float(psi.mean())
                Xc_aug = np.c_[np.ones(Xc.shape[0]), Xc]
                e_c = yc - (Xc_aug @ beta_c)
                try:
                    Gc_inv = np.linalg.solve(Xc_aug.T @ Xc_aug, np.eye(Xc_aug.shape[1]))
                except np.linalg.LinAlgError:
                    Gc_inv = np.linalg.pinv(Xc_aug.T @ Xc_aug)
                X_aug = np.c_[np.ones(X.shape[0]), X]
                xbar1 = np.mean(X_aug[Dg == 1, :], axis=0)
                w_ctrl = np.zeros_like(ps_safe)
                w_ctrl[Dg == 0] = ps_safe[Dg == 0] / np.maximum(1.0 - ps_safe[Dg == 0], 1e-12)
                sens_or = -(xbar1 + np.sum(w_ctrl[Dg == 0].reshape(-1, 1) * X_aug[Dg == 0, :], axis=0) / max(float(n_treat_local), 1.0))
                psi_or = np.zeros_like(psi_centered)
                psi_or[Dg == 0] = -(Xc_aug @ (Gc_inv @ sens_or.reshape(-1, 1))).reshape(-1) * e_c
                psi_centered = psi_centered + psi_or - float(psi_or.mean())
                if X_ps is not None and X_ps.shape[0] == psi_centered.shape[0]:
                    score = (Dg - ps_safe).reshape(-1, 1) * X_ps
                    Ihat = (X_ps.T * (ps_safe * (1.0 - ps_safe)).reshape(1, -1)) @ X_ps
                    try:
                        Iinv = np.linalg.solve(Ihat, np.eye(Ihat.shape[0]))
                    except np.linalg.LinAlgError:
                        Iinv = np.linalg.pinv(Ihat)
                    resid_dr = dY - m0
                    sens_ps = np.mean(w0.reshape(-1, 1) * resid_dr.reshape(-1, 1) * X_ps, axis=0)
                    IF_gamma = score @ Iinv
                    psi_ps = (IF_gamma @ sens_ps.reshape(-1, 1)).reshape(-1)
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
            used_idx = sub.index.to_numpy(dtype=int)
            # multiplier bootstrap draws: att* = att + sum_i ψ_i * W_i (per-column b)
            att_star = att + (psi.reshape(-1, 1) * W[used_idx, :]).sum(axis=0, keepdims=False)
            att_star = att_star.reshape(1, -1)

            tau = int(t) - int(g)
            att_rows.append((int(g), int(t), int(tau), float(att), int(n1)))
            att_star_cells.append(att_star)
            used_index.append(used_idx)
            cells_order.append((int(g), int(t)))

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
        if np.any(post_mask_tau):
            post_att, (lo_s, hi_s) = post_aggregate_uniform_band(
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
            "NoAnalyticPValues": True,
            "Bootstrap": f"IF common-multiplier; policy={getattr(self.boot, 'policy', None)}; dist={getattr(self.boot, 'dist', None)}",
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
                "policy": str(getattr(self.boot, "policy", "bootstrap")),
                "dist": getattr(self.boot, "dist", None),
                "preset": getattr(self.boot, "preset", None),
                "kind": "uniform",
                "level": band_level,
                "B": int(att_tau_star.shape[1])
                if isinstance(att_tau_star, np.ndarray)
                else int(model_info.get("B", 0)),
                "estimator": "eventstudy",
                "context": (
                    "did" if str(getattr(self.boot, "preset", "")).lower() == "did" else "eventstudy"
                ),
            },
        }

        return EstimationResult(
            params=plot_df.set_index("tau")["att"],
            bands=bands,
            n_obs=n_unique,
            model_info=model_info,
            extra={
                "att_gt": att_gt,
                "att_tau": plot_df,  # For DDD compatibility
                "att_tau_star": att_tau_star,
                "bands_source": "bootstrap",
                "se_source": "bootstrap",
                "group_size_map_inferred": group_size,
            },
        )


EventStudyCS = CallawaySantAnnaES
__all__ = ["CallawaySantAnnaES", "EventStudyCS"]
