"""Spatial Difference-in-Differences.

This module estimates direct and spillover effects using spatial weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    from collections.abc import Sequence

__all__ = ["SpatialDID", "SpatialDIDResult"]

from lineareg.utils.helpers import event_tau, time_to_pos
from lineareg.utils.eventstudy_helpers import pret_for_cohort
from lineareg.utils.helpers import prev_time

LOGGER = logging.getLogger(__name__)


@dataclass
class SpatialDIDResult:
    """Container for Spatial DID estimates.

    Attributes
    ----------
    direct_tau : pd.DataFrame
        Direct effect estimates.
    spill_tau : pd.DataFrame
        Spillover effect estimates.
    bands_direct, bands_spill, bands_beta_s : tuple
        Bootstrap confidence bands.
    model_info : dict
        Model metadata.
    """

    direct_tau: pd.DataFrame
    spill_tau: pd.DataFrame
    beta_s_tau: pd.DataFrame  # raw beta_S (not multiplied by mean exposure) with bands
    # Each bands_* now carries three pairs: (pre, post, full)
    bands_direct: tuple[
        tuple[pd.Series, pd.Series],
        tuple[pd.Series, pd.Series],
        tuple[pd.Series, pd.Series],
    ]
    bands_spill: tuple[
        tuple[pd.Series, pd.Series],
        tuple[pd.Series, pd.Series],
        tuple[pd.Series, pd.Series],
    ]
    bands_beta_s: tuple[
        tuple[pd.Series, pd.Series],
        tuple[pd.Series, pd.Series],
        tuple[pd.Series, pd.Series],
    ]
    model_info: dict[str, object]


class SpatialDID:
    """Spatial Difference-in-Differences estimator.

    Estimates direct and spillover effects using spatial weights matrices.
    Inference via restricted-residual bootstrap (WCR).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        id_name: str,
        t_name: str,
        cohort_name: str,
        treat_name: str,
        y_name: str,
        event_time_name: str | None = None,
        W: la.Matrix,
        row_normalized: bool = True,
        center_at: int = -1,
        anticipation: int = 0,
        base_period: str = "varying",
        control_group: str = "notyet",
        alpha: float = 0.05,
        boot: BootConfig | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        spatial_coords: np.ndarray | None = None,
        spatial_radius: float | None = None,
        residual_type: str = "restricted",
        tau_weight: str = "group",
        rank_policy: str = "stata",
        s_mode: str = "cohort",
    ) -> None:
        self.id_name = str(id_name)
        self.t_name = str(t_name)
        self.cohort_name = str(cohort_name)
        self.treat_name = str(treat_name)
        self.y_name = str(y_name)
        self.event_time_name = None if event_time_name is None else str(event_time_name)
        self.W = W
        self.row_normalized = bool(row_normalized)
        self.center_at = int(center_at)
        self.anticipation = int(anticipation)
        if self.anticipation < 0:
            raise ValueError("anticipation must be >= 0.")
        base_period_norm = str(base_period).lower()
        if base_period_norm not in {"varying", "universal"}:
            raise ValueError("base_period must be 'varying' or 'universal'")
        self.base_period = base_period_norm
        cg_norm = str(control_group).lower().replace("_", "").replace("-", "").replace("treated", "")
        if cg_norm not in {"never", "notyet"}:
            raise ValueError("control_group must be 'never' or 'notyet'.")
        self.control_group = cg_norm
        self.alpha = float(alpha)

        self.boot = boot
        self.cluster_ids = cluster_ids
        self.space_ids = space_ids
        self.time_ids = time_ids
        self.spatial_coords = spatial_coords
        self.spatial_radius = spatial_radius
        if residual_type not in {"restricted", "unrestricted"}:
            msg = 'residual_type must be "restricted" or "unrestricted".'
            raise ValueError(msg)
        self.residual_type = residual_type
        self.tau_weight = str(tau_weight).lower()
        if self.tau_weight not in {"group", "equal", "treated_t"}:
            msg = "tau_weight must be one of {'group','equal','treated_t'}."
            raise ValueError(msg)
        # validate rank policy
        self._rank_policy = str(rank_policy).lower()
        if self._rank_policy not in {"stata", "r"}:
            raise ValueError("rank_policy must be one of {'stata','r'}.")
        self.s_mode = str(s_mode).lower()
        if self.s_mode not in {"cohort", "treated_now"}:
            raise ValueError("s_mode must be 'cohort' (cohort g spillover with t>=g) or 'treated_now' (all currently treated).")

    # ------------------------------------------------------------------
    def _event_time(self, df: pd.DataFrame) -> pd.Series:
        if self.event_time_name and self.event_time_name in df.columns:
            return df[self.event_time_name].astype(int)
        cohort_clean = pd.to_numeric(df[self.cohort_name], errors="coerce").fillna(0).astype(int)
        return (df[self.t_name].astype(int) - cohort_clean).astype(int)

    def _first_difference(self, df: pd.DataFrame) -> pd.DataFrame:
        dfx = df[[self.id_name, self.t_name, self.y_name]].copy()
        dfx = dfx.sort_values([self.id_name, self.t_name])
        dfx["_dY"] = dfx.groupby(self.id_name)[self.y_name].diff()
        return df.merge(
            dfx[[self.id_name, self.t_name, "_dY"]],
            on=[self.id_name, self.t_name],
            how="left",
        )

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame | None = None) -> EstimationResult:
        """Fit Spatial DID model.

        Estimates direct and spillover effects with bootstrap confidence bands.
        """
        if df is None:
            df = getattr(self, "_formula_df", None)
            if df is None:
                raise ValueError(
                    "Provide df explicitly or instantiate via from_formula().",
                )
        else:
            self._formula_df = df

        # keep original order to permit strict realignment of any external IDs
        df0 = df.copy()
        df0["_origpos"] = np.arange(df0.shape[0], dtype=np.int64)
        df = df0.copy()
        df["_tau"] = self._event_time(df)
        df = df.sort_values([self.id_name, self.t_name]).reset_index(drop=True)

        # After sorting, any observation-level arrays provided externally must be
        # realigned using the original row positions. This is critical for
        # clustered/spatial bootstrap correctness.
        perm = df["_origpos"].to_numpy(dtype=np.int64)

        def _align_obs_array(arr, name: str):
            if arr is None:
                return None
            a = np.asarray(arr)
            if a.shape[0] != df0.shape[0]:
                raise ValueError(
                    f"{name} must have length n_obs={df0.shape[0]} (one value per row). Got {a.shape[0]}.",
                )
            return a[perm]

        cluster_ids_aligned = _align_obs_array(self.cluster_ids, "cluster_ids")
        space_ids_aligned = _align_obs_array(self.space_ids, "space_ids")
        time_ids_aligned = _align_obs_array(self.time_ids, "time_ids")
        spatial_coords_aligned = None
        if self.spatial_coords is not None:
            sc = np.asarray(self.spatial_coords, dtype=np.float64)
            if sc.ndim != 2 or sc.shape[0] != df0.shape[0]:
                raise ValueError(
                    f"spatial_coords must be (n_obs x 2) with n_obs={df0.shape[0]}; got shape {sc.shape}.",
                )
            spatial_coords_aligned = sc[perm, :]

        # strict integrity checks
        if df.duplicated(subset=[self.id_name, self.t_name]).any():
            raise ValueError("Duplicate (id,time) rows detected.")
        # cohort must be unique per id (ignore missing/zero cohorts)
        _c = df[[self.id_name, self.cohort_name]].dropna()
        cohort_c_clean = pd.to_numeric(_c[self.cohort_name], errors="coerce").fillna(0).astype(int)
        _c = _c[cohort_c_clean > 0]
        if not _c.empty:
            chk = _c.groupby(self.id_name)[self.cohort_name].nunique()
            if int((chk > 1).sum()) > 0:
                raise ValueError("Cohort must be unique per id.")

        # canonical ids / times and maps to wide indices
        # SpatialDID assumes integer-valued time/cohort indices (event-time is t-g).
        # Enforce this strictly so that base period and anticipation semantics are unambiguous.
        try:
            df[self.t_name] = pd.to_numeric(df[self.t_name], errors="raise").astype(int)
        except Exception as exc:
            raise ValueError("SpatialDID requires an integer-valued time index.") from exc
        try:
            df[self.cohort_name] = (
                pd.to_numeric(df[self.cohort_name], errors="coerce").fillna(0).astype(int)
            )
        except Exception as exc:
            raise ValueError("SpatialDID requires an integer-valued cohort index.") from exc

        ids = np.asarray(sorted(df[self.id_name].unique()))
        times = np.asarray(sorted(df[self.t_name].unique()), dtype=int)
        id_map = {v: i for i, v in enumerate(ids)}
        t_map = time_to_pos(times)

        n_ids = len(ids)
        n_times = len(times)

        # Build wide outcome matrix Y (n_ids x n_times)
        Y = np.full((n_ids, n_times), np.nan, dtype=np.float64)
        # and a treated indicator matrix from the raw data (for validation)
        treated_raw = np.zeros((n_ids, n_times), dtype=np.float64)
        for _, row in df.iterrows():
            i = id_map[row[self.id_name]]
            tt = t_map[row[self.t_name]]
            val = row.get(self.y_name, np.nan)
            try:
                Y[i, tt] = float(val)
            except (TypeError, ValueError):
                Y[i, tt] = np.nan
            tr = row.get(self.treat_name, 0)
            treated_raw[i, tt] = 1.0 if int(tr) == 1 else 0.0

        # Enforce balanced panel for SpatialDID (strict policy)
        if np.isnan(Y).any():
            raise ValueError("SpatialDID requires a balanced panel. Unbalanced panel detected.")

        # Cohort vector per id (G): cohort assigned to id or 0 if missing/<=0
        G = np.zeros(n_ids, dtype=int)
        cohort_series = df.drop_duplicates(subset=[self.id_name]).set_index(
            self.id_name,
        )[self.cohort_name]
        for v, i in id_map.items():
            try:
                c = int(cohort_series.loc[v])
            except (KeyError, TypeError, ValueError):
                c = 0
            G[i] = c
        cohorts = np.unique(G)

        # Staggered-adoption treatment path implied by cohorts: D_it = 1[t >= G_i] for G_i>0.
        treated_now = np.zeros((n_ids, n_times), dtype=np.float64)
        for i in range(n_ids):
            gi = int(G[i])
            if gi <= 0:
                continue
            for tt, tval in enumerate(times):
                treated_now[i, tt] = 1.0 if int(tval) >= gi else 0.0

        # Strict validation: provided treat_name must match the implied staggered-adoption path.
        # This avoids silently mixing inconsistent D definitions across direct/spill components.
        if np.any(np.abs(treated_raw - treated_now) > 0.0):
            raise ValueError(
                "treat_name is inconsistent with cohort timing (expected D_it = 1[t>=g_i]). "
                "Provide a consistent treatment indicator or omit treat_name and rely on cohorts.",
            )

        # Validate W and compute exposure S = W * treated_now (row-normalize if requested)
        if isinstance(self.W, pd.DataFrame):
            Wdf = self.W.copy()
            ids_sorted = ids
            if set(Wdf.index) != set(ids_sorted) or set(Wdf.columns) != set(ids_sorted):
                raise ValueError("W index/columns must match the set of ids.")
            Wd = Wdf.loc[ids_sorted, ids_sorted].to_numpy(dtype=np.float64, copy=False)
        else:
            Wd = la.to_dense(self.W).astype(np.float64, copy=False)
        # enforce zero diagonal (no self-exposure)
        self_loops = float(np.sum(np.diag(Wd) != 0.0))
        np.fill_diagonal(Wd, 0.0)
        # non-negativity check
        if (Wd < 0.0).any():
            raise ValueError("W must be non-negative.")

        # optional row-normalize (strict)
        row_sums = Wd.sum(axis=1)
        zero_rows = int(np.sum(row_sums == 0.0))
        if not self.row_normalized:
            nz = row_sums > 0.0
            Wd[nz, :] = Wd[nz, :] / row_sums[nz].reshape(-1, 1)
        # verify approximately row-normalized
        elif np.max(np.abs(row_sums - 1.0)) > 1e-10:
            raise ValueError(
                "row_normalized=True is declared but row sums deviate from 1. Set row_normalized=False to normalize inside.",
            )

        # Exposure matrix S = W * treated_now (kept for diagnostics only)
        S = la.dot(Wd, treated_now)

        # Define wide indices *before* any use in multiplier grouping
        df["_i"] = df[self.id_name].map(id_map).astype(int)
        df["_tt"] = df[self.t_name].map(t_map).astype(int)

        # Prepare bootstrap multipliers with spatial block bootstrap if spatial_coords provided.
        # For spatial DID, proper spatial inference requires spatial block bootstrap.
        if self.spatial_coords is not None:
            spatial_coords_arr = spatial_coords_aligned
            if spatial_coords_arr is None:
                raise ValueError("spatial_coords alignment failed unexpectedly.")
            if self.spatial_radius is None:
                raise ValueError(
                    "spatial_radius is required when spatial_coords is provided."
                )
            # Use spatial block bootstrap: form spatial clusters and generate multipliers
            n_boot = getattr(self.boot, "n_boot", bt.DEFAULT_BOOTSTRAP_ITERATIONS) if self.boot else bt.DEFAULT_BOOTSTRAP_ITERATIONS
            seed = getattr(self.boot, "seed", None) if self.boot else None
            policy = getattr(self.boot, "policy", "boottest") if self.boot else "boottest"

            W_mult, log_mult, spatial_clusters = bt.spatial_distance_multipliers(
                spatial_coords_arr,
                radius=self.spatial_radius,
                n_boot=n_boot,
                seed=seed,
                policy=policy,
            )
            W_obs = np.asarray(W_mult, dtype=np.float64)
            boot_cfg = None
            W_obs_log = log_mult
        elif self.boot is None:
            # Do not force policy/enumeration here; inject only aligned ID arrays
            # If both space_ids and time_ids are provided (space×time multiway),
            # disable enumeration by default to match boottest parity for multiway.
            if (space_ids_aligned is not None) and (time_ids_aligned is not None):
                boot_cfg = BootConfig(
                    cluster_ids=cluster_ids_aligned,
                    space_ids=space_ids_aligned,
                    time_ids=time_ids_aligned,
                    use_enumeration=False,
                    enumeration_mode="disabled",
                )
            else:
                boot_cfg = BootConfig(
                    cluster_ids=cluster_ids_aligned,
                    space_ids=space_ids_aligned,
                    time_ids=time_ids_aligned,
                )
            # BootConfig.make_multipliers returns (DataFrame, log)
            W_obs_df, _log = boot_cfg.make_multipliers(n_obs=len(df))
            W_obs_log = _log
            # accept either DataFrame or ndarray
            try:
                W_obs = W_obs_df.to_numpy(dtype=np.float64)
            except (AttributeError, TypeError, ValueError):
                W_obs = np.asarray(W_obs_df, dtype=np.float64)
        else:
            # Respect user BootConfig: replace only IDs with aligned versions
            # Build kwargs from existing BootConfig dataclass fields
            bkw = {
                field_name: getattr(self.boot, field_name)
                for field_name in self.boot.__dataclass_fields__
            }
            if cluster_ids_aligned is not None:
                bkw["cluster_ids"] = cluster_ids_aligned
            if space_ids_aligned is not None:
                bkw["space_ids"] = space_ids_aligned
            if time_ids_aligned is not None:
                bkw["time_ids"] = time_ids_aligned
            # If caller provided both space_ids and time_ids, conservatively
            # disable enumeration to avoid unintended enumeration for multiway.
            if (space_ids_aligned is not None) and (time_ids_aligned is not None):
                bkw["use_enumeration"] = False
                bkw["enumeration_mode"] = "disabled"
            boot_cfg = BootConfig(**bkw)
            # BootConfig.make_multipliers returns (DataFrame, log)
            W_obs_df, _log = boot_cfg.make_multipliers(n_obs=len(df))
            W_obs_log = _log
            # accept either DataFrame or ndarray
            try:
                W_obs = W_obs_df.to_numpy(dtype=np.float64)
            except (AttributeError, TypeError, ValueError):
                W_obs = np.asarray(W_obs_df, dtype=np.float64)

        # Wrap multipliers in a DataFrame for reproducibility.
        # IMPORTANT: do NOT recenter/rescale multiplier columns here.
        # The library's wild/multiplier definitions match boottest/fwildclusterboot,
        # which avoids per-column recentering for finite B.
        if isinstance(locals().get("W_obs_df", None), pd.DataFrame):
            W_obs_df_used = locals()["W_obs_df"].copy()
        else:
            W_obs_df_used = pd.DataFrame(
                W_obs,
                columns=[f"b{i}" for i in range(int(W_obs.shape[1]))],
            )

        # NOTE: we intentionally do NOT enforce multiplier-constancy on (unit,D,S)
        # groups here. Multiplier grouping/structure should be configured by
        # the BootConfig (cluster_ids / multiway_ids) passed by the caller.

        # (indices were assigned above)

        def control_mask_strict(t_int: int, base_int: int) -> np.ndarray:
            m = max(int(t_int), int(base_int))
            if self.control_group == "never":
                return G == 0
            # not-yet-treated at time m with anticipation: treated cohorts with G <= m+anticipation are excluded
            return (G == 0) | ((m + int(self.anticipation)) < G)

        # Containers
        atts_direct: list[
            tuple[int, int, int, float, float]
        ] = []  # (g,t,tau,β_D,n_treat)
        atts_spill: list[
            tuple[int, int, int, float, float]
        ] = []  # (g,t,tau,spill,n_exposed)
        atts_beta_s: list[
            tuple[int, int, int, float, float]
        ] = []  # (g,t,tau,β_S,n_exposed)
        direct_boot: list[tuple[int, int, np.ndarray, float]] = []
        spill_boot: list[tuple[int, int, np.ndarray, float]] = []
        bs_boot: list[tuple[int, int, np.ndarray, float]] = []
        used_rows: set[int] = set()

        # Iterate cells (single canonical loop)
        for g in np.sort(cohorts):
            if int(g) <= 0:
                continue
            treat_units = int(g) == G
            if not np.any(treat_units):
                continue
            pret_g = pret_for_cohort(int(g), times, anticipation=int(self.anticipation))
            if pret_g is None:
                continue
            for t in times:
                if self.base_period == "varying" and int(t) < int(g):
                    base_time = prev_time(int(t), times)
                else:
                    base_time = pret_g
                if base_time is None or int(base_time) not in t_map:
                    continue
                ctrl_units = control_mask_strict(int(t), int(base_time))
                sample_units = treat_units | ctrl_units
                if not np.any(sample_units):
                    continue
                tt = int(t_map[int(t)])
                # rows where time==t and unit in sample_units
                row_mask = (df[self.t_name].to_numpy() == int(t)) & sample_units[
                    df["_i"].to_numpy()
                ]
                sub = df.loc[
                    row_mask, [self.treat_name, self.cohort_name, "_i", "_tt"],
                ].copy()
                sub = sub[sub[self.treat_name].isin([0, 1])]
                if sub.empty:
                    continue
                base_time = int(g) - 1
                # base_time already validated above; keep a local int for indexing
                base_time = int(base_time)
                tt_now = sub["_tt"].to_numpy(dtype=int)
                Yi_now = Y[sub["_i"].to_numpy(dtype=int), tt_now]
                Yi_base = Y[sub["_i"].to_numpy(dtype=int), t_map[base_time]]
                ok = np.isfinite(Yi_now) & np.isfinite(Yi_base)
                if not np.any(ok):
                    continue
                # strict row subsetting by finite long-diff
                sub = sub.loc[ok, :].copy()
                for rid in sub.index.to_numpy(dtype=int):
                    used_rows.add(int(rid))
                y = (Yi_now[ok] - Yi_base[ok]).reshape(-1, 1)
                tau_val = int(event_tau(int(t), int(g), t_map))
                cohort_sub_clean = pd.to_numeric(sub[self.cohort_name], errors="coerce").fillna(0).astype(int).to_numpy()
                cohort_indicator = (cohort_sub_clean.reshape(-1, 1) == int(g)).astype(np.float64)
                # In long-difference DID, treated indicator is cohort membership (D_g), not current treatment status.
                Z = cohort_indicator
                unit_indices = sub["_i"].to_numpy(dtype=int)
                if self.s_mode == "treated_now":
                    # Exposure to *all currently treated* units at time t.
                    s_source = treated_now[:, tt].astype(np.float64)
                else:
                    # Cohort-specific exposure: neighbors belonging to cohort g.
                    # For t<g this is a placebo exposure definition; for t>=g it matches treated neighbors.
                    s_source = (G == int(g)).astype(np.float64)
                Svec_raw = la.dot(Wd[unit_indices, :], s_source.reshape(-1, 1))
                exposed_controls = (Svec_raw.reshape(-1) > 0.0) & (Z.reshape(-1) == 0.0)
                n_exposed = float(np.sum(exposed_controls))
                mean_S_exposed = (
                    float(la.col_mean(Svec_raw.reshape(-1)[exposed_controls].reshape(-1, 1))[0])
                    if np.any(exposed_controls)
                    else 1.0
                )
                Svec = np.zeros_like(Svec_raw)
                if np.any(exposed_controls) and mean_S_exposed > 0:
                    Svec[exposed_controls.reshape(-1, 1)] = (
                        Svec_raw[exposed_controls.reshape(-1, 1)] / mean_S_exposed
                    )
                var_Z = float(np.var(Z, ddof=0))
                var_S = float(np.var(Svec, ddof=0))
                if var_Z < 1e-12:
                    continue

                # Spillover on controls is identified only when there exist exposed controls.
                # When not identified, we still estimate the direct effect with the reduced model,
                # but we do NOT treat spillover as 0 (leave missing at aggregation level).
                use_spill = (n_exposed > 0.0) and (var_S >= 1e-12)
                if use_spill:
                    X = la.hstack([np.ones((y.shape[0], 1), dtype=np.float64), Z, Svec])
                else:
                    X = la.hstack([np.ones((y.shape[0], 1), dtype=np.float64), Z])
                try:
                    # R/Stata parity: pivoted-QR with rank policy
                    beta = la.solve(X, y, method="qr", rank_policy=self._rank_policy)
                except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
                    LOGGER.debug(
                        "Skipping tau=%s due to solver failure: %s",
                        tau_val,
                        exc,
                    )
                    continue
                beta_D = float(beta[1, 0]) if beta.shape[0] > 1 else np.nan
                beta_S = float(beta[2, 0]) if (use_spill and beta.shape[0] > 2) else np.nan
                tau = int(tau_val)
                n_treat = float(np.sum(Z))
                atts_direct.append((int(g), int(t), tau, beta_D, n_treat))
                if use_spill:
                    atts_spill.append((int(g), int(t), tau, beta_S, n_exposed))
                    atts_beta_s.append((int(g), int(t), tau, beta_S, n_exposed))

                # Bootstrap draws for the cell (observation-level multipliers prebuilt)
                W_cell = W_obs[sub.index.to_numpy(dtype=int), :]
                if self.residual_type == "restricted":
                    # --- D-only restriction ---
                    R_D = np.array([[0.0, 1.0, 0.0]], dtype=np.float64) if use_spill else np.array([[0.0, 1.0]], dtype=np.float64)
                    Gmat = la.tdot(X)
                    Ginv_RT_D = la.solve(Gmat, R_D.T, sym_pos=True)
                    lam_D = la.solve(
                        la.dot(R_D, Ginv_RT_D), la.dot(R_D, beta), sym_pos=True,
                    )
                    beta_RD = beta - la.dot(Ginv_RT_D, lam_D)
                    yhat_RD = la.dot(X, beta_RD)
                    resid_RD = y - yhat_RD
                    Ystar_D = yhat_RD + resid_RD * W_cell
                    # --- S-only restriction ---
                    if use_spill:
                        R_S = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
                        Ginv_RT_S = la.solve(Gmat, R_S.T, sym_pos=True)
                        lam_S = la.solve(
                            la.dot(R_S, Ginv_RT_S), la.dot(R_S, beta), sym_pos=True,
                        )
                        beta_RS = beta - la.dot(Ginv_RT_S, lam_S)
                        yhat_RS = la.dot(X, beta_RS)
                        resid_RS = y - yhat_RS
                        Ystar_S = yhat_RS + resid_RS * W_cell
                    else:
                        Ystar_S = Ystar_D
                else:
                    yhat = la.dot(X, beta)
                    resid = y - yhat
                    Ystar_D = yhat + resid * W_cell
                    Ystar_S = Ystar_D

                try:
                    beta_star_D = la.solve(
                        X, Ystar_D, method="qr", rank_policy=self._rank_policy,
                    )
                    beta_star_S = la.solve(
                        X, Ystar_S, method="qr", rank_policy=self._rank_policy,
                    )
                except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
                    LOGGER.debug(
                        "Skipping bootstrap refit for (g=%s, t=%s) due to failure: %s",
                        g,
                        t,
                        exc,
                    )
                    continue
                betaD_star = (
                    beta_star_D[1:2, :]
                    if beta_star_D.shape[0] > 1
                    else np.full((1, beta_star_D.shape[1]), np.nan)
                )
                betaS_star = (
                    beta_star_S[2:3, :]
                    if beta_star_S.shape[0] > 2
                    else np.zeros((1, beta_star_S.shape[1]), dtype=np.float64)
                )
                direct_boot.append((int(g), int(t), betaD_star.reshape(-1), n_treat))
                if use_spill:
                    spill_boot.append((int(g), int(t), betaS_star.reshape(-1), n_exposed))
                    bs_boot.append((int(g), int(t), betaS_star.reshape(-1), n_exposed))

        if not atts_direct:
            raise RuntimeError("No valid (g,t) cells for Spatial DID.")

        # Build DataFrames
        df_dir = pd.DataFrame(
            atts_direct, columns=["g", "t", "tau", "beta_D", "n_w"],
        )  # direct coefficient
        df_spi = pd.DataFrame(
            atts_spill, columns=["g", "t", "tau", "spill", "n_w"],
        )  # spill interpreted
        df_bs = pd.DataFrame(
            atts_beta_s, columns=["g", "t", "tau", "beta_Sc", "n_w"],
        )  # spill on controls (base)

        # τ aggregation with weights (do not floor n_w; zero-exposure rows excluded above)
        # group size n_g for "group" weights
        # R did/csdid compatibility: N_g = number of unique ids with cohort == g
        group_size: dict[int, int] = {}
        cohort_all_clean = pd.to_numeric(df[self.cohort_name], errors="coerce").fillna(0).astype(int).to_numpy()
        cohort_vals = np.sort(np.unique(cohort_all_clean))
        for g in cohort_vals:
            if int(g) <= 0:
                continue
            ids_g = df.loc[cohort_all_clean == int(g), self.id_name].unique()
            group_size[int(g)] = len(ids_g)

        def _agg(df_in: pd.DataFrame, val_col: str, kind: str) -> pd.DataFrame:
            if df_in.empty:
                return pd.DataFrame({"tau": [], "params": []})

            def _w_for_chunk(d: pd.DataFrame) -> float:
                mode = self.tau_weight
                if mode == "equal":
                    w = np.ones(d.shape[0], dtype=np.float64)
                elif mode == "group":
                    w = np.array(
                        [group_size.get(int(g), 0) for g in d["g"].to_numpy(dtype=int)],
                        dtype=np.float64,
                    )
                else:  # "treated_t" : default direct uses treated_t; spill falls back to group
                    tau_val = (
                        int(d["tau"].iloc[0])
                        if "tau" in d.columns and len(d) > 0
                        else 0
                    )
                    if kind == "direct" and tau_val < 0:
                        # pre-period cells have zero treated-at-time counts; fall back to cohort size
                        w = np.array(
                            [
                                group_size.get(int(g), 0)
                                for g in d["g"].to_numpy(dtype=int)
                            ],
                            dtype=np.float64,
                        )
                    elif kind == "direct":
                        w = d["n_w"].to_numpy(
                            dtype=np.float64,
                        )  # number treated in cell (time t)
                    else:
                        w = np.array(
                            [
                                group_size.get(int(g), 0)
                                for g in d["g"].to_numpy(dtype=int)
                            ],
                            dtype=np.float64,
                        )
                wsum = float(np.sum(w))
                if wsum <= 0:
                    return float("nan")
                return float(np.sum(w * d[val_col].to_numpy(dtype=np.float64)) / wsum)

            # Avoid pandas groupby.apply reduction FutureWarning by explicit iteration
            rows = []
            # preserve natural order of tau appearance
            for tau_val, dchunk in df_in.groupby("tau", sort=False):
                rows.append({"tau": int(tau_val), "params": _w_for_chunk(dchunk)})
            return pd.DataFrame(rows)

        dir_tau = (
            _agg(df_dir, "beta_D", kind="direct")
            .sort_values("tau")
            .reset_index(drop=True)
        )
        spi_tau = (
            _agg(df_spi, "spill", kind="spill")
            .sort_values("tau")
            .reset_index(drop=True)
        )
        bs_tau = (
            _agg(df_bs, "beta_Sc", kind="spill")
            .sort_values("tau")
            .reset_index(drop=True)
        )
        if isinstance(dir_tau, pd.DataFrame) and not dir_tau.empty:
            tau_grid = pd.Index(dir_tau["tau"].astype(int).tolist(), name="tau")

            def _align(df_in: pd.DataFrame | None, fill: float) -> pd.DataFrame:
                if df_in is None or df_in.empty:
                    return pd.DataFrame(
                        {
                            "tau": tau_grid,
                            "params": np.full(len(tau_grid), fill, dtype=np.float64),
                        },
                    )
                df_tmp = df_in.set_index("tau").reindex(tau_grid)
                df_tmp["params"] = df_tmp["params"].fillna(fill)
                df_tmp = df_tmp.reset_index()
                df_tmp["tau"] = df_tmp["tau"].astype(int)
                return df_tmp.sort_values("tau").reset_index(drop=True)

            # For spillover/beta_S series, missing τ means spillover is not identified
            # (e.g., no exposed controls for that τ). Keep as NaN rather than forcing 0.
            spi_tau = _align(spi_tau, np.nan)
            bs_tau = _align(bs_tau, np.nan)

        # Normalize all event-time paths at the base period (tau == center_at).
        base_tau = int(self.center_at)

        def _center_series(df_in: pd.DataFrame) -> float:
            base_rows = df_in.index[df_in["tau"].astype(int) == base_tau]
            if base_rows.size != 1:
                raise ValueError(
                    f"Expected exactly one base_tau={base_tau} row; got {base_rows.size}.",
                )
            base_row = int(base_rows[0])
            theta_base = float(df_in.loc[base_row, "params"])
            if not np.isfinite(theta_base):
                raise ValueError(
                    f"Non-finite base-period estimate at tau={base_tau}.",
                )
            df_in["params"] = df_in["params"].astype(float) - theta_base
            df_in.loc[base_row, "params"] = 0.0
            return theta_base

        theta_base_dir = _center_series(dir_tau)

        def _center_series_allow_nan_base(df_in: pd.DataFrame) -> float:
            base_rows = df_in.index[df_in["tau"].astype(int) == base_tau]
            if base_rows.size != 1:
                raise ValueError(
                    f"Expected exactly one base_tau={base_tau} row; got {base_rows.size}.",
                )
            base_row = int(base_rows[0])
            theta_base = float(df_in.loc[base_row, "params"])
            # If base is missing (unidentified), enforce normalization by fixing it to 0.
            if not np.isfinite(theta_base):
                df_in.loc[base_row, "params"] = 0.0
                theta_base = 0.0
            df_in["params"] = df_in["params"].astype(float) - float(theta_base)
            df_in.loc[base_row, "params"] = 0.0
            return float(theta_base)

        theta_base_spill = _center_series_allow_nan_base(spi_tau)
        theta_base_bs = _center_series_allow_nan_base(bs_tau)

        # Bootstrap aggregation
        def _aggregate_boot(
            cell_boot: list[tuple[int, int, np.ndarray, float]],
            df_in: pd.DataFrame,
            kind: str,
        ) -> tuple[np.ndarray, list[int]]:
            tau_vals = sorted(df_in["tau"].unique().tolist())
            K = len(tau_vals)
            B = len(cell_boot[0][2]) if cell_boot else 0
            out = np.zeros((K, B), dtype=np.float64)
            for j, tau in enumerate(tau_vals):
                rows = df_in[df_in["tau"] == tau]
                accum = np.zeros(B, dtype=np.float64)
                wsum = 0.0
                for _, r in rows.iterrows():
                    g_val = int(r["g"])
                    t_val = int(r["t"])
                    match = next(
                        (cb for cb in cell_boot if cb[0] == g_val and cb[1] == t_val),
                        None,
                    )
                    if match is None:
                        continue
                    (_, _, vec, w_cell) = match
                    # use the same τ-aggregation weight rule
                    if self.tau_weight == "equal":
                        w = 1.0
                    elif self.tau_weight == "group":
                        w = float(group_size.get(g_val, 0))
                    elif kind == "direct":
                        # Pre-period τ: group-size weights; post: treated-at-time weights
                        if int(tau) < int(self.center_at):
                            w = float(group_size.get(g_val, 0))
                        else:
                            w = float(w_cell)
                    else:
                        w = float(group_size.get(g_val, 0))
                    accum += w * vec
                    wsum += w
                if wsum > 0:
                    out[j, :] = accum / wsum
            return out, tau_vals

        dir_tau_star, dir_star_grid = _aggregate_boot(
            direct_boot, df_dir, kind="direct",
        )
        spi_tau_star, spi_star_grid = _aggregate_boot(spill_boot, df_spi, kind="spill")
        bs_tau_star, bs_star_grid = _aggregate_boot(bs_boot, df_bs, kind="spill")

        def _align_star_matrix(star: np.ndarray, tau_vals: list[int]) -> np.ndarray:
            if dir_tau.empty:
                return star
            target = [int(t) for t in dir_tau["tau"].tolist()]
            B = star.shape[1] if star.ndim == 2 else 0
            aligned = np.full((len(target), B), np.nan, dtype=np.float64)
            idx_map = {int(t): i for i, t in enumerate(tau_vals)}
            for i, tau in enumerate(target):
                j = idx_map.get(int(tau))
                if j is not None and star.size:
                    aligned[i, :] = star[j, :]
            return aligned

        dir_tau_star = _align_star_matrix(dir_tau_star, dir_star_grid)
        spi_tau_star = _align_star_matrix(spi_tau_star, spi_star_grid)
        bs_tau_star = _align_star_matrix(bs_tau_star, bs_star_grid)

        def _center_star(att_tau_df: pd.DataFrame, star: np.ndarray) -> np.ndarray:
            if star.size == 0:
                return star
            tau_arr = att_tau_df["tau"].to_numpy(dtype=int)
            base_locs = np.flatnonzero(tau_arr == base_tau)
            if base_locs.size != 1:
                raise ValueError(
                    f"Expected exactly one base_tau={base_tau} in bootstrap grid; got {base_locs.size}.",
                )
            bidx = int(base_locs[0])
            base_draw = star[bidx, :].copy()
            # If base is not identified (NaN draws), keep normalization as 0.
            if not np.isfinite(base_draw).all():
                base_draw = np.zeros_like(base_draw)
            star = star - base_draw[None, :]
            star[bidx, :] = 0.0
            return star

        dir_tau_star = _center_star(dir_tau, dir_tau_star)
        spi_tau_star = _center_star(spi_tau, spi_tau_star)
        bs_tau_star = _center_star(bs_tau, bs_tau_star)

        # Bands helper (pre/post/full) using sup-t bootstrap
        def _bands(
            att_tau_df: pd.DataFrame,
            att_tau_star: np.ndarray,
        ) -> tuple[
            tuple[pd.Series, pd.Series],
            tuple[pd.Series, pd.Series],
            tuple[pd.Series, pd.Series],
        ]:
            tau_array = att_tau_df["tau"].to_numpy(dtype=int)
            tau_to_row = {tau: i for i, tau in enumerate(sorted(tau_array))}
            star_ordered = (
                np.vstack([att_tau_star[tau_to_row[tau], :] for tau in tau_array])
                if att_tau_star.size
                else np.zeros((len(tau_array), 0))
            )
            pre_mask = tau_array < self.center_at
            # STRICT: exclude baseline (tau == center_at) from post
            post_mask = tau_array > self.center_at
            # full (exclude baseline): sup-t over all non-baseline τ
            full_mask = tau_array != self.center_at

            def _one(mask: np.ndarray) -> tuple[pd.Series, pd.Series]:
                if not np.any(mask) or star_ordered.shape[1] == 0:
                    return pd.Series(dtype=float), pd.Series(dtype=float)
                theta_all = att_tau_df.loc[mask, "params"].to_numpy(dtype=np.float64)
                th_star_all = star_ordered[mask, :]
                ok = np.isfinite(theta_all) & np.all(np.isfinite(th_star_all), axis=1)
                if not np.any(ok):
                    return pd.Series(dtype=float), pd.Series(dtype=float)
                theta = theta_all[ok]
                th_star = th_star_all[ok, :]
                lo, hi = bt.uniform_confidence_band(
                    theta,
                    th_star,
                    alpha=self.alpha,
                    studentize="bootstrap",
                    context="eventstudy",
                )
                idx = att_tau_df.index[mask][ok]
                return pd.Series(lo, index=idx), pd.Series(hi, index=idx)

            pre_pair = _one(pre_mask)
            post_pair = _one(post_mask)
            full_pair = _one(full_mask)
            return pre_pair, post_pair, full_pair

        bands_direct = _bands(dir_tau, dir_tau_star)
        bands_spill = _bands(spi_tau, spi_tau_star)
        bands_beta_s = _bands(bs_tau, bs_tau_star)

        # --- post-period aggregated ATE (tau >= center_at) with bootstrap bands (no p-values) ---
        def _post_agg(
            att_tau_df: pd.DataFrame, att_star: np.ndarray, df_cells: pd.DataFrame,
        ) -> tuple[float, tuple[float, float]]:
            if att_tau_df.empty or att_star.size == 0:
                return np.nan, (np.nan, np.nan)
            # STRICT: aggregate over post τ only (exclude baseline)
            taus = att_tau_df.loc[att_tau_df["tau"] > self.center_at, "tau"].to_numpy(
                dtype=int,
            )
            if taus.size == 0:
                return np.nan, (np.nan, np.nan)
            # PostATE aggregation strictly follows tau_weight
            mode = self.tau_weight
            if mode == "equal":
                w = np.ones(taus.size, dtype=np.float64)
            elif mode == "group":
                # For each tau, find associated cohorts and sum their group sizes
                w_list = []
                for tau_val in taus:
                    cohorts_for_tau = df_cells[df_cells["tau"] == tau_val]["g"].unique()
                    w_tau = sum(group_size.get(int(g), 0) for g in cohorts_for_tau)
                    w_list.append(w_tau)
                w = np.array(w_list, dtype=np.float64)
            # direct: treated_t; spill: fallback to group
            elif "n_w" in df_cells.columns:
                w = np.array(
                    [df_cells.loc[df_cells["tau"] == t, "n_w"].sum() for t in taus],
                    dtype=np.float64,
                )
            else:
                # For spill, fallback to group
                w_list = []
                for tau_val in taus:
                    cohorts_for_tau = df_cells[df_cells["tau"] == tau_val]["g"].unique()
                    w_tau = sum(group_size.get(int(g), 0) for g in cohorts_for_tau)
                    w_list.append(w_tau)
                w = np.array(w_list, dtype=np.float64)
            if w.sum() <= 0:
                return np.nan, (np.nan, np.nan)
            w = w / w.sum()
            tau_to_param = att_tau_df.set_index("tau")["params"].to_dict()
            vals = np.asarray(
                [float(tau_to_param[t]) for t in taus],
                dtype=np.float64,
            )
            ok = np.isfinite(vals)
            if not np.any(ok):
                return np.nan, (np.nan, np.nan)
            taus = taus[ok]
            vals = vals[ok]
            w = w[ok]
            wsum = float(w.sum())
            if wsum <= 0:
                return np.nan, (np.nan, np.nan)
            w = w / wsum
            theta = float(np.sum(w * vals))
            rowpos = {t: i for i, t in enumerate(att_tau_df["tau"].to_list())}
            rows = np.array([rowpos[t] for t in taus], dtype=int)
            star = (w.reshape(-1, 1) * att_star[rows, :]).sum(axis=0)
            # Use bootstrap studentization for Post ATE band via centralized helper.
            lo_b, hi_b = bt.uniform_confidence_band(
                np.array([theta], dtype=np.float64),
                star.reshape(1, -1),
                alpha=self.alpha,
                studentize="bootstrap",
                context="eventstudy",
            )
            return theta, (float(lo_b[0]), float(hi_b[0]))

        post_direct, post_direct_band = _post_agg(dir_tau, dir_tau_star, df_dir)
        # Use beta_S (unscaled) for post-aggregation spillover inference
        post_spill, post_spill_band = _post_agg(bs_tau, bs_tau_star, df_bs)
        post_beta_s, post_beta_s_band = _post_agg(bs_tau, bs_tau_star, df_bs)

        def _attach(df_series: pd.DataFrame, bands) -> pd.DataFrame:
            (lo_pre, hi_pre), (lo_post, hi_post), (lo_full, hi_full) = bands
            out = df_series.copy()
            for c in [
                "pre_lower_95",
                "pre_upper_95",
                "post_lower_95",
                "post_upper_95",
                "full_lower_95",
                "full_upper_95",
            ]:
                out[c] = np.nan
            for idx, val in lo_pre.items():
                out.loc[idx, "pre_lower_95"] = val
                out.loc[idx, "pre_upper_95"] = hi_pre[idx]
            for idx, val in lo_post.items():
                out.loc[idx, "post_lower_95"] = val
                out.loc[idx, "post_upper_95"] = hi_post[idx]
            # write true full sup-t band (non-baseline)
            for idx, val in lo_full.items():
                out.loc[idx, "full_lower_95"] = val
                out.loc[idx, "full_upper_95"] = hi_full[idx]
            return out

        dir_tau_out = _attach(dir_tau, bands_direct)
        # Use beta_S (unscaled) for spillover parameter series with valid CIs
        spi_tau_out = _attach(
            bs_tau, bands_spill,
        )  # Display beta_S, not scaled spillover
        bs_tau_out = _attach(bs_tau, bands_beta_s)

        # number of bootstrap replications used & store multipliers for reproducibility
        B = int(W_obs.shape[1])
        W_used = W_obs_df_used

        bootstrap_desc = (
            "wild residual (WCR/WCU), shared multipliers by "
            + (
                "cluster x time"
                if (self.cluster_ids is not None and self.time_ids is not None)
                else (
                    "cluster"
                    if self.cluster_ids is not None
                    else (
                        "calendar time" if self.time_ids is not None else "observation"
                    )
                )
            )
            + "; policy=delegated_to_BootConfig_or_user"
        )

        spill_interp = (
            "β_S^C (controls) and β_S^C x mean(S | exposed controls); "
            "β_S^T=β_S^C+β_DS (available in cell-level beta)"
        )

        info: dict[str, object] = {
            "Estimator": "EventStudy: Spatial-DID",
            "RowNormalizedW": self.row_normalized,
            "ResidualType": self.residual_type,
            "Bootstrap": bootstrap_desc,

            "CenterAt": self.center_at,
            "ZeroRowsW": zero_rows,
            "SelfLoops": self_loops,
            "SpillInterpretation": spill_interp,
            "B": int(B),
            "PostATT": post_direct,
            "PostSpill": post_spill,
            "PostBetaS": post_beta_s,
            "PostAggregates": {
                "direct": {"value": post_direct, "band95": post_direct_band},
                "spill": {"value": post_spill, "band95": post_spill_band},
                "beta_s": {"value": post_beta_s, "band95": post_beta_s_band},
            },
            "TauWeight": self.tau_weight,
            "GroupSizeMap": group_size,
        }
        if W_obs_log is not None:
            info["MultipliersLog"] = W_obs_log
        if dir_tau_star.shape[1] > 1:
            post_mask = dir_tau["tau"].to_numpy(dtype=int) > self.center_at
            if np.any(post_mask):
                post_draws = np.nanmean(dir_tau_star[post_mask, :], axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    info["PostATT_se"] = float(np.std(post_draws, ddof=1)) if post_draws.size > 1 else 0.0
        # Compute PostSpill_se from bootstrap draws
        if bs_tau_star.shape[1] > 1:
            post_mask_spill = bs_tau["tau"].to_numpy(dtype=int) > self.center_at
            if np.any(post_mask_spill):
                post_spill_draws = np.nanmean(bs_tau_star[post_mask_spill, :], axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    info["PostSpill_se"] = float(np.std(post_spill_draws, ddof=1)) if post_spill_draws.size > 1 else 0.0
        # Add diagnostics about exposure matrix W and S among controls
        try:
            row_sums = Wd.sum(axis=1)
            W_norm_flag = (
                "row-stochastic"
                if np.allclose(row_sums, 1.0, atol=1e-10)
                else "not-row-stochastic"
            )
        except (AttributeError, TypeError, ValueError) as exc:
            LOGGER.debug("Unable to summarize row sums for W: %s", exc)
            W_norm_flag = "unknown"
        # Fraction of rows whose row-sum is (nearly) equal to the median row-sum.
        try:
            rs = Wd.sum(axis=1)
            med = float(np.median(rs))
            frac_close = float(np.mean(np.isclose(rs, med, atol=1e-8)))
        except (TypeError, ValueError) as exc:
            LOGGER.debug("Unable to compute W diagnostics: %s", exc)
            frac_close = float("nan")
        try:
            # S controls: S values for units that are never-treated (G==0) and have positive exposure
            S_controls_vals = S[(G == 0) & (S > 0.0)].ravel()
            S_stats = {
                "mean_S_controls": float(np.nanmean(S_controls_vals))
                if S_controls_vals.size
                else float("nan"),
                "var_S_controls": float(np.nanvar(S_controls_vals))
                if S_controls_vals.size
                else float("nan"),
            }
        except (FloatingPointError, ValueError) as exc:
            LOGGER.debug("Unable to compute spillover stats: %s", exc)
            S_stats = {"mean_S_controls": None, "var_S_controls": None}
        info.update({"W_norm": W_norm_flag, "S_stats": S_stats})
        info.update({"W_RowSumFractionCloseToMedian": frac_close})

        # --- STANDARD EstimationResult export (direct effects) ---
        # Convert tuple bands to standard dict format for summary.py/plots.py compatibility
        # STANDARDIZATION: All three series (direct, spill, beta_s) get pre/post/full uniform bands
        (
            (lo_pre_direct, hi_pre_direct),
            (lo_post_direct, hi_post_direct),
            (lo_full_direct, hi_full_direct),
        ) = bands_direct
        # Reindex band Series by actual tau values so plotting can match by tau
        try:
            tau_dir_vals = dir_tau.set_index("tau").index
            lo_pre_direct.index = lo_pre_direct.index.map(
                lambda i: int(dir_tau.loc[i, "tau"]),
            )
            hi_pre_direct.index = hi_pre_direct.index.map(
                lambda i: int(dir_tau.loc[i, "tau"]),
            )
            lo_post_direct.index = lo_post_direct.index.map(
                lambda i: int(dir_tau.loc[i, "tau"]),
            )
            hi_post_direct.index = hi_post_direct.index.map(
                lambda i: int(dir_tau.loc[i, "tau"]),
            )
            lo_full_direct.index = lo_full_direct.index.map(
                lambda i: int(dir_tau.loc[i, "tau"]),
            )
            hi_full_direct.index = hi_full_direct.index.map(
                lambda i: int(dir_tau.loc[i, "tau"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.debug("Failed to realign direct uniform bands: %s", exc)
        # Attach provenance metadata required by plotting API (uniform/sup-t)
        _bands_meta = {
            "origin": "bootstrap",
            "policy": str(getattr(boot_cfg, "policy", "bootstrap"))
            if "boot_cfg" in locals()
            else "bootstrap",
            "dist": getattr(boot_cfg, "dist", None) if "boot_cfg" in locals() else None,
            "kind": "uniform",
            "level": 95,
            "B": int(B),
            # mark as DiD/event-study family for uniform-band plots
            "estimator": "did",
        }
        standard_bands_direct = {
            "pre": pd.DataFrame({"lower": lo_pre_direct, "upper": hi_pre_direct}),
            "post": pd.DataFrame({"lower": lo_post_direct, "upper": hi_post_direct}),
            "full": pd.DataFrame({"lower": lo_full_direct, "upper": hi_full_direct}),
            "__meta__": _bands_meta,
        }

        (
            (lo_pre_spill, hi_pre_spill),
            (lo_post_spill, hi_post_spill),
            (lo_full_spill, hi_full_spill),
        ) = bands_spill
        # Ensure index is tau (int) to avoid NaN band issues
        for s in (
            lo_pre_spill,
            hi_pre_spill,
            lo_post_spill,
            hi_post_spill,
            lo_full_spill,
            hi_full_spill,
        ):
            if isinstance(s, pd.Series):
                s.index = s.index.map(lambda i: int(spi_tau.loc[i, "tau"]))
        standard_bands_spill = {
            "pre": pd.DataFrame({"lower": lo_pre_spill, "upper": hi_pre_spill}),
            "post": pd.DataFrame({"lower": lo_post_spill, "upper": hi_post_spill}),
            "full": pd.DataFrame({"lower": lo_full_spill, "upper": hi_full_spill}),
            "post_scalar_spill": pd.DataFrame(
                {
                    "lower": [float(post_spill_band[0])],
                    "upper": [float(post_spill_band[1])],
                },
            ),
            "__meta__": _bands_meta,
        }

        (lo_pre_bs, hi_pre_bs), (lo_post_bs, hi_post_bs), (lo_full_bs, hi_full_bs) = (
            bands_beta_s
        )
        try:
            lo_pre_bs.index = lo_pre_bs.index.map(lambda i: int(bs_tau.loc[i, "tau"]))
            hi_pre_bs.index = hi_pre_bs.index.map(lambda i: int(bs_tau.loc[i, "tau"]))
            lo_post_bs.index = lo_post_bs.index.map(lambda i: int(bs_tau.loc[i, "tau"]))
            hi_post_bs.index = hi_post_bs.index.map(lambda i: int(bs_tau.loc[i, "tau"]))
            lo_full_bs.index = lo_full_bs.index.map(lambda i: int(bs_tau.loc[i, "tau"]))
            hi_full_bs.index = hi_full_bs.index.map(lambda i: int(bs_tau.loc[i, "tau"]))
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.debug("Failed to realign beta_s uniform bands: %s", exc)
        standard_bands_beta_s = {
            "pre": pd.DataFrame({"lower": lo_pre_bs, "upper": hi_pre_bs}),
            "post": pd.DataFrame({"lower": lo_post_bs, "upper": hi_post_bs}),
            "full": pd.DataFrame({"lower": lo_full_bs, "upper": hi_full_bs}),
            "post_scalar_beta_s": pd.DataFrame(
                {
                    "lower": [float(post_beta_s_band[0])],
                    "upper": [float(post_beta_s_band[1])],
                },
            ),
            "__meta__": _bands_meta,
        }

        tau_idx = dir_tau_out.set_index("tau").index.to_numpy(dtype=int)
        se_source_val = None
        if dir_tau_star.shape[1] > 1:
            se_vals = bt.bootstrap_se(dir_tau_star)
            se_series = pd.Series(se_vals, index=tau_idx)
            if int(self.center_at) in se_series.index:
                se_series.loc[int(self.center_at)] = 0.0
            se_source_val = "bootstrap"
        else:
            se_series = None

        # Primary return: EstimationResult for direct effects (standard format)
        return EstimationResult(
            params=dir_tau_out.set_index("tau")["params"],
            se=se_series,
            bands={
                **standard_bands_direct,
                "post_scalar": pd.DataFrame(
                    {
                        "lower": [float(post_direct_band[0])],
                        "upper": [float(post_direct_band[1])],
                    },
                ),
                "post_scalar_spill": pd.DataFrame(
                    {
                        "lower": [float(post_spill_band[0])],
                        "upper": [float(post_spill_band[1])],
                    },
                ),
            },
            n_obs=len(used_rows),
            model_info={
                **info,
                "EffectType": "Direct",
                "Note": "Use .extra['custom_result'] for full SpatialDIDResult with spill effects",
                "CenterAt": int(self.center_at),
                "UniformBandsStandardized": {
                    "description": "All three series (direct, spill, beta_s) use pre/post/full uniform bands with bootstrap studentization",
                    "series": ["direct_tau", "spill_tau", "beta_s_tau"],
                    "bands_available": ["pre", "post", "full"],
                    "studentization": "bootstrap (B+1) sup-t over non-baseline τ",
                },
            },
            extra={
                "custom_result": SpatialDIDResult(
                    direct_tau=dir_tau_out,
                    spill_tau=spi_tau_out,
                    beta_s_tau=bs_tau_out,
                    bands_direct=bands_direct,
                    bands_spill=bands_spill,
                    bands_beta_s=bands_beta_s,
                    model_info=info,
                ),
                "spill_tau": spi_tau_out,
                "beta_s_tau": bs_tau_out,
                "spill_tau_star": spi_tau_star,
                "boot_config": boot_cfg if "boot_cfg" in locals() else None,
                "W_multipliers_inference": W_used,
                "multipliers_log": W_obs_log if "W_obs_log" in locals() else None,
                "bands_source": "bootstrap",
                "se_source": se_source_val,
                "bands_spill_standard": standard_bands_spill,
                "bands_beta_s_standard": standard_bands_beta_s,
                "post_scalar_direct": pd.DataFrame(
                    {
                        "lower": [float(post_direct_band[0])],
                        "upper": [float(post_direct_band[1])],
                    },
                ),
                "post_scalar_spill": pd.DataFrame(
                    {
                        "lower": [float(post_spill_band[0])],
                        "upper": [float(post_spill_band[1])],
                    },
                ),
                "post_scalar_beta_s": pd.DataFrame(
                    {
                        "lower": [float(post_beta_s_band[0])],
                        "upper": [float(post_beta_s_band[1])],
                    },
                ),
                "boot_cluster_requested": "spatial_distance"
                if self.spatial_coords is not None
                else ("space_time" if (space_ids_aligned is not None and time_ids_aligned is not None) else ("cluster" if cluster_ids_aligned is not None else "iid")),
            },
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
        W: la.Matrix | None = None,
        boot: BootConfig | None = None,
        **kwargs,
    ) -> SpatialDID:
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
            default_boot_kwargs={
                "policy": "boottest",
                "enumeration_mode": "boottest",
                "dist": "standard_normal",
            },
        )
        meta.attrs["_formula_df"] = df_use

        kw = dict(kwargs) if kwargs else {}
        id_name = id_col if id_col is not None else kw.pop("id_name", None)
        t_name = time_col if time_col is not None else kw.pop("t_name", None)
        cohort_name = kw.pop("cohort_name", None)
        treat_name = kw.pop("treat_name", None)
        y_name = kw.pop("y_name", None)
        # Infer y_name from formula LHS if not explicitly provided
        if y_name is None and formula is not None and "~" in formula:
            y_name = formula.split("~", 1)[0].strip()
        kw.pop("W", None)
        kw.pop("boot", None)

        boot_to_use = boot_eff if boot_eff is not None else boot

        inst = cls(
            id_name=id_name,
            t_name=t_name,
            cohort_name=cohort_name,
            treat_name=treat_name,
            y_name=y_name,
            W=W,
            boot=boot_to_use,
            **kw,
        )
        attach_formula_metadata(inst, meta)
        inst._formula_df = df_use
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
        W: la.Matrix | None = None,
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
            W=W,
            boot=boot,
            **kwargs,
        )
        extra = fit_kwargs or {}
        return est.fit(**extra)
