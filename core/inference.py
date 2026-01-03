"""Inference utilities for event-study estimators.

This module provides tools for computing simultaneous uniform confidence bands and
aggregated Average Treatment Effects on the Treated (ATT) inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from lineareg.core import bootstrap as bt

Band = tuple[pd.Series, pd.Series]


def compute_uniform_bands(
    att_tau: pd.DataFrame,
    att_tau_star: NDArray[np.float64],
    base_tau: int,
    alpha: float,
) -> tuple[pd.DataFrame, Band, Band, Band]:
    """Compute simultaneous sup-t confidence bands.

    Calculates uniform confidence bands for pre-treatment, post-treatment,
    and all periods using the provided bootstrap statistics.
    """
    if not isinstance(att_tau, pd.DataFrame):
        raise TypeError("att_tau must be a pandas DataFrame")
    if "tau" not in att_tau.columns or "att" not in att_tau.columns:
        raise ValueError("att_tau must contain columns {'tau','att'}")

    att_tau_star = np.asarray(att_tau_star, dtype=np.float64)
    if att_tau_star.ndim != 2:
        B = 0
    else:
        # Contract: att_tau_star is (K x B) where K == number of taus.
        # If the array is transposed (B x K), repair it deterministically.
        K = int(att_tau.shape[0])
        if att_tau_star.shape[0] != K and att_tau_star.shape[1] == K:
            att_tau_star = att_tau_star.T
        if att_tau_star.shape[0] != K:
            raise ValueError(
                "att_tau_star must have shape (K,B) with K=len(att_tau); "
                f"got {att_tau_star.shape} with K={K}.",
            )
        B = int(att_tau_star.shape[1])
    if B < 2:
        att_tau = att_tau.copy()
        att_tau["ci_lower"] = np.nan
        att_tau["ci_upper"] = np.nan
        att_tau["ci_lower_full"] = np.nan
        att_tau["ci_upper_full"] = np.nan
        empty_band = (pd.Series([], dtype=float), pd.Series([], dtype=float))
        return att_tau, empty_band, empty_band, empty_band

    taus = att_tau["tau"].to_numpy(dtype=int)
    pre_mask = taus < int(base_tau)
    post_mask = taus > int(base_tau)
    full_mask = taus != int(base_tau)

    def _band(mask: NDArray[np.bool_]) -> Band:
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
    att_tau_star: NDArray[np.float64],
    base_tau: int,
    tau_weight: str = "group",
    alpha: float = 0.05,
    group_size_map: dict[int, int] | None = None,
) -> tuple[float, tuple[float, float], float]:
    """Compute aggregated post-treatment ATT and scalar confidence interval.

    Aggregates period-specific ATTs using specified weights (e.g., 'group' size)
    and computes a corresponding bootstrap-based confidence interval.
    """
    if not isinstance(att_gt, pd.DataFrame):
        raise TypeError("att_gt must be a pandas DataFrame")
    if not isinstance(att_tau, pd.DataFrame):
        raise TypeError("att_tau must be a pandas DataFrame")
    if "tau" not in att_tau.columns or "att" not in att_tau.columns:
        raise ValueError("att_tau must contain columns {'tau','att'}")

    att_tau_star = np.asarray(att_tau_star, dtype=np.float64)
    if att_tau_star.ndim != 2:
        return float("nan"), (float("nan"), float("nan")), float("nan")

    # Contract: (K x B) where K == len(att_tau)
    K = int(att_tau.shape[0])
    if att_tau_star.shape[0] != K and att_tau_star.shape[1] == K:
        att_tau_star = att_tau_star.T
    if att_tau_star.shape[0] != K:
        raise ValueError(
            "att_tau_star must have shape (K,B) with K=len(att_tau); "
            f"got {att_tau_star.shape} with K={K}.",
        )
    if int(att_tau_star.shape[1]) < 2:
        return float("nan"), (float("nan"), float("nan")), float("nan")

    taus = att_tau["tau"].to_numpy(dtype=int)
    sel = taus > int(base_tau)
    if not np.any(sel):
        return float("nan"), (float("nan"), float("nan")), float("nan")
    if tau_weight not in {"equal", "group", "treated_t"}:
        raise ValueError("tau_weight must be one of {'equal','group','treated_t'}.")

    # When 'group' weighting is requested and no mapping is provided, attempt
    # to infer sensible group sizes from att_gt if possible.
    if tau_weight == "equal":
        w_by_tau = pd.Series(1.0, index=att_tau.loc[sel, "tau"]).groupby(level=0).sum()
        w = w_by_tau.reindex(att_tau.loc[sel, "tau"]).to_numpy(dtype=np.float64)
    elif tau_weight == "group":
        if group_size_map is None:
            # Conservative fallback: use max treated count observed for each cohort
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
        return float("nan"), (float("nan"), float("nan")), float("nan")
    w_norm = (w / s).reshape(-1, 1)

    sel_idx = np.flatnonzero(sel)
    theta = float(
        np.sum(w_norm.reshape(-1) * att_tau.loc[sel, "att"].to_numpy(dtype=np.float64)),
    )
    post_star = (w_norm * att_tau_star[sel_idx, :]).sum(axis=0).reshape(1, -1)

    post_se = float(bt.bootstrap_se(post_star)[0]) if post_star.size else float("nan")

    lo, hi = bt.uniform_confidence_band(
        np.array([[theta]], dtype=np.float64),
        post_star,
        alpha=alpha,
        studentize="bootstrap",
        context="eventstudy",
    )
    return float(theta), (float(lo[0]), float(hi[0])), post_se
