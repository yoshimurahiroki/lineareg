"""Plot utilities.

Visualizes event studies, spatial DiD, and balance checks with bootstrap
confidence bands.
"""

from __future__ import annotations

import re
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lineareg.estimators.base import EstimationResult
else:  # pragma: no cover - used only for typing
    Sequence = tuple  # type: ignore[assignment]
    EstimationResult = Any  # type: ignore[assignment,misc]

__all__ = [
    "balance_plot",
    "event_study_auto_plot",
    "event_study_percentile_plot",
    "event_study_plot",
    "spatial_did_plot",
]


def _get_uniform_band_for_tau_from_dict(  # noqa: PLR0911, PLR0913
    bands: dict,
    tau_value,
    *,
    tau_grid: pd.Index | None = None,
    pos: int | None = None,
    center_at: int = 0,
    which_band: Literal["full", "pre", "post", "auto"] = "auto",
):
    """Retrieve uniform band (lo, hi) for a specific tau value.

    Handles dicts with keys 'pre','post','full' containing DataFrames, dicts,
    or arrays. Returns None if unavailable.
    """
    # choose side
    if which_band == "auto":
        try:
            t = float(tau_value)
            if t < float(center_at):
                side = "pre"
            elif t > float(center_at):
                side = "post"
            else:
                return None
        except (TypeError, ValueError):
            side = "post"
    else:
        side = which_band
    band = bands.get(side)
    if band is None:
        band = bands.get("full")
    if band is None:
        return None

    # Compute position within the selected band when fallback-by-position is needed.
    # For 'full' bands, baseline (tau == center_at) is excluded; for 'pre'/'post' masks,
    # only the respective side is kept. This aligns positional indexing even when the band
    # array/DataFrame has fewer rows than the full tau grid.
    def _band_pos() -> int | None:
        if tau_grid is None or pos is None:
            return pos
        try:
            vals = _parse_tau_index(pd.Index(tau_grid))
            if which_band == "pre" or (
                which_band == "auto" and float(tau_value) < float(center_at)
            ):
                mask = vals < float(center_at)
            elif which_band == "post" or (
                which_band == "auto" and float(tau_value) > float(center_at)
            ):
                mask = vals > float(center_at)
            else:  # full
                mask = vals != float(center_at)
            if pos < 0 or pos >= mask.shape[0] or not mask[pos]:
                # If the current tau is not included in this band, no position
                return None
            # position within the compressed band
            return int(np.sum(mask[:pos]))
        except (TypeError, ValueError):
            return pos

    pos_in_band = _band_pos()

    def _pick(vec) -> float | None:  # noqa: PLR0911
        if isinstance(vec, pd.Series):
            # 1) Exact index match (int/float)
            if tau_value in vec.index:
                with suppress(KeyError, TypeError, ValueError):
                    return float(vec.loc[tau_value])
            # 2) Tolerant match: parse index labels such as 'tau=-2'
            with suppress(TypeError, ValueError):
                tv = float(tau_value)
                idx_vals = _parse_tau_index(pd.Index(vec.index))
                if np.isfinite(tv) and np.isfinite(idx_vals).any():
                    hit = np.isclose(idx_vals, tv, atol=1e-10, rtol=1e-12)
                    if np.any(hit):
                        j = int(np.flatnonzero(hit)[0])
                        return float(vec.iloc[j])
            # 3) Fallback: positional alignment within the band
            if pos_in_band is not None and 0 <= pos_in_band < vec.shape[0]:
                with suppress(IndexError, TypeError, ValueError):
                    return float(vec.iloc[pos_in_band])
                return None
        arr = np.asarray(vec)
        if arr.ndim == 0:
            return float(arr)
        # If this vector is aligned to the full tau grid, use the original position.
        if tau_grid is not None and arr.ndim == 1 and arr.shape[0] == len(tau_grid):
            if pos is not None and 0 <= pos < arr.shape[0]:
                with suppress(IndexError, TypeError, ValueError):
                    return float(arr[int(pos)])
            return None
        # Otherwise assume it's aligned to the selected (compressed) band.
        if pos_in_band is not None and 0 <= pos_in_band < arr.shape[0]:
            return float(arr[pos_in_band])
        return None

    # DataFrame with 'lower'/'upper' (or legacy 'lo'/'hi')
    if isinstance(band, pd.DataFrame):
        cols = {c.lower(): c for c in band.columns}
        # Accept synonyms: 'lo' -> 'lower', 'hi' -> 'upper'
        if "lower" not in cols and "lo" in cols:
            cols["lower"] = cols["lo"]
        if "upper" not in cols and "hi" in cols:
            cols["upper"] = cols["hi"]
        if "lower" in cols and "upper" in cols:
            # exact match first
            if tau_value in band.index:
                return float(band.loc[tau_value, cols["lower"]]), float(
                    band.loc[tau_value, cols["upper"]],
                )
            # tolerant numeric match (e.g., '-2' vs '-2.0')
            with suppress(TypeError, ValueError):
                idx_vals = np.asarray(band.index, dtype=float)
                if np.isfinite(idx_vals).all() and np.isfinite(float(tau_value)):
                    hit = np.isclose(idx_vals, float(tau_value), atol=1e-10, rtol=1e-12)
                    if np.any(hit):
                        j = int(np.flatnonzero(hit)[0])
                        return float(band.iloc[j][cols["lower"]]), float(
                            band.iloc[j][cols["upper"]],
                        )
            jpos = pos_in_band if pos_in_band is not None else pos
            if jpos is not None and 0 <= jpos < band.shape[0]:
                return float(band.iloc[jpos][cols["lower"]]), float(
                    band.iloc[jpos][cols["upper"]],
                )
            return None
    # dict {'lower':..., 'upper':...} (or legacy 'lo'/'hi')
    if isinstance(band, dict):
        lo = _pick(band.get("lower", band.get("lo", None)))
        hi = _pick(band.get("upper", band.get("hi", None)))
        if (lo is not None) and (hi is not None):
            return lo, hi
        return None
    # list/tuple/ndarray
    if isinstance(band, (list, tuple, np.ndarray)):
        arr = np.asarray(band)
        if arr.ndim == 1 and arr.size == 2 and all(np.isscalar(x) for x in arr):
            return float(arr[0]), float(arr[1])
        if arr.ndim == 2:
            if arr.shape[0] == 2 and pos is not None and 0 <= pos < arr.shape[1]:
                return float(arr[0, pos]), float(arr[1, pos])
            if arr.shape[1] == 2 and pos is not None and 0 <= pos < arr.shape[0]:
                return float(arr[pos, 0]), float(arr[pos, 1])
    return None


def _parse_tau_index(idx: pd.Index) -> np.ndarray:
    """Robustly convert index labels (possibly strings like 'tau=-2' or 'D-2') to float taus.
    Non-parsable labels become np.nan.
    """
    out: list[float] = []
    # Accept canonical forms only: 'tau=-2', 'event_time=-2', 'et=-2', or Stata-like 'D-2'
    pat_tau = re.compile(
        r"^(?:\s*(?:tau|event[_ ]?time|et)\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*|D(-?\d+))$",
        re.IGNORECASE,
    )
    for lab in idx.tolist():
        if isinstance(lab, (int, float, np.integer, np.floating)):
            out.append(float(lab))
            continue
        s = str(lab).strip()
        # Try plain numeric strings first (e.g., '-2', '0', '3.0')
        try:
            out.append(float(s))
            continue
        except (TypeError, ValueError):
            # Not a numeric string - try pattern match below
            s = s  # noqa: PLW0127
        m = pat_tau.match(s)
        if m:
            g = m.group(1) or m.group(2)
            try:
                out.append(float(g))
            except (TypeError, ValueError):
                out.append(np.nan)
        else:
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def event_study_plot(  # noqa: PLR0913
    es_df: pd.DataFrame | Any,
    title: str = "Event Study",
    xlabel: str = "Event time (t - g)",
    ylabel: str = "Estimated ATT",
    which_band: Literal["full", "pre", "post", "auto"] = "full",
    level: int = 95,
    show_zero: bool = True,
    vline_at: float | None = None,
    center_at: int = 0,
    bands: dict | None = None,
    post_ate_line: float | None = None,
    post_ate_band: tuple[float, float] | None = None,
    fix_inversions: bool = True,
    hide_base: bool = True,
    ax: plt.Axes | None = None,
    # --- NEW: color/label customization for pre/post segments ---
    pre_color: str | None = None,
    post_color: str | None = None,
    pre_label: str = "ATT (pre)",
    post_label: str = "ATT (post)",
    band_label_pre: str = "Uniform band (pre)",
    band_label_post: str = "Uniform band (post)",
    band_alpha: float = 0.20,
):
    """Plot an event-study series with uniform bands **provided by the estimator's bootstrap API**.
    Strict policy: this function does **not** compute or infer bands from columns; pass `bands`
    as returned by `core.bootstrap.uniform_confidence_band` (keys: 'pre','post','full').
    """
    # Accept either a DataFrame with a 'params' column OR an estimator result
    # with attributes (params, bands, model_info). This keeps notebooks simple.
    res_obj = None
    if hasattr(es_df, "params") and isinstance(es_df.params, (pd.Series, pd.DataFrame)):
        res_obj = es_df  # estimator result-like object
        # Build a minimal es_df DataFrame from result params
        params_series = (
            res_obj.params
            if isinstance(res_obj.params, pd.Series)
            else res_obj.params.iloc[:, 0]
        )
        es_df = pd.DataFrame(
            {"params": params_series.to_numpy()}, index=params_series.index,
        )
        # Auto-fill bands/center_at when not provided explicitly
        if bands is None and hasattr(res_obj, "bands"):
            bsrc = res_obj.bands or {}
            if isinstance(bsrc, dict):
                # keep only expected keys plus meta
                bands = {
                    k: v
                    for k, v in bsrc.items()
                    if k in {"pre", "post", "full", "__meta__"}
                }
        if (center_at is None or center_at == 0) and hasattr(res_obj, "model_info"):
            with suppress(TypeError, ValueError):
                ca = int(res_obj.model_info.get("CenterAt", center_at))
                center_at = ca
    # DataFrame path validation and legacy compatibility
    if "params" not in es_df.columns:
        # Legacy support: accept 'att' and treat it as params
        if "att" in es_df.columns:
            es_df = es_df.rename(columns={"att": "params"}).copy()
        else:
            msg = "es_df must contain a 'params' column (or legacy 'att')."
            raise KeyError(msg)
    # robust tau parsing from index labels (supports "tau=-2", "D-2", numeric labels)
    tau = _parse_tau_index(es_df.index)
    y = es_df["params"].to_numpy(dtype=np.float64)

    # Do not silently plot unparseable event-time labels.
    if np.any(~np.isfinite(tau)):
        bad = [str(ix) for ix, tv in zip(es_df.index.tolist(), tau) if not np.isfinite(tv)]
        preview = ", ".join(bad[:8]) + ("" if len(bad) <= 8 else ", …")
        raise ValueError(
            "Event-study index labels must be numeric taus (or 'tau=-2'/'D-2' forms). "
            f"Unparseable labels: {preview}",
        )

    if bands is None:
        raise ValueError(
            "Pass `bands` as a dict from the estimator's bootstrap API (keys: 'pre','post','full'). "
            "Column-based band fallbacks are disallowed.",
        )
    # Parse metadata from bands
    meta = (bands.get("__meta__") if isinstance(bands, dict) else None) or {}
    kind = str(meta.get("kind", "")).lower()
    # Enforce estimator context for event-study uniform-band plotting
    est = str(meta.get("estimator", "")).lower()
    allowed = {"did", "eventstudy", "event-study", "sunab", "synthetic", "sdid", "rct"}
    if est and est not in allowed:
        raise ValueError(
            f"estimator='{est}' is not allowed for uniform bands plotting.",
        )
    if kind not in {"uniform", "percentile", ""}:
        raise ValueError(
            "event_study_plot requires uniform (sup-t) or percentile bands.",
        )
    # Require bootstrap actually ran (B>0 recorded by estimator)
    Bv = meta.get("B", 0)
    if not (isinstance(Bv, (int, float)) and Bv > 0):
        raise ValueError("Bands require bootstrap runs (meta['B']>0).")
    # Build band arrays depending on kind
    ylo = np.full_like(tau, np.nan, dtype=np.float64)
    yhi = np.full_like(tau, np.nan, dtype=np.float64)

    def _required_mask_for_bands() -> np.ndarray:
        # Which taus are expected to have bands depends on which_band.
        c = float(center_at)
        if which_band == "pre":
            req = tau < c
        elif which_band == "post":
            req = tau > c
        else:
            # full/auto: expect both sides but never baseline
            req = ~np.isclose(tau, c, atol=1e-10)
        if hide_base:
            req = req & (~np.isclose(tau, c, atol=1e-10))
        return req

    req_mask = _required_mask_for_bands()
    if str(kind) == "percentile":
        # Expect bands['percentile'] as a DataFrame with index matching tau grid
        p = bands.get("percentile") if isinstance(bands, dict) else None
        if isinstance(p, pd.DataFrame):
            cols = {c.lower(): c for c in p.columns}
            lo_col = cols.get("lower") or cols.get("lo")
            hi_col = cols.get("upper") or cols.get("hi")
            if lo_col is None or hi_col is None:
                raise ValueError(
                    "Percentile bands must have 'lower' and 'upper' columns.",
                )
            # ensure numeric tau index for matching
            try:
                p_idx_vals = _parse_tau_index(p.index)
            except (TypeError, ValueError):
                p_idx_vals = np.asarray(p.index, dtype=float)
            for i, tval in enumerate(tau):
                hit = np.isclose(p_idx_vals, float(tval), atol=1e-10, rtol=1e-12)
                if np.any(hit):
                    j = int(np.flatnonzero(hit)[0])
                    ylo[i] = float(np.asarray(p.iloc[j][lo_col]).reshape(-1)[0])
                    yhi[i] = float(np.asarray(p.iloc[j][hi_col]).reshape(-1)[0])
        # Require full coverage for the requested tau subset.
        missing = req_mask & (~np.isfinite(ylo) | ~np.isfinite(yhi))
        if np.any(missing):
            miss_tau = tau[missing]
            preview = ", ".join(map(lambda z: f"{z:g}", miss_tau[:8])) + (
                "" if miss_tau.size <= 8 else ", …"
            )
            raise ValueError(
                "Percentile bands could not be aligned to the tau grid for all requested taus. "
                f"Missing taus: {preview}",
            )
    else:
        # Uniform bands path; be tolerant to extra keys in dict
        if isinstance(bands, dict):
            bands = {
                k: v
                for k, v in bands.items()
                if k in {"__meta__", "pre", "post", "full"}
            }
        pairs: list[tuple[float, float] | None] = [None] * len(tau)
        for i, tval in enumerate(tau):
            pairs[i] = _get_uniform_band_for_tau_from_dict(
                bands,
                tval,
                tau_grid=es_df.index,
                pos=i,
                center_at=center_at,
                which_band=which_band,
            )
        ylo = np.array(
            [p[0] if p is not None else np.nan for p in pairs], dtype=np.float64,
        )
        yhi = np.array(
            [p[1] if p is not None else np.nan for p in pairs], dtype=np.float64,
        )
        # Require coverage for the requested tau subset.
        missing = req_mask & (~np.isfinite(ylo) | ~np.isfinite(yhi))
        if np.any(missing):
            miss_tau = tau[missing]
            preview = ", ".join(map(lambda z: f"{z:g}", miss_tau[:8])) + (
                "" if miss_tau.size <= 8 else ", …"
            )
            raise ValueError(
                "Uniform bands could not be matched to the tau grid for all requested taus. "
                f"Missing taus: {preview}",
            )

    # detect inversions/inconsistencies
    # Optionally hide the base period band (tau == center_at)
    if hide_base and np.isfinite(center_at):
        base_mask = np.isfinite(tau) & np.isclose(tau, float(center_at), atol=1e-10)
        ylo[base_mask] = np.nan
        yhi[base_mask] = np.nan

    bad_order = np.any(ylo > yhi)
    bad_inside = np.any((y < ylo) | (y > yhi))
    if bad_order or bad_inside:
        if fix_inversions:
            # auto-correct inversions by sorting pairs
            lo = np.minimum(ylo, yhi)
            hi = np.maximum(ylo, yhi)
            ylo, yhi = lo, hi
        else:
            msg = (
                "Band columns appear inconsistent (lower>upper or estimate outside band). "
                "Set fix_inversions=True to auto-correct."
            )
            raise ValueError(msg)

    # ensure sort by tau
    idx = np.argsort(tau)
    tau, y, ylo, yhi = tau[idx], y[idx], ylo[idx], yhi[idx]

    # Use provided axes or create a new figure/axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    # --- NEW: pre/post segmentation with optional color/label customization ---
    pre_mask = tau < float(center_at)
    post_mask = tau > float(center_at)
    center_mask = np.isclose(tau, float(center_at), atol=1e-10)

    line_pre = None
    line_post = None
    if np.any(pre_mask):
        (line_pre,) = ax.plot(
            tau[pre_mask],
            y[pre_mask],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            label=pre_label,
            color=pre_color,
        )
    if np.any(post_mask):
        (line_post,) = ax.plot(
            tau[post_mask],
            y[post_mask],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            label=post_label,
            color=post_color,
        )
    if np.any(center_mask):
        ax.plot(tau[center_mask], y[center_mask], marker="o", linestyle="None")

    # Determine display level from bands meta (fallback to arg)
    level_from_meta = None
    level_from_meta = None
    with suppress(TypeError, ValueError):
        level_from_meta = int((bands.get("__meta__", {}) or {}).get("level", None))
    disp_level = (
        level_from_meta
        if isinstance(level_from_meta, int) and 0 < level_from_meta < 100
        else int(level)
    )
    # Fill uniform bands, split by pre/post so colors follow segments
    pre_fill_color = (
        pre_color
        if pre_color is not None
        else (line_pre.get_color() if line_pre is not None else None)
    )
    post_fill_color = (
        post_color
        if post_color is not None
        else (line_post.get_color() if line_post is not None else None)
    )

    # pre band
    if np.any(pre_mask):
        ylo_pre = ylo.copy()
        yhi_pre = yhi.copy()
        ylo_pre[~pre_mask] = np.nan
        yhi_pre[~pre_mask] = np.nan
        if not (np.all(np.isnan(ylo_pre)) and np.all(np.isnan(yhi_pre))):
            ax.fill_between(
                tau,
                ylo_pre,
                yhi_pre,
                alpha=band_alpha,
                color=pre_fill_color,
                label=f"{band_label_pre} ({disp_level}%)",
            )
    # post band
    if np.any(post_mask):
        ylo_post = ylo.copy()
        yhi_post = yhi.copy()
        ylo_post[~post_mask] = np.nan
        yhi_post[~post_mask] = np.nan
        if not (np.all(np.isnan(ylo_post)) and np.all(np.isnan(yhi_post))):
            ax.fill_between(
                tau,
                ylo_post,
                yhi_post,
                alpha=band_alpha,
                color=post_fill_color,
                label=f"{band_label_post} ({disp_level}%)",
            )

    if show_zero:
        ax.axhline(0.0, linestyle="--", linewidth=0.8)
    # default vertical line at center_at when vline_at not provided
    vpos = float(center_at) if vline_at is None else float(vline_at)
    ax.axvline(vpos, linestyle=":", linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5)
    # Legend without policy/provenance title (per display policy)
    ax.legend(frameon=False)
    # optional post-ATE line and band
    if post_ate_line is not None:
        # mark an approximate post-ATE position (end of tau)
        ax.hlines(post_ate_line, tau[0] - 1, tau[-1] + 1, colors="0.2", linestyles=":")
        if post_ate_band is not None:
            if (not isinstance(post_ate_band, (list, tuple))) or (
                len(post_ate_band) != 2
            ):
                msg = "post_ate_band must be a (lo, hi) tuple."
                raise ValueError(msg)
            lo, hi = post_ate_band
            if hi < lo:
                lo, hi = hi, lo
            ax.fill_between([tau[0] - 1, tau[-1] + 1], [lo, lo], [hi, hi], color="0.92")

    return fig, ax


def spatial_did_plot(  # noqa: PLR0913
    res: Any,
    *,
    which: Literal["direct", "spill", "beta_s"] = "direct",
    title: str | None = None,
    center_at: int | None = None,
    which_band: Literal["full", "pre", "post", "auto"] = "full",
    level: int = 95,
    ax: plt.Axes | None = None,
    hide_base: bool = True,
    pre_color: str | None = None,
    post_color: str | None = None,
    pre_label: str | None = None,
    post_label: str | None = None,
    band_alpha: float = 0.20,
):
    """Plot Spatial DiD direct or spill τ-series with uniform bands.

    Automatically detects standardized spill/beta_s bands from the estimator result
    (extra['bands_spill_standard'] / extra['bands_beta_s_standard']). Falls back to
    the main result params/bands for direct effects.
    """
    # Determine source series and bands
    series_df = None
    bands = None
    if which == "direct":
        # Use main params and bands
        params_series = (
            res.params
            if isinstance(res.params, pd.Series)
            else pd.Series([], dtype=float)
        )
        series_df = pd.DataFrame(
            {"params": params_series.to_numpy()}, index=params_series.index,
        )
        bands = getattr(res, "bands", None)
        if center_at is None:
            center_at = 0
            with suppress(TypeError, ValueError):
                center_at = int(res.model_info.get("CenterAt", 0))
        if title is None:
            title = "Spatial DiD: Direct"
        pre_label = pre_label or "Direct (pre)"
        post_label = post_label or "Direct (post)"
    elif which == "spill":
        extra = getattr(res, "extra", {}) or {}
        spi = extra.get("spill_tau")
        bands = extra.get("bands_spill_standard")
        if isinstance(spi, pd.DataFrame):
            series_df = spi.rename(columns={"params": "params"})
        else:
            raise ValueError(
                "Result does not contain spill series in extra['spill_tau'].",
            )
        if center_at is None:
            center_at = 0
            with suppress(TypeError, ValueError):
                center_at = int(res.model_info.get("CenterAt", 0))
        if title is None:
            title = "Spatial DiD: Spillover"
        pre_label = pre_label or "Spill (pre)"
        post_label = post_label or "Spill (post)"
    else:  # beta_s
        extra = getattr(res, "extra", {}) or {}
        bs = extra.get("beta_s_tau")
        bands = extra.get("bands_beta_s_standard")
        if isinstance(bs, pd.DataFrame):
            series_df = bs.rename(columns={"params": "params"})
        else:
            raise ValueError(
                "Result does not contain beta_s series in extra['beta_s_tau'].",
            )
        if center_at is None:
            center_at = 0
            with suppress(TypeError, ValueError):
                center_at = int(res.model_info.get("CenterAt", 0))
        if title is None:
            title = r"Spatial DiD: $\\beta_S$"
        pre_label = pre_label or r"$\\beta_S$ (pre)"
        post_label = post_label or r"$\\beta_S$ (post)"

    # Delegate to event_study_plot with the determined series and bands
    fig, ax = event_study_plot(
        series_df,
        title=title,
        which_band=which_band,
        level=level,
        ax=ax,
        hide_base=hide_base,
        center_at=center_at if center_at is not None else 0,
        bands=bands,
        pre_color=pre_color,
        post_color=post_color,
        pre_label=pre_label or "ATT (pre)",
        post_label=post_label or "ATT (post)",
        band_alpha=band_alpha,
    )
    return fig, ax


def event_study_auto_plot(  # noqa: PLR0913
    res_or_df: Any,
    *,
    which: Literal["auto", "direct", "spill", "beta_s", "both"] = "auto",
    which_band: Literal["full", "pre", "post", "auto"] = "full",
    level: int = 95,
    center_at: int | None = None,
    hide_base: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
    pre_color: str | None = None,
    post_color: str | None = None,
    band_alpha: float = 0.20,
):
    """Dispatch to the appropriate event-study plotting function.

    Rules
    - Spatial-DID: use spatial_did_plot; when which="both", produce a 1x2 panel with direct and spill.
    - If bands kind is percentile or bands has a 'percentile' key, use event_study_percentile_plot.
    - Otherwise, use event_study_plot (uniform bands expected via bands['pre'/'post'/'full']).
    """
    # Try to detect estimator family
    est_name = str(
        getattr(getattr(res_or_df, "model_info", {}), "get", lambda *_: "")(
            "Estimator", "",
        )
        if hasattr(res_or_df, "model_info")
        else getattr(res_or_df, "Estimator", ""),
    ).lower()
    is_spatial = "spatial" in est_name
    # Spatial path
    if is_spatial or which in {"direct", "spill", "beta_s", "both"}:
        # both panels: direct + spill
        if which == "both":
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            spatial_did_plot(
                res_or_df,
                which="direct",
                which_band=which_band,
                level=level,
                ax=axes[0],
                hide_base=hide_base,
                pre_color=pre_color,
                post_color=post_color,
                band_alpha=band_alpha,
            )
            spatial_did_plot(
                res_or_df,
                which="spill",
                which_band=which_band,
                level=level,
                ax=axes[1],
                hide_base=hide_base,
                pre_color=pre_color,
                post_color=post_color,
                band_alpha=band_alpha,
            )
            if title:
                fig.suptitle(title)
            for ax_i in axes:
                ax_i.legend(frameon=False)
            fig.tight_layout()
            return fig, axes
        # single series
        return spatial_did_plot(
            res_or_df,
            which=(which if which in {"direct", "spill", "beta_s"} else "direct"),
            which_band=which_band,
            level=level,
            ax=ax,
            hide_base=hide_base,
            pre_color=pre_color,
            post_color=post_color,
            band_alpha=band_alpha,
        )
    # Non-spatial: infer percentile vs uniform (SC/SDID prefer percentile)
    bands = getattr(res_or_df, "bands", None)
    if isinstance(bands, dict):
        meta = bands.get("__meta__", {}) or {}
        kind = str(meta.get("kind", "")).lower()
        if ("percentile" in bands) or (kind == "percentile"):
            # align percentile bands DataFrame with params index when possible
            pct = bands.get("percentile")
            if (
                hasattr(pct, "reindex")
                and hasattr(res_or_df, "params")
                and isinstance(res_or_df.params, pd.Series)
            ):
                with suppress(AttributeError, TypeError, ValueError):
                    pct_aligned = pct.reindex(res_or_df.params.index).dropna()
                    if pct_aligned.shape[0] == res_or_df.params.shape[0]:
                        bands = dict(bands)
                        bands["percentile"] = pct_aligned
            ax_out = event_study_percentile_plot(
                res_or_df,
                level=level,
                ax=ax,
                hide_base=hide_base,
                title=title,
            )
            return ax_out.figure, ax_out
    # Default: uniform bands plot
    # Center_at: prefer provided; else from result
    if center_at is None and hasattr(res_or_df, "model_info"):
        center_at = 0
        with suppress(TypeError, ValueError):
            center_at = int(res_or_df.model_info.get("CenterAt", 0))
    fig, ax2 = event_study_plot(
        res_or_df,
        title=(title or "Event Study"),
        which_band=which_band,
        level=level,
        show_zero=True,
        center_at=(center_at if center_at is not None else 0),
        bands=(
            getattr(res_or_df, "bands", None) if hasattr(res_or_df, "bands") else None
        ),
        hide_base=hide_base,
        ax=ax,
        pre_color=pre_color,
        post_color=post_color,
        band_alpha=band_alpha,
    )
    return fig, ax2


def event_study_percentile_plot(  # noqa: PLR0913
    tau: np.ndarray | Any,
    att: np.ndarray | None = None,
    bands: dict | None = None,
    *,
    level: int = 95,
    ax: plt.Axes | None = None,
    hide_base: bool = True,
    center_at: float = 0.0,
    fix_inversions: bool = True,
    title: str | None = None,
):
    """Plot pointwise percentile CI (e.g., SC/SDID placebo).

    Backwards-compatible API:
      - Old style: event_study_percentile_plot(tau, att, bands, ax=...)
      - New style: event_study_percentile_plot(result_object, ax=..., title=...)
    """
    # Accept a result-like object and extract (tau, att, bands)
    if att is None and bands is None and hasattr(tau, "params"):
        res_obj = tau
        # Prefer event-time path when available (e.g., SDID stores it in extra['att_t'])
        est_name = str(getattr(res_obj, "model_info", {}).get("Estimator", "")).lower()
        used_series = None
        if "sdid" in est_name:
            with suppress(AttributeError):
                att_t = (getattr(res_obj, "extra", {}) or {}).get("att_t")
                if isinstance(att_t, pd.Series) and att_t.size > 1:
                    # Use denominator to mask out times with no data
                    den = (getattr(res_obj, "extra", {}) or {}).get("att_t_den")
                    if isinstance(den, pd.Series) and den.index.equals(att_t.index):
                        valid = den > 0
                        att_t = att_t.where(valid)
                    used_series = att_t.dropna()
        if used_series is None:
            # Build tau from params index via robust parser used elsewhere
            params_series = (
                res_obj.params
                if isinstance(res_obj.params, pd.Series)
                else res_obj.params.iloc[:, 0]
            )
            used_series = params_series
        idx = used_series.index
        tau_vals = _parse_tau_index(pd.Index(idx))
        # keep finite taus and their corresponding ATT
        mask = np.isfinite(tau_vals)
        tau = tau_vals[mask]
        att = used_series.to_numpy(dtype=float)[mask]
        # center-at from model_info if available
        with suppress(TypeError, ValueError):
            center_at = float(res_obj.model_info.get("CenterAt", center_at))
        # SDID does not always have a CenterAt; prefer first cohort as base if present
        with suppress(TypeError, ValueError):
            if "sdid" in est_name and (not np.isfinite(center_at) or center_at == 0.0):
                cohorts = getattr(res_obj, "model_info", {}).get("Cohorts")
                if isinstance(cohorts, (list, tuple)) and len(cohorts) > 0:
                    center_at = float(min(cohorts))
        # bands from result (percentile)
        bsrc = getattr(res_obj, "bands", None)
        bands = bsrc if isinstance(bsrc, dict) else None
        # Align percentile band to the used series index when possible (e.g., SDID uses att_t over valid times)
        if isinstance(bands, dict) and isinstance(
            bands.get("percentile"), pd.DataFrame,
        ):
            with suppress(AttributeError, TypeError, ValueError):
                pct = bands["percentile"]
                # reindex to used_series index and drop rows that don't align
                pct_aligned = pct.reindex(pd.Index(used_series.index)).dropna()
                # keep shape if alignment was successful
                if pct_aligned.shape[0] == used_series.shape[0]:
                    bands = dict(bands)
                    bands["percentile"] = pct_aligned
    # Validate band input
    if bands is None or not isinstance(bands, dict) or "percentile" not in bands:
        raise ValueError(
            "Provide bands['percentile'] from estimator (placebo percentile CIs).",
        )
    meta = bands.get("__meta__", {})
    est = str(meta.get("estimator", "")).lower()
    if est and est not in {"synthetic", "sdid", "rct", "placebo"}:
        raise ValueError(f"percentile bands are not supported for estimator='{est}'.")
    if str(meta.get("kind", "")).lower() != "percentile":
        raise ValueError(
            "Percentile CI bands require meta.kind='percentile'; got '{}'.".format(
                meta.get("kind", "missing")
            ),
        )
    if "level" in meta:
        level = int(meta["level"])
    band = bands["percentile"]
    ylo = np.asarray(band["lower"], float)
    yhi = np.asarray(band["upper"], float)
    # Ensure shapes align exactly to avoid silent mis-plotting
    if ylo.shape != yhi.shape:
        raise ValueError(
            "bands['percentile']['lower'] and ['upper'] must have identical shapes.",
        )
    if ylo.shape[0] != tau.shape[0]:
        raise ValueError(
            "Length mismatch: len(percentile bands) must equal len(tau) and len(att).",
        )
    if att.shape[0] != tau.shape[0]:
        raise ValueError("Length mismatch: len(att) must equal len(tau).")
    if fix_inversions:
        sw = yhi < ylo
        if np.any(sw):
            ylo2 = ylo.copy()
            ylo = np.minimum(ylo2, yhi)
            yhi = np.maximum(ylo2, yhi)
    tau_ord = np.argsort(tau)
    tau = tau[tau_ord]
    att = att[tau_ord]
    ylo = ylo[tau_ord]
    yhi = yhi[tau_ord]
    if hide_base:
        base = float(center_at)
        keep = ~(np.isclose(tau, base))
        tau, att, ylo, yhi = tau[keep], att[keep], ylo[keep], yhi[keep]
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    ax.axhline(0.0, color="0.3", lw=1, ls="--")
    ax.fill_between(tau, ylo, yhi, alpha=0.25, label=f"Percentile CI ({level}%)")
    ax.plot(tau, att, lw=1.8)
    ax.set_xlabel("Event time (tau)")
    ax.set_ylabel("ATT(tau)")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    return ax


def balance_plot(  # noqa: PLR0913
    X: pd.DataFrame | np.ndarray | dict | EstimationResult,
    treat: pd.Series | np.ndarray | None = None,
    *,
    w_after: np.ndarray | None = None,
    w_before: np.ndarray | None = None,
    covariate_names: Sequence[str] | None = None,
    standardize: bool = True,
    absval: bool = True,
    order_by_after: bool = True,
    thresholds: tuple[float, float] | None = (0.1, 0.2),
    ax: plt.Axes | None = None,
    label: str | None = None,
):
    """Covariate balance plot (SMD) à la cobalt::love.plot (R).
    - X: covariates used in treatment prediction (columns) or an EstimationResult/dict payload
    - treat: 0/1 treatment indicator (when X is a matrix/DataFrame)
    - w_before: optional base weights (default: uniform 1)
    - w_after:  weights after DR/PS adjustment (default: None -> before only)
    No SE/p-values are computed; purely descriptive standardized differences.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if treat is None and w_after is None and w_before is None:
        payload = None
        if hasattr(X, "extra"):
            meta = (getattr(X, "extra", {}) or {}).get("balance_plot") or {}
            payloads = meta.get("payloads") or []
            if payloads:
                selected = None
                if label is not None:
                    for pld in payloads:
                        if pld.get("name") == label:
                            selected = pld
                            break
                if selected is None:
                    default_name = meta.get("default")
                    if default_name is not None:
                        for pld in payloads:
                            if pld.get("name") == default_name:
                                selected = pld
                                break
                payload = selected or (payloads[0] if payloads else None)
        elif isinstance(X, dict):
            payload = X
        if payload is None:
            raise ValueError(
                "balance_plot: unable to infer covariate payload from the provided object.",
            )
        covariate_names = covariate_names or payload.get("covariate_names")
        return balance_plot(
            payload.get("X"),
            payload.get("group"),
            w_after=payload.get("w_after"),
            w_before=payload.get("w_before"),
            covariate_names=covariate_names,
            standardize=standardize,
            absval=absval,
            order_by_after=order_by_after,
            thresholds=thresholds,
            ax=ax,
        )
    if hasattr(X, "values"):
        Xv = np.asarray(X.values, float)
        names = list(getattr(X, "columns", []))
    else:
        Xv = np.asarray(X, float)
        names = []
    if covariate_names is not None:
        names = list(covariate_names)
    if not names:
        names = [f"x{j}" for j in range(Xv.shape[1])]
    if len(names) != Xv.shape[1]:
        raise ValueError("covariate_names length must match number of columns in X.")
    t = np.asarray(treat, int).reshape(-1)
    if not np.array_equal(np.unique(t), np.array([0, 1])):
        raise ValueError("treat must be a {0,1} indicator.")
    n = Xv.shape[0]
    if Xv.shape[0] != t.shape[0]:
        raise ValueError("X and treat must have the same number of rows.")
    w0 = (
        np.ones(n, float)
        if w_before is None
        else np.asarray(w_before, float).reshape(-1)
    )
    if w0.shape[0] != n:
        raise ValueError("w_before length mismatch.")
    if np.any(w0 < 0):
        raise ValueError("w_before must be nonnegative.")
    if not np.isfinite(float(np.sum(w0))) or float(np.sum(w0)) <= 0:
        raise ValueError("w_before must sum to a positive finite value.")

    def _smd(w):
        # groupwise normalization (cobalt default): within each group sum(ww)=1
        w = w / np.sum(w)
        wt_raw = w[t == 1]
        wc_raw = w[t == 0]
        if wt_raw.size == 0 or wc_raw.size == 0:
            raise ValueError("Both treatment groups must be present to compute SMD.")
        # Prevent zero-sum weights (indicates a missing group or degenerate weighting)
        swt = float(np.sum(wt_raw))
        swc = float(np.sum(wc_raw))
        if swt <= 0 or swc <= 0:
            raise ValueError(
                "Within-group weights must sum to a positive value for both groups.",
            )
        wt = wt_raw / swt
        wc = wc_raw / swc

        def m(v, ww):
            return np.sum(ww * v)  # sum(ww)=1

        mu_t = np.array([m(Xv[t == 1, j], wt) for j in range(Xv.shape[1])])
        mu_c = np.array([m(Xv[t == 0, j], wc) for j in range(Xv.shape[1])])

        # pooled SD with Bessel correction using group-normalized weights
        def varw(v, ww):
            mu = np.sum(ww * v)
            denom = 1.0 - float(np.sum(ww**2))
            if not np.isfinite(denom) or denom <= 0:
                raise ValueError(
                    "Insufficient effective sample size in a group (denominator <= 0).",
                )
            return np.sum(ww * (v - mu) ** 2) / denom

        s2_t = np.array([varw(Xv[t == 1, j], wt) for j in range(Xv.shape[1])])
        s2_c = np.array([varw(Xv[t == 0, j], wc) for j in range(Xv.shape[1])])
        sd_pool = np.sqrt(0.5 * (s2_t + s2_c) + 1e-12)
        return (mu_t - mu_c) / (sd_pool if standardize else 1.0)

    if w_after is not None:
        w_after = np.asarray(w_after, float).reshape(-1)
        if w_after.shape[0] != n:
            raise ValueError("w_after length mismatch.")
        if np.any(w_after < 0):
            raise ValueError("w_after must be nonnegative.")
        if not np.isfinite(float(np.sum(w_after))) or float(np.sum(w_after)) <= 0:
            raise ValueError("w_after must sum to a positive finite value.")
    smd_before = _smd(w0)
    smd_after = _smd(w_after) if w_after is not None else None
    if absval:
        smd_before = np.abs(smd_before)
        if smd_after is not None:
            smd_after = np.abs(smd_after)
    # order by after (descending), else by before
    key = smd_after if (order_by_after and smd_after is not None) else smd_before
    ord_idx = np.argsort(-key)  # descending
    names = [names[i] for i in ord_idx]
    smd_before = smd_before[ord_idx]
    if smd_after is not None:
        smd_after = smd_after[ord_idx]
    ax = ax or plt.gca()
    y = np.arange(len(names))
    ax.scatter(smd_before, y, marker="o", label="Before", zorder=3)
    if smd_after is not None:
        ax.scatter(smd_after, y, marker="s", label="After", zorder=3)
    if thresholds:
        for th in thresholds:
            ax.axvline(th, color="0.8", lw=1, ls="--")
            ax.axvline(-th, color="0.8", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Standardized mean difference" + (" (absolute)" if absval else ""))
    ax.invert_yaxis()
    ax.legend(frameon=False)
    return ax
