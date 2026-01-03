"""Summary tables and diagnostics output.

Generates publication-ready tables with bootstrap inference.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from tabulate import tabulate

from lineareg.core import bootstrap as bt
# Keep package-relative imports centralized in helpers; use absolute package paths.
from lineareg.estimators.base import EstimationResult
from lineareg.spatial.spatial import moran_i as _moran_i
from lineareg.utils.helpers import (
    collect_extra as _collect_extra,
)
from lineareg.utils.helpers import (
    collect_info as _collect_info,
)
from lineareg.utils.helpers import (
    collect_param_index as _collect_param_index,
)
from lineareg.utils.helpers import (
    escape_latex as _escape_latex,
)
from lineareg.utils.helpers import (
    filter_and_order_params as _filter_and_order_params,
)
from lineareg.utils.helpers import (
    hline_placeholder as _hline_placeholder,
)
from lineareg.utils.helpers import (
    pretty_term as _pretty_term,
)

__all__ = ["diagnostics", "modelsummary", "weakiv_table"]


# helpers are centralized in `utils/helpers.py`; imported above


def modelsummary(  # noqa: PLR0913
    results: list[EstimationResult],
    model_names: list[str] | None = None,
    *,
    skip_missing: bool = False,
    # Parameter selection and ordering
    params: list[str] | None = None,  # explicit ordered list; no regex
    include: list[str] | None = None,  # OR of regex or exact matches
    exclude: list[str] | None = None,  # OR of regex or exact matches
    sort: Literal[
        "alpha", "tau", "none",
    ] = "alpha",  # default alphabetic; use "tau" for ES
    strict: bool = True,  # error if a specified param is missing
    # Footer rows selection and ordering
    footer_keys: list[str]
    | None = None,  # order of model_info keys; None uses sorted union
    footer_strict: bool = False,  # if True, show only footer_keys
    coef_format: str = ".6g",
    se_format: str = ".6g",
    col_width: int = 18,
    # Output formatting
    output: str = "text",  # {"text","latex"}
    latex_booktabs: bool = True,
    escape_latex: bool = True,
    # LaTeX tuning (optional)
    latex_colspec: str | None = None,  # e.g., 'l*{5}{c}' or None to auto 'l' + 'c'*k
    latex_resizebox: bool = False,  # wrap the tabular in \resizebox{\linewidth}{!}{...}
    latex_width: str = "\\linewidth",  # used when latex_resizebox=True
    truncate: bool = True,  # truncate long labels to col_width with ellipsis
    name_style: str = "paper",  # {"paper"} future extension point
    # Policy note control (default True: reiterate bootstrap-only policy)
    show_policy_note: bool = True,
    stats: Sequence[tuple[Any, ...]] | None = None,
    moran_weights: Sequence[Any] | dict[Any, Any] | None = None,
    moran_type: str | Sequence[str] | dict[Any, str] | None = None,
    # Event-study convenience: hide base period (tau == CenterAt)
    hide_baseline: bool = False,
) -> str:
    """Build publication-ready model summary.

    Formats coefficients, standard errors, and confidence intervals. Explicitly
    excludes analytic p-values and significance stars.
    """
    if model_names is None:
        model_names = [f"({i + 1})" for i in range(len(results))]

    def _expand_rct(_results: list[EstimationResult], _names: list[str] | None):
        """Detect a single 'RCT (multi-contrast) ...' result whose params are labelled like
        'RA_ATE__<contrast>' and split it into multiple column-wise pseudo models:
            columns: '<EST>-<TARGET>' (e.g., 'RA-ATE', 'IPW-ATT', ...)
            rows:    '<contrast>' (suffix after '__', e.g., '1_vs_0', with optional '_S<k>')
        Each pseudo model carries uniform bands restricted to its own rows so that
        CI rows in the summary and stats like Moran's I align under each estimator column.
        """
        if _names is None:
            _names = [f"({i + 1})" for i in range(len(_results))]
        out_r: list[EstimationResult] = []
        out_n: list[str] = []
        for _res, _nm in zip(_results, _names):
            # Skip None results
            if _res is None:
                continue
            est_label = str(_res.model_info.get("Estimator", "")).lower()
            if "rct" not in est_label:
                out_r.append(_res)
                out_n.append(_nm)
                continue
            # Parse param names like 'RA_ATE__contrast' into (key='RA-ATE', row='contrast')
            try:
                pser = _res.params if isinstance(_res.params, pd.Series) else None
                if pser is None:
                    out_r.append(_res)
                    out_n.append(_nm)
                    continue
                names = list(map(str, list(pser.index)))
                parsed: dict[str, list[tuple[str, float]]] = {}
                # optional bands to be sliced per pseudo model
                bdict = _res.bands if isinstance(_res.bands, dict) else {}
                lo_ser = None
                hi_ser = None
                if isinstance(bdict.get("uniform"), dict):
                    lo_ser = bdict["uniform"].get("lower")
                    hi_ser = bdict["uniform"].get("upper")
                for lab in names:
                    if "__" not in lab:
                        # keep non-family params in a dedicated bucket (rare)
                        key = "OTHER"
                        row = lab
                    else:
                        left, right = lab.split("__", 1)
                        tok = left.split("_")
                        key = (
                            (tok[0].upper() + "-" + tok[1].upper())
                            if len(tok) >= 2
                            else left.upper()
                        )
                        row = right
                    parsed.setdefault(key, []).append((row, float(pser[lab])))
                # Build one pseudo-model per <EST>-<TARGET> key
                for key, items in parsed.items():
                    rows = [r for (r, _) in items]
                    vals = [v for (_, v) in items]
                    # Assemble params Series indexed by contrast rows
                    params_sub = pd.Series(
                        vals, index=pd.Index(rows, name="param"), name="params",
                    )
                    # Slice uniform bands for these rows if available
                    bands_sub = None
                    if (lo_ser is not None) and (hi_ser is not None):
                        try:
                            lo_vals = []
                            hi_vals = []
                            for row, _ in items:
                                # original index in lo/hi is full label; reconstruct it:
                                # prefer exact '<LEFT>__<RIGHT>' if present, else fallback by scan
                                # Since we know key = 'EST-TGT', rebuild LEFT token:
                                left = key.replace("-", "_")
                                full_label = f"{left}__{row}"
                                if (
                                    hasattr(lo_ser, "loc")
                                    and (full_label in lo_ser.index)
                                    and (full_label in hi_ser.index)
                                ):
                                    lo_vals.append(float(lo_ser.loc[full_label]))
                                    hi_vals.append(float(hi_ser.loc[full_label]))
                                else:
                                    # defensive: search first matching suffix '__row'
                                    def _first_by_suffix(s, *, target=row):
                                        for ix in getattr(s, "index", []):
                                            if str(ix).endswith(f"__{target}"):
                                                return float(s.loc[ix])
                                        return float("nan")

                                    lo_vals.append(_first_by_suffix(lo_ser))
                                    hi_vals.append(_first_by_suffix(hi_ser))
                            lo_sub = pd.Series(
                                lo_vals, index=params_sub.index, name="lower",
                            )
                            hi_sub = pd.Series(
                                hi_vals, index=params_sub.index, name="upper",
                            )
                            bands_sub = {
                                "uniform": {
                                    "lower": lo_sub,
                                    "upper": hi_sub,
                                    "alpha": _res.bands.get("uniform", {}).get(
                                        "alpha", _res.model_info.get("Alpha"),
                                    ),
                                },
                            }
                        except (AttributeError, KeyError, TypeError, ValueError):
                            bands_sub = _res.bands
                    # Copy model_info with a clearer Estimator name for the column
                    info = dict(_res.model_info)
                    info["Estimator"] = f"RCT: {key}"
                    info["ParamNames"] = list(params_sub.index)
                    info["FamilySize"] = len(params_sub)
                    # Preserve bootstrap meta in extra
                    extra = dict(_res.extra or {})
                    # Construct pseudo result
                    out_r.append(
                        EstimationResult(
                            params=params_sub,
                            se=(
                                None
                                if _res.se is None
                                else (
                                    _res.se.loc[params_sub.index]
                                    if hasattr(_res.se, "loc")
                                    else None
                                )
                            ),
                            bands=(bands_sub if bands_sub is not None else _res.bands),
                            n_obs=_res.n_obs,
                            model_info=info,
                            extra=extra,
                        ),
                    )
                    out_n.append(f"{_nm} {key}")
            except (AttributeError, KeyError, TypeError, ValueError):
                # On any parsing trouble, fall back to the original result
                out_r.append(_res)
                out_n.append(_nm)
        return out_r, out_n

    def _expand_spatial(_results: list[EstimationResult], _names: list[str] | None):
        if _names is None:
            _names = [f"({i + 1})" for i in range(len(_results))]
        out_r: list[EstimationResult] = []
        out_n: list[str] = []
        for _res, _nm in zip(_results, _names):
            # Skip None results
            if _res is None:
                continue
            est = str(_res.model_info.get("Estimator", "")).lower()
            if ("spatial" in est) and ("did" in est):
                out_r.append(_res)
                out_n.append(f"{_nm} (Direct)")
                extra = _res.extra or {}
                spill_obj = extra.get("spill_tau")
                if spill_obj is not None:
                    try:
                        if isinstance(spill_obj, dict):
                            spill_obj = pd.DataFrame(spill_obj)
                        spill_df = (
                            spill_obj if isinstance(spill_obj, pd.DataFrame) else None
                        )
                        if spill_df is not None and {"tau", "params"}.issubset(
                            spill_df.columns,
                        ):
                            spill_series = spill_df.set_index("tau")["params"]
                            bands_raw = extra.get("bands_spill_standard") or {}
                            bands = {
                                k: v
                                for k, v in bands_raw.items()
                                if k in {"pre", "post", "full", "post_scalar_spill"}
                            }
                            if bands:
                                meta = bands.setdefault("__meta__", {})
                                meta.setdefault("kind", "uniform")
                                meta.setdefault("estimator", "eventstudy")
                            spill_se = None
                            try:
                                spill_boot_star = extra.get("spill_tau_star")
                                if spill_boot_star is not None and isinstance(spill_boot_star, np.ndarray) and spill_boot_star.shape[1] > 1:
                                    spill_se = pd.Series(bt.bootstrap_se(spill_boot_star), index=spill_series.index, name="se")
                                elif _res.se is not None:
                                    spill_se = _res.se
                            except (KeyError, TypeError, ValueError, AttributeError):
                                spill_se = None
                            info = dict(_res.model_info)
                            info["Estimator"] = f"{info.get('Estimator', '')} (Spill)"
                            if "PostSpill" in info:
                                info["PostATT"] = info.get("PostSpill")
                                info.pop("PostSpill", None)
                            # Copy PostATT_se from PostSpill if available
                            if "PostSpill_se" in _res.model_info:
                                info["PostATT_se"] = _res.model_info["PostSpill_se"]
                            # (if "PostATT_se" already exists, keep it unchanged)
                            spill_extra = dict(_res.extra) if _res.extra else {}
                            spill_extra["se_source"] = extra.get("se_source", "bootstrap")
                            spill_res = EstimationResult(
                                params=spill_series,
                                se=spill_se,
                                bands=bands if bands else None,
                                n_obs=_res.n_obs,
                                model_info=info,
                                extra=spill_extra,
                            )
                            out_r.append(spill_res)
                            out_n.append(f"{_nm} (Spill)")
                            continue
                    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                        # Spill extraction failed - fall through to default handling
                        out_r = out_r  # noqa: PLW0127
                # fallback: no spill column emitted
            else:
                out_r.append(_res)
                out_n.append(_nm)
        return out_r, out_n

    results, model_names = _expand_rct(results, model_names)
    results, model_names = _expand_spatial(results, model_names)

    def _resolve_option(option: Any, idx: int, name: str) -> Any:
        if isinstance(option, dict):
            for key in (name, idx, str(idx)):
                if key in option:
                    return option[key]
            return option.get("_default")
        if isinstance(option, (list, tuple)):
            return option[idx] if idx < len(option) else None
        return option

    def _format_stat_value(val: Any) -> str:
        if val is None:
            return ""
        try:
            if isinstance(val, (int, float, np.floating)):
                if not np.isfinite(float(val)):
                    return ""
                return f"{float(val):{coef_format}}"
        except (TypeError, ValueError):
            # Not convertible to float - fall through to str()
            val = val  # noqa: PLW0127
        return str(val)

    def _stat_moran(  # noqa: PLR0911
        res: EstimationResult, *, weights: Any, moran_type: str = "cross_section",
    ) -> float:
        """Calculate Moran's I for residuals.

        Parameters
        ----------
        res : EstimationResult
            Result object containing residuals
        weights : array-like
            Spatial weight matrix
        moran_type : str
            Type of Moran's I: "cross_section" (or "cs") or "panel"

        """
        if weights is None:
            return float("nan")

        # Get residuals - check multiple possible locations
        resid = (res.extra or {}).get("residuals")
        if resid is None:
            resid = (res.extra or {}).get("resid")
        if resid is None:
            resid = (res.extra or {}).get("u_inference")  # OLS/GLS/IV/GMM store here
        if resid is None:
            # Try as direct attribute
            resid = getattr(res, "resid", None)
        if resid is None:
            resid = getattr(res, "residuals", None)
        if resid is None:
            return float("nan")

        arr = np.asarray(resid, dtype=np.float64).reshape(-1)

        # Validate weights matrix
        try:
            W = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError):
            return float("nan")

        # Normalize moran_type
        mtype = str(moran_type).lower().strip()

        if mtype in {"cross_section", "cs", "crosssection"}:
            # Cross-sectional Moran's I. Accept two conformability modes:
            # 1) W matches residual length directly (pure cross-section)
            # 2) Panel-style residuals (n_units * T) with W of shape (n_units, n_units):
            #    treat as block-diagonal W (one block per time), i.e., compute
            #    I = (N / (T*S0)) * (sum_t e_t' W e_t) / (sum_t e_t' e_t).
            if W.shape == (arr.size, arr.size):
                try:
                    return float(_moran_i(arr, W))
                except (TypeError, ValueError, ZeroDivisionError):
                    return float("nan")
            # Panel-shaped fallback using id/time if available
            extra = res.extra or {}
            id_vals = extra.get("id")
            time_vals = extra.get("time")
            if id_vals is None or time_vals is None:
                return float("nan")
            try:
                id_arr = np.asarray(id_vals).reshape(-1)
                time_arr = np.asarray(time_vals).reshape(-1)
                # Validate lengths
                if id_arr.size != arr.size or time_arr.size != arr.size:
                    return float("nan")
                # Build per-time slices; require complete panels for stability
                times_unique = np.unique(time_arr)
                # Determine unit ordering per period and ensure it matches W (sorted by id)
                units_unique = np.unique(id_arr)
                n_units = units_unique.size
                if W.shape[0] != W.shape[1] or W.shape[0] != n_units:
                    return float("nan")
                # Precompute S0 for a single block and scale by number of complete periods
                # Use numpy sum on row sums to avoid densifying scipy.sparse
                from lineareg.core import linalg as _la

                one = np.ones((n_units, 1), dtype=np.float64)
                S0_block = float(
                    np.sum(np.asarray(_la.dot(W, one)).reshape(-1), dtype=np.float64),
                )
                num_sum = 0.0
                den_sum = 0.0
                T_complete = 0
                for t in times_unique:
                    mask_t = time_arr == t
                    # Build vector e_t aligned by sorted unit id
                    ids_t = id_arr[mask_t]
                    if ids_t.size != n_units:
                        # skip incomplete periods
                        continue
                    order = np.argsort(ids_t)
                    e_t = arr[mask_t][order].reshape(-1, 1)
                    # e_t' W e_t and e_t' e_t
                    We_t = _la.dot(W, e_t)
                    num_arr = _la.crossprod(e_t, We_t)
                    den_arr = _la.crossprod(e_t, e_t)
                    num_sum += float(
                        num_arr.item() if hasattr(num_arr, "item") else num_arr,
                    )
                    den_sum += float(
                        den_arr.item() if hasattr(den_arr, "item") else den_arr,
                    )
                    T_complete += 1
                if T_complete == 0 or S0_block == 0.0 or den_sum == 0.0:
                    return float("nan")
                N_total = float(n_units * T_complete)
                S0_total = float(T_complete) * S0_block
                return float((N_total / S0_total) * (num_sum / den_sum))
            except (
                ArithmeticError,
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                return float("nan")

        elif mtype in {"panel"}:
            # Panel Moran's I
            # Need id and time information from extra
            extra = res.extra or {}
            id_vals = extra.get("id")
            time_vals = extra.get("time")

            if id_vals is None or time_vals is None:
                # Fallback to cross-sectional if panel info unavailable
                if W.shape != (arr.size, arr.size):
                    return float("nan")
                try:
                    return float(_moran_i(arr, W))
                except (TypeError, ValueError, ZeroDivisionError):
                    return float("nan")

            try:
                # Import panel version of Moran's I
                from lineareg.spatial.spatial import moran_i_panel

                id_arr = np.asarray(id_vals)
                time_arr = np.asarray(time_vals)

                return float(moran_i_panel(arr, W, id_arr, time_arr))
            except (
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                # Fallback to cross-sectional
                if W.shape != (arr.size, arr.size):
                    return float("nan")
                try:
                    return float(_moran_i(arr, W))
                except (TypeError, ValueError, ZeroDivisionError):
                    return float("nan")
        else:
            # Default to cross-sectional
            if W.shape != (arr.size, arr.size):
                return float("nan")
            try:
                return float(_moran_i(arr, W))
            except (TypeError, ValueError, ZeroDivisionError):
                return float("nan")

    norm_stats: list[tuple[str, Any, dict[str, Any]]] = []
    if stats:
        for entry in stats:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                raise ValueError("stats entries must be (label, callable[, options]).")
            label = str(entry[0])
            func = entry[1]
            if not callable(func):
                raise TypeError("stats callable must be callable.")
            options = entry[2] if len(entry) > 2 else {}
            norm_stats.append((label, func, dict(options or {})))

    # Add Moran's I statistics if weights are provided
    if moran_weights is not None:
        # Normalize moran_type to handle different input formats
        def _get_moran_type(idx: int, name: str) -> str:  # noqa: PLR0911
            if moran_type is None:
                return "cross_section"
            if isinstance(moran_type, str):
                return moran_type
            if isinstance(moran_type, dict):
                # Try by name first, then by index
                if name in moran_type:
                    return str(moran_type[name])
                if idx in moran_type:
                    return str(moran_type[idx])
                if str(idx) in moran_type:
                    return str(moran_type[str(idx)])
                return "cross_section"
            if isinstance(moran_type, (list, tuple)):
                return (
                    str(moran_type[idx]) if idx < len(moran_type) else "cross_section"
                )
            return "cross_section"

        # Add a SINGLE Moran's I stat row that will be computed for all models
        # Store the configuration in a way that allows per-model lookup
        norm_stats.append(
            (
                "Moran's I",
                _stat_moran,
                {
                    "weights": moran_weights,
                    "moran_type": moran_type,
                    "_get_moran_type": _get_moran_type,
                },
            ),
        )
    # Strictly validate output mode
    if output not in {"text", "latex"}:
        raise ValueError("output must be either 'text' or 'latex'.")
    # LaTeX escape (column headers)
    if output == "latex" and escape_latex:
        model_names = [_escape_latex(n) for n in model_names]

    # Helper: decide if this result should display uniform bands (ES/Synth/SDID only)
    def _is_es_uniform_band(res: EstimationResult) -> bool:
        est = str(res.model_info.get("Estimator", "")).lower()
        allowed_name = any(
            key in est for key in ["did", "eventstudy", "synthetic", "sdid", "ddd"]
        )
        has_bands = (
            hasattr(res, "bands")
            and isinstance(res.bands, dict)
            and (("pre" in res.bands) or ("post" in res.bands) or ("full" in res.bands))
        )
        if not (allowed_name and has_bands):
            return False
        # If BandType metadata is present, require it to indicate 'uniform' or 'percentile'.
        # Percentile bands (from placebo inference) are valid for SC/SDID.
        btype = str(res.model_info.get("BandType", "")).lower()
        if btype and ("uniform" not in btype) and ("percentile" not in btype):
            return False
        # Require bootstrap to have been run (recorded in model_info['B'])
        Bv = res.model_info.get("B", 0)
        return isinstance(Bv, (int, float)) and Bv > 0

    def _is_rct_uniform_band(res: EstimationResult) -> bool:
        est = str(res.model_info.get("Estimator", "")).lower()
        if "rct" not in est:
            return False
        b = getattr(res, "bands", None)
        if not (isinstance(b, dict) and ("uniform" in b)):
            return False
        # Require bootstrap to have been run (recorded in model_info['B'])
        Bv = res.model_info.get("B", 0)
        return isinstance(Bv, (int, float)) and Bv > 0

    # Estimator family helpers for formatting policy
    def _is_core_linear(res: EstimationResult) -> bool:
        est = str(res.model_info.get("Estimator", "")).lower()
        return any(
            key in est for key in ["ols", "iv (2sls)", "gmm", "gls", "qr", "sar-2sls"]
        )

    def _is_policy_ci_family(res: EstimationResult) -> bool:
        # Families where we prefer CI rows (and suppress SE rows): RCT, ES/DR/DDD/SpatialDID, Synthetic, SDID
        est = str(res.model_info.get("Estimator", "")).lower()
        if "rct" in est:
            return True
        if _is_es_uniform_band(res):
            return True
        band_type = str(res.model_info.get("BandType", "")).lower()
        Bv = res.model_info.get("B", 0)
        if (
            band_type
            and band_type not in {"none", ""}
            and isinstance(Bv, (int, float))
            and Bv > 0
        ):
            return True
        if any(key in est for key in ["synthetic", "sdid"]):
            bands = getattr(res, "bands", None)
            if isinstance(bands, dict):
                has_pct = (
                    bands.get("percentile")
                    or bands.get("per_tau")
                    or bands.get("post_scalar")
                )
                if has_pct is not None and isinstance(Bv, (int, float)) and Bv > 0:
                    return True
        return False

    def _uniform_band_str(  # noqa: PLR0911
        res: EstimationResult, param: object, fmt: str,
    ) -> str:
        if not _is_es_uniform_band(res):
            return ""
        # Accept bootstrap origins including placebo (SDID/SC)
        extra = res.extra or {}
        origin = ""
        # Prefer estimator-provided bands metadata.
        if isinstance(getattr(res, "bands", None), dict):
            with suppress(AttributeError, KeyError, TypeError):
                origin = str((res.bands.get("__meta__") or {}).get("origin", "")).lower()
        # Fall back to extra-stored meta for older results.
        if not origin:
            origin = str((extra.get("boot_meta") or {}).get("origin", "")).lower()
        # Normalize common aliases.
        origin = origin.replace("_", "-")
        allowed_origins = {
            "bootstrap",
            "wild",
            "multiplier",
            "rademacher",
            "mammen",
            "webb",
            "placebo",
            # SyntheticControl/SDID nomenclature
            "permutation",
            "wild-if",
        }
        if origin and origin not in allowed_origins:
            return ""
        # robustly parse an event-time integer from a parameter label. Accepts
        # labels like -2, "-2", "tau=-2", "D-2", "tau:-2" etc.
        tau = None
        if isinstance(param, (int, np.integer)):
            tau = int(param)
        else:
            s = str(param).strip()
            # Strict recognition: whole-string integer (e.g. '-2'), or explicit
            # event-time labels like 'tau=-2', 'tau:-2', or 'D-2'. Avoid matching
            # coefficients like 'x1' or 'v_2'.
            m = re.match(r"^(-?\d+)\s*$", s)
            if m:
                try:
                    tau = int(m.group(1))
                except (TypeError, ValueError):
                    tau = None
            else:
                # Accept only explicit event-time tokens; disallow embedded digits like 'x1'
                m = re.match(
                    r"^(?:(?:tau|event[_ ]?time|et)[:=]\s*(-?\d+)|D(-?\d+))$",
                    s,
                    flags=re.IGNORECASE,
                )
                if m:
                    try:
                        tau = int(m.group(1) or m.group(2))
                    except (TypeError, ValueError):
                        tau = None
        if tau is None:
            return ""
        ca = res.model_info.get("CenterAt", 0)
        try:
            center_at = int(ca) if ca is not None and str(ca).lower() != "nan" else 0
        except (TypeError, ValueError):
            center_at = 0
        # Do not show CI for the base period (tau == center_at)
        if np.isclose(float(tau), float(center_at), atol=1e-10):
            return ""
        side = "pre" if tau < center_at else "post"
        bands = getattr(res, "bands", None)
        if not (isinstance(bands, dict) and bands):
            return ""
        # Compute position of param in res.params.index for array-based band extraction
        try:
            pos = (
                list(res.params.index).index(param) if param in res.params.index else -1
            )
        except ValueError:
            pos = -1

        # helper: fetch lower/upper for the specific tau from a DataFrame/dict
        # Returns (lo, hi) for the provided tau_val or (None, None) if unavailable.
        # Accepts DataFrame (index=tau), dict+Series (index=tau), dict+array (position-fallback).
        def _pick_from(  # noqa: PLR0911
            obj: object, tau_val: int, pos: int,
        ) -> tuple[float | None, float | None]:
            try:
                if isinstance(obj, pd.DataFrame):
                    # try index == tau, else 'tau' column
                    if tau_val in obj.index:
                        row = obj.loc[tau_val]
                    elif "tau" in obj.columns:
                        row = obj.loc[obj["tau"] == tau_val].iloc[0]
                    else:
                        return (None, None)
                    # Accept synonyms: 'lo' -> 'lower', 'hi' -> 'upper'
                    lo_key = (
                        "lower"
                        if "lower" in row.index
                        else ("lo" if "lo" in row.index else None)
                    )
                    hi_key = (
                        "upper"
                        if "upper" in row.index
                        else ("hi" if "hi" in row.index else None)
                    )
                    if lo_key is None or hi_key is None:
                        return (None, None)
                    lo = float(np.asarray(row[lo_key]).reshape(-1)[0])
                    hi = float(np.asarray(row[hi_key]).reshape(-1)[0])
                    return (lo, hi)
                if isinstance(obj, dict):
                    # dict {'lower': Series/array, 'upper': Series/array} or legacy {'lo', 'hi'}
                    lo_obj = obj.get("lower", obj.get("lo"))
                    hi_obj = obj.get("upper", obj.get("hi"))
                    if isinstance(lo_obj, (list, np.ndarray, pd.Series)) and isinstance(
                        hi_obj, (list, np.ndarray, pd.Series),
                    ):
                        # 1) Try index==tau for Series (tau-matching)
                        if isinstance(lo_obj, pd.Series) and isinstance(
                            hi_obj, pd.Series,
                        ):
                            if (tau_val in lo_obj.index) and (tau_val in hi_obj.index):
                                return (
                                    float(lo_obj.loc[tau_val]),
                                    float(hi_obj.loc[tau_val]),
                                )
                        # 2) Fallback: by position (matches res.params ordering)
                        lo_arr = np.asarray(lo_obj).reshape(-1)
                        hi_arr = np.asarray(hi_obj).reshape(-1)
                        if 0 <= pos < lo_arr.size and pos < hi_arr.size:
                            return (float(lo_arr[pos]), float(hi_arr[pos]))
            except (IndexError, KeyError, TypeError, ValueError):
                return (None, None)
            return (None, None)

        # prefer side-specific band, else fall back to 'full'
        lo, hi = (None, None)
        side_band = bands.get(side, None) if isinstance(bands, dict) else None
        if side_band is not None:
            lo, hi = _pick_from(side_band, tau, pos)
        if (lo is None or hi is None) and isinstance(bands, dict) and ("full" in bands):
            lo, hi = _pick_from(bands["full"], tau, pos)
        return (
            f"[{lo:{fmt}}, {hi:{fmt}}]" if (lo is not None and hi is not None) else ""
        )

    def _agg_band_str(res: EstimationResult, param_name: str, fmt: str) -> str:
        """Aggregated ATE band extractor. Prefers the canonical model_info key
        '{param}_Band' and falls back to suffixed variants such as
        '{param}_Band90'. Accepts dicts, objects with attributes, or array-like
        [lo, hi]. If no model_info band is found, falls back to
        res.bands['post_scalar'] when available (DataFrame with columns
        {'lower','upper'} or dict/array-like). For Spatial DiD, also accepts
        'post_scalar_direct' for the canonical PostATT row.
        """
        # Aggregated scalar bands can arise from uniform ES inference (event-study)
        # AND from percentile/permutation (placebo) inference in SC/SDID.
        Bv = res.model_info.get("B", 0)
        if not (isinstance(Bv, (int, float)) and Bv > 0):
            return ""
        # canonical key (introduced in the unified API refresh)
        key = f"{param_name}_Band"
        band = res.model_info.get(key, None)
        if band is None:
            band_level = res.model_info.get("BandLevel")
            if band_level is not None:
                label = (f"{band_level:.6g}").rstrip("0").rstrip(".")
                alt_key = f"{param_name}_Band{label}"
                band = res.model_info.get(alt_key)

            if band is None:
                # Legacy fallback: keys that (i) contain param_name and (ii) end with 'Band' or 'BandXX'
                patt = re.compile(
                    rf"{re.escape(param_name)}.*_(?:Band|band)(\d+)?$", re.IGNORECASE,
                )
                for k, v in res.model_info.items():
                    if isinstance(k, str) and patt.search(k):
                        band = v
                        break

        # Helper to normalize lower/upper from various containers
        def _lo_hi_from_band_obj(obj: object) -> tuple[float | None, float | None]:
            lo = hi = None
            if isinstance(obj, dict):
                lo = obj.get("lo") or obj.get("lower")
                hi = obj.get("hi") or obj.get("upper")
            elif isinstance(obj, pd.DataFrame):
                cols = {c.lower(): c for c in obj.columns}
                if "lower" in cols and "upper" in cols:
                    try:
                        lo = float(np.asarray(obj[cols["lower"]]).reshape(-1)[0])
                        hi = float(np.asarray(obj[cols["upper"]]).reshape(-1)[0])
                    except (IndexError, TypeError, ValueError):
                        lo = hi = None
                elif "lo" in cols and "hi" in cols:
                    try:
                        lo = float(np.asarray(obj[cols["lo"]]).reshape(-1)[0])
                        hi = float(np.asarray(obj[cols["hi"]]).reshape(-1)[0])
                    except (IndexError, TypeError, ValueError):
                        lo = hi = None
            else:
                lo = getattr(obj, "lo", None) or getattr(obj, "lower", None)
                hi = getattr(obj, "hi", None) or getattr(obj, "upper", None)
                if lo is None or hi is None:
                    try:
                        arr = np.asarray(obj).reshape(-1)
                        if arr.size >= 2:
                            lo, hi = float(arr[0]), float(arr[1])
                    except (TypeError, ValueError):
                        lo = hi = None
            try:
                lo = float(np.asarray(lo).reshape(-1)[0]) if lo is not None else None
                hi = float(np.asarray(hi).reshape(-1)[0]) if hi is not None else None
            except (TypeError, ValueError):
                lo = hi = None
            return lo, hi

        if band is None:
            # Secondary fallback: estimator-provided bands dict, prefer 'post_scalar'
            bdict = getattr(res, "bands", None)
            if isinstance(bdict, dict):
                cand = None
                # Primary generic key: only for PostATT-like
                if param_name in {"PostATT", "PostATT_Diff"} and (
                    "post_scalar" in bdict
                ):
                    cand = bdict.get("post_scalar")
                # Spatial DiD may provide split scalars
                est_lower = str(res.model_info.get("Estimator", "")).lower()
                if cand is None and ("spatial" in est_lower):
                    if param_name in {"PostATT", "PostATT_Diff"}:
                        cand = (
                            bdict.get("post_scalar_direct")
                            or bdict.get("post_scalar")
                            or bdict.get("post_scalar_spill")
                        )
                    elif param_name == "PostSpill":
                        cand = bdict.get("post_scalar_spill")
                if cand is not None:
                    band = cand
            # SC/SDID may store post_scalar in extra instead of bands
            if band is None:
                extra = res.extra or {}
                if param_name in {"PostATT", "PostATT_Diff"}:
                    cand = extra.get("post_scalar")
                    if cand is not None:
                        band = cand
        if band is None:
            return ""
        lo, hi = _lo_hi_from_band_obj(band)
        if lo is None or hi is None:
            return ""
        return f"[{lo:{fmt}}, {hi:{fmt}}]"

    # Truncation helper (preserves LaTeX escaping as plain text; avoids trailing backslash)
    def _truncate_text(s: str, n: int) -> str:
        try:
            if (not truncate) or (n is None) or (n <= 0):
                return s
            if len(s) <= n:
                return s
            cut = s[: max(0, n - 1)]  # leave room for ellipsis
            # Avoid ending on a solitary backslash which would escape the ellipsis
            if cut.endswith("\\"):
                cut = cut.removesuffix("\\")
            return cut + "…"
        except (AttributeError, TypeError, ValueError):
            return s

    # Rows: parameter, SE row (if bootstrap), CI row for ES families only
    raw_pool = _collect_param_index(results, skip_missing=skip_missing)

    # Auto-detect event-study models and use sort="tau" when appropriate
    # If user did not explicitly set sort, check if all models are event-study type
    effective_sort = sort
    if sort == "alpha":
        # Check if all non-None results are event-study family
        es_keywords = {"did", "eventstudy", "event_study", "synthetic", "sdid", "ddd", "rct"}
        all_es = True
        for res in results:
            if res is None:
                continue
            est_lbl = str(res.model_info.get("Estimator", "")).lower().replace("-", "").replace(" ", "")
            if not any(k in est_lbl for k in es_keywords):
                all_es = False
                break
        if all_es and len([r for r in results if r is not None]) > 0:
            effective_sort = "tau"

    # Special handling for PostATT: if explicitly requested in params, temporarily add to pool
    if params is not None and any(
        str(p) in {"PostATT", "PostATT_Diff"} for p in params
    ):
        # Add PostATT to raw_pool temporarily so filter doesn't fail
        if "PostATT" not in raw_pool:
            raw_pool = [*list(raw_pool), "PostATT"]

    raw_params = _filter_and_order_params(
        raw_pool,
        params=params,
        include=include,
        exclude=exclude,
        sort=effective_sort,
        strict=strict,
    )

    # Optionally drop the base period (tau == CenterAt) from event-study tables
    if hide_baseline and effective_sort == "tau":
        # collect all CenterAt values advertised by ES-family models
        center_vals: set[int] = set()
        for res in results:
            est_lbl = str(res.model_info.get("Estimator", "")).lower()
            if any(k in est_lbl for k in ["did", "eventstudy", "sdid", "synthetic", "ddd"]):
                ca = res.model_info.get("CenterAt", None)
                try:
                    if ca is not None and str(ca).lower() != "nan":
                        center_vals.add(int(ca))
                except (TypeError, ValueError):
                    # CenterAt not convertible to int - skip
                    ca = ca  # noqa: PLW0127

        def _parse_tau(label: object) -> int | None:
            if isinstance(label, (int, np.integer)):
                return int(label)
            s = str(label).strip()
            m = re.match(r"^(-?\d+)\s*$", s)
            if m:
                try:
                    return int(m.group(1))
                except (TypeError, ValueError):
                    return None
            m = re.match(r"^(?:(?:tau|event[_ ]?time|et)[:=]\s*(-?\d+)|D(-?\d+))$", s, flags=re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1) or m.group(2))
                except (TypeError, ValueError):
                    return None
            return None

        if center_vals:
            raw_params = [p for p in raw_params if (_parse_tau(p) not in center_vals)]

    # Normalize intercept naming across models to avoid duplicate '(Intercept)' vs 'cons'
    def _norm_name(x: object) -> object:
        """Normalize intercept naming across dialects to a unified 'cons'.

        Maps any of {(Intercept), Intercept, _cons, const} -> 'cons'.
        Also unify post-period aggregate synonyms to a single 'PostATT'.
        Spatial spill aggregate normalized to 'PostSpill'.
        """
        if isinstance(x, str):
            s = x.strip()
            if s in {"(Intercept)", "Intercept", "_cons", "const"}:
                return "cons"
            # Unify post-period aggregate names across estimators
            if s.lower().replace(" ", "") in {
                "attpost",
                "postatt",
                "post_att",
                "post-att",
            }:
                return "PostATT"
            # DDD reports PostATT_Diff; display as PostATT per project policy
            if s in {"PostATT_Diff", "PostATTdiff", "PostATT- Diff"}:
                return "PostATT"
            # Spatial spill aggregate legacy labels -> 'PostSpill'
            if s.lower().replace(" ", "") in {
                "postspill",
                "post_spill",
                "postatt(spill)",
                "postattspill",
            }:
                return "PostSpill"
        return x

    # Stable de-duplication after normalization while preserving original order
    norm_params: list[object] = []
    seen_norm = set()
    for rp in raw_params:
        npn = _norm_name(rp)
        if npn not in seen_norm:
            seen_norm.add(npn)
            norm_params.append(npn)
    raw_params = norm_params
    table_data: list[list[str]] = []
    # map raw -> pretty display name (search uses raw)
    param_pairs: list[tuple[Any, str]] = []
    for p in raw_params:
        disp = _pretty_term(p, style=name_style)
        if output == "latex" and escape_latex:
            disp = _escape_latex(disp)
        # Only truncate aggressively for LaTeX output to improve column fit
        if output == "latex":
            disp = _truncate_text(disp, col_width)
        param_pairs.append((p, disp))

    # Only estimators belonging to ES/Synth/SDID/DDD families can show SE rows
    def _is_es_family(res: EstimationResult) -> bool:
        est = str(res.model_info.get("Estimator", "")).lower()
        return any(
            key in est for key in ["did", "eventstudy", "synthetic", "sdid", "ddd"]
        )

    # Helper to resolve per-result parameter name synonyms (e.g., cons vs (Intercept))
    def _resolve_name(res: EstimationResult, name: object) -> object | None:
        """Resolve a normalized parameter name to the estimator-specific key.

        Handles intercept synonyms across dialects: {(Intercept), Intercept, _cons, const}.
        """
        # Direct hit
        if name in res.params.index:
            return name
        # Intercept normalization: map unified 'cons' to any dialect-specific key present
        if name == "cons":
            for cand in ("cons", "_cons", "(Intercept)", "Intercept", "const"):
                if cand in res.params.index:
                    return cand
        # Also allow the reverse: if a dialect-specific intercept label was requested, map to 'cons' if present
        if isinstance(name, str) and name.strip() in {
            "_cons",
            "(Intercept)",
            "Intercept",
            "const",
        }:
            if "cons" in res.params.index:
                return "cons"
            for cand in ("_cons", "(Intercept)", "Intercept", "const"):
                if cand in res.params.index:
                    return cand
        # PostATT unification: map to any estimator-specific synonym
        if name == "PostATT":
            for cand in ("PostATT", "ATT post", "ATT post ", "PostATT_Diff"):
                if cand in res.params.index:
                    return cand
        return None

    for raw_name, disp_name in param_pairs:
        # display name already escaped/truncated above
        row = [disp_name]
        se_row = [""]  # SE row (between coefficient and CI for ES family)
        ci_row = [""]  # CI row (uniform bands for ES family)
        for res in results:
            key = _resolve_name(res, raw_name)

            # Special handling for PostATT when it's not in params.index but in model_info
            if key is None and str(raw_name) in {"PostATT", "PostATT_Diff"}:
                # Try to get PostATT from model_info
                val = res.model_info.get("PostATT", "")
                if val != "":
                    row.append(f"{float(val):{coef_format}}")
                    # Try to get SE for PostATT from model_info first
                    se_val = None
                    if _is_policy_ci_family(res):
                        postatt_se = res.model_info.get("PostATT_se") or res.model_info.get("post_att_se")
                        if postatt_se is not None:
                            try:
                                se_val = float(postatt_se)
                            except (TypeError, ValueError):
                                se_val = None
                        if se_val is None:
                            extra = res.extra or {}
                            postatt_se = extra.get("PostATT_se") or extra.get("post_att_se")
                            if postatt_se is not None:
                                try:
                                    se_val = float(postatt_se)
                                except (TypeError, ValueError):
                                    se_val = None
                        if se_val is None:
                            s = getattr(res, "se", None)
                            if isinstance(s, pd.Series):
                                for cand in ("PostATT", "ATT post", "PostATT_Diff"):
                                    if cand in s.index:
                                        try:
                                            se_val = float(s[cand])
                                            break
                                        except (KeyError, TypeError, ValueError):
                                            # Continue to next candidate
                                            se_val = se_val  # noqa: PLW0127
                    se_row.append(f"({se_val:{se_format}})" if se_val is not None and np.isfinite(se_val) and abs(se_val) > 1e-10 else "")
                    # Try to get aggregated CI
                    if _is_policy_ci_family(res):
                        ci_row.append(_agg_band_str(res, "PostATT", coef_format))
                    else:
                        ci_row.append("")
                else:
                    row.append("")
                    se_row.append("")
                    ci_row.append("")
                continue

            if key is not None:
                coef = float(res.params[key])
                if not np.isfinite(coef):
                    # For ES-style outputs, prefer blank over 'nan' in table
                    est_name__ = str(res.model_info.get("Estimator", "")).lower()
                    if any(
                        k in est_name__
                        for k in ["did", "eventstudy", "sdid", "synthetic", "ddd"]
                    ):
                        row.append("")
                    else:
                        row.append(f"{coef:{coef_format}}")
                else:
                    row.append(f"{coef:{coef_format}}")

                def _se_allowed(r: EstimationResult) -> bool:
                    Bv = r.model_info.get("B", 0)
                    if not (isinstance(Bv, (int, float)) and Bv > 0):
                        return False
                    extra = r.extra or {}
                    # Check se_source first (ES/DID/SpatialDID/RCT use this)
                    se_source = str(extra.get("se_source", "")).lower()
                    if se_source in {"bootstrap", "wild_bootstrap", "multiplier"}:
                        return True
                    boot_meta = (
                        extra.get("boot_meta") or extra.get("bootstrap_meta") or {}
                    )
                    if isinstance(boot_meta, dict):
                        origin = str(boot_meta.get("origin", "")).lower()
                        origin = origin.replace("_", "-")
                        # Accept only known bootstrap origins
                        if origin and (
                            origin
                            not in {
                                "bootstrap",
                                "wild",
                                "multiplier",
                                "rademacher",
                                "mammen",
                                "webb",
                                # SyntheticControl/SDID
                                "permutation",
                                "wild-if",
                                "placebo",
                            }
                        ):
                            return False
                    return True

                # For ES/DID/Synth/SDID/RCT families: build both SE row and CI row
                ci_cell = ""
                se_cell = ""
                if _is_policy_ci_family(res):
                    # Get SE for ES family
                    se_val = None
                    s = getattr(res, "se", None)
                    if isinstance(s, pd.Series) and _se_allowed(res):
                        try:
                            if key is not None and key in s.index:
                                se_val = float(s[key])
                            elif raw_name in s.index:
                                se_val = float(s[raw_name])
                        except (KeyError, TypeError, ValueError):
                            se_val = None
                    # Only show SE if non-zero and finite (skip baseline period)
                    if se_val is not None and np.isfinite(se_val) and abs(se_val) > 1e-10:
                        se_cell = f"({se_val:{se_format}})"

                    # Get CI (uniform bands) for ES family
                    ci_cell = _uniform_band_str(res, raw_name, coef_format)
                    # Aggregated post-period effects when requested as parameters
                    if (not ci_cell) and str(raw_name) in {"PostATT", "PostATT_Diff"}:
                        ci_cell = _agg_band_str(res, str(raw_name), coef_format)
                    # RCT uniform per-parameter
                    if (not ci_cell) and _is_rct_uniform_band(res):
                        try:
                            bands = getattr(res, "bands", None)
                            if isinstance(bands, dict):
                                u = bands.get("uniform")
                                if isinstance(u, dict):
                                    idx = list(res.params.index).index(raw_name)
                                    lo = np.asarray(u.get("lower")).reshape(-1)[idx]
                                    hi = np.asarray(u.get("upper")).reshape(-1)[idx]
                                    ci_cell = f"[{float(lo):{coef_format}}, {float(hi):{coef_format}}]"
                        except (ValueError, KeyError, IndexError, TypeError):
                            ci_cell = ""
                    # SC/SDID: percentile bands fallback（pointwise）
                    if not ci_cell:
                        est_name = str(res.model_info.get("Estimator", "")).lower()
                        if any(k in est_name for k in ["synthetic", "sdid"]):
                            try:
                                bands = getattr(res, "bands", None)
                                if isinstance(bands, dict):
                                    p = (
                                        bands.get("full")
                                        or bands.get("percentile")
                                        or bands.get("per_tau")
                                    )
                                    param_label = key if key is not None else raw_name
                                    idx = None
                                    try:
                                        idx = list(res.params.index).index(param_label)
                                    except ValueError:
                                        idx = None
                                    if hasattr(p, "loc") and param_label is not None:
                                        try:
                                            lo_v = float(p.loc[param_label, "lower"])
                                            hi_v = float(p.loc[param_label, "upper"])
                                            if np.isfinite(lo_v) and np.isfinite(hi_v):
                                                ci_cell = f"[{lo_v:{coef_format}}, {hi_v:{coef_format}}]"
                                        except (KeyError, IndexError, TypeError, ValueError):
                                            ci_cell = ""
                                            try:
                                                tau_lab = int(param_label)
                                                lo_v = float(p.loc[tau_lab, "lower"])
                                                hi_v = float(p.loc[tau_lab, "upper"])
                                                if np.isfinite(lo_v) and np.isfinite(hi_v):
                                                    ci_cell = f"[{lo_v:{coef_format}}, {hi_v:{coef_format}}]"
                                            except (KeyError, TypeError, ValueError):
                                                ci_cell = ""
                                    elif isinstance(p, dict) and (idx is not None and idx >= 0):
                                        lo_arr = np.asarray(p.get("lower"))
                                        hi_arr = np.asarray(p.get("upper"))
                                        if lo_arr.size > idx and hi_arr.size > idx:
                                            lo_v = float(np.asarray(lo_arr).reshape(-1)[idx])
                                            hi_v = float(np.asarray(hi_arr).reshape(-1)[idx])
                                            if np.isfinite(lo_v) and np.isfinite(hi_v):
                                                ci_cell = f"[{lo_v:{coef_format}}, {hi_v:{coef_format}}]"
                            except (ValueError, KeyError, IndexError, TypeError):
                                ci_cell = ""
                    se_row.append(se_cell)
                    ci_row.append(ci_cell)
                else:
                    # Bootstrap SE policy for non-ES families (core linear, SAR, etc.)
                    se_val = None
                    bse = (res.extra or {}).get("bootstrap_se")
                    if bse is not None:
                        if isinstance(bse, pd.Series):
                            try:
                                if key is not None and key in bse.index:
                                    se_val = float(bse[key])
                                elif raw_name in bse.index:
                                    se_val = float(bse[raw_name])
                            except (KeyError, TypeError, ValueError):
                                se_val = None
                        else:
                            try:
                                arr = np.asarray(bse).reshape(-1)
                                idx = list(res.params.index).index(
                                    key if key is not None else raw_name,
                                )
                                if 0 <= idx < arr.size:
                                    se_val = float(arr[idx])
                            except (IndexError, TypeError, ValueError):
                                se_val = None
                    else:
                        s = getattr(res, "se", None)
                        if isinstance(s, pd.Series) and _se_allowed(res):
                            try:
                                if key is not None and key in s.index:
                                    se_val = float(s[key])
                                elif raw_name in s.index:
                                    se_val = float(s[raw_name])
                            except (KeyError, TypeError, ValueError):
                                se_val = None
                    if (se_val is not None) and _se_allowed(res):
                        se_row.append(f"({se_val:{se_format}})")
                    else:
                        se_row.append("")
                    ci_row.append("")
            else:
                row.append("")
                se_row.append("")
                ci_row.append("")
        table_data.append(row)
        if any(se_row[1:]):
            table_data.append(se_row)
        if any(ci_row[1:]):
            table_data.append(ci_row)

    # Before building the footer, ensure ES-family aggregated PostATT is visible
    # in the main coefficient table even when params lacks 'PostATT'.
    def _append_postatt_if_missing() -> None:
        # Add PostATT row only when (a) at least one model is ES-family with PostATT
        # in model_info, and (b) 'PostATT' is not already present in raw_params.
        if "PostATT" in raw_params:
            return
        has_postatt = False
        for r in results:
            if r is None:
                continue
            v = r.model_info.get("PostATT")
            try:
                if v is not None and str(v) != "":
                    fv = float(v)
                    if np.isfinite(fv):
                        has_postatt = True
                        break
            except (TypeError, ValueError):
                continue
        if not has_postatt:
            return
        # Build a row for PostATT value, SE, and aggregated CI
        row = ["PostATT"]
        se_row = [""]
        ci_row = [""]
        for res in results:
            # Prefer a unified value from model_info if present; else try synonyms in params
            val = res.model_info.get("PostATT", "")
            if val == "":
                # try params synonyms
                syn = _resolve_name(res, "PostATT")
                if syn is not None:
                    try:
                        val = float(res.params[syn])
                    except (KeyError, TypeError, ValueError):
                        val = ""
            row.append("" if val == "" else f"{float(val):{coef_format}}")
            # Try to get SE for PostATT from model_info first, then extra
            se_val = None
            postatt_se = res.model_info.get("PostATT_se") or res.model_info.get("post_att_se")
            if postatt_se is not None:
                try:
                    se_val = float(postatt_se)
                except (TypeError, ValueError):
                    se_val = None
            if se_val is None:
                extra = res.extra or {}
                postatt_se = extra.get("PostATT_se") or extra.get("post_att_se")
                if postatt_se is not None:
                    try:
                        se_val = float(postatt_se)
                    except (TypeError, ValueError):
                        se_val = None
            if se_val is None:
                s = getattr(res, "se", None)
                if isinstance(s, pd.Series):
                    for cand in ("PostATT", "ATT post", "PostATT_Diff", "post_ATT"):
                        if cand in s.index:
                            try:
                                se_val = float(s[cand])
                                break
                            except (KeyError, TypeError, ValueError):
                                # Continue to next candidate
                                se_val = se_val  # noqa: PLW0127
            se_row.append(f"({se_val:{se_format}})" if se_val is not None and np.isfinite(se_val) and abs(se_val) > 1e-10 else "")
            # Aggregated CI (from model_info bands)
            band_text = _agg_band_str(res, "PostATT", coef_format)
            ci_row.append(band_text)
        table_data.append(row)
        if any(se_row[1:]):
            table_data.append(se_row)
        if any(ci_row[1:]):
            table_data.append(ci_row)

    _append_postatt_if_missing()

    # After building the core table rows, remove any lingering duplicate PostATT-like
    # entries that escaped normalization (defensive). Keep only the unified 'PostATT'.
    def _is_postatt_like(label: str) -> bool:
        s = str(label).strip()
        return s == "PostATT" or s.lower().replace(" ", "") in {
            "attpost",
            "post_att",
            "post-att",
        }

    deduped: list[list[str]] = []
    seen_postatt = False
    for row in table_data:
        if not row:
            continue
        lab = row[0]
        if _is_postatt_like(lab):
            if seen_postatt:
                continue  # drop duplicates
            seen_postatt = True
            # Force the label to the canonical 'PostATT'
            row[0] = "PostATT"
        deduped.append(row)
    table_data = deduped

    if norm_stats:
        for label, func, options in norm_stats:
            stat_row = [label]
            for idx, res in enumerate(results):
                # Special handling for Moran's I with per-model configuration
                if label == "Moran's I" and "_get_moran_type" in (options or {}):
                    _get_mtype = options["_get_moran_type"]
                    weights_for_model = _resolve_option(
                        options["weights"], idx, model_names[idx],
                    )
                    if weights_for_model is not None:
                        mtype = _get_mtype(idx, model_names[idx])
                        kwargs = {"weights": weights_for_model, "moran_type": mtype}
                    else:
                        # No weights for this model - show empty
                        stat_row.append("")
                        continue
                else:
                    kwargs = {
                        k: _resolve_option(v, idx, model_names[idx])
                        for k, v in (options or {}).items()
                        if k != "_get_moran_type"
                    }
                try:
                    value = func(res, **kwargs)
                except (
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    ZeroDivisionError,
                    np.linalg.LinAlgError,
                ):
                    value = np.nan
                stat_row.append(_format_stat_value(value))
            table_data.append(stat_row)

    # Footer: observations and a minimal set of model_info key-values
    # Default policy: show only essential keys and hide empty or non-informative rows.
    footer_rows: list[list[str]] = []
    footer_rows.append(["Observations", *[str(res.n_obs) for res in results]])

    # Determine candidate footer keys
    if footer_keys is None:
        # Minimal default set tuned for common estimators
        # Note: omit PostATT to avoid duplication with the main table row
        candidate_keys = [
            "B",  # show bootstrap reps only when > 0
            "OverID_df",  # GMM/IV/SAR2SLS only when > 0
            "MoranI",  # SAR2SLS diagnostic
            # 'PostATT' intentionally omitted to avoid duplication
            # 'PostATT_Diff' intentionally omitted; unify on 'PostATT'
            # Spatial spill aggregate presented via separate (Spill) column; omit row
            "Constraints",  # only when not empty/False
        ]
        if moran_weights is not None:
            candidate_keys = [k for k in candidate_keys if k != "MoranI"]
    else:
        candidate_keys = list(footer_keys)

    # Enforce: never show p-values, stars, or any p-value-like artifacts in the footer.
    banned_footer_keys = {
        "p", "pval", "p_value", "pvalue", "pvalues",
        "p>|t|", "p>|z|", "p>|chi2|", "p>|f|",
        "stars", "signif", "significance",
    }

    # Build an ordered unique list from candidate keys only
    ordered_keys = []
    seen_keys = set()
    for key in candidate_keys:
        if key in seen_keys:
            continue
        # Skip banned keys (case-insensitive)
        k_low = str(key).strip().lower()
        if k_low in banned_footer_keys:
            continue
        seen_keys.add(key)
        ordered_keys.append(key)

    # Helper: determine if a row has informative content according to key semantics
    def _informative_row(key: str, vals: list[object]) -> bool:
        # Convert values to primitives
        norm: list[object] = []
        for v in vals:
            if v is None:
                norm.append(None)
            elif isinstance(v, (int, float)):
                norm.append(v)
            elif isinstance(v, str) and v.strip() != "":
                # interpret numeric strings when possible
                try:
                    norm.append(float(v))
                except (TypeError, ValueError):
                    norm.append(v.strip())
            else:
                norm.append(str(v).strip())
        # Per-key rules
        if key.lower() == "b":
            return any((isinstance(x, (int, float)) and float(x) > 0) for x in norm)
        if key.lower() == "overid_df":
            return any((isinstance(x, (int, float)) and float(x) > 0) for x in norm)
        if key.lower() == "morani":
            return any(
                (isinstance(x, (int, float)) and np.isfinite(float(x))) for x in norm
            )
        if key.lower() in {"postatt", "postatt_diff", "postspill"}:
            return any(
                (isinstance(x, (int, float)) and np.isfinite(float(x))) for x in norm
            )
        if key.lower() == "constraints":
            # show only if any model has non-empty and not a literal False
            return any((str(x).lower() not in {"", "none", "false"}) for x in norm)
        # Fallback: any non-empty, non-"nan" string
        return any((x is not None and str(x).lower() not in {"", "nan"}) for x in norm)

    # Append filtered footer rows (skip banned keys and empty rows)
    for key in ordered_keys:
        # Skip keys that violate policy (already filtered above, but double-check)
        k_low = str(key).strip().lower()
        if k_low in banned_footer_keys:
            continue
        raw_vals = [res.model_info.get(key, "") for res in results]
        if not _informative_row(str(key), raw_vals):
            continue
        lab = str(key)
        if output == "latex" and escape_latex:
            lab = _escape_latex(lab)
        lab = _truncate_text(lab, col_width) if output == "latex" else lab
        vals: list[str] = []
        # Special policy: for Constraints, display 'True' where applicable and blank otherwise
        is_constraints = str(key).strip().lower() == "constraints"
        for v in raw_vals:
            if is_constraints:
                cell = (
                    "True"
                    if (isinstance(v, (bool, int, float)) and bool(v))
                    or (isinstance(v, str) and v.strip().lower() in {"true", "yes"})
                    else ""
                )
                if output == "latex" and escape_latex:
                    cell = _escape_latex(cell)
                cell = _truncate_text(cell, col_width) if output == "latex" else cell
                vals.append(cell)
            else:
                sv = str(v)
                if output == "latex" and escape_latex:
                    sv = _escape_latex(sv)
                sv = _truncate_text(sv, col_width) if output == "latex" else sv
                vals.append(sv)
        footer_rows.append([lab, *vals])
    if output == "latex":
        # LaTeX: concatenate body and footer inside table. Use a single-row
        # placeholder so post-processing can replace it with \midrule.
        table_all = [
            *table_data,
            _hline_placeholder(model_names),
            *footer_rows,
        ]
        headers = [""] + [
            (_escape_latex(n) if escape_latex else str(n)) for n in model_names
        ]
        if truncate:
            headers = [headers[0]] + [_truncate_text(h, col_width) for h in headers[1:]]
        tablefmt = "latex_booktabs" if latex_booktabs else "latex"
        table = cast(
            "str",
            tabulate(table_all, headers=headers, stralign="center", tablefmt=tablefmt),
        )
        # Post-process: replace the MSMIDRULE token row(s) with a single \midrule
        if "MSMIDRULE" in table:
            lines = table.splitlines()
            out_lines = []
            for ln in lines:
                # Strict replacement: only replace a placeholder row that begins with
                # MSMIDRULE and contains exactly (1 + len(model_names)) placeholders.
                if ln.strip().startswith("MSMIDRULE") and ln.count("MSMIDRULE") == (
                    1 + len(model_names)
                ):
                    out_lines.append(r"\midrule")
                    continue
                out_lines.append(ln)
            table = "\n".join(out_lines)
        # Column spec override (e.g., ensure centered numeric columns)
        with suppress(re.error, TypeError, ValueError):
            k = len(model_names)
            default_colspec = "l" + ("c" * k)
            colspec = (
                latex_colspec
                if isinstance(latex_colspec, str) and latex_colspec.strip()
                else default_colspec
            )
            # Build replacement without f-string to avoid brace escaping issues
            repl = "\\1" + colspec + "}"
            table = re.sub(r"(\\begin\{tabular\}\{)([^}]*)\}", repl, table, count=1)
        # Optional resizebox wrapper
        if latex_resizebox:
            table = f"\\resizebox{{{latex_width}}}{{!}}{{\n{table}\n}}"
        return table
    # text output: render body and footer together to ensure column alignment
    headers = ["", *list(model_names)]
    # Combine rows so tabulate computes a single consistent column layout.
    sep = ["" for _ in headers]  # blank separator row
    table_all = [*table_data, sep, *footer_rows]
    tbl = cast("str", tabulate(table_all, headers=headers, stralign="center"))
    # Historical note about bootstrap-only policy has been removed per user request.
    # Retain the keyword for backward compatibility but suppress the message.
    if show_policy_note:
        _ = None  # no-op to preserve parameter semantics without emitting text
    return tbl


def diagnostics(  # noqa: PLR0913
    results: list[EstimationResult],
    model_names: list[str] | None = None,
    *,
    output: str = "text",  # {"text","latex"}
    latex_booktabs: bool = True,
    escape_latex: bool = True,
    col_width: int = 18,
) -> str:
    """Build a compact diagnostics table for weak-IV and overidentification tests.

    Output
    ------
    - Weak-IV diagnostic: min_partial_F (Sanderson-Windmeijer), mean_partial_F if available.
    - Overidentification: Hansen's J statistic (value only) and df when provided.
    - Spatial diagnostic (if present): Moran's I.
    - Bootstrap reps (B) for transparency.

    Note:
    ----
    No p-values or star annotations are reported here either.

    """
    if model_names is None:
        model_names = [f"({i + 1})" for i in range(len(results))]
    if output not in {"text", "latex"}:
        raise ValueError("output must be either 'text' or 'latex'.")
    # Escape/truncate headers for LaTeX
    headers = [""] + [
        (_escape_latex(n) if (output == "latex" and escape_latex) else str(n))
        for n in model_names
    ]
    if output == "latex":
        headers = [headers[0]] + [
            h if len(h) <= col_width else h[: col_width - 1] + "…" for h in headers[1:]
        ]

    rows: list[list[str]] = []
    rows.append(
        _collect_extra(
            results, "first_stage_stats.min_partial_F", "Weak IV: min partial F",
        ),
    )
    rows.append(
        _collect_extra(
            results, "first_stage_stats.mean_partial_F", "Weak IV: mean partial F",
        ),
    )
    rows.append(
        _collect_extra(
            results, "first_stage_stats.bootstrap_robust_f", "First-stage: robust F (bootstrap)",
        ),
    )
    rows.append(
        _collect_extra(results, "first_stage_stats.cd_min_eig", "Cragg-Donald min eig"),
    )
    rows.append(
        _collect_extra(results, "first_stage_stats.kp_min_eig", "KP rk Wald min eig"),
    )
    rows.append(
        _collect_extra(
            results, "first_stage_stats.kp_rk_lm_min_eig", "KP rk LM min eig",
        ),
    )
    # Accept either J_stat in extra or OverID_stat in extra/OverID_df in model_info
    j_row_extra = _collect_extra(results, "J_stat", "Hansen J")
    # If no extra values, fall back to model_info OverID_stat key
    j_row_info = _collect_info(results, "OverID_stat", "Hansen J")
    # Merge: prefer extra values if present else model_info values
    merged_j = [
        j_row_extra[0],
        *(
            j_row_extra[j] or j_row_info[j]
            for j in range(1, len(j_row_extra))
        ),
    ]
    rows.append(merged_j)
    rows.append(_collect_info(results, "OverID_df", "OverID df"))
    # Spatial diagnostic (accept either extra.moran_I or model_info.MoranI)
    spatial_row_extra = _collect_extra(results, "moran_I", "Moran's I (resid)")
    spatial_row_info = _collect_info(results, "MoranI", "Moran's I (resid)")
    # Merge: prefer extra if available else model_info
    merged_spatial = [
        spatial_row_extra[0],
        *(
            spatial_row_extra[j] or spatial_row_info[j]
            for j in range(1, len(spatial_row_extra))
        ),
    ]
    rows.append(merged_spatial)
    # Event-study aggregated effects if present
    rows.append(_collect_info(results, "PostATT", "PostATT"))
    rows.append(_collect_info(results, "Constraints", "Constraints"))
    rows.append(_collect_info(results, "B", "Bootstrap B"))

    # Drop rows with no informative content across all models
    def _nonempty(row: list[str]) -> bool:
        # row[0] is label; check remaining cells
        cells = [str(c).strip().lower() for c in row[1:]]
        # Treat '', 'nan' as empty
        return any(c not in {"", "nan"} for c in cells)

    rows = [r for r in rows if _nonempty(r)]

    if output == "latex":
        # Escape/truncate row labels for LaTeX
        for r in rows:
            if r and isinstance(r[0], str):
                lab = r[0]
                lab = _escape_latex(lab) if escape_latex else lab
                if len(lab) > col_width:
                    lab = lab[: col_width - 1] + "…"
                r[0] = lab
        tablefmt = "latex_booktabs" if latex_booktabs else "latex"
        return cast(
            "str", tabulate(rows, headers=headers, stralign="center", tablefmt=tablefmt),
        )
    # text
    return cast("str", tabulate(rows, headers=headers, stralign="center"))


def weakiv_table(res: EstimationResult) -> pd.DataFrame:
    """Return a compact weak-IV table for a single EstimationResult.

    This function emits values only (no p-values): SW partial F per-endogenous,
    min partial F, Cragg-Donald min eigenvalue, KP min eigenvalue, and OverID J value.
    """
    fs = (res.extra or {}).get("first_stage_stats", {})
    rows: list[dict[str, float | str]] = []

    def _get_value(keys: Sequence[str]) -> float | None:
        for key in keys:
            if key in fs:
                return float(fs[key])
        lower_map = {str(k).lower(): k for k in fs}
        for key in keys:
            actual = lower_map.get(key.lower())
            if actual is not None:
                try:
                    return float(fs[actual])
                except (TypeError, ValueError):
                    return None
        return None

    min_f = _get_value(("min_partial_F", "minF", "sw_min_partial_f", "f_sw_min"))
    if min_f is not None:
        rows.append({"metric": "SW min partial F", "value": min_f})

    boot_rf = _get_value(("bootstrap_robust_f", "boot_robust_f", "robust_first_stage_f"))
    if boot_rf is not None:
        rows.append({"metric": "Robust first-stage F (bootstrap)", "value": boot_rf})
    # per-endogenous SW entries
    sw_entry_added = False
    for k, v in fs.items():
        if k.startswith("F_SW_partial_"):
            rows.append(
                {
                    "metric": k.replace("F_SW_partial_", "SW partial F (") + ")",
                    "value": float(v),
                },
            )
            sw_entry_added = True
    if not sw_entry_added:
        sw_list = fs.get("SW_partial_F_list") or fs.get("sw_partial_f_list")
        if isinstance(sw_list, Sequence):
            for item in sw_list:
                if not isinstance(item, Sequence) or len(item) < 2:
                    continue
                name, val = item[0], item[1]
                try:
                    rows.append(
                        {"metric": f"SW partial F ({name})", "value": float(val)},
                    )
                    sw_entry_added = True
                except (TypeError, ValueError):
                    continue
    for keys, label in [
        (
            ("cd_min_eig", "CD_min_eig", "cragg_donald_min_eig"),
            "Cragg-Donald min eigenvalue",
        ),
        (
            ("kp_min_eig", "KP_min_eig", "KP_rk_min_eig", "kp_rk_min_eig"),
            "KP min eigenvalue",
        ),
    ]:
        val = _get_value(keys)
        if val is not None:
            rows.append({"metric": label, "value": val})
    j = (res.extra or {}).get("J_stat", (res.extra or {}).get("OverID_stat", None))
    if j is not None:
        rows.append({"metric": "OverID (J) value", "value": float(j)})
    return pd.DataFrame(rows)


# _filter_and_order_params is provided by libs/lineareg/utils/helpers.py

# Pretty-printer utilities are implemented in libs/lineareg/utils/helpers.py
