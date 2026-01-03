"""Shared helper utilities.

Table formatting, parameter selection, and event-time utilities for
estimators and output modules.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Iterable, Sequence

    from lineareg.estimators.base import EstimationResult
else:
    Iterable = Sequence = tuple  # type: ignore[assignment]
    EstimationResult = Any  # type: ignore[misc, assignment]

__all__ = [
    "collect_extra",
    "collect_info",
    "collect_param_index",
    "escape_latex",
    "filter_and_order_params",
    "hline_placeholder",
    "kv_row",
    "pretty_term",
    "time_to_pos",
    "event_tau",
    "prev_time",
]

import numpy as np


def time_to_pos(times: Sequence[Any] | np.ndarray) -> dict[Any, int]:
    """Map time values to their sorted index position."""
    return {t: i for i, t in enumerate(times)}


def event_tau(t: Any, g: Any, t2pos: dict) -> int:
    """Compute event time tau = t - g, falling back to positional difference."""
    try:
        return int(t) - int(g)
    except (TypeError, ValueError):
        return t2pos[t] - t2pos[g]


def prev_time(t: Any, times_sorted: np.ndarray) -> Any | None:
    """Find the previous time period in a sorted array."""
    idx = np.searchsorted(times_sorted, t) - 1
    if idx < 0:
        return None
    return times_sorted[idx]


def collect_param_index(results: Sequence[EstimationResult], *, skip_missing: bool = False) -> list[Any]:
    """Return an ordered list of parameter identifiers appearing across results.

    Parameters are ordered by first appearance across the provided
    ``EstimationResult`` objects. When ``skip_missing`` is True, only parameters
    present in *all* results are returned.
    """
    total = len(results)
    counts: dict[Any, int] = {}
    ordered: list[Any] = []
    for res in results:
        params = getattr(res, "params", None)
        if params is None:
            continue
        for name in list(params.index):
            counts[name] = counts.get(name, 0) + 1
            if name not in ordered:
                ordered.append(name)
    if skip_missing and total:
        ordered = [name for name in ordered if counts.get(name, 0) == total]
    return ordered


def _pattern_matches(label: Any, pattern: str) -> bool:
    """Return True when ``label`` matches the provided pattern."""
    text = str(label)
    try:
        return bool(re.search(pattern, text))
    except re.error:
        return text == pattern


def _tau_sort_key(label: Any) -> tuple[int, float, str]:
    """Build a sortable key for event-time labels.

    Recognises integers, floats, and tokens like ``tau=-2`` or ``D-2``.
    Non-parsable labels fallback to alphabetical ordering.
    """
    if isinstance(label, (int, float)):
        return (0, float(label), str(label))
    text = str(label).strip()
    try:
        val = float(text)
        return (0, val, text)
    except ValueError:
        pass
    m = re.match(r"^(?:tau|event[_ ]?time|et)\s*[:=]\s*(-?\d+(?:\.\d+)?)$", text, flags=re.IGNORECASE)
    if m:
        try:
            return (0, float(m.group(1)), text)
        except ValueError:
            return (1, 0.0, text.lower())
    m = re.match(r"^D(-?\d+)$", text, flags=re.IGNORECASE)
    if m:
        try:
            return (0, float(m.group(1)), text)
        except ValueError:
            return (1, 0.0, text.lower())
    return (1, 0.0, text.lower())


def filter_and_order_params(  # noqa: PLR0913
    raw_pool: Sequence[Any],
    *,
    params: Sequence[Any] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    sort: str = "alpha",
    strict: bool = True,
) -> list[Any]:
    """Filter and order parameter identifiers for summary tables.

    Parameters
    ----------
    raw_pool
        Iterable of raw parameter names (e.g., from :func:`collect_param_index`).
    params / include / exclude
        Manual controls for selecting parameters. ``params`` takes precedence
        and preserves the provided ordering. ``include`` and ``exclude`` accept
        regular expressions (falling back to literal comparison when a pattern
        cannot be compiled).
    sort
        ``"alpha"`` (default) sorts alphabetically, ``"tau"`` sorts by event
        time, and ``"none"`` preserves the incoming order.
    strict
        When True, missing entries referenced in ``params`` raise ``ValueError``.

    """
    # Base set of available parameters (preserves first appearance ordering).
    seen: dict[Any, None] = {}
    for name in raw_pool:
        if name not in seen:
            seen[name] = None
    available = list(seen.keys())

    selected: list[Any] = []
    missing: list[Any] = []
    if params is not None:
        for name in params:
            if name in available:
                if name not in selected:
                    selected.append(name)
            elif strict:
                missing.append(name)
            else:
                selected.append(name)
        if missing:
            joined = ", ".join(str(m) for m in missing)
            raise ValueError(f"Requested parameter(s) not found: {joined}")
    else:
        selected = available.copy()

    def _filter(pool: Sequence[Any], patterns: Sequence[str], *, mode: str) -> list[Any]:
        out: list[Any] = []
        for name in pool:
            match = any(_pattern_matches(name, pat) for pat in patterns)
            if (mode == "include" and match) or (mode == "exclude" and not match):
                out.append(name)
            elif (mode == "include" and not match) or (mode == "exclude" and match):
                continue
        return out

    if include:
        selected = _filter(selected, include, mode="include")
        if strict and not selected:
            raise ValueError("include patterns filtered out all parameters.")
    if exclude:
        selected = _filter(selected, exclude, mode="exclude")

    if sort.lower() == "alpha":
        selected = sorted(selected, key=lambda x: str(x).lower())
    elif sort.lower() == "tau":
        selected = sorted(selected, key=_tau_sort_key)
    elif sort.lower() != "none":
        raise ValueError("sort must be one of {'alpha','tau','none'}")
    return selected


def escape_latex(obj: Any) -> str:
    """Minimal LaTeX escaping (consistent with tabulate's expectations)."""
    text = str(obj)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def pretty_term(
    name: Any, *, style: str = "paper",
) -> str:
    """Lightweight pretty-printer for parameter labels.

    Currently this performs minimal cleanup (replace underscores with spaces for
    ``style='paper'``). LaTeX mode simply forwards the string; escaping is
    handled by :func:`escape_latex` upstream to avoid double-escaping.
    """
    # LaTeX handling is performed upstream by escape_latex to avoid double-escaping
    text = str(name)
    if style == "paper":
        text = text.replace("_", " ")
    # LaTeX mode is handled upstream by escape_latex; here we simply return
    # the text unchanged when latex is requested (avoid double-escaping).
    return text


def _format_value(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.6g}"
    return "" if val is None else str(val)


def collect_info(results: Sequence[EstimationResult], key: str, label: str | None = None) -> list[str]:
    """Collect values from ``res.model_info[key]`` across results."""
    row: list[str] = [label or key]
    for res in results:
        info = getattr(res, "model_info", {}) or {}
        val = info.get(key, "")
        row.append(_format_value(val) if val != "" else "")
    return row


def _traverse_path(root: Any, path: Sequence[str]) -> Any:
    current = root
    for part in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part, None)
        else:
            current = getattr(current, part, None)
    return current


def collect_extra(results: Sequence[EstimationResult], path: str, label: str | None = None) -> list[str]:
    """Collect nested values from ``res.extra`` following ``path`` dotted notation."""
    keys = path.split(".")
    row: list[str] = [label or path]
    for res in results:
        extra = getattr(res, "extra", {}) or {}
        val = _traverse_path(extra, keys)
        row.append(_format_value(val) if val not in (None, "") else "")
    return row


def kv_row(label: Any, values: Iterable[Any], *, value_width: int = 18, label_width: int = 18) -> str:
    """Format a key/value row for plain-text summaries.

    ``values`` should be an iterable of already-rendered strings. Missing values
    are represented as empty strings.
    """
    label_str = str(label)
    rendered = [str(v) if v is not None else "" for v in values]
    value_fmt = " ".join(f"{val:<{value_width}}" for val in rendered)
    return f"{label_str:<{label_width}} {value_fmt}".rstrip()


def hline_placeholder(model_names: Sequence[str]) -> list[str]:
    """Build a placeholder row used by LaTeX post-processing to insert a \\midrule.

    The placeholder is detected and replaced downstream in
    ``lineareg.output.summary``.
    """
    n_cols = len(list(model_names)) + 1  # include stub column
    return ["MSMIDRULE"] * n_cols
