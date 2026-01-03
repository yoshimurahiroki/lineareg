"""Dialect-specific intercept column handling.

R, Stata, and statsmodels constant column placement and collinearity
detection. Replicates model.matrix, _cons, and add_constant semantics.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["add_constant"]

# Constants
_NDIM_2D = 2
_CONST_TOL = 1e-12


def add_constant(  # noqa: PLR0913
    X: np.ndarray,
    var_names: Sequence[str] | None = None,
    *,
    position: str | None = None,
    const_name: str | None = None,
    force_name: str | None = None,
    allow_na_in_constant: bool = True,
    row_mask: np.ndarray | None = None,
    dialect: str | None = None,
    include_intercept: bool = True,
    statsmodels_has_constant: str | None = None,
    warn_on_existing_constant: bool = True,
    tol: float = _CONST_TOL,
    rank_policy: str | None = None,
) -> tuple[np.ndarray, list[str], str]:
    """Add intercept column with strict R/Stata/statsmodels compliance.

    R dialect: (Intercept) at front, existing constants normalized
    Stata dialect: _cons at back, existing constants normalized
    statsmodels: has_constant='add'|'skip'|'raise' semantics

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
        Design matrix
    var_names : Sequence[str] | None
        Variable names
    position : str | None
        Intercept position ('front'|'back'); dialect sets default
    const_name : str | None
        Intercept name; dialect sets default
    force_name : str | None
        Override const_name
    allow_na_in_constant : bool
        For statsmodels: allow NaN in masked rows
    row_mask : np.ndarray | None
        Boolean mask for active rows
    dialect : str | None
        'r'|'stata'|'statsmodels'; default 'stata'
    include_intercept : bool
        Whether to add intercept
    statsmodels_has_constant : str | None
        'add'|'skip'|'raise' for statsmodels dialect
    warn_on_existing_constant : bool
        Warn about existing constant columns

    Returns
    -------
    X_out : np.ndarray
        Matrix with intercept
    names_out : list[str]
        Variable names with intercept
    const_name_out : str
        Intercept name used

    Notes
    -----
    R/Stata: If all-ones column exists, it is normalized (position/name).
    Other constant columns are dropped (collinearity). If no all-ones column,
    other constants dropped and new intercept added at dialect position.

    statsmodels: Strictly follows has_constant parameter.

    """
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != _NDIM_2D:
        msg = "X must be 2D."
        raise ValueError(msg)
    n, k = X.shape

    names = _normalize_variable_names(var_names, k)
    _validate_unique_names(names)

    dialect_str = _validate_dialect(dialect)
    const_name_final = _get_const_name(dialect_str, const_name, force_name)

    _validate_position_for_dialect(position, dialect_str)

    active = _create_active_mask(row_mask, n)
    tol_eff = _determine_tolerance(rank_policy, tol)
    ones_idx, const_idx = _find_constant_cols(X, active, tol_eff)

    if not include_intercept:
        return _handle_no_intercept(
            X,
            names,
            const_name_final,
            ones_idx,
            const_idx,
            warn_on_existing_constant,
        )

    intercept_col = _make_intercept_col(
        n, dialect_str, row_mask, active, allow_na_in_constant,
    )

    if k == 0:
        return intercept_col, [const_name_final], const_name_final

    _check_reserved_name_conflicts(
        const_name_final, names, ones_idx, const_idx, dialect_str,
    )

    if dialect_str == "r":
        Xo, No, Co = _handle_r(
            X,
            names,
            const_name_final,
            ones_idx,
            const_idx,
            warn_on_existing_constant,
            intercept_col,
        )
        _assert_no_name_collision(No)
        return Xo, No, Co

    if dialect_str == "stata":
        Xo, No, Co = _handle_stata(
            X,
            names,
            const_name_final,
            ones_idx,
            const_idx,
            warn_on_existing_constant,
            intercept_col,
        )
        _assert_no_name_collision(No)
        return Xo, No, Co

    # statsmodels
    Xo, No, Co = _handle_statsmodels(
        X,
        names,
        const_name_final,
        ones_idx,
        const_idx,
        statsmodels_has_constant,
        intercept_col,
        position=position or "front",
    )
    _handle_statsmodels_name_collision(No, Co, statsmodels_has_constant)
    return Xo, No, Co


def _normalize_variable_names(var_names: Sequence[str] | None, k: int) -> list[str]:
    """Normalize variable names to a list of strings."""
    if var_names is None:
        return [f"x{i}" for i in range(k)]

    names = [str(nm) for nm in var_names]
    if len(names) != k:
        msg = f"var_names length ({len(names)}) does not match X columns ({k})."
        raise ValueError(msg)
    return names


def _validate_unique_names(names: list[str]) -> None:
    """Ensure all variable names are unique."""
    if len(names) != len(set(names)):
        # Find first duplicate
        dup_seen = set()
        for nm in names:
            if nm in dup_seen:
                raise ValueError(
                    f"Duplicate variable name in var_names: '{nm}'. "
                    "Provide unique names.",
                )
            dup_seen.add(nm)


def _validate_dialect(dialect: str | None) -> str:
    """Validate and normalize dialect string."""
    dialect_str = "stata" if dialect is None else str(dialect).lower()
    if dialect_str not in {"r", "stata", "statsmodels"}:
        raise ValueError(
            f"Invalid dialect: '{dialect_str}'. "
            "Must be one of: 'r', 'stata', 'statsmodels'.",
        )
    return dialect_str


def _validate_position_for_dialect(position: str | None, dialect: str) -> None:
    """Validate position parameter is compatible with dialect."""
    if position is None:
        return

    pos = str(position).lower()
    if pos not in {"front", "back"}:
        raise ValueError("position must be 'front' or 'back'.")

    if dialect == "r" and pos != "front":
        raise ValueError("R dialect requires position='front'.")
    if dialect == "stata" and pos != "back":
        raise ValueError("Stata dialect requires position='back'.")


def _create_active_mask(row_mask: np.ndarray | None, n: int) -> np.ndarray:
    """Create boolean mask for active rows."""
    if row_mask is None:
        return np.ones(n, dtype=bool)

    active = np.asarray(row_mask, dtype=bool).reshape(-1)
    if active.shape[0] != n:
        raise ValueError(
            f"row_mask length ({active.shape[0]}) does not match X rows ({n}).",
        )
    return active


def _determine_tolerance(rank_policy: str | None, tol: float) -> float:
    """Determine numerical tolerance based on rank policy."""
    if rank_policy is None:
        return float(tol)

    rp = str(rank_policy).lower()
    if rp not in {"r", "stata"}:
        raise ValueError("rank_policy must be one of {'r','stata'} or None.")

    return 1e-7 if rp == "r" else 1e-10


def _check_reserved_name_conflicts(
    const_name: str,
    names: list[str],
    ones_idx: list[int],
    const_idx: list[int],
    dialect: str,
) -> None:
    """Check if a non-constant column uses a reserved intercept name."""
    if const_name not in names:
        return

    j = int(names.index(const_name))
    if j in ones_idx or j in const_idx:
        return

    if dialect == "r":
        raise ValueError(
            "A non-constant column uses the reserved name '(Intercept)'. "
            "Rename the variable.",
        )
    if dialect == "stata":
        raise ValueError(
            "A non-constant column uses the reserved name '_cons'. "
            "Rename the variable.",
        )
    raise ValueError(
        f"A non-constant column uses the reserved intercept name '{const_name}'. "
        "Rename the variable.",
    )


def _handle_statsmodels_name_collision(
    names: list[str],
    const_name: str,
    statsmodels_has_constant: str | None,
) -> None:
    """Handle statsmodels-specific name collision rules."""
    sm_mode = (
        "skip"
        if statsmodels_has_constant is None
        else str(statsmodels_has_constant).lower()
    )

    if sm_mode == "add":
        # Allow duplicate 'const' name only
        if const_name != "const":
            raise ValueError(
                "statsmodels 'add' mode only permits duplicating "
                "the canonical name 'const'.",
            )
        _assert_no_name_collision(
            names,
            allow_duplicate=True,
            allowed_duplicate_name=const_name,
        )
    else:
        _assert_no_name_collision(names)


def _get_const_name(
    dialect: str,
    const_name: str | None,
    force_name: str | None,
) -> str:
    """Get constant column name per dialect."""
    # Enforce dialect-specific canonical names. R and Stata have fixed
    # intercept names and do not permit overriding. For statsmodels, the
    # canonical name is 'const' and any forced name must equal that.
    if force_name:
        if dialect in {"r", "stata"}:
            raise ValueError(
                "force_name is not allowed for R/Stata dialects; names are fixed.",
            )
        # statsmodels dialect: enforce 'const'
        if dialect == "statsmodels" and str(force_name) != "const":
            raise ValueError(
                "statsmodels dialect requires const_name='const' when using force_name.",
            )
        return str(force_name)
    if dialect == "r":
        return "(Intercept)"
    if dialect == "stata":
        return "_cons"
    return const_name if const_name else "const"


def _find_constant_cols(
    X: np.ndarray,
    active: np.ndarray,
    tol: float,
) -> tuple[list[int], list[int]]:
    """Find indices of all-ones and other constant columns.

    Detection is performed only on the active rows. The tolerance `tol`
    should come from the caller and is usually derived from the configured
    `rank_policy` (e.g. 1e-7 for R, 1e-10 for Stata).
    """
    k = X.shape[1]
    ones_idx: list[int] = []
    const_idx: list[int] = []

    for j in range(k):
        col = X[:, j]
        vals = col[active]

        if vals.size == 0:
            continue

        # Convert to float; skip non-numeric columns
        try:
            vf = vals.astype(float)
        except (ValueError, TypeError):
            continue

        if not np.all(np.isfinite(vf)):
            continue

        # Check if all-ones column (max deviation from 1)
        if _is_all_ones(vf, tol):
            ones_idx.append(j)
            continue

        # Check if constant column (range near zero)
        if _is_constant(vf, tol):
            const_idx.append(j)

    return ones_idx, const_idx


def _is_all_ones(values: np.ndarray, tol: float) -> bool:
    """Check if array is all ones within tolerance."""
    max_deviation = float(np.max(np.abs(values - 1.0)))
    return max_deviation <= float(tol)


def _is_constant(values: np.ndarray, tol: float) -> bool:
    """Check if array has constant values within tolerance."""
    vmax = float(np.max(values))
    vmin = float(np.min(values))
    return np.isfinite(vmax) and np.isfinite(vmin) and (vmax - vmin) <= float(tol)


def _handle_no_intercept(  # noqa: PLR0913
    X: np.ndarray,
    names: list[str],
    const_name: str,
    ones_idx: list[int],
    const_idx: list[int],
    warn: bool,
) -> tuple[np.ndarray, list[str], str]:
    """Handle include_intercept=False case."""
    return X, names, const_name


def _make_intercept_col(
    n: int,
    _dialect: str,
    _row_mask: np.ndarray | None,
    _active: np.ndarray,
    _allow_na: bool,
) -> np.ndarray:
    """Build intercept column.

    By default the intercept is a column of ones.

    If a row mask is provided and ``_allow_na`` is True, rows that are not
    active will be filled with NaN. This prevents masked-out rows from being
    accidentally treated as valid in downstream computations that may not
    consistently apply the row mask.
    """
    out = np.ones((n, 1), dtype=np.float64)
    if _row_mask is not None and bool(_allow_na):
        out[~_active, 0] = np.nan
    return out


def _handle_r(  # noqa: PLR0913
    X: np.ndarray,
    names: list[str],
    const_name: str,
    ones_idx: list[int],
    const_idx: list[int],
    warn: bool,
    intercept_col: np.ndarray,
) -> tuple[np.ndarray, list[str], str]:
    """Handle R dialect: (Intercept) always at front."""
    if warn and (ones_idx or const_idx):
        warnings.warn(
            "Existing constant column(s) detected; normalizing to R intercept semantics.",
            RuntimeWarning,
            stacklevel=2,
        )
    if ones_idx:
        return _normalize_existing_ones_r(
            X,
            names,
            const_name,
            ones_idx,
            const_idx,
            warn,
        )
    return _add_new_intercept_r(
        X,
        names,
        const_name,
        const_idx,
        warn,
        intercept_col,
    )


def _handle_stata(  # noqa: PLR0913
    X: np.ndarray,
    names: list[str],
    const_name: str,
    ones_idx: list[int],
    const_idx: list[int],
    warn: bool,
    intercept_col: np.ndarray,
) -> tuple[np.ndarray, list[str], str]:
    """Handle Stata dialect: _cons always at back."""
    if warn and (ones_idx or const_idx):
        warnings.warn(
            "Existing constant column(s) detected; normalizing to Stata intercept semantics.",
            RuntimeWarning,
            stacklevel=2,
        )
    if ones_idx:
        return _normalize_existing_ones_stata(
            X,
            names,
            const_name,
            ones_idx,
            const_idx,
            warn,
        )
    return _add_new_intercept_stata(
        X,
        names,
        const_name,
        const_idx,
        warn,
        intercept_col,
    )


def _normalize_existing_ones_r(  # noqa: PLR0913
    X: np.ndarray,
    names: list[str],
    const_name: str,
    ones_idx: list[int],
    const_idx: list[int],
    warn: bool,
) -> tuple[np.ndarray, list[str], str]:
    """R: Adopt first all-ones column, move to front, drop others."""
    if warn and (len(ones_idx) > 1 or const_idx):
        warnings.warn(
            "Dropping redundant constant column(s) to avoid collinearity in R dialect.",
            RuntimeWarning,
            stacklevel=2,
        )
    X_work = X.copy()
    names_work = list(names)
    j = int(ones_idx[0])

    drop = sorted(list(const_idx) + list(ones_idx[1:]), reverse=True)
    for t in drop:
        X_work = np.delete(X_work, t, axis=1)
        del names_work[t]
        if t < j:
            j -= 1

    if j != 0:
        col = X_work[:, j : j + 1].copy()
        X_work = np.delete(X_work, j, axis=1)
        del names_work[j]
        X_work = np.column_stack([col, X_work])
        names_work = [const_name, *names_work]
    else:
        names_work[0] = const_name
    return X_work, names_work, const_name


def _normalize_existing_ones_stata(  # noqa: PLR0913
    X: np.ndarray,
    names: list[str],
    const_name: str,
    ones_idx: list[int],
    const_idx: list[int],
    warn: bool,
) -> tuple[np.ndarray, list[str], str]:
    """Stata: Adopt first all-ones column, move to back, drop others."""
    if warn and (len(ones_idx) > 1 or const_idx):
        warnings.warn(
            "Dropping redundant constant column(s) to avoid collinearity in Stata dialect.",
            RuntimeWarning,
            stacklevel=2,
        )
    X_work = X.copy()
    names_work = list(names)
    j = int(ones_idx[0])

    drop = sorted(list(const_idx) + list(ones_idx[1:]), reverse=True)
    for t in drop:
        X_work = np.delete(X_work, t, axis=1)
        del names_work[t]
        if t < j:
            j -= 1

    last = X_work.shape[1] - 1
    if j != last:
        col = X_work[:, j : j + 1].copy()
        X_work = np.delete(X_work, j, axis=1)
        del names_work[j]
        X_work = np.column_stack([X_work, col])
        names_work = [*names_work, const_name]
    else:
        names_work[-1] = const_name
    return X_work, names_work, const_name


def _add_new_intercept_r(  # noqa: PLR0913
    X: np.ndarray,
    names: list[str],
    const_name: str,
    const_idx: list[int],
    warn: bool,
    intercept_col: np.ndarray,
) -> tuple[np.ndarray, list[str], str]:
    """R: Add new intercept at front after dropping other constants."""
    X_out = X.copy()
    names_out = list(names)

    if warn and const_idx:
        warnings.warn(
            "Dropping existing constant column(s) and adding an explicit R intercept.",
            RuntimeWarning,
            stacklevel=2,
        )

    for t in sorted(const_idx, reverse=True):
        X_out = np.delete(X_out, t, axis=1)
        del names_out[t]

    X_out = np.column_stack([intercept_col, X_out])
    names_out = [const_name, *names_out]
    return X_out, names_out, const_name


def _add_new_intercept_stata(  # noqa: PLR0913
    X: np.ndarray,
    names: list[str],
    const_name: str,
    const_idx: list[int],
    warn: bool,
    intercept_col: np.ndarray,
) -> tuple[np.ndarray, list[str], str]:
    """Stata: Add new intercept at back after dropping other constants."""
    X_out = X.copy()
    names_out = list(names)

    if warn and const_idx:
        warnings.warn(
            "Dropping existing constant column(s) and adding an explicit Stata _cons.",
            RuntimeWarning,
            stacklevel=2,
        )

    for t in sorted(const_idx, reverse=True):
        X_out = np.delete(X_out, t, axis=1)
        del names_out[t]

    X_out = np.column_stack([X_out, intercept_col])
    names_out = [*names_out, const_name]
    return X_out, names_out, const_name


def _handle_statsmodels(  # noqa: PLR0913
    X: np.ndarray,
    names: list[str],
    const_name: str,
    ones_idx: list[int],
    const_idx: list[int],
    has_constant: str | None,
    intercept_col: np.ndarray,
    position: str = "front",
) -> tuple[np.ndarray, list[str], str]:
    """Handle statsmodels dialect with has_constant semantics.

    Parameters
    ----------
    position : 'front'|'back'
        Where to place the intercept when adding. statsmodels historically
        allowed both prepend=True/False; we mirror that behavior here.

    """
    mode = "skip" if has_constant is None else str(has_constant).lower()
    if mode not in {"add", "skip", "raise"}:
        msg = "statsmodels_has_constant must be 'add'|'skip'|'raise'."
        raise ValueError(msg)

    has_const = bool(ones_idx or const_idx)
    if mode == "raise" and has_const:
        offenders = [names[j] for j in (ones_idx + const_idx)]
        raise ValueError(f"A constant column is already present: {offenders!r}.")
    if mode == "skip" and has_const:
        return X, names, const_name

    # 'add': always add (statsmodels allows duplication). Placement follows position.
    if str(position).lower() == "back":
        X_out = np.column_stack([X, intercept_col])
        names_out = [*names, const_name]
    else:
        X_out = np.column_stack([intercept_col, X])
        names_out = [const_name, *names]
    return X_out, names_out, const_name


def _assert_no_name_collision(
    names_out: list[str],
    *,
    allow_duplicate: bool = False,
    allowed_duplicate_name: str | None = None,
) -> None:
    """Disallow duplicate column names unless caller explicitly allows duplication.

    Parameters
    ----------
    names_out
        Final column names to validate.
    allow_duplicate
        If True, skip duplicate-name validation (only used for statsmodels 'add' mode
        which intentionally permits a duplicate 'const' column). For all other
        callers, duplicates raise an error to prevent silent overwrites.

    """
    if allow_duplicate:
        # permit duplication only for the specified intercept name (e.g., 'const')
        if allowed_duplicate_name is None:
            raise ValueError(
                "allowed_duplicate_name must be provided when allow_duplicate=True.",
            )
        seen = set()
        for nm in names_out:
            if nm in seen and nm != allowed_duplicate_name:
                raise ValueError(
                    f"Duplicate column name detected: '{nm}'. Only '{allowed_duplicate_name}' may duplicate in statsmodels 'add' mode.",
                )
            seen.add(nm)
        return
    if len(names_out) != len(set(names_out)):
        # find first duplicate for clarity
        seen = set()
        dup = None
        for nm in names_out:
            if nm in seen:
                dup = nm
                break
            seen.add(nm)
        raise ValueError(
            f"Duplicate column name detected: '{dup}'. Consider renaming variables.",
        )
