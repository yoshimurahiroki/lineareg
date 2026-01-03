"""Formula parser for lineareg.

Patsy-based formula parsing with lag/lead/diff, spatial lag, fixed effects,
IV/GMM clauses, and cluster variable handling.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import patsy

from lineareg.core import linalg as la
from lineareg.utils.instruments import parse_iv_formula

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable
else:  # pragma: no cover
    Iterable = tuple  # type: ignore[assignment]

_FE_PAT = re.compile(r"fe\((?P<inside>.+?)\)")
_LAG_PAT = re.compile(
    r"lag\((?P<var>[^,]+?),(?P<h>\s*-?\d+\s*)\)",
)  # will validate h>0 explicitly
_LAG_RANGE_PAT = re.compile(
    r"lag\((?P<var>[^,]+?),\s*(?P<a>-?\d+)\s*:\s*(?P<b>-?\d+)\s*\)",
)
_LEAD_PAT = re.compile(
    r"lead\((?P<var>[^,]+?),(?P<h>\s*-?\d+\s*)\)",
)  # will validate h>0 explicitly
_DIFF_PAT = re.compile(
    r"diff\((?P<var>[^,]+?)(?:,(?P<h>\s*\d+\s*))?\)",
)  # will validate h>0 explicitly
_SLAG_PAT = re.compile(
    r"slag\(\s*(?P<var>[^,]+?)\s*(?:,\s*W\s*=\s*(?:['\"]?(?P<Wkey>[^'\"),]+)['\"]?))?"
    r"(?:,\s*(?:normalize|style)\s*=\s*(?P<norm>[^,)]+))?\)",
)
_STATA_LFD_PAT = re.compile(
    r"\b(?P<op>[LFD])(?P<k>\d*)\.(?P<var>[^+\-*/\s]+?)\b", re.IGNORECASE,
)
_CLUSTER_PAT = re.compile(
    r"cluster\((?P<vars>.+?)\)", re.IGNORECASE | re.DOTALL | re.MULTILINE,
)

_R_LFD_FUNC_PAT = re.compile(
    r"\b(?P<op>[LFD])\s*\(\s*(?P<var>[^,()]+)\s*(?:,\s*(?P<h>\d+)\s*)?\)", re.IGNORECASE,
)
_GMMW_PAT = re.compile(
    r"gmm\(\s*W\s*=\s*(?:['\"]?(?P<Wkey>[A-Za-z0-9_\.\-]+)['\"]?)\s*\)",
    re.IGNORECASE,
)


def _validate_w_meta(W: Any, *, n: int | None = None) -> None:
    """Lightweight validation for spatial weight matrices used by the formula parser.

    This function performs defensive checks without densifying large sparse
    matrices: it ensures W has a 2-D shape attribute, is square, and (when
    requested) conforms to an expected size `n`.

    It raises ValueError for obvious shape mismatches. The goal is to fail
    fast for common user mistakes (wrongly ordered W, incorrect dimension).
    """
    # Accept numpy ndarray, scipy.sparse.spmatrix, or pandas.DataFrame-like objects
    shp = getattr(W, "shape", None)
    if shp is None:
        msg = "Provided W does not expose a shape attribute; supply a matrix-like object (ndarray, DataFrame, or SciPy sparse)."
        raise ValueError(msg)
    if not (isinstance(shp, (tuple, list)) and len(shp) == 2):
        msg = "W must be two-dimensional (n x n) spatial weights matrix."
        raise ValueError(msg)
    rows, cols = int(shp[0]), int(shp[1])
    if rows != cols:
        msg = f"W must be square (n x n); found shape={shp}."
        raise ValueError(msg)
    if n is not None and rows != int(n):
        msg = f"W dimension {rows} does not match the number of observations/rows ({int(n)})."
        raise ValueError(msg)


def _factorize_interaction_cols(cols: Iterable[pd.Series]) -> np.ndarray:
    # Explicitly preserve NA semantics: any factor with NA in a row yields code -1
    arrs = [c.to_numpy() for c in cols]
    mi = pd.MultiIndex.from_arrays(arrs, names=None)
    codes, _ = mi.factorize(sort=False)
    # Force rows where any input was NA to be -1 (consistent exclusion semantics)
    na_mask = np.zeros(codes.shape[0], dtype=bool)
    for a in arrs:
        na_mask |= pd.isna(a)
    codes = np.where(na_mask, -1, codes)
    return codes.astype(np.int64, copy=False)


def _require_id_time(
    df: pd.DataFrame, id_name: str | None, t_name: str | None,
) -> tuple[str, str]:
    if not id_name or not t_name:
        msg = "lag/lead/diff operations require both id_name and t_name to be set."
        raise ValueError(msg)
    if id_name not in df.columns or t_name not in df.columns:
        msg = "id_name or t_name not found in the provided DataFrame."
        raise ValueError(msg)
    return id_name, t_name


def _sorted_by_id_time(df: pd.DataFrame, id_name: str, t_name: str) -> pd.DataFrame:
    return df.sort_values([id_name, t_name], kind="mergesort")


def _slag(
    x: np.ndarray, W: Any, *, normalize: str = "row", zero_row_policy: str = "zero",
) -> np.ndarray:
    """Spatial lag helper (S = W x) that is sparse-aware and uses core.linalg
    for matrix operations.

    Parameters
    ----------
    x : np.ndarray
        Column or 1d array of length n containing the variable to be lagged.
    W : Matrix-like
        Spatial weights matrix (dense ndarray or SciPy sparse). Must be
        conformable with `x` (n x n).
    normalize : {"row", "none"}
        If "row", row-normalize W by its row sums before multiplying.
        If "none", use W as provided. (Only these options are supported.)

    Returns
    -------
    np.ndarray
        1d numpy array of length n with the spatial lag W x. If input `x` is
        length-zero, an empty array is returned. This routine avoids
        densifying W when possible by delegating the product to
        `lineareg.core.linalg.dot`.

    Notes
    -----
    - This helper deliberately does minimal preprocessing: callers are
      responsible for NA handling and aligning indices. It only guarantees
      shape and numeric dtype for downstream numeric routines.
    - Numerical stability: no inverses are taken. Row-normalization divides
      by row sums; zero row-sums are treated as ones to avoid NaN and to
      reflect a convention (units with no neighbors have lag zero).

    """
    # Ensure input vector is numeric and finite where applicable; preserve NaNs
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    n = int(x_arr.shape[0])
    if n == 0:
        return np.asarray([], dtype=np.float64)

    # Validate W shape defensively without forcing dense representation
    if not hasattr(W, "shape") or len(W.shape) != 2:
        msg = "W must be a 2-dimensional weight matrix (n x n)"
        raise ValueError(msg)
    if int(W.shape[0]) != int(W.shape[1]) or int(W.shape[0]) != n:
        msg = "W must be square and conformable with x (n x n)"
        raise ValueError(msg)

    # Accept R/Stata styles: 'W','U','B','C' as aliases; keep backward-compat for 'row'/'none'
    norm_map = {"row": "W", "none": "U", "w": "W", "u": "U", "b": "B", "c": "C"}
    norm_std = norm_map.get(
        str(normalize).strip().lower(), str(normalize).strip().upper(),
    )
    if norm_std not in {"W", "U", "B", "C"}:
        msg = "normalize/style must be one of {'row'|'none' (aliases of 'W'|'U')} or {'W','U','B','C'}."
        raise ValueError(msg)

    # Validate zero_row_policy
    if zero_row_policy not in ("zero", "warn", "error"):
        msg = "zero_row_policy must be one of {'zero','warn','error'}"
        raise ValueError(msg)

    # Validate W meta defensively (sparse-safe) before any operations
    _validate_w_meta(W, n=n)

    if norm_std == "B":
        # Binary adjacency: nonzero -> 1 (sparse-safe)
        try:
            import scipy.sparse as sps  # type: ignore
        except ImportError:
            sps = None  # type: ignore[assignment]
        if sps is not None and isinstance(W, sps.spmatrix):
            try:
                Wb = W.sign()
            except AttributeError:
                Wb = (np.asarray(W) != 0).astype(np.float64)
        else:
            Wb = (np.asarray(W) != 0).astype(np.float64)
        Wx = la.dot(Wb, x_arr)
        slag_col = Wx
    elif norm_std == "C":
        # Global standardization: sum_ij w_ij scaled so that total equals n
        n = int(x_arr.shape[0])
        try:
            import scipy.sparse as sps  # type: ignore
        except ImportError:
            sps = None  # type: ignore[assignment]
        if sps is not None and isinstance(W, sps.spmatrix):
            s0 = W.sum()
        else:
            s0 = float(np.asarray(W).sum())
        if s0 == 0:
            raise ValueError("Global standardization undefined for sum(W)=0.")
        fac = n / float(s0)
        Wx = la.dot(W, x_arr)
        slag_col = fac * Wx
    elif norm_std == "W":
        one = np.ones((n, 1), dtype=np.float64)
        rsum = np.asarray(la.dot(W, one)).reshape(-1, 1)
        zero_rows = np.isclose(rsum.reshape(-1), 0.0)
        if np.any(zero_rows):
            if zero_row_policy == "error":
                msg = "W contains rows with zero sum (units with no neighbors). Set zero_row_policy to 'zero' to proceed."
                raise ValueError(msg)
        rsum_safe = np.where(rsum == 0.0, 1.0, rsum)
        Wx = la.dot(W, x_arr)
        slag_col = Wx.reshape(-1, 1) / rsum_safe
    else:  # 'U'
        slag_col = la.dot(W, x_arr)

    return np.asarray(slag_col, dtype=np.float64).reshape(-1)


def _cleanup_rhs(rhs: str) -> str:
    """Remove empty/duplicate additive operators after dropping specials (e.g., fe(...)) so
    that Patsy receives a valid additive formula. Guarantees non-empty output (returns
    "0" if empty) to avoid syntax errors.
    """
    s = re.sub(r"\s*\+\s*", " + ", rhs)
    # collapse multiple consecutive pluses
    s = re.sub(r"(?:\s*\+\s*){2,}", " + ", s)
    s = s.strip()
    s = re.sub(r"^\+\s*", "", s)
    s = re.sub(r"\s*\+$", "", s)
    return s if s else "0"


class FormulaParser:
    """High-rigor formula parser for extended econometric syntax.

    Extensions
    ----------
    fe(a + b + a:b)
        Additive multi-dimensional fixed effects (a and b absorbed separately) with
        explicit interaction a:b only if user specifies it. This prevents accidental
        over-absorption (cell FE) and mirrors reghdfe/fixest semantics.
    lag(x,h) / lead(x,h) / diff(x,h)
        Panel-aware time-series operators. We *materialize* the transformed columns
        after sorting by (id,time) and replace their occurrences in the RHS with
        Q("lag(x,h)") etc., so Patsy treats them as ordinary variables (no runtime
        evaluation of unknown functions).
        Note: diff(x,h) is the h-th ORDER difference (Stata D^h / fixest::D),
        NOT base R's diff(x, lag=h).
    slag(x, W=K, normalize=row|none)
        Spatial lag using weight matrix referenced by key K in ``W_dict``. The created
        column is uniquely named ``slag(x)[K,norm]`` to avoid collisions across
        distinct W matrices or normalization choices. ASSUMPTION: The row ordering of
        the provided DataFrame already matches the ordering underlying W (if W was
        constructed from the same (id,time) sorted panel). When time-series specials
        are used we explicitly sort by (id,time) prior to constructing slag columns;
        users must ensure W reflects that sorted order.
    IV clause
        Parsed separately (``iv=`` argument) to recover endogenous regressors and
        instrument sets via AST without relying on fragile string heuristics.
    cluster(...), gmm(W="key")
        Option string parsing with strict validation: cluster variables must exist
        in the provided DataFrame; GMM weight key recorded as ``gmm_W_key``.

    Design Principles
    -----------------
    1. Deterministic naming for all generated columns (reproducibility).
    2. No implicit creation of interactions among FE dimensions.
    3. Fail-fast validation for missing id/time when time-series specials appear.
    4. All heavy algebra delegated to core.linalg; only lightweight SVD/factorize
       via pandas/NumPy where unavoidable.
    """

    def __init__(  # noqa: PLR0913
        self,
        data: pd.DataFrame,
        *,
        id_name: str | None = None,
        t_name: str | None = None,
        warn_if_no_id: bool = True,
        W_dict: dict[str, Any] | None = None,
        zero_row_policy: str = "zero",
    ) -> None:
        """Initialize parser with optional panel identifiers and spatial weight dict."""
        self.data = data.copy()
        self.id_name = id_name
        self.t_name = t_name
        self.warn_if_no_id = warn_if_no_id
        self.W_dict = {} if W_dict is None else W_dict
        # Validate zero_row_policy value
        if zero_row_policy not in ("zero", "warn", "error"):
            msg = "zero_row_policy must be one of {'zero','warn','error'}"
            raise ValueError(msg)
        self.zero_row_policy = zero_row_policy

        # PATCH: Pre-compute W row sums for validation (R spdep::listw metadata)
        # Enables early detection of row-normalization issues and alignment errors.
        # Store row-sum vectors for each spatial weight matrix to support strict
        # validation of normalize='row' at slag() materialization time without
        # repeated computation. Sparse-safe: uses core.linalg.dot exclusively.
        self._W_row_sums: dict[str, np.ndarray | None] = {}
        for key, W_obj in self.W_dict.items():
            if (
                hasattr(W_obj, "shape")
                and len(W_obj.shape) == 2
                and W_obj.shape[0] == W_obj.shape[1]
            ):
                try:
                    n = int(W_obj.shape[0])
                    one = np.ones((n, 1), dtype=np.float64)
                    rsum = np.asarray(la.dot(W_obj, one)).reshape(-1)
                    self._W_row_sums[key] = rsum
                except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                    # Catch any sparse/dense conversion errors gracefully
                    self._W_row_sums[key] = None
            else:
                self._W_row_sums[key] = None

    def parse(
        self,
        formula: str,
        *,
        iv: str | None = None,
        options: str | None = None,
        enforce_drop: bool = True,
    ) -> dict[str, Any]:
        """Parse formula and attached specifications into design components.

        Returns dict with keys: X, y, var_names, fe_codes_list, iv_* fields, and
        cluster_vars / gmm_W_key when provided.
        """
        if "~" not in formula:
            msg = "Formula must contain '~'."
            raise ValueError(msg)
        # Split LHS/RHS (LHS may contain time-series specials; support like R/Stata)
        y_raw, rhs_raw = [s.strip() for s in formula.split("~", 1)]
        # Strict: require unique index to avoid ambiguous remapping back to original data
        if self.data.index.has_duplicates:
            msg = "Input DataFrame index must be unique for deterministic row mapping."
            raise ValueError(msg)

        # --- Materialize LHS time-series specials first to ensure frame contains any generated columns
        df_after_lhs, y_rewritten = self._apply_time_series_ops(self.data, y_raw)
        m_y = re.search(r'Q\("([^\"]+)"\)', y_rewritten)
        y_name = m_y.group(1) if m_y else y_rewritten
        if y_name not in df_after_lhs.columns:
            if y_raw in df_after_lhs.columns:
                y_name = y_raw
            else:
                msg = f"Response variable '{y_raw}' not found (after materialization)."
                raise KeyError(msg)

        # Keep an *unmodified* RHS (minus fe(...)) for IV endogenous detection to avoid
        # losing symbol-level dependencies via Q("...") substitution.
        rhs_no_fe_original_tokens = _FE_PAT.sub("", rhs_raw)

        # Now materialize time-series/spatial operators on frame that includes LHS transforms
        df_work, rhs_rewritten = self._apply_time_series_ops(df_after_lhs, rhs_raw)

        # Build FE codes from the rewritten RHS so Q(...) tokens are visible to FE parsing
        fe_codes_list = self._build_fe_codes(df_work, rhs_rewritten)

        # Strip fe(...) and cleanup RHS for Patsy
        rhs_no_fe_dirty = _FE_PAT.sub("", rhs_rewritten)
        rhs_no_fe = _cleanup_rhs(rhs_no_fe_dirty)

        # Determine valid row mask from FE codes (codes >= 0). If no FE
        # dimensions present, mask is all True. Use explicit names to avoid
        # confusion with later Patsy NA drops.
        if fe_codes_list is None:
            row_mask_fe = np.ones(df_work.shape[0], dtype=bool)
        else:
            row_mask_fe = np.ones(df_work.shape[0], dtype=bool)
            for codes in fe_codes_list:
                row_mask_fe &= codes >= 0  # -1 denotes missing level -> exclude

        # If requested, apply FE-based mask to working DataFrame but DO NOT
        # reset the index. We must preserve original row indices so that
        # downstream callers can map back to the original data.
        if enforce_drop and (not np.all(row_mask_fe)):
            df_work = df_work.loc[row_mask_fe]
            if fe_codes_list is not None:
                fe_codes_list = [codes[row_mask_fe] for codes in fe_codes_list]

        # Build design matrix via Patsy on the (possibly masked) working frame
        X, var_names, kept_idx = self._patsy_matrix(df_work, rhs_no_fe)
        y = df_work.loc[kept_idx, y_name].to_numpy(dtype=np.float64).reshape(-1, 1)

        # Synchronize FE codes to Patsy-kept rows using positional indexer.
        if fe_codes_list is not None:
            pos = df_work.index.get_indexer(kept_idx)
            if pos.min() < 0:
                msg = "Internal error: Patsy kept indices not found in working frame index."
                raise RuntimeError(msg)
            fe_codes_list = [codes[pos] for codes in fe_codes_list]

        # Final row mapping back to original data
        row_index_labels = kept_idx
        row_pos = self.data.index.get_indexer(row_index_labels)
        if np.any(row_pos < 0):
            msg = "Internal index mapping failure: some kept rows are not found in original index."
            raise RuntimeError(msg)
        mask_final = np.zeros(self.data.shape[0], dtype=bool)
        mask_final[row_pos] = True

        # IV parsing (optional)
        iv_endog: list[str] = []
        iv_instr_user: list[str] = []
        iv_instr_full = ""
        if iv:
            iv_endog, iv_instr_user, iv_instr_full = parse_iv_formula(
                iv, rhs_no_fe_original_tokens,
            )

        opt = self._parse_options(options or "")

        # Strict: if cluster(...) was provided, materialize cluster ids aligned to kept rows
        cluster_ids_used = None
        if "cluster_vars" in opt:
            cids_ordered: list[tuple[str, np.ndarray]] = []
            for v in opt["cluster_vars"]:
                if v not in df_work.columns:
                    msg = f"Cluster variable '{v}' not found after materialization."
                    raise KeyError(msg)
                s = df_work.loc[kept_idx, v]
                if s.isna().any():
                    where = list(np.flatnonzero(s.isna().to_numpy()))
                    msg = f"Cluster variable '{v}' has NA in kept rows at positions {where[:5]}..."
                    raise ValueError(msg)
                cats = pd.unique(s)
                codes = pd.Categorical(s, categories=cats, ordered=False).codes.astype(
                    np.int64, copy=False,
                )
                cids_ordered.append((v, codes))
            # single var -> ndarray, multiple -> ordered tuple of (name, codes)
            cluster_ids_used = (
                cids_ordered[0][1] if len(cids_ordered) == 1 else tuple(cids_ordered)
            )

        # If a GMM weight key was provided in options, resolve it here to the actual
        # matrix-like object stored in self.W_dict so downstream GMM routines can
        # access the canonical weight without re-parsing string tokens.
        gmm_W_obj = None
        if "gmm_W_key" in opt:
            gkey = opt["gmm_W_key"]
            if gkey not in self.W_dict:
                msg = f"gmm weight key '{gkey}' not found in parser W_dict."
                raise KeyError(msg)
            gmm_W_obj = self.W_dict[gkey]
            _validate_w_meta(gmm_W_obj)

        # Intercept handling: we always ask Patsy to drop the intercept and then
        # later re-add a constant via utils.auto_constant (for consistent naming/placement).
        # Therefore, we must correctly detect when the user requested *no intercept*
        # (R/Patsy/Stata semantics): e.g., `0 + x`, `x + 0`, or `x - 1`.
        rhs_s = str(rhs_raw).strip()
        no_intercept = bool(
            re.search(r"(?<!\d)\s*-\s*1\b|\+\s*0\b", rhs_s)
            or re.match(r"^0\b", rhs_s)
            or re.match(r"^0\s*\+", rhs_s)
        )
        include_intercept = not no_intercept

        return {
            "X": X,
            "y": y,
            "var_names": var_names,
            "fe_codes_list": fe_codes_list,
            "iv_endog": iv_endog,
            "iv_instr_user": iv_instr_user,
            "iv_instr_full": iv_instr_full,
            "row_mask_valid": mask_final,
            "row_index_used": row_index_labels.to_numpy(),
            "cluster_ids_used": cluster_ids_used,
            "gmm_W": gmm_W_obj,
            "include_intercept": include_intercept,
            **opt,
        }

    def _patsy_matrix(
        self, df: pd.DataFrame, rhs: str,
    ) -> tuple[np.ndarray, list[str], pd.Index]:
        # Use Patsy with NA_action=drop to mirror R/Stata model.matrix behavior.
        # Intercept/constant columns must be suppressed here: constant handling
        # is centralized in utils.auto_constant.add_constant to ensure a
        # deterministic, library-wide convention (constant placed first).
        na = patsy.NAAction(on_NA="drop")
        # Force removal of the intercept from the model matrix (-1). Wrap rhs
        # to preserve grouping and operator precedence.
        # If RHS is empty ("0"), pass "-1" (remove intercept) explicitly for Patsy clarity.
        rhs_no_intercept = "-1" if rhs.strip() == "0" else f"({rhs}) - 1"
        design = patsy.dmatrix(
            rhs_no_intercept, df, NA_action=na, return_type="dataframe",
        )
        X = design.to_numpy(dtype=np.float64)
        names = list(design.design_info.column_names)
        return np.asarray(X, dtype=np.float64, order="C"), names, design.index

    def patsy_matrix(
        self,
        df: pd.DataFrame,
        rhs: str,
    ) -> tuple[np.ndarray, list[str], pd.Index]:
        """Public wrapper exposing the Patsy design matrix helper."""
        return self._patsy_matrix(df, rhs)

    def _apply_time_series_ops(
        self, df: pd.DataFrame, rhs: str,
    ) -> tuple[pd.DataFrame, str]:
        """Materialize lag/lead/diff/slag specials and rewrite RHS with Q("...")."""
        out = df.copy()
        rewritten = rhs

        # --- (0) normalize Stata L/F/D notation into R form -----------------
        def _stata_to_r(match: re.Match) -> str:
            op = match.group("op").upper()
            kst = match.group("k")
            var = match.group("var")
            k = int(kst) if (kst and kst.isdigit()) else 1
            if op == "L":
                return f"lag({var}, {k})"
            if op == "F":
                return f"lead({var}, {k})"
            return f"diff({var}, {k})"

        rhs_norm = _STATA_LFD_PAT.sub(_stata_to_r, rhs)
        # --- (-1) Stata factor/categorical notation -> Patsy-compatible ----------
        # ib#.<var> (reference level) and i.<var> (categorical), c.<var> (continuous)
        _IB_PAT = re.compile(r"\bib(?P<ref>-?\d+)\.(?P<var>[A-Za-z_][A-Za-z0-9_]*)\b")
        _I_PAT = re.compile(r"\bi\.(?P<var>[A-Za-z_][A-Za-z0-9_]*)\b")
        _C_PAT = re.compile(r"\bc\.(?P<var>[A-Za-z_][A-Za-z0-9_]*)\b")
        # interactions: '##' -> '*' and single '#' -> ':'
        temp = re.sub(r"(?<!#)##(?!#)", "*", rhs_norm)
        temp = re.sub(r"(?<!#)#(?!#)", ":", temp)
        temp = _IB_PAT.sub(
            lambda m: f"C({m.group('var')}, Treatment(reference={int(m.group('ref'))}))",
            temp,
        )
        temp = _I_PAT.sub(lambda m: f"C({m.group('var')})", temp)
        temp = _C_PAT.sub(lambda m: f"{m.group('var')}", temp)

        # Accept R/Fixest function aliases: L(x,k), F(x,k), D(x,k)
        def _rfunc_to_r(match: re.Match) -> str:
            op = match.group("op").upper()
            var = match.group("var")
            h = match.group("h")
            k = int(h) if h else 1
            return {
                "L": f"lag({var},{k})",
                "F": f"lead({var},{k})",
                "D": f"diff({var},{k})",
            }[op]

        rhs_norm = _R_LFD_FUNC_PAT.sub(_rfunc_to_r, temp)
        rewritten = rhs_norm

        uses_ts = any(
            p.search(rewritten)
            for p in (_LAG_PAT, _LEAD_PAT, _DIFF_PAT, _LAG_RANGE_PAT)
        )
        uses_sp = bool(_SLAG_PAT.search(rewritten))
        # Fail-fast: time-series operators require both id and time to be set.
        if uses_ts and (self.id_name is None or self.t_name is None):
            msg = "lag/lead/diff require both id_name and t_name; global shift fallback is not permitted."
            raise ValueError(msg)
        # Fail-fast: spatial lag requires panel identifiers for deterministic row alignment
        if uses_sp and (self.id_name is None or self.t_name is None):
            msg = (
                "slag() requires both id_name and t_name for deterministic row alignment to W. "
                "Stata spmatrix and R spdep require explicit row ordering; "
                "provide id_name and t_name to ensure W matches your data ordering."
            )
            raise ValueError(msg)
        # If either time-series operators or spatial lag operators appear and
        # panel identifiers are provided, we always canonical-sort by (id,time).
        # This unifies the earlier inconsistent messaging about W alignment and
        # guarantees that spatial W may be validated against a deterministic
        # ordering.
        if (
            (uses_ts or uses_sp)
            and (self.id_name is not None)
            and (self.t_name is not None)
        ):
            idn, tn = _require_id_time(out, self.id_name, self.t_name)
            out = _sorted_by_id_time(out, idn, tn)
            # strict: no duplicate (id,time) allowed
            dup = out.duplicated(subset=[idn, tn])
            if bool(dup.any()):
                msg = "Duplicate (id,time) rows detected; time operators require unique observation per (id,time)."
                raise ValueError(msg)

        def _time_indexed_lag(x: pd.Series, k: int) -> pd.Series:
            if self.id_name is None or self.t_name is None:
                msg = "Panel lag/lead/diff require both id_name and t_name."
                raise ValueError(msg)
            idn = self.id_name
            tn = self.t_name
            var_name = x.name if x.name else "__var__"
            df_temp = out[[idn, tn]].copy()
            df_temp[var_name] = x.to_numpy()
            t_ser = df_temp[tn]
            # Strict policy: datetime-like time variables make a k-step lag ambiguous
            # without an explicit frequency; do not guess (e.g. by subtracting days).
            sample_non_na = None
            if t_ser.size:
                try:
                    sample_non_na = t_ser.dropna().iloc[0]
                except (IndexError, KeyError):
                    sample_non_na = None
            if (
                pd.api.types.is_datetime64_any_dtype(t_ser)
                or pd.api.types.is_timedelta64_dtype(t_ser)
                or isinstance(sample_non_na, (pd.Timestamp, np.datetime64))
            ):
                raise ValueError(
                    "Datetime-like time variables are not supported for lag/lead/diff. "
                    "Provide an integer/numeric period index (e.g., year=2010,2011,...) or a pandas PeriodIndex converted to integers."
                )

            if pd.api.types.is_period_dtype(t_ser):
                t_shifted = t_ser - k if k > 0 else t_ser + abs(k)
            else:
                # Use numeric time for both the shifted target and lookup key to
                # ensure exact t-k matching (gaps yield missing as in Stata ts operators).
                try:
                    t_num = pd.to_numeric(t_ser, errors="raise")
                except Exception as exc:
                    raise ValueError(
                        "Time variable must be numeric/integer (or Period) for lag/lead/diff.",
                    ) from exc
                df_temp[tn] = t_num
                t_shifted = t_num.to_numpy() - k if k > 0 else t_num.to_numpy() + abs(k)
            df_temp["__t_target__"] = t_shifted
            df_lookup = df_temp[[idn, tn, var_name]].rename(columns={tn: "__t_target__", var_name: "__lagged__"})
            merged = df_temp.merge(df_lookup, on=[idn, "__t_target__"], how="left")
            return pd.Series(merged["__lagged__"].to_numpy(), index=x.index)

        # Handle lag/lead/diff by materializing deterministic column names
        if uses_ts:
            # expand lag ranges (lag(var, a:b)) into individual lag columns first
            for m in list(_LAG_RANGE_PAT.finditer(rewritten)):
                var = m.group("var").strip()
                a = int(m.group("a"))
                b = int(m.group("b"))
                if a <= 0 or b <= 0 or a > b:
                    raise ValueError("lag(x, a:b) requires integers 1<=a<=b.")
                if var not in out.columns:
                    raise ValueError(f"lag(): unknown variable '{var}' in formula.")
                repl = []
                for h in range(a, b + 1):
                    new = f"__L{h}__:{var}" if h != 1 else f"__L__:{var}"
                    if new not in out.columns:
                        out[new] = _time_indexed_lag(out[var], h)
                    repl.append(f'Q("{new}")')
                rewritten = re.sub(
                    re.escape(m.group(0)), " + ".join(repl), rewritten, count=1,
                )
            # lag (single k)
            for m in list(_LAG_PAT.finditer(rewritten)):
                var = m.group("var").strip()
                h = int(m.group("h"))
                if h <= 0:
                    msg = "lag(x,h) requires h>=1 (positive integer)."
                    raise ValueError(msg)
                new = f"__L{h}__:{var}" if h != 1 else f"__L__:{var}"
                if var not in out.columns:
                    msg = f"lag(): unknown variable '{var}' in formula."
                    raise ValueError(msg)
                if new not in out.columns:
                    out[new] = _time_indexed_lag(out[var], h)
                rewritten = re.sub(
                    re.escape(m.group(0)), f'Q("{new}")', rewritten, count=1,
                )
            # lead
            for m in list(_LEAD_PAT.finditer(rewritten)):
                var = m.group("var").strip()
                h = int(m.group("h"))
                if h <= 0:
                    msg = "lead(x,h) requires h>=1 (positive integer)."
                    raise ValueError(msg)
                new = f"__F{h}__:{var}" if h != 1 else f"__F__:{var}"
                if var not in out.columns:
                    msg = f"lead(): unknown variable '{var}' in formula."
                    raise ValueError(msg)
                if new not in out.columns:
                    out[new] = _time_indexed_lag(out[var], -h)
                rewritten = re.sub(
                    re.escape(m.group(0)), f'Q("{new}")', rewritten, count=1,
                )
            # diff (iterated differences) — naming consistency: always D{k}:{var}
            for m in list(_DIFF_PAT.finditer(rewritten)):
                var = m.group("var").strip()
                k = int(m.group("h") or "1")
                if k <= 0:
                    msg = "diff(x,h) requires h>=1 (positive integer)."
                    raise ValueError(msg)
                if var not in out.columns:
                    msg = f"diff(): unknown variable '{var}' in formula."
                    raise ValueError(msg)
                base = out[var]
                for _ in range(k):
                    base = base - _time_indexed_lag(base, 1)
                # Consistent naming for all orders, simplifies downstream checks.
                new = f"__D{k}__:{var}"
                if new not in out.columns:
                    out[new] = base
                rewritten = re.sub(
                    re.escape(m.group(0)), f'Q("{new}")', rewritten, count=1,
                )
        # spatial lags (operate on the rewritten RHS so any prior normalization/replacements
        # are respected; order independent of panel sort; relies on aligned rows post-sort)
        for m in list(_SLAG_PAT.finditer(rewritten)):
            var = m.group("var").strip()
            Wkey: str = str(m.group("Wkey") or "").strip()
            norm: str = str(m.group("norm") or "row").strip().lower()
            # Early validation of normalize parameter for better error messages
            if norm not in ("row", "none"):
                msg = f"slag(..., normalize=...) must be 'row' or 'none'; got '{norm}'."
                raise ValueError(msg)
            if not Wkey:
                msg = 'slag(var, W="W_key") requires W specification.'
                raise ValueError(msg)
            if Wkey not in self.W_dict:
                msg = f"Spatial weight key '{Wkey}' not found in W_dict."
                raise KeyError(msg)

            # Prepare vector and W object
            if var not in out.columns:
                raise KeyError(
                    f"slag(): unknown variable '{var}' in formula after materialization.",
                )
            vec = out[var].to_numpy(dtype=np.float64).reshape(-1)
            W_obj = self.W_dict[Wkey]

            # Require explicit, strict alignment when panel identifiers are available.
            # If id/time are provided for the parser, the user MUST supply W as a
            # pandas.DataFrame with index/columns matching the (id,time) sorted rows.
            id_name_opt: str | None = self.id_name
            t_name_opt: str | None = self.t_name
            if id_name_opt is not None and t_name_opt is not None:
                # `out` has been canonical-sorted by (id,time) above when either
                # time or spatial operators exist. Enforce strict DataFrame matching
                # so that W rows/cols align exactly with the (id,time)-sorted order.
                if not isinstance(W_obj, pd.DataFrame):
                    msg = (
                        "With id_name and t_name set, spatial W must be a pandas.DataFrame "
                        "whose index/columns exactly match the (id,time)-sorted rows."
                    )
                    raise TypeError(
                        msg,
                    )
                # Support exact MultiIndex matching: allow DataFrame whose index/columns
                # are a MultiIndex of (id,time) or simple single-level index of the
                # concatenated "id|time" strings. This mirrors typical R spdep usage
                # when panels are flattened into a single spatial order.
                if isinstance(W_obj.index, pd.MultiIndex) and isinstance(
                    W_obj.columns, pd.MultiIndex,
                ):
                    # Build the target MultiIndex preserving dtypes.
                    target_mi = pd.MultiIndex.from_frame(
                        out[[id_name_opt, t_name_opt]],
                        names=W_obj.index.names if W_obj.index.names is not None else None,
                    )
                    if not (W_obj.index.equals(target_mi) and W_obj.columns.equals(target_mi)):
                        msg = f"Spatial weight DataFrame MultiIndex index/columns do not match parser row ordering for key '{Wkey}'."
                        raise ValueError(msg)
                else:
                    # Single-level string keys ("id|time") are allowed only if the
                    # W DataFrame already uses strings (no implicit coercion) and
                    # the constructed keys are collision-free.
                    row_keys = (
                        out[id_name_opt].astype(str) + "|" + out[t_name_opt].astype(str)
                    ).to_list()
                    if len(set(row_keys)) != len(row_keys):
                        raise ValueError(
                            "Ambiguous (id,time) -> string key mapping for spatial W alignment; "
                            "use a MultiIndex (id,time) for W to avoid collisions.",
                        )
                    if any("|" in k for k in row_keys):
                        raise ValueError(
                            "The character '|' appears in id/time string representations; "
                            "cannot use 'id|time' alignment keys. Use MultiIndex for W.",
                        )
                    w_index = list(W_obj.index)
                    w_cols = list(W_obj.columns)
                    if not all(isinstance(v, str) for v in w_index + w_cols):
                        raise TypeError(
                            "Spatial weight DataFrame must use MultiIndex (id,time) or single-level string keys 'id|time'.",
                        )
                    if w_index != row_keys or w_cols != row_keys:
                        msg = f"Spatial weight DataFrame index/columns do not match parser row ordering for key '{Wkey}'."
                        raise ValueError(msg)
                # Keep W as DataFrame object (do not densify). Downstream helpers handle sparse/DF/ndarray.
                W_arr = W_obj
            else:
                # No explicit panel mapping available: accept numpy/sparse or DataFrame, but
                # enforce squareness and matching row/column counts to avoid silent misalignment.
                if getattr(W_obj, "shape", None) is None or len(W_obj.shape) != 2:
                    msg = "Spatial weight must be a 2-dimensional array-like or DataFrame."
                    raise TypeError(msg)
                if W_obj.shape[0] != W_obj.shape[1]:
                    msg = f"Spatial weight '{Wkey}' must be square (n x n). Found shape={W_obj.shape}."
                    raise ValueError(msg)
                # Explicitly check both row and column dimensions against the vector
                if W_obj.shape[0] != vec.shape[0] or W_obj.shape[1] != vec.shape[0]:
                    msg = f"Spatial weight matrix '{Wkey}' shape mismatch: expected ({vec.shape[0]},{vec.shape[0]}), found {W_obj.shape}."
                    raise ValueError(msg)
                # Preserve original object type (DataFrame/ndarray/sparse) to avoid densification
                W_arr = W_obj

            # --- Strict spatial checks (always) ---
            # zero diagonal (sparse-safe O(nnz) for SciPy sparse matrices)
            d = None
            try:
                import scipy.sparse as sps  # type: ignore
            except ImportError:
                sps = None  # type: ignore[assignment]
            if sps is not None and isinstance(W_arr, sps.spmatrix):
                coo = W_arr.tocoo()
                mask_diag = coo.row == coo.col
                if np.any(mask_diag) and np.any(coo.data[mask_diag] != 0):
                    raise ValueError(
                        f"Spatial W '{Wkey}' must have zero diagonal (no self-loops).",
                    )
            else:
                try:
                    d = np.asarray(
                        np.diag(np.asarray(W_arr)), dtype=np.float64,
                    ).reshape(-1)
                except (TypeError, ValueError):
                    d = None
            if d is not None and np.any(np.abs(d) > 0.0):
                raise ValueError(
                    f"Spatial W '{Wkey}' must have zero diagonal (no self-loops).",
                )
            # elementwise nonnegativity check (sparse-safe fallback)
            try:
                import scipy.sparse as sps  # type: ignore
            except ImportError:
                sps = None  # type: ignore[assignment]
            if sps is not None and isinstance(W_arr, sps.spmatrix):
                if W_arr.data.size and np.min(W_arr.data) < 0:
                    raise ValueError(
                        f"Spatial W '{Wkey}' must be elementwise nonnegative.",
                    )
            elif np.any(np.asarray(W_arr) < 0):
                raise ValueError(
                    f"Spatial W '{Wkey}' must be elementwise nonnegative.",
                )

            cname = f"slag({var})[{Wkey},{norm}]"
            # Reuse already generated spatial lag if present
            already = cname in out.columns

            # Normalization checks (strict, numerically robust): when normalize='row',
            # require row sums ≈ 1 within absolute+relative tolerance
            # Tolerances for row-sum ≈ 1 checks (spdep/spatialreg practical range)
            atol = 1e-7
            rtol = 1e-10
            if norm == "row":
                # Prefer cached row-sums if available and size-matching to avoid
                # repeated matrix-vector multiplications (cache computed in __init__).
                rsum_cached = self._W_row_sums.get(Wkey, None)
                if rsum_cached is not None and rsum_cached.shape[0] == vec.shape[0]:
                    rsum = np.asarray(rsum_cached, dtype=np.float64).reshape(-1)
                else:
                    # Fallback: compute row sums once via core.linalg (sparse-safe)
                    rsum = la.dot(
                        W_arr, np.ones((vec.shape[0], 1), dtype=np.float64),
                    ).reshape(-1)
                # Identify zero rows (allowed) and validate non-zero rows are close to 1
                zero_rows = np.isclose(rsum, 0.0, atol=atol)
                if not np.all(zero_rows):
                    deviations = np.abs(rsum[~zero_rows] - 1.0)
                    max_dev = float(np.max(deviations)) if deviations.size else 0.0
                    if max_dev > (atol + rtol):
                        if max_dev > 1e-3:
                            msg = (
                                f"Row sums of W '{Wkey}' deviate substantially from 1 (max |sum-1|={max_dev:.3e}, >{1e-3:.1e}). "
                                f"Check construction and ordering of W. If row-normalization is intended, apply it before passing W."
                            )
                            raise ValueError(msg)
                # Row-normalization diagnostics only
                # (zero-diagonal/nonnegativity already enforced above in a sparse-safe way)

            # Compute spatial lag using core helper; pass vector as 1-d array
            # Propagate parser-level zero_row_policy to the helper to ensure
            # consistent behavior across all spatial-lag materializations.
            if not already:
                out[cname] = _slag(
                    vec, W_arr, normalize=norm, zero_row_policy=self.zero_row_policy,
                )
            rewritten = re.sub(
                re.escape(m.group(0)), f'Q("{cname}")', rewritten, count=1,
            )
        return out, rewritten

    def _build_fe_codes(self, df: pd.DataFrame, rhs: str) -> list[np.ndarray] | None:
        """Extract additive FE code arrays from zero or more fe(...) groups.

        Each fe(...) group may contain additive factors separated by '+'. Explicit
        interactions a:b must be declared and are treated as additional FE *only for
        that interaction* (not expanded into main effects). Duplicate factor tokens
        across multiple fe(...) occurrences are ignored to maintain idempotence.
        """
        matches = list(_FE_PAT.finditer(rhs))
        if not matches:
            return None
        fe_list: list[np.ndarray] = []
        seen: set[str] = set()
        for m in matches:
            inside = m.group("inside")
            terms = [t.strip() for t in re.split(r"\s*\+\s*", inside) if t.strip()]
            for t in terms:
                if t in seen:
                    continue
                if ":" in t:
                    parts = [s.strip() for s in t.split(":") if s.strip()]
                    if len(parts) < 2:
                        msg = f"Interaction spec '{t}' must be at least a:b."
                        raise ValueError(msg)
                    # Validate each component exists in the DataFrame and raise a clear error
                    for p in parts:
                        if p not in df.columns:
                            raise KeyError(
                                f"fe() interaction term '{p}' not found in data columns.",
                            )
                    # Support N-way cell FEs (a:b:c:...). Factorize via MultiIndex
                    # to obtain unique cell identifiers. Preserve NA semantics.
                    cols = [df[p].astype("category") for p in parts]
                    codes = _factorize_interaction_cols(cols)
                    fe_list.append(codes.astype(np.int64, copy=False))
                else:
                    if t not in df.columns:
                        raise KeyError(f"fe() term '{t}' not found in data columns.")
                    cat = df[t].astype("category")
                    codes = cat.cat.codes.to_numpy(dtype=np.int64, copy=False)
                    # Policy: rows with NA factor levels are DROPPED (consistent with Stata/R default)
                    # Instead of introducing a synthetic level, we mark their indices for exclusion.
                    # The calling estimator can drop rows where any FE code < 0.
                    fe_list.append(codes)
                seen.add(t)
        return fe_list if fe_list else None

    def _parse_options(self, options: str) -> dict[str, Any]:
        opt: dict[str, Any] = {}
        if not options:
            return opt
        ms = list(_CLUSTER_PAT.finditer(options))
        if len(ms) > 1:
            raise ValueError("cluster(...) may appear at most once.")
        if ms:
            raw = ms[0].group("vars")
            vars_ = [v.strip() for v in re.split(r"[+,\s]+", raw) if v.strip()]
            missing = [v for v in vars_ if v not in self.data.columns]
            if missing:
                msg = f"Cluster variable(s) not in data: {missing}"
                raise KeyError(msg)
            opt["cluster_vars"] = vars_
        mats = list(_GMMW_PAT.finditer(options))
        if len(mats) > 1:
            msg = "gmm(W=...) may appear at most once in options."
            raise ValueError(msg)
        if mats:
            opt["gmm_W_key"] = mats[0].group("Wkey")
        # Capture raw constraints(...) specification if present. We keep the
        # exact body string and defer matrix construction until variable names
        # are finalized (e.g., after add_constant in estimators).
        _CONSTRAINTS_PAT = re.compile(
            r"constraints\((?P<body>.*?)\)", re.IGNORECASE | re.DOTALL,
        )
        m = _CONSTRAINTS_PAT.search(options)
        if m:
            body = m.group("body").strip()
            if (len(body) >= 2) and ((body[0] == body[-1]) and body[0] in "\"'"):
                body = body[1:-1]
            opt["constraints_raw"] = body
        return opt
