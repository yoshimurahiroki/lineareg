"""Linear equality constraints parsing and solving.

Builds R b = q from user specifications and solves via QR/SVD.
Supports 2SLS and batch solving for constrained estimation.
"""

from __future__ import annotations

import logging
import re
import warnings

import numpy as np

from lineareg.core import linalg as la

LOGGER = logging.getLogger(__name__)

__all__ = ["solve_constrained", "solve_constrained_batch", "stack_constraints"]
__all__ += ["build_rq_from_string", "solve_constrained_2sls"]


def stack_constraints(
    R_list: tuple[np.ndarray, ...],
    q_list: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack multiple equality constraints:
    R_i b = q_i  -->  [R_1; ...; R_m] b = [q_1; ...; q_m]
    """
    if len(R_list) != len(q_list):
        msg = "R_list and q_list must have the same length."
        raise ValueError(msg)
    R_blocks: list[np.ndarray] = []
    q_blocks: list[np.ndarray] = []
    k_ref: int | None = None
    q_cols_ref: int | None = None
    for R, q in zip(R_list, q_list):
        Ri = np.asarray(R, dtype=np.float64, order="C")
        qi = np.asarray(q, dtype=np.float64, order="C")
        # Check column consistency across all R blocks
        if k_ref is None:
            k_ref = int(Ri.shape[1])
        elif int(Ri.shape[1]) != k_ref:
            msg = f"All R blocks must have the same number of columns: got {Ri.shape[1]} vs {k_ref}."
            raise ValueError(msg)
        # Strictly require q to have matching number of rows as Ri
        if qi.ndim == 1:
            qi = qi.reshape(-1, 1)
        if qi.shape[0] != Ri.shape[0]:
            msg = f"q rows ({qi.shape[0]}) must match R rows ({Ri.shape[0]})."
            raise ValueError(msg)
        # Enforce identical number of columns across all q blocks
        if q_cols_ref is None:
            q_cols_ref = int(qi.shape[1])
        elif int(qi.shape[1]) != q_cols_ref:
            msg = f"All q blocks must have the same number of columns: got {qi.shape[1]} vs {q_cols_ref}."
            raise ValueError(msg)
        R_blocks.append(Ri)
        q_blocks.append(qi)
    # Use NumPy stacks to avoid relying on non-standard la.vstack in all environments
    return np.vstack(R_blocks), np.vstack(q_blocks)


# ---------- NEW: parse linear equalities into (R, q) ----------
_NUM = r"(?:[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"


def _split_constraints_items(body: str) -> list[str]:
    """Split a constraints body by ';' or ',' but not inside brackets or quotes.
    Returns a list of constraint strings.
    """
    items, buf, depth, in_s, in_d = [], [], 0, False, False
    for ch in body:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif not (in_s or in_d):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth = max(0, depth - 1)
            elif ch in ",;" and depth == 0:
                s = "".join(buf).strip()
                # Strict: do not silently accept empty/trailing separators
                if not s:
                    raise ValueError(
                        "Empty/trailing constraint separator detected; check for extra ',' or ';'.",
                    )
                items.append(s)
                buf = []
                continue
        buf.append(ch)
    s = "".join(buf).strip()
    if s:
        items.append(s)
    return items


def _parse_side_to_coeffs(
    side: str, var_names: list[str], const_aliases: tuple[str, ...],
) -> tuple[dict[str, float], float]:
    """Parse a linear expression (LHS or RHS) into (coeff_dict, const_scalar).

    Supported tokens:
      - _b[variable name]  (Stata-style exact name)
      - bare identifiers matching var_names
      - numeric constants
      - optional numeric multiplier like '2 * _b[x]'
    Returns a dictionary mapping variable -> coefficient and a numeric constant.
    """
    s = side.strip()
    s = s.replace("\u2212", "-")
    coeffs: dict[str, float] = {}
    const_val: float = 0.0
    if s and s[0] not in "+-":
        s = "+" + s
    var_pat = re.compile(
        rf"(?P<sign>[+-])\s*(?:(?P<c>{_NUM})\s*\*?\s*)?(?:_b\[(?P<bv>.+?)\]|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|(?P<num>{_NUM}))",
    )
    pos = 0
    for m in var_pat.finditer(s):
        if m.start() != pos:
            frag = s[pos : m.start()].strip()
            if frag:
                raise ValueError(f"Unrecognized token in constraint: '{frag}'")
        pos = m.end()
        sign = -1.0 if m.group("sign") == "-" else +1.0
        c = float(m.group("c")) if m.group("c") is not None else 1.0
        if m.group("bv") is not None:
            name = m.group("bv")
            # Prefer an exact '_cons' column if present; otherwise fall back
            # to the first configured const_alias that appears in var_names.
            if name.strip() == "_cons":
                if "_cons" in var_names:
                    name = "_cons"
                else:
                    for al in const_aliases:
                        if al in var_names:
                            name = al
                            break
            if name not in var_names:
                raise KeyError(f"_b[{name}] not found among regressors.")
            coeffs[name] = coeffs.get(name, 0.0) + sign * c
        elif m.group("id") is not None:
            ident = m.group("id")
            target = None
            if ident in var_names or (ident in const_aliases and ident in var_names):
                target = ident
            elif ident == "_cons":
                # Prefer literal '_cons' if it exists; otherwise use first matching alias
                if "_cons" in var_names:
                    target = "_cons"
                else:
                    for al in const_aliases:
                        if al in var_names:
                            target = al
                            break
            if target is None:
                raise KeyError(
                    f"Variable '{ident}' not found; use _b[exact name] for special columns.",
                )
            coeffs[target] = coeffs.get(target, 0.0) + sign * c
        elif m.group("num") is not None:
            val = float(m.group("num"))
            const_val += sign * val
    if pos != len(s):
        tail = s[pos:].strip()
        if tail:
            raise ValueError(f"Unparsed tail in constraint: '{tail}'")
    return coeffs, const_val


def build_rq_from_string(
    spec_raw: str,
    var_names: list[str],
    *,
    const_aliases: tuple[str, ...] = ("const", "Intercept", "_cons"),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert semicolon/comma-separated equalities into (R, q).

    Example: "_b[x1] = 0; 2*_b[x2] + _b[x3] = 1; x4 = x5"
    Returns (R, q, labels) where R is (m x k) and q is (m x 1).
    """
    items = _split_constraints_items(spec_raw)
    if not items:
        raise ValueError("constraints(...) is empty.")
    k = len(var_names)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    labels: list[str] = []
    for raw in items:
        eq = raw.strip()
        if not eq:
            continue
        # Accept '==' as an alternate equality token (R-style) but normalize
        eq = eq.replace("==", "=")
        parts = eq.split("=")
        if len(parts) != 2:
            raise ValueError(f"Constraint must contain one '=' : '{eq}'")
        left, right = parts[0].strip(), parts[1].strip()
        cl, al = _parse_side_to_coeffs(left, var_names, const_aliases)
        cr, ar = _parse_side_to_coeffs(right, var_names, const_aliases)
        row = np.zeros((k,), dtype=np.float64)
        for name, val in cl.items():
            row[var_names.index(name)] += val
        for name, val in cr.items():
            row[var_names.index(name)] -= val
        rconst = float(ar - al)
        if not np.any(np.abs(row) > 0):
            if abs(rconst) > 0:
                raise ValueError(
                    f"Infeasible constraint '0 = {rconst}' generated from '{eq}'.",
                )
            continue
        rows.append(row.reshape(1, -1))
        rhs.append(rconst)
        labels.append(eq)
    if not rows:
        R = np.zeros((0, k), dtype=np.float64)
        q = np.zeros((0, 1), dtype=np.float64)
        return R, q, labels
    R = np.vstack(rows)
    q = np.asarray(rhs, dtype=np.float64).reshape(-1, 1)
    R2, q2, keep_rows = _prune_redundant_rows_with_index(R, q, rtol=1e-10)
    labels2 = [labels[int(i)] for i in keep_rows]
    return R2, q2, labels2


def solve_constrained_2sls(  # noqa: PLR0913
    Qz: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    q: np.ndarray,
    rank_policy: str | None = None,
) -> np.ndarray:
    """Solve constrained 2SLS via KKT on the projected system:
        min_b ||Qz' y - Qz' X b||^2  s.t. R b = q
    Returns b_hat. No standard errors (policy).

    Notes
    -----
    - Qz is expected to be an (n x r) orthonormal basis for the column space of Z.
    - X and y may be dense or sparse; use core.linalg for matrix ops where possible.

    """
    Qz = np.asarray(Qz, dtype=np.float64, order="C")
    X = np.asarray(X, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64, order="C")
    R = np.asarray(R, dtype=np.float64, order="C")
    q = np.asarray(q, dtype=np.float64, order="C").reshape(-1, 1)
    # Finite check: reject NaN/Inf early to match R/Stata strictness
    for name, arr in (
        ("Qz", Qz),
        ("X", X),
        ("y", y.reshape(-1, 1)),
        ("R", R),
        ("q", q),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values (NaN/Inf).")
    # Dimensionality & orthonormality checks (strict)
    n = int(X.shape[0])
    if Qz.shape[0] != n or y.reshape(-1, 1).shape[0] != n:
        raise ValueError("Qz, X, and y must have the same number of rows.")
    # Qz'Qz ≈ I
    I_r = la.eye(Qz.shape[1])
    ortho_res = la.to_dense(la.dot(Qz.T, Qz) - I_r)
    # tolerance depends on rank_policy for parity with R/Stata
    ortho_tol = 1e-8
    if rank_policy is not None:
        rp = str(rank_policy).lower()
        if rp not in {"r", "stata"}:
            raise ValueError("rank_policy must be one of {'r','stata'} or None.")
        ortho_tol = 1e-6 if rp == "r" else 1e-8
    if float(np.max(np.abs(ortho_res))) > ortho_tol:
        raise ValueError("Qz must have (approximately) orthonormal columns: Qz'Qz ≈ I.")

    # Project onto Qz: Xp = Qz' X, yp = Qz' y
    Xp = la.dot(Qz.T, X)
    yp = la.dot(Qz.T, y.reshape(-1, 1))

    # Form normal-equation blocks on the projected system (small r x r matrices)
    # Use la.tdot / la.crossprod when available; fallback to explicit dense products
    try:
        XtX = la.tdot(Xp)
        Xty = la.crossprod(Xp, yp)
    except (AttributeError, TypeError, np.linalg.LinAlgError) as exc:
        # Dense fallback (theoretically equivalent; clearer numeric path)
        LOGGER.debug("Falling back to dense projected normal equations: %s", exc)
        XtX = la.dot(Xp.T, Xp)
        Xty = la.dot(Xp.T, yp)

    # KKT system (remove historical factor-of-two):
    # [XtX  R'; R 0] [b; lambda] = [Xty; q]
    k = Xp.shape[1]
    # Ensure R has correct number of columns to match projected X
    if R.shape[1] != k:
        raise ValueError(f"R must have {k} columns to match projected X (Xp).")
    zeros_m = np.zeros((R.shape[0], R.shape[0]), dtype=np.float64)
    A11 = XtX
    A12 = R.T
    A21 = R
    A_kkt = np.block([[A11, A12], [A21, zeros_m]])
    rhs = np.vstack([Xty, q])
    # choose verification tolerance by rank_policy (R/Stata parity)
    vr = 1e-10
    if rank_policy is not None:
        rp = str(rank_policy).lower()
        if rp not in {"r", "stata"}:
            raise ValueError("rank_policy must be one of {'r','stata'} or None.")
        vr = 1e-7 if rp == "r" else 1e-10
    try:
        sol = la.solve(A_kkt, rhs, sym_pos=False)
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as primary_error:
        LOGGER.debug("KKT solve failed; switching to SPD fallback: %s", primary_error)
        # ---- SPD route with pruning (strict parity with solve_constrained) ----
        # Optional pruning of redundant constraints
        R_loc, q_loc = _prune_redundant_rows(
            R,
            q,
            rtol=(1e-7 if (rank_policy and str(rank_policy).lower() == "r") else 1e-10),
        )
        m_loc = R_loc.shape[0]
        # Symmetrize and stabilize XtX
        XtX_sym = 0.5 * (XtX + XtX.T)
        try:
            la.solve(XtX_sym, la.eye(XtX_sym.shape[0]), sym_pos=True)
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
            LOGGER.debug("Stabilizing XtX_sym due to solver issue: %s", exc)
            XtX_sym = XtX_sym + 1e-12 * la.eye(XtX_sym.shape[0])
        # Block elimination on stabilized XtX
        RHS = la.column_stack([R_loc.T, Xty])
        S = la.solve(XtX_sym, RHS, sym_pos=True)
        V = S[:, :m_loc]
        Y = S[:, m_loc:]
        M = la.dot(R_loc, V)
        rhs2 = la.dot(R_loc, Y) - q_loc
        lam = la.solve(M, rhs2, sym_pos=True)
        b = Y - la.dot(V, lam)
        # Strict verification
        rb = la.dot(R_loc, b)
        tol_loc = vr * (1.0 + np.max(np.abs(la.to_dense(q_loc))))
        if np.max(np.abs(la.to_dense(rb - q_loc))) > tol_loc:
            # ---- Final KKT fallback (saddle point) ----
            zeros_m = np.zeros((m_loc, m_loc), dtype=np.float64)
            A11 = XtX_sym
            A12 = R_loc.T
            A21 = R_loc
            A_kkt2 = np.block([[A11, A12], [A21, zeros_m]])
            B = np.vstack([Xty, q_loc])
            # Try to use a symmetric-indefinite LDL^T factorization when available
            # for better numerical parity with dedicated solvers. Prefer SciPy's
            # ldl if installed; otherwise fall back to a direct dense solve.
            try:
                # avoid a hard dependency on SciPy; import only if present
                from scipy.linalg import ldl as scipy_ldl  # type: ignore

                # scipy.linalg.ldl returns (L, D, perm) with A[perm][:, perm] = L D L^T
                L, D, perm = scipy_ldl(A_kkt2, lower=False)
                # apply permutation to RHS
                Bp = B[perm, :]
                # solve L D L^T x = Bp by forward/back substitutions
                y = np.linalg.solve(L, Bp)
                z = np.linalg.solve(D, y)
                x_perm = np.linalg.solve(L.T, z)
                x = np.empty_like(x_perm)
                x[perm, :] = x_perm
                sol2 = x
            except (ImportError, np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
                LOGGER.debug("Falling back to dense KKT solve: %s", exc)
                try:
                    sol2 = la.solve(A_kkt2, B, sym_pos=False)
                except (np.linalg.LinAlgError, RuntimeError, ValueError) as dense_error:
                    raise RuntimeError(
                        "solve_constrained_2sls: unable to solve final KKT system with LDL or dense solver",
                    ) from dense_error
            b = sol2[:k, :]
            lam = sol2[k:, :]
            rb2 = la.dot(R_loc, b)
            tol_final = vr * (1.0 + np.max(np.abs(la.to_dense(q_loc))))
            if np.max(np.abs(la.to_dense(rb2 - q_loc))) > tol_final:
                # Constraint still not satisfied exactly. Instead of silently
                # regularizing, follow rank_policy for degenerate imputation:
                #   'r' -> raise with NaN-filled coefficients (explicit)
                #   'stata' -> impute zeros for exactly-degenerate coefficients (Stata-like)
                rp_impute = None
                if rank_policy is not None:
                    rp_impute = str(rank_policy).lower()
                if rp_impute == "r":
                    # produce NaN-filled coefficients to indicate degeneracy
                    raise RuntimeError(
                        "solve_constrained_2sls: constraint not satisfied after fallback KKT; "
                        "degenerate-case encountered (rank_policy='r'): coefficients set to NaN.",
                    ) from primary_error
                if rp_impute == "stata":
                    # Stata historically fills exact degeneracies with zeros for some routines
                    b = np.zeros_like(b)
                else:
                    # Unknown policy: fail loudly rather than guessing
                    raise RuntimeError(
                        "solve_constrained_2sls: constraint not satisfied after fallback KKT.",
                    ) from primary_error
        # Stationarity on projected system also for SPD/fallback
        res_p = la.to_dense(yp - la.dot(Xp, b))
        stat = (-1.0) * la.crossprod(Xp, res_p) + la.dot(R_loc.T, lam)
        s_norm = float(np.max(np.abs(la.to_dense(stat))))
        s_tol = vr * (1.0 + np.max(np.abs(la.to_dense(la.crossprod(Xp, res_p)))))
        if s_norm > s_tol:
            raise RuntimeError(
                "solve_constrained_2sls: stationarity violated after SPD/fallback.",
            ) from None
        return b.reshape(-1)
    else:
        b = sol[:k, :]
        lam = sol[k:, :]
        rb = la.dot(R, b)
        tol = vr * (1.0 + np.max(np.abs(la.to_dense(q))))
        if np.max(np.abs(la.to_dense(rb - q))) > tol:
            msg = "solve_constrained_2sls: constraint violation after KKT."
            raise RuntimeError(msg)
        res_p = la.to_dense(yp - la.dot(Xp, b))
        stat = (-1.0) * la.crossprod(Xp, res_p) + la.dot(R.T, lam)
        s_norm = float(np.max(np.abs(la.to_dense(stat))))
        s_tol = vr * (1.0 + np.max(np.abs(la.to_dense(la.crossprod(Xp, res_p)))))
        if s_norm > s_tol:
            raise RuntimeError(
                "solve_constrained_2sls: KKT stationarity violated on projected system.",
            )
        return b.reshape(-1)


def _weighted_normal_equations(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray | None,
    *,
    prefer_WX_first: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute XtWX and X'Wy with (optional) weights.

    - W is None: unweighted.
    - W is vector of nonnegative weights (length n): analytic weights (diag(W)).
    - W is (n x n) symmetric weights matrix.

    prefer_WX_first controls the order for matrix W:
        - False:  form X'W (k x n), then multiply by X (more cache-friendly when k<<n).
        - True:   form WX (n x k), then X' (better for sparse W).
    """
    X = np.asarray(X, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64, order="C")
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if W is None:
        return la.tdot(X), la.crossprod(X, y)

    Wd = np.asarray(W, dtype=np.float64, order="C")
    if Wd.ndim == 1:
        if Wd.shape[0] != X.shape[0]:
            msg = "Length of weight vector W must equal number of rows of X."
            raise ValueError(msg)
        if np.any(Wd < 0.0):
            msg = "Vector weights must be nonnegative."
            raise ValueError(msg)
        ws = np.sqrt(Wd).reshape(-1, 1)
        Xw = la.hadamard(X, ws)
        yw = la.hadamard(y, ws)
        return la.tdot(Xw), la.crossprod(Xw, yw)

    # general matrix W
    # Strictly require (n x n) shape for matrix W to avoid silent shape bugs.
    if Wd.ndim != 2 or Wd.shape[0] != X.shape[0] or Wd.shape[1] != X.shape[0]:
        msg = "Matrix W must be of shape (n,n) with n = number of rows of X."
        raise ValueError(msg)
    auto_prefer = False
    if not prefer_WX_first:
        # Auto toggle for sparse inputs (heuristic: presence of indptr/indices or scipy.sparse types)
        if hasattr(Wd, "indptr") or hasattr(Wd, "indices"):
            auto_prefer = True
    use_WX = prefer_WX_first or auto_prefer
    if use_WX:
        WX = la.dot(Wd, X)  # sparse-aware
        XtWX = la.crossprod(X, WX)
        Wy = la.dot(Wd, y)
        Xty = la.crossprod(X, Wy)
        return XtWX, Xty
    XtW = la.crossprod(X, Wd)
    return la.dot(XtW, X), la.dot(XtW, y)


def _prune_redundant_rows(
    R: np.ndarray, q: np.ndarray, *, rtol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Prune linearly dependent constraint rows via QR on R^T with column pivoting.

    R is (m x k). We compute QR on R^T (k x m) with column pivoting; the pivot
    indices correspond to rows of R. We keep the first `rank` pivoted columns
    (i.e., rows of R) whose diag(R_up) exceed tolerance, returning a deterministic
    maximal independent subset. This mirrors how Stata/R decide independent
    constraint rows using QR pivoting.
    """
    if R.size == 0:
        return R, q
    Rt = np.asarray(R.T, dtype=np.float64, order="C")  # (k x m)
    # Use core.linalg.qr with pivoting; expect (Q, R_up, piv) when pivoting=True
    try:
        qr_res = la.qr(Rt, pivoting=True, mode="economic")
    except (AttributeError, TypeError, np.linalg.LinAlgError, RuntimeError) as exc:
        LOGGER.debug("QR with pivoting unavailable; using thin QR: %s", exc)
        qr_res = la.qr(Rt, pivoting=False, mode="economic")
    if isinstance(qr_res, tuple) and len(qr_res) == 3:
        _Q, R_up, piv = qr_res
    else:
        _Q, R_up = qr_res
        # no pivoting: piv is identity ordering of columns of Rt -> rows of R
        piv = np.arange(R.shape[0], dtype=np.int64)
    R_up_dense = la.to_dense(R_up)
    if R_up_dense.size == 0:
        return R[:0, :], q[:0, :]
    diagR = np.abs(np.diag(R_up_dense))
    tol = float(diagR.max()) * float(rtol) if diagR.size else 0.0
    rank = int(np.sum(diagR > tol))
    if rank <= 0:
        return R[:0, :], q[:0, :]
    keep_rows = np.sort(piv[:rank].astype(np.int64))
    return R[keep_rows, :], q[keep_rows, :]


def _prune_redundant_rows_with_index(
    R: np.ndarray, q: np.ndarray, *, rtol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like _prune_redundant_rows, but also returns the kept row indices.

    This is required when the caller maintains parallel metadata (e.g., labels)
    that must stay aligned with the pruned constraint system.
    """
    if R.size == 0:
        return R, q, np.zeros((0,), dtype=np.int64)
    Rt = np.asarray(R.T, dtype=np.float64, order="C")
    try:
        qr_res = la.qr(Rt, pivoting=True, mode="economic")
    except (AttributeError, TypeError, np.linalg.LinAlgError, RuntimeError) as exc:
        LOGGER.debug("QR with pivoting unavailable; using thin QR: %s", exc)
        qr_res = la.qr(Rt, pivoting=False, mode="economic")
    if isinstance(qr_res, tuple) and len(qr_res) == 3:
        _Q, R_up, piv = qr_res
    else:
        _Q, R_up = qr_res
        piv = np.arange(R.shape[0], dtype=np.int64)
    R_up_dense = la.to_dense(R_up)
    if R_up_dense.size == 0:
        keep_rows = np.zeros((0,), dtype=np.int64)
        return R[:0, :], q[:0, :], keep_rows
    diagR = np.abs(np.diag(R_up_dense))
    tol = float(diagR.max()) * float(rtol) if diagR.size else 0.0
    rank = int(np.sum(diagR > tol))
    if rank <= 0:
        keep_rows = np.zeros((0,), dtype=np.int64)
        return R[:0, :], q[:0, :], keep_rows
    keep_rows = np.sort(piv[:rank].astype(np.int64))
    return R[keep_rows, :], q[keep_rows, :], keep_rows


def solve_constrained(  # noqa: PLR0911, PLR0913
    X: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    q: np.ndarray,
    *,
    W: np.ndarray | None = None,
    # Weight policy: forbid weights by default to prevent misuse by OLS/IV/QR callers.
    weight_policy: str = "forbid",
    symmetrize_W: bool = True,
    symmetry_atol: float = 1e-10,
    symmetry_rtol: float = 1e-8,
    prune_redundant: bool = False,
    rank_policy: str | None = None,
    scale_constraints: bool = False,
    prefer_WX_first: bool = False,
    ridge: float = 0.0,
    return_diagnostics: bool = False,
    # ---- STRICT: post-solve verification (R/Stata reproducibility) ----
    verify_constraints: bool = True,
    verify_rtol: float = 1e-8,
    verify_atol: float = 1e-12,
) -> np.ndarray:
    """Preferred SPD route (with fallbacks)
    ------------------------------------
    1) (optional) prune redundant rows of R by QR pivoting (if prune_redundant=True)
    2) Try SPD route using A=(X'WX)^{-1} or XtWX_sym; otherwise fall back to KKT.
    3) Enforce R b = q within (verify_rtol, verify_atol) if verify_constraints=True.

    Notes
    -----
    - Length compatibility of W is validated up front.
    - For matrix W, either symmetrize or validate approximate symmetry.

    """

    def _fail_runtime(message: str) -> None:
        raise RuntimeError(message) from None
    X = np.asarray(X, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64, order="C")
    R = np.asarray(R, dtype=np.float64, order="C")
    q = np.asarray(q, dtype=np.float64, order="C")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    # Enforce single right-hand-side for this solver
    q = q.reshape(R.shape[0], -1)
    if q.shape[1] != 1:
        raise ValueError(
            "solve_constrained expects a single-column q (one right-hand side). Use solve_constrained_batch for multiple q.",
        )

    # Finite check (reject NaN/Inf early to match R/Stata strictness)
    for name, arr in (("X", X), ("y", y), ("R", R), ("q", q)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{name} contains non-finite values (NaN/Inf) which are not allowed.",
            )

    n, k = X.shape
    if R.shape[1] != k:
        msg = "R must have the same number of columns as X."
        raise ValueError(msg)
    if X.shape[0] != y.shape[0]:
        msg = "X and y must have the same number of rows."
        raise ValueError(msg)

    # Validate/sanitize W
    W_eff: np.ndarray | None = None
    sym_used = False
    if W is not None:
        # Enforce module-level weight policy guard: only allow explicit paths (GLS/GMM)
        if str(weight_policy).lower() != "allow":
            msg = "Observation weights W are forbidden here. Only GLS/GMM routes may set weight_policy='allow'."
            raise ValueError(msg)
        Wd = np.asarray(W, dtype=np.float64, order="C")
        if Wd.ndim == 1:
            if Wd.shape[0] != n:
                msg = "Length of weight vector W must equal number of rows of X."
                raise ValueError(msg)
            if np.any(Wd < 0.0):
                msg = "Vector weights must be nonnegative."
                raise ValueError(msg)
            W_eff = Wd
        else:
            if symmetrize_W:
                W_eff = 0.5 * (Wd + Wd.T)
                sym_used = True
            else:
                if not np.allclose(Wd, Wd.T, atol=symmetry_atol, rtol=symmetry_rtol):
                    msg = (
                        "Matrix W must be approximately symmetric "
                        "(or set symmetrize_W=True to force symmetrization)."
                    )
                    raise ValueError(
                        msg,
                    )
                W_eff = Wd
            # Enforce PSD (allow tiny negatives due to numerical roundoff)
            try:
                lam_min = float(np.min(la.eigvalsh(W_eff)))
            except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
                LOGGER.debug("Unable to compute eigenvalues of W for PSD check: %s", exc)
                lam_min = float("nan")
            if not (np.isnan(lam_min)) and lam_min < -1e-12:
                msg = f"W must be (approximately) PSD; min eigenvalue={lam_min:.3e}."
                raise ValueError(msg)

    # Try QR-based inverse for XtWX. If unavailable, fall back to normal-equation routes.
    spd_added_ridge = False
    A: np.ndarray | None
    try:
        A = la.xtwx_inv_via_qr(X, weights=W_eff)
    except (AttributeError, TypeError, np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        LOGGER.debug("xtwx_inv_via_qr unavailable; falling back to normal equations: %s", exc)
        A = None
    else:
        if not np.all(np.isfinite(A)):
            msg = "Non-finite entries in xtwx_inv_via_qr result"
            raise RuntimeError(msg) from None

    # Eagerly compute XtWX and X'Wy once; downstream logic depends on them.
    XtWX, Xty = _weighted_normal_equations(X, y, W_eff, prefer_WX_first=prefer_WX_first)
    XtWX_sym = 0.5 * (XtWX + XtWX.T)

    # Optional ridge regularization (constrained ridge when constraints present).
    # This is an opt-in numerical/stochastic regularization knob; ridge=0 preserves
    # the exact constrained OLS/GLS estimator.
    ridge_val = float(ridge)
    if ridge_val < 0.0:
        raise ValueError("ridge must be nonnegative.")
    if ridge_val > 0.0:
        XtWX_sym = XtWX_sym + ridge_val * la.eye(k)
        # xtwx_inv_via_qr (if available) does not incorporate ridge, so disable
        # that shortcut to keep results consistent with the regularized system.
        A = None
        spd_added_ridge = True

    # Optional pruning of redundant constraints for numerical stability
    pruned0 = 0
    if prune_redundant and R.shape[0] > 0:
        m_before = int(R.shape[0])
        rtol_prune = 1e-10
        if rank_policy is not None:
            rp = str(rank_policy).lower()
            if rp not in {"r", "stata"}:
                raise ValueError("rank_policy must be one of {'r','stata'} or None.")
            rtol_prune = 1e-7 if rp == "r" else 1e-10
        R, q = _prune_redundant_rows(R, q, rtol=rtol_prune)
        pruned0 = m_before - int(R.shape[0])

    # Optional: scale constraint rows for numerical conditioning (solution invariant)
    if R.size and scale_constraints:
        # Compute 2-norm of each row safely; guard zero or non-finite norms
        # Row-wise 2-norm without numpy.linalg to comply with matrix-op routing policy
        s = np.sqrt(np.sum(R * R, axis=1, keepdims=True))
        s[~np.isfinite(s)] = 1.0
        s[s == 0.0] = 1.0
        R = R / s
        q = q / s
    m = R.shape[0]
    # Unconstrained case: simple solve
    if m == 0:
        if A is not None:
            beta = la.dot(A, Xty)
        else:
            try:
                beta = la.solve(XtWX_sym, Xty, sym_pos=True)
            except (np.linalg.LinAlgError, RuntimeError, ValueError, TypeError):
                # XtWX not SPD: prefer QR with Stata rank-policy for W=None or
                # vector W; force KKT fallback for full matrix W. Never apply
                # ridge regularization as that would change the estimator.
                if (W_eff is None) or (hasattr(W_eff, "ndim") and W_eff.ndim == 1):
                    if W_eff is None:
                        Xw = X
                        yw = y
                    else:
                        sw = np.sqrt(W_eff).reshape(-1, 1)
                        Xw = la.hadamard(X, sw)
                        yw = la.hadamard(y, sw)
                    beta = la.solve(Xw, yw, method="qr", rank_policy="stata")
                else:
                    # Full matrix W (GLS) and XtWX not SPD/symmetric ->
                    # Normal equations are theoretically inappropriate:
                    # force KKT saddle-point solution for exact constrained estimate
                    msg = "KKT_fallback_enforced"
                    raise RuntimeError(msg) from None

        if return_diagnostics:
            return {
                "beta": np.asarray(beta, dtype=np.float64, order="C"),
                "ridge_added": spd_added_ridge,
                "pruned_rows": 0,
                "rank_R": 0,
                "kkt_fallback": False,
            }
        return np.asarray(beta, dtype=np.float64, order="C").reshape(-1)

    # -------- Constrained case: prefer SPD route via A (if available) or XtWX; else KKT --------
    try:
        # Build RHS using la.column_stack to keep operations inside core.linalg where possible
        RHS = la.column_stack([R.T, Xty])  # (k x (m + r))
        if A is not None:
            S = la.dot(A, RHS)
        else:
            try:
                # SPD check on symmetric part and solve using symmetric part
                la.solve(XtWX_sym, la.eye(XtWX_sym.shape[0]), sym_pos=True)
                S = la.solve(XtWX_sym, RHS, sym_pos=True)
            except (np.linalg.LinAlgError, RuntimeError, ValueError, TypeError):
                # XtWX not SPD in constrained case: force KKT fallback.
                # Never apply ridge regularization as that would change
                # the estimator from constrained OLS/GLS to ridge.
                msg = "KKT_fallback_enforced"
                raise RuntimeError(msg) from None

        V = S[:, :m]
        Y = S[:, m:]
        M = la.dot(R, V)
        # Unscaled system: solve M lam = (R Y - q)
        rhs = la.dot(R, Y) - q
        try:
            lam = la.solve(M, rhs, sym_pos=True)
            # b = Y - V lam
            b = Y - la.dot(V, lam)  # ensure b always computed on success
            # Strict post-check: constraints must be satisfied
            if verify_constraints:
                rb = la.dot(R, b)
                rnorm = float(np.max(np.abs(la.to_dense(rb - q))))
                # verification tolerances aligned with rank_policy for R/Stata parity
                vr = verify_rtol
                if rank_policy is not None:
                    vr = 1e-7 if rank_policy.lower() == "r" else 1e-10
                scale = float(
                    max(verify_atol, vr * (1.0 + np.max(np.abs(la.to_dense(q))))),
                )
                if not (rnorm <= scale):
                    msg = f"Constraint violation after SPD route: max|Rb-q|={rnorm:.3e} > tol={scale:.3e}"
                    _fail_runtime(msg)
                # Stationarity check for SPD route: enforce the KKT gradient
                # -X' W (y - X b) + R' λ ≈ 0 using the same effective weight W_eff
                # used in the SPD computations. Compute X'W(y - Xb) without
                # densifying full W when possible (preserve sparse/diag cases).
                res = y - la.dot(X, b)
                if (W_eff is None) or (np.ndim(W_eff) == 1):
                    # Unweighted or diagonal weights (vector): use crossprod
                    if W_eff is None:
                        XtWres = la.crossprod(X, res)
                    else:
                        # X' diag(w) res via Hadamard on columns of X with w
                        XtWres = la.crossprod(la.hadamard(X, W_eff.reshape(-1, 1)), res)
                else:
                    # Full matrix W: compute X' (W res) directly
                    XtWres = la.dot(X.T, la.dot(W_eff, res))

                stat = la.to_dense((-1.0) * XtWres + la.dot(R.T, lam))
                stat_norm = float(np.max(np.abs(stat)))
                # Reuse `scale` tolerance from the feasibility check above
                if float(stat_norm) > scale:
                    _fail_runtime(
                        f"Stationarity violation after SPD route: max|∇L|={float(stat_norm):.3e} > tol={scale:.3e}",
                    )
        except (np.linalg.LinAlgError, RuntimeError, ValueError, TypeError):
            # For strict theoretical correctness we do NOT accept a pinv-based
            # λ which only approximately enforces R b = q. Instead, prefer to
            # deterministically prune dependent rows and retry; if pruning not
            # enabled, force KKT fallback by raising to outer except.
            if not prune_redundant:
                _fail_runtime("KKT_fallback_enforced")
            # If pruning requested, prune and recompute M/rhs then solve.
            R, q = _prune_redundant_rows(R, q)
            m = R.shape[0]
            if m == 0:
                # no constraints remain
                if A is not None:
                    b = la.dot(A, Xty)
                else:
                    # Use the symmetrized XtWX for numerical consistency
                    b = la.solve(XtWX_sym, Xty, sym_pos=True)
                if return_diagnostics:
                    return {
                        "beta": np.asarray(b, dtype=np.float64, order="C"),
                        "ridge_added": spd_added_ridge,
                        "pruned_rows": int(pruned0),
                        "rank_R": int(m),
                        "symmetrized_W": sym_used,
                        "kkt_fallback": False,
                    }
                return np.asarray(b, dtype=np.float64, order="C").reshape(-1)
            RHS = la.column_stack([R.T, Xty])
            if A is not None:
                S = la.dot(A, RHS)
            else:
                # Use the symmetrized XtWX for numerical consistency
                S = la.solve(XtWX_sym, RHS, sym_pos=True)
            V = S[:, :m]
            Y = S[:, m:]
            M = la.dot(R, V)
            rhs = la.dot(R, Y) - q
            lam = la.solve(M, rhs, sym_pos=True)
            b = Y - la.dot(V, lam)
            if verify_constraints:
                rb = la.dot(R, b)
                rnorm = float(np.max(np.abs(la.to_dense(rb - q))))
                scale = float(
                    max(
                        verify_atol,
                        verify_rtol * (1.0 + np.max(np.abs(la.to_dense(q)))),
                    ),
                )
                if not (rnorm <= scale):
                    msg = f"Constraint violation after pruning: max|Rb-q|={rnorm:.3e} > tol={scale:.3e}"
                    _fail_runtime(msg)
            if return_diagnostics:
                return {
                    "beta": np.asarray(b, dtype=np.float64, order="C"),
                    "ridge_added": spd_added_ridge,
                    "pruned_rows": int(pruned0),
                    "rank_R": int(m),
                    "symmetrized_W": sym_used,
                    "kkt_fallback": False,
                }
        return np.asarray(b, dtype=np.float64, order="C").reshape(-1)
    except (np.linalg.LinAlgError, RuntimeError, ValueError, TypeError):
        # ---- KKT fallback: solve full saddle-point (saddle-point) system ----
        # We construct and solve the KKT linear system directly when SPD
        # elimination fails. This mirrors Stata's `cnsreg` implementation:
        # the augmented system
        #
        #   [ XtWX   R' ] [ b ] = [ X'Wy ]
        #   [  R     0  ] [ λ ]   [  q   ]
        #
        # enforces stationarity and feasibility simultaneously. We keep
        # this branch as a strict, Stata-compatible fallback for exact
        # constrained estimates and KKT verification.
        zeros_m = np.zeros((m, m), dtype=np.float64)
        # ensure XtWX and Xty are available
        if XtWX is None or Xty is None:
            XtWX, Xty = _weighted_normal_equations(
                X, y, W_eff, prefer_WX_first=prefer_WX_first,
            )
        A11 = XtWX_sym
        A12 = R.T
        A21 = R
        A_kkt = np.block([[A11, A12], [A21, zeros_m]])
        B = np.vstack([Xty, q])
        sol = la.solve(A_kkt, B, sym_pos=False)
        b = sol[:k, :]
        lam = sol[k:, :]
        if verify_constraints:
            # Check R b = q
            rb = la.dot(R, b)
            rnorm = float(np.max(np.abs(la.to_dense(rb - q))))
            scale = float(
                max(verify_atol, verify_rtol * (1.0 + np.max(np.abs(la.to_dense(q))))),
            )
            if not (rnorm <= scale):
                msg = f"Constraint violation after KKT fallback: max|Rb-q|={rnorm:.3e} > tol={scale:.3e}"
                _fail_runtime(msg)
            # KKT stationarity: - X'W(y - X b) + R' lam ≈ 0
            # ensure XtWX/Xty available
            if XtWX is None or Xty is None:
                XtWX, Xty = _weighted_normal_equations(
                    X, y, W_eff, prefer_WX_first=prefer_WX_first,
                )
            # compute stationarity residual (theoretical first-order condition):
            # X'W X b - X'W y + R' lam = - X'W(y - X b) + R' lam ≈ 0
            Xb = la.dot(X, b)
            y_minus_Xb = la.to_dense(y - Xb)
            xtw_y_minus_xb = la.xty(
                X, y_minus_Xb, weights=(W_eff if W_eff is not None else None),
            )
            stat = (-1.0) * xtw_y_minus_xb + la.dot(R.T, lam)
            stat_norm = float(np.max(np.abs(la.to_dense(stat))))
            stat_scale = float(
                max(
                    verify_atol,
                    verify_rtol * (1.0 + np.max(np.abs(la.to_dense(xtw_y_minus_xb)))),
                ),
            )
            if not (stat_norm <= stat_scale):
                msg = f"KKT stationarity violated: max|- X'W(y-Xb)+R'lam|={stat_norm:.3e} > tol={stat_scale:.3e}"
                _fail_runtime(msg)
        if return_diagnostics:
            return {
                "beta": np.asarray(b, dtype=np.float64, order="C"),
                "ridge_added": spd_added_ridge,
                "pruned_rows": int(pruned0),
                "rank_R": int(m),
                "kkt_fallback": True,
                "symmetrized_W": sym_used,
            }
        return np.asarray(b, dtype=np.float64, order="C").reshape(-1)


def solve_constrained_batch(  # noqa: PLR0913
    X: np.ndarray,
    Y_batch: np.ndarray,
    R: np.ndarray,
    q_batch: np.ndarray | None,
    *,
    W: np.ndarray | None = None,
    symmetrize_W: bool = True,
    symmetry_atol: float = 1e-10,
    symmetry_rtol: float = 1e-8,
    prune_redundant: bool = False,
    prefer_WX_first: bool = False,
    ridge: float = 0.0,
    weight_policy: str = "forbid",
) -> np.ndarray:
    # Delegate heavy linear algebra to core.linalg's centralized batched solver
    X_arr = np.asarray(X, dtype=np.float64, order="C")
    Y_arr = np.asarray(Y_batch, dtype=np.float64, order="C")
    R_arr = np.asarray(R, dtype=np.float64, order="C")
    q_arr = (
        None if q_batch is None else np.asarray(q_batch, dtype=np.float64, order="C")
    )
    return la.solve_constrained_batch(
        X_arr,
        Y_arr,
        R_arr,
        q_arr,
        W=W,
        symmetrize_W=symmetrize_W,
        symmetry_atol=symmetry_atol,
        symmetry_rtol=symmetry_rtol,
        prune_redundant=prune_redundant,
        prefer_WX_first=prefer_WX_first,
        ridge=ridge,
        weight_policy=weight_policy,
    )
