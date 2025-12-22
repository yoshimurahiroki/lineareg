"""Randomized Controlled Trial (RCT) estimators with bootstrap-only inference.

Scope and policy
----------------
- Supports stratified designs and family-wise simultaneous inference across
    multi-hypothesis contrasts. Estimators include Regression Adjustment (RA)
    and Hajek-IPW; no analytical SEs, p-values, or critical values.
- Inference is bootstrap-only with positive multipliers (e.g., Exp(1)). No
    pairs bootstrap. Studentization is bootstrap-based. Default B=2000 with the
    B+1 quantile rule.
- Linear algebra uses :mod:`lineareg.core.linalg` only (no normal equations).
- No uniform bands beyond the specified family-wise sup-t bands for the RCT
    hypothesis family; DiD-style event-study bands are not exposed here.

Comments and docstrings are English-only by policy.
"""

# rct.py (full replacement)
# Stratified RCT: multi-hypothesis (family-wise) uniform bootstrap confidence bands
# - Supports arbitrary contrasts between a chosen treatment arm and a user-specified control set,
#   optionally restricted to any subset/combination of strata, with optional per-stratum expansion.
# - Estimators: Regression Adjustment (RA) and Hájek-IPW (IPW), both without analytic SE/p/critical values.
# - Bootstrap: positive-multiplier Exp(1) only; no pairs bootstrap. Studentization is purely bootstrap-based.
# - Uniform band: sup-t across the ENTIRE hypothesis family (simultaneous coverage).
# - Rank policy: {"stata","r"} forwarded to QR solver; no normal equations; no analytic SE.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

import numpy as np

# Project-internal linear algebra and result container
from lineareg.core import (
    linalg as la,  # must provide la.solve(..., method="qr", rank_policy={'stata','r'})
)
from lineareg.estimators.base import (
    EstimationResult,  # value-only container (no analytic p, criticals, etc.)
)

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]
MatrixLike = Union[np.ndarray, Sequence[Sequence[float]]]

# --------- utilities: shape & validation ---------


def _as_1d(x: ArrayLike, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite.")
    return a.astype(np.float64)


def _as_int_1d(x: ArrayLike, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    return a


def _as_2d(x: MatrixLike | None, n: int, name: str) -> np.ndarray:
    if x is None:
        return np.zeros((n, 0), dtype=np.float64)
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if a.shape[0] != n:
        raise ValueError(f"{name} has {a.shape[0]} rows; expected {n}.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite.")
    return a


def _check_same_length(*args: tuple[np.ndarray, str]) -> int:
    n = args[0][0].shape[0]
    for arr, nm in args:
        if arr.shape[0] != n:
            raise ValueError(f"Length mismatch: {nm} has {arr.shape[0]}, expected {n}.")
    return n


# --------- strata dummies ---------


def _build_strata_dummies(
    strata: ArrayLike | None, n: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[Any, int]]:
    """Return (S, codes, map) where:
      S: (n x (L-1)) full-rank dummies dropping the first code;
      codes: (n,) numerical codes aligned with original order (or None if strata is None)
      map: original_label -> integer_code (0..L-1)

    The ordering of levels preserves first-appearance order (R factor / Stata i.var
    semantics) rather than lexicographic sorting.
    """
    if strata is None:
        return np.zeros((n, 0), dtype=np.float64), None, {}
    s_raw = np.asarray(strata)
    if s_raw.shape[0] != n:
        raise ValueError("strata length mismatch.")
    # preserve first-appearance ordering
    code_map: dict[Any, int] = {}
    levels: list[Any] = []
    codes = np.empty(n, dtype=int)
    for i, v in enumerate(s_raw):
        if v not in code_map:
            code_map[v] = len(code_map)
            levels.append(v)
        codes[i] = code_map[v]
    L = len(levels)
    if L <= 1:
        return np.zeros((n, 0), dtype=np.float64), codes, code_map
    S = np.zeros((n, L - 1), dtype=np.float64)
    for k in range(1, L):
        S[:, k - 1] = (codes == k).astype(np.float64)
    return S, codes, code_map


# --------- Exp(1) multipliers ---------


def _draw_exp_multipliers(
    n: int, B: int, seed: int | None = None, strata_codes: np.ndarray | None = None,
) -> np.ndarray:
    """Draw positive Exp(1) multipliers. If strata_codes is given, assign one weight per stratum (stratum-clustered).
    Returns (n x B) array of positive weights.
    """
    rng = np.random.default_rng(seed)
    W = np.empty((n, B), dtype=np.float64)
    if strata_codes is None:
        W[:] = rng.exponential(scale=1.0, size=(n, B))
    else:
        uniq = np.unique(strata_codes)
        for b in range(B):
            g_w = rng.exponential(scale=1.0, size=uniq.shape[0])
            for j, g in enumerate(uniq):
                W[strata_codes == g, b] = g_w[j]
    return W


# --------- assignment probabilities (multiarm, optional by-stratum) ---------


def _resolve_pi_for_contrast(
    a: np.ndarray,
    strata_codes: np.ndarray | None,
    p: float | dict[Any, Any] | None,
    treat_arm: Any,
    control_arms: Sequence[Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-observation assignment probabilities for a given contrast (treat vs control set).
    Accepts p as:
      - None (raises if IPW requested),
      - scalar for binary {0,1} (π_t = p, Π0 = 1-p),
      - dict arm->prob,
      - dict strata->(dict arm->prob)
    Returns (pi_t, Pi0, mask_valid), each length n.
      mask_valid selects units eligible for this contrast (observed arm ∈ {treat} union control).
    """
    n = a.shape[0]
    if control_arms is None:
        control_arms = [v for v in np.unique(a) if v != treat_arm]
    control_arms_set = set(control_arms)
    use_mask = np.array(
        [(aa == treat_arm) or (aa in control_arms_set) for aa in a], dtype=bool,
    )

    pi_t = np.full(n, np.nan, dtype=np.float64)
    Pi0 = np.full(n, np.nan, dtype=np.float64)

    if p is None:
        # IPW cannot be computed; RA only case may still call this function -> fill NaN and return.
        return pi_t, Pi0, use_mask

    # helper to fill per-observation probabilities
    def fill_probs_for_row(i: int):
        if isinstance(p, (float, int)):
            # binary-only convenience
            pt = float(p)
            if not (0.0 < pt < 1.0):
                raise ValueError("scalar p must lie in (0,1).")
            pi_t[i] = pt
            Pi0[i] = 1.0 - pt
        elif isinstance(p, dict):
            if (
                strata_codes is not None
                and strata_codes[i] in p
                and isinstance(p[strata_codes[i]], dict)
            ):
                # by-stratum dict-of-dicts
                pdict = p[strata_codes[i]]
                if treat_arm not in pdict:
                    raise ValueError(
                        f"p[{strata_codes[i]}] must include key for treat_arm={treat_arm!r}.",
                    )
                pi_t[i] = float(pdict[treat_arm])
                # control mixture probability = sum p(c) over c in control set
                Pi0[i] = float(sum(float(pdict[c]) for c in control_arms_set))
            else:
                # global dict arm->prob
                if treat_arm not in p:
                    raise ValueError(f"p must include key for treat_arm={treat_arm!r}.")
                pi_t[i] = float(p[treat_arm])
                Pi0[i] = float(sum(float(p[c]) for c in control_arms_set))
        else:
            raise TypeError(
                "Unsupported p type. Use float, dict[arm->prob], or dict[stratum]->dict[arm->prob].",
            )
        # basic checks
        if not (
            np.isfinite(pi_t[i])
            and np.isfinite(Pi0[i])
            and pi_t[i] > 0.0
            and Pi0[i] > 0.0
            and pi_t[i] + Pi0[i] <= 1.0000000001
        ):
            # Allow other arms outside {treat, control}; hence π_t + Π0 can be < 1 if there are >2 arms.
            if not (
                np.isfinite(pi_t[i])
                and np.isfinite(Pi0[i])
                and pi_t[i] > 0.0
                and Pi0[i] > 0.0
            ):
                raise ValueError("Nonpositive or non-finite assignment probabilities.")

    idx = np.where(use_mask)[0]
    for i in idx:
        fill_probs_for_row(i)

    return pi_t, Pi0, use_mask


# --------- RA: arm-wise OLS/WLS on common design Z=[1,S,X] ---------


def _ra_pair_effects(  # noqa: PLR0913
    *,
    y: np.ndarray,
    a: np.ndarray,
    treat_arm: Any,
    control_arms: Sequence[Any],
    X: np.ndarray,
    S: np.ndarray,
    weights: np.ndarray | None,
    rank_policy: str,
    use_mask: np.ndarray | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """Regression adjustment ATE/ATT/ATC with overlap enforcement.

    Estimation and averaging are both restricted to `use_mask` if provided.
    Strata with zero count in either group are excluded (no extrapolation).
    Returns (ATE, ATT, ATC, diff_vector) where diff_vector is defined on the
    restricted sample (rows satisfying treatment/control & use_mask).
    """
    control_arms_set = set(control_arms)
    # initial eligibility mask for this contrast
    m0 = (a == treat_arm) | np.isin(a, list(control_arms_set))
    m = m0 & use_mask if use_mask is not None else m0
    if np.sum(m) == 0:
        return np.nan, np.nan, np.nan, np.full(0, np.nan)

    y_m = y[m]
    X_m = X[m, :]
    S_m = S[m, :]
    d_m = (a[m] == treat_arm).astype(np.int8)
    const = np.ones((y_m.shape[0], 1), dtype=np.float64)
    Z = np.column_stack([const, S_m, X_m])
    w = (
        None
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)[m]
    )

    # detect overlap at the strata level for the restricted sample: remove strata without both groups
    # compute strata codes if S has zero cols then treat as single stratum
    if S_m.shape[1] == 0:
        ok_idx = np.ones(Z.shape[0], dtype=bool)
    else:
        # reconstruct simple code by comparing rows across S_m columns
        # Create numeric strata id by dotting with powers of two is fragile; instead group by tuple
        strata_tuples = [tuple(row) for row in S_m.tolist()]
        uniq = {}
        codes = np.empty(len(strata_tuples), dtype=int)
        for i, t in enumerate(strata_tuples):
            if t not in uniq:
                uniq[t] = len(uniq)
            codes[i] = uniq[t]
        L = int(codes.max()) + 1 if codes.size > 0 else 0
        has_t = np.zeros(L, dtype=bool)
        has_c = np.zeros(L, dtype=bool)
        for s in range(L):
            hit = codes == s
            if np.any(hit):
                has_t[s] = np.any(d_m & hit)
                has_c[s] = np.any((~d_m) & hit)
        ok_strata = set(np.where(has_t & has_c)[0])
        ok_idx = np.array(
            [codes[i] in ok_strata for i in range(len(codes))], dtype=bool,
        )

    # If no strata have both treated and control, undefined
    if not np.any(ok_idx):
        return np.nan, np.nan, np.nan, np.full(0, np.nan)

    # Restrict to OK observations
    Z_ok = Z[ok_idx, :]
    y_ok = y_m[ok_idx]
    d_ok = d_m[ok_idx]

    beta1 = _group_ols(
        y_ok, Z_ok, (d_ok == 1), w[ok_idx] if w is not None else None, rank_policy,
    )
    beta0 = _group_ols(
        y_ok, Z_ok, (d_ok == 0), w[ok_idx] if w is not None else None, rank_policy,
    )
    if np.any(~np.isfinite(beta1)) or np.any(~np.isfinite(beta0)):
        return np.nan, np.nan, np.nan, np.full(0, np.nan)

    mu1 = la.dot(Z_ok, beta1)
    mu0 = la.dot(Z_ok, beta0)
    diff = (mu1 - mu0).reshape(-1)

    ate = float(np.mean(diff)) if diff.size > 0 else np.nan
    att = float(np.mean(diff[d_ok == 1])) if (d_ok == 1).any() else np.nan
    atc = float(np.mean(diff[d_ok == 0])) if (d_ok == 0).any() else np.nan
    return ate, att, atc, diff


def _group_ols(
    y: np.ndarray,
    Z: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray | None,
    rank_policy: str,
) -> np.ndarray:
    idx = np.where(mask)[0]
    if idx.size == 0:
        # no observations in this group: forbid extrapolation by returning NaNs
        return np.full(Z.shape[1], np.nan, dtype=np.float64)
    Zi = Z[idx, :]
    yi = y[idx].reshape(-1, 1)
    if weights is not None:
        wi = weights[idx]
        if np.any(wi <= 0.0) or not np.all(np.isfinite(wi)):
            raise ValueError("bootstrap weights must be positive & finite.")
        rootw = np.sqrt(wi).reshape(-1, 1)
        Zi = Zi * rootw
        yi = yi * rootw
    return la.solve(Zi, yi, method="qr", rank_policy=str(rank_policy).lower()).reshape(
        -1,
    )


# -------------------- Propensity score helpers (cross-fit multiclass) --------------------
def _cross_fit_ps_multiclass(ps_learner, A, X, *, k_folds: int, seed: int | None):
    """Cross-fit probabilites for multiclass treatment A using user-supplied sklearn-like
    estimator `ps_learner` implementing predict_proba. Returns (P, levels) where P is
    (n x G) probability matrix and levels is the class order corresponding to columns.
    """
    from sklearn.base import clone

    A = np.asarray(A)
    n = A.shape[0]
    # preserve first-appearance order for levels
    levels = []
    pos = {}
    inv = np.empty(n, dtype=int)
    for i, a in enumerate(A):
        if a not in pos:
            pos[a] = len(pos)
            levels.append(a)
        inv[i] = pos[a]
    G = len(levels)
    P = np.zeros((n, G), dtype=np.float64)

    if k_folds <= 1:
        clf = clone(ps_learner)
        clf.fit(X, A)
        proba = clf.predict_proba(X)
        if proba.shape[1] != G:
            raise ValueError(
                "predict_proba column count does not match number of treatment levels.",
            )
        P[:] = proba
        return P, levels

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.arange(n), A):
        clf = clone(ps_learner)
        clf.fit(X[tr, :], A[tr])
        proba = clf.predict_proba(X[te, :])
        if proba.shape[1] != G:
            raise ValueError(
                "predict_proba column count does not match number of treatment levels.",
            )
        P[te, :] = proba
    return P, levels


def _resolve_estimated_pi_for_contrast(A, P, levels, treat_arm, control_arms):
    """From multiclass probability matrix P (n x G) and ordered levels, extract
    unit-level pi_t and Pi0 for a contrast.
    """
    n, G = P.shape
    if np.asarray(A).shape[0] != n:
        raise ValueError("Observed assignments and probability matrix must align.")
    pi_t = np.full(n, np.nan, dtype=np.float64)
    Pi0 = np.full(n, np.nan, dtype=np.float64)
    pos = {levels[g]: g for g in range(G)}
    if treat_arm not in pos:
        raise ValueError("treat_arm not found in estimated PS levels.")
    tcol = pos[treat_arm]
    pi_t = P[:, tcol]
    ctrl = (
        list(control_arms)
        if control_arms is not None
        else [lv for lv in levels if lv != treat_arm]
    )
    cpos = [pos[c] for c in ctrl]
    Pi0 = np.sum(P[:, cpos], axis=1)
    return pi_t, Pi0


# --------- IPW: Hájek normalizations (multiarm pairwise) ---------


def _ipw_pair_effects(  # noqa: PLR0913
    y: np.ndarray,
    a: np.ndarray,
    treat_arm: Any,
    control_arms: Sequence[Any],
    pi_t: np.ndarray,
    Pi0: np.ndarray,
    use_mask: np.ndarray,
    w: np.ndarray | None,
) -> tuple[float, float, float]:
    """Compute ATE, ATT, ATC for pair (treat_arm vs control_arms) using Hájek IPW on subset m=use_mask.
    π_t and Π0 are per-observation assign probs for treat and control mixture in this contrast.
    """
    m = use_mask
    if m.sum() == 0:
        return np.nan, np.nan, np.nan
    # positive multipliers (bootstrap) or ones
    ww = (
        np.ones(y.shape[0], dtype=np.float64)
        if w is None
        else np.asarray(w, dtype=np.float64).reshape(-1)
    )
    yv = y
    av = a

    treat = av == treat_arm
    control = np.isin(av, list(control_arms))
    # ATE:
    w1 = ww * (treat & m) / np.clip(pi_t, 1e-12, np.inf)
    w0 = ww * (control & m) / np.clip(Pi0, 1e-12, np.inf)
    mu1 = np.sum(w1 * yv) / np.sum(w1) if np.sum(w1) > 0.0 else np.nan
    mu0 = np.sum(w0 * yv) / np.sum(w0) if np.sum(w0) > 0.0 else np.nan
    ate = mu1 - mu0 if (np.isfinite(mu1) and np.isfinite(mu0)) else np.nan

    # ATT (on treat_arm): counterfactual for treated via inverse-odds within stratum-neutral design
    w0_att = (
        ww
        * (control & m)
        * (np.clip(pi_t, 1e-12, np.inf) / np.clip(Pi0, 1e-12, np.inf))
    )
    mu0_att = np.sum(w0_att * yv) / np.sum(w0_att) if np.sum(w0_att) > 0.0 else np.nan
    mu1_att = np.mean(yv[treat & m]) if (treat & m).any() else np.nan
    att = (
        mu1_att - mu0_att if (np.isfinite(mu1_att) and np.isfinite(mu0_att)) else np.nan
    )

    # ATC (on controls set): mirror construction
    w1_atc = (
        ww * (treat & m) * (np.clip(Pi0, 1e-12, np.inf) / np.clip(pi_t, 1e-12, np.inf))
    )
    mu1_atc = np.sum(w1_atc * yv) / np.sum(w1_atc) if np.sum(w1_atc) > 0.0 else np.nan
    mu0_atc = np.mean(yv[control & m]) if (control & m).any() else np.nan
    atc = (
        mu1_atc - mu0_atc if (np.isfinite(mu1_atc) and np.isfinite(mu0_atc)) else np.nan
    )

    return ate, att, atc


def _ipw_unit_weights(
    a: np.ndarray,
    treat_arm: Any,
    control_arms: Sequence[Any],
    pi_t: np.ndarray,
    Pi0: np.ndarray,
) -> np.ndarray:
    """Construct observation-level IPW weights for a given contrast.

    Parameters
    ----------
    a :
        Observed treatment assignments.
    treat_arm :
        Label of the treated arm for this contrast.
    control_arms :
        Sequence of control arm labels.
    pi_t :
        Probability of being assigned to the treated arm.
    Pi0 :
        Probability of being assigned to any of the control arms in the contrast.

    """
    weights = np.full_like(pi_t, np.nan, dtype=np.float64)
    treat_mask = a == treat_arm
    ctrl_mask = np.isin(a, list(control_arms))
    weights[treat_mask] = 1.0 / np.clip(pi_t[treat_mask], 1e-12, np.inf)
    weights[ctrl_mask] = 1.0 / np.clip(Pi0[ctrl_mask], 1e-12, np.inf)
    return weights


# --------- uniform band (sup-t over family) ---------


def _uniform_band_from_bootstrap(
    theta: np.ndarray,
    theta_star: np.ndarray,
    alpha: float,
    zero_se_tol: float = 1e-12,
    zero_se_rel: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """theta: (K,) vector of point estimates across the hypothesis family
    theta_star: (K, B) bootstrap replicates (re-estimated)
    Returns (lo, hi, se_boot) with simultaneous sup-t band (B+1 correction).
    """
    K, B = theta_star.shape
    if theta.shape[0] != K:
        raise ValueError("theta and theta_star must agree on the hypothesis dimension.")
    # bootstrap sd per component (ddof=1), using centered replicates
    diffs = theta_star - theta.reshape(-1, 1)
    se = np.nanstd(diffs, axis=1, ddof=1)
    # handle zero/near-zero se -> exclude from Tmax and keep degenerate band at point
    use = (
        np.isfinite(se)
        & (se > 0)
        & (se > zero_se_tol)
        & (se > zero_se_rel * np.max(se))
    )
    if not np.any(use):
        # all zero -> band reduces to point
        return theta.copy(), theta.copy(), se
    T = np.full((np.count_nonzero(use), B), np.nan, dtype=np.float64)
    uu = np.where(use)[0]
    for j, k in enumerate(uu):
        T[j, :] = diffs[k, :] / se[k]
    Tmax = np.nanmax(np.abs(T), axis=0)
    Tmax_sorted = np.sort(Tmax)
    # (B+1)*(1-alpha) quantile with finite-sample correction (no p-values/criticals output)
    rank = int(np.ceil((B + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), B - 1)
    c = Tmax_sorted[rank]
    lo = theta - c * se
    hi = theta + c * se
    # for non-usable components (se==0), set bands to point value
    lo[~use] = theta[~use]
    hi[~use] = theta[~use]
    return lo, hi, se


# --------- main estimator class ---------


class RCT:
    """Stratified RCT uniform bootstrap bands for arbitrary multi-hypothesis families.

    Parameters
    ----------
    alpha : float
        Family-wise noncoverage level (uniform band), default 0.05.
    B : int
        Number of bootstrap draws (Exp(1) multipliers), default 999.
    rank_policy : {'stata','r'}
        Rank decision policy forwarded to QR solver.
    bootstrap_by_stratum : bool
        If True, use one multiplier per stratum (clustered within stratum). Else IID per unit.
    seed : Optional[int]
        Random seed for multipliers.

    """

    def __init__(  # noqa: PLR0913
        self,
        alpha: float = 0.05,
        B: int = 999,
        rank_policy: str = "stata",
        bootstrap_by_stratum: bool = True,
        seed: int | None = None,
        *,
        ps_learner: object | None = None,
        ps_X: MatrixLike | None = None,
        n_folds: int = 5,
        fold_seed: int | None = None,
        trim_ps: float = 0.0,
        trim_mode: str = "clip",
    ) -> None:
        self.alpha = float(alpha)
        self.B = int(B)
        self.rank_policy = str(rank_policy).lower()
        if self.rank_policy not in {"stata", "r"}:
            raise ValueError("rank_policy must be 'stata' or 'r'.")
        self.bootstrap_by_stratum = bool(bootstrap_by_stratum)
        self.seed = seed
        # Propensity score estimation settings
        self.ps_learner = ps_learner
        self.ps_X = ps_X
        self.n_folds = int(n_folds)
        self.fold_seed = fold_seed
        self.trim_ps = float(trim_ps)
        if self.trim_ps < 0.0 or self.trim_ps >= 0.5:
            raise ValueError("trim_ps must be in [0,0.5).")
        self.trim_mode = str(trim_mode).lower()
        if self.trim_mode not in {"clip", "drop"}:
            raise ValueError("trim_mode must be 'clip' or 'drop'.")
        self._results: EstimationResult | None = None

    @property
    def results(self) -> EstimationResult | None:
        return self._results

    def fit(  # noqa: PLR0913
        self,
        y: ArrayLike,
        a: ArrayLike,  # arm labels (categorical; e.g., 0/1, or strings)
        *,
        X: MatrixLike | None = None,
        strata: ArrayLike | None = None,
        p: float | dict[Any, Any] | None = None,
        var_names: Sequence[str] | None = None,
        # Multi-hypothesis family specification:
        # contrasts: list of dict with keys:
        #   - 'treat_arm': required
        #   - 'control_arms': optional list (default: all except treat)
        #   - 'strata_subset': optional list of strata labels to include
        #   - 'by_stratum': bool, if True, expand into per-stratum hypotheses
        #   - 'estimators': subset of {'ra','ipw'} (default: both)
        #   - 'targets': subset of {'ate','att','atc'} (default: ('ate','att'))
        #   - 'name': optional label prefix for this contrast
        contrasts: list[dict[str, Any]] | None = None,
    ) -> EstimationResult:
        yv = _as_1d(y, "y")
        av = _as_int_1d(a, "a")
        n = _check_same_length((yv, "y"), (av, "a"))
        Xv = _as_2d(X, n, "X")
        Sv, strata_codes, strata_map = _build_strata_dummies(strata, n)
        _ = _check_same_length((yv, "y"), (av, "a"), (Sv, "S"), (Xv, "X"))

        # Resolve variable names
        if var_names is None:
            var_names = [f"x{j}" for j in range(Xv.shape[1])]
        elif len(var_names) != Xv.shape[1]:
            raise ValueError("var_names length mismatch with X columns.")

        # Default contrasts: binary case a∈{0,1} -> 1 vs {0}, overall, RA only (IPW requires p), targets {ATE, ATT}
        if contrasts is None:
            arms = np.unique(av)
            if arms.size != 2:
                raise ValueError("Provide 'contrasts' for multi-arm designs.")
            treat_default = arms.max()
            control_default = [x for x in arms if x != treat_default]
            contrasts = [
                {
                    "treat_arm": treat_default,
                    "control_arms": control_default,
                    "strata_subset": None,
                    "by_stratum": False,
                    "estimators": ["ra"],
                    "targets": ["ate", "att"],
                    "name": f"{treat_default}_vs_{'+'.join(map(str, control_default))}",
                },
            ]

        # Pre-draw bootstrap multipliers
        strata_ids_for_boot = strata_codes if self.bootstrap_by_stratum else None
        W = _draw_exp_multipliers(
            n, self.B, seed=self.seed, strata_codes=strata_ids_for_boot,
        )

        # If p == 'estimate', perform cross-fit PS estimation once up-front and
        # apply trimming/clipping rules that synchronize rows and bootstrap multipliers.
        estimated_P = None
        estimated_levels = None
        if isinstance(p, str) and p == "estimate":
            if self.ps_learner is None:
                raise ValueError(
                    "p='estimate' requires self.ps_learner to be set (sklearn-like estimator with predict_proba).",
                )
            if self.ps_X is None:
                raise ValueError(
                    "ps_X must be provided when p='estimate' (required for PS learning).",
                )
            Xps = np.asarray(self.ps_X, dtype=np.float64)
            if Xps.shape[0] != n:
                raise ValueError("ps_X row count must equal n observations.")
            if not hasattr(self.ps_learner, "predict_proba"):
                raise ValueError(
                    "ps_learner must implement predict_proba (sklearn-like multi-class classifier).",
                )
            k = max(2, int(self.n_folds))
            P, levels = _cross_fit_ps_multiclass(
                self.ps_learner, av, Xps, k_folds=k, seed=self.fold_seed,
            )
            # Trimming / clipping
            if self.trim_ps > 0.0:
                if self.trim_mode == "clip":
                    P = np.clip(P, self.trim_ps, 1.0 - self.trim_ps)
                else:
                    keep = np.all(
                        (self.trim_ps <= P) & (1.0 - self.trim_ps >= P), axis=1,
                    )
                    if not np.any(keep):
                        raise ValueError(
                            "All observations trimmed by trim_ps/drop settings.",
                        )
                    # synchronize primary arrays and bootstrap multipliers to kept rows
                    av = av[keep]
                    yv = yv[keep]
                    Xv = Xv[keep, :]
                    Sv = Sv[keep, :]
                    strata_codes = (
                        strata_codes[keep] if strata_codes is not None else None
                    )
                    W = W[keep, :]
                    P = P[keep, :]
                    n = av.shape[0]
            estimated_P = P
            estimated_levels = levels

        # Build the family vector (names, theta), and replicates theta*
        family_names: list[str] = []
        theta_list: list[float] = []
        balance_payloads: list[dict[str, Any]] = []

        # Precompute S dummies for each strata subset mask to avoid repeated slicing overhead
        # (we still slice per contrast because treatment/control sets differ)
        # Iterate contrasts in order and build both family_names (point estimates) and balance payloads
        for cc in contrasts:
            treat_arm = cc["treat_arm"]
            control_arms = cc.get("control_arms", None)
            by_stratum = bool(cc.get("by_stratum", False))
            estimators = [str(s).lower() for s in cc.get("estimators", ["ra"])]
            targets = [str(t).lower() for t in cc.get("targets", ("ate", "att"))]
            name_prefix = str(
                cc.get(
                    "name",
                    f"{treat_arm}_vs_{'+'.join(map(str, control_arms)) if control_arms else 'rest'}",
                ),
            )
            # strata subset mask (use numeric strata_codes for robust masking after trimming/drop)
            if cc.get("strata_subset", None) is None:
                subset_mask = np.ones(n, dtype=bool)
                strata_levels = (
                    np.unique(strata_codes) if strata_codes is not None else None
                )
            else:
                if strata_codes is None:
                    raise ValueError("strata_subset specified but no strata provided.")
                allowed = set(cc["strata_subset"])
                allowed_codes = np.array(
                    [strata_map[s] for s in allowed if s in strata_map], dtype=int,
                )
                subset_mask = np.isin(strata_codes, allowed_codes)
                strata_levels = allowed_codes if by_stratum else None

            # Resolve probabilities once for this contrast. If p=='estimate' use estimated P.
            if isinstance(p, str) and p == "estimate":
                if estimated_P is None:
                    raise RuntimeError("Internal error: estimated PS matrix missing.")
                pi_t, Pi0 = _resolve_estimated_pi_for_contrast(
                    av, estimated_P, estimated_levels, treat_arm, control_arms,
                )
                # build mask of units eligible for this contrast
                use_mask_all = np.array(
                    [
                        (aa == treat_arm)
                        or (
                            aa
                            in (
                                control_arms
                                if control_arms is not None
                                else [x for x in np.unique(av) if x != treat_arm]
                            )
                        )
                        for aa in av
                    ],
                    dtype=bool,
                )
            else:
                pi_t, Pi0, use_mask_all = _resolve_pi_for_contrast(
                    av, strata_codes, p, treat_arm, control_arms,
                )
            # Overall (restricted to subset)
            overall_mask = subset_mask & use_mask_all

            def push_one(
                est: str,
                tgt: str,
                val: float,
                tag: str | None = None,
                *,
                prefix: str = name_prefix,
            ):
                nm = f"{est.upper()}_{tgt.upper()}__{prefix}" + (
                    f"{tag}" if tag else ""
                )
                family_names.append(nm)
                theta_list.append(float(val))

            # ---------- Overall ----------
            # RA: compute on restricted sample using use_mask to avoid index mismatches
            if "ra" in estimators:
                ate_ra, att_ra, atc_ra, _ = _ra_pair_effects(
                    y=yv,
                    a=av,
                    treat_arm=treat_arm,
                    control_arms=(
                        control_arms
                        if control_arms is not None
                        else [x for x in np.unique(av) if x != treat_arm]
                    ),
                    X=Xv,
                    S=Sv,
                    weights=None,
                    rank_policy=self.rank_policy,
                    use_mask=overall_mask,
                )
                if "ate" in targets:
                    push_one("ra", "ate", ate_ra)
                if "att" in targets:
                    push_one("ra", "att", att_ra)
                if "atc" in targets:
                    push_one("ra", "atc", atc_ra)

            # IPW
            if "ipw" in estimators:
                if p is None:
                    raise ValueError(
                        "IPW requested but 'p' (assignment probabilities) is missing.",
                    )
                ate_ipw, att_ipw, atc_ipw = _ipw_pair_effects(
                    y=yv,
                    a=av,
                    treat_arm=treat_arm,
                    control_arms=(
                        control_arms
                        if control_arms is not None
                        else [x for x in np.unique(av) if x != treat_arm]
                    ),
                    pi_t=pi_t,
                    Pi0=Pi0,
                    use_mask=overall_mask,
                    w=None,
                )
                if "ate" in targets:
                    push_one("ipw", "ate", ate_ipw)
                if "att" in targets:
                    push_one("ipw", "att", att_ipw)
                if "atc" in targets:
                    push_one("ipw", "atc", atc_ipw)
                # Capture balance payload (once per contrast)
                ctrl_list = (
                    control_arms
                    if control_arms is not None
                    else [x for x in np.unique(av) if x != treat_arm]
                )
                weights_ipw = _ipw_unit_weights(av, treat_arm, ctrl_list, pi_t, Pi0)
                payload_name = f"{name_prefix}_IPW"
                if np.isfinite(weights_ipw[overall_mask]).all() and overall_mask.any():
                    if not any(p.get("name") == payload_name for p in balance_payloads):
                        payload_entry = {
                            "name": payload_name,
                            "X": Xv[overall_mask, :].astype(np.float64, copy=False),
                            "group": (av[overall_mask] == treat_arm).astype(
                                int, copy=False,
                            ),
                            "w_before": np.ones(
                                int(overall_mask.sum()), dtype=np.float64,
                            ),
                            "w_after": weights_ipw[overall_mask].astype(
                                np.float64, copy=False,
                            ),
                            "covariate_names": list(var_names),
                            "metadata": {
                                "contrast": name_prefix,
                                "type": "ipw",
                                "treat_arm": treat_arm,
                            },
                        }
                        balance_payloads.append(payload_entry)

            # ---------- Per-stratum expansion ----------
            if by_stratum and strata_levels is not None and strata_levels.size > 0:
                for s_code in strata_levels:
                    m_s = (strata_codes == s_code) & use_mask_all
                    tag = f"_S{int(s_code)}"
                    # RA per stratum: reuse diff via re-fit on that stratum subset for exactness
                    if "ra" in estimators:
                        # limit data to stratum s_code
                        sel = strata_codes == s_code
                        ate_s, att_s, atc_s, _ = _ra_pair_effects(
                            y=yv[sel],
                            a=av[sel],
                            treat_arm=treat_arm,
                            control_arms=(
                                control_arms
                                if control_arms is not None
                                else [x for x in np.unique(av[sel]) if x != treat_arm]
                            ),
                            X=Xv[sel, :],
                            S=Sv[sel, :],
                            weights=None,
                            rank_policy=self.rank_policy,
                        )
                        if "ate" in targets:
                            push_one("ra", "ate", ate_s, tag)
                        if "att" in targets:
                            push_one("ra", "att", att_s, tag)
                        if "atc" in targets:
                            push_one("ra", "atc", atc_s, tag)
                    if "ipw" in estimators:
                        if p is None:
                            raise ValueError("IPW requested but 'p' is missing.")
                        ate_s, att_s, atc_s = _ipw_pair_effects(
                            y=yv,
                            a=av,
                            treat_arm=treat_arm,
                            control_arms=(
                                control_arms
                                if control_arms is not None
                                else [x for x in np.unique(av) if x != treat_arm]
                            ),
                            pi_t=pi_t,
                            Pi0=Pi0,
                            use_mask=m_s,
                            w=None,
                        )
                        if "ate" in targets:
                            push_one("ipw", "ate", ate_s, tag)
                        if "att" in targets:
                            push_one("ipw", "att", att_s, tag)
                        if "atc" in targets:
                            push_one("ipw", "atc", atc_s, tag)

        # Now K is known (keep theta) -- fill of bootstrap replicates is done with the plan below
        theta = np.array(theta_list, dtype=np.float64)

        # The above per-draw loop advanced 'row' incorrectly because we inserted sequentially.
        # Reconstruct thetastar by re-evaluating contrasts for ALL b in a second pass, but this time
        # filling per-block slices computed from the names order. To keep code compact and correct,
        # rebuild using a single names-order iterator:

        # Recompute replicates accurately (deterministic order over family_names):
        # Prepare index plan: for each entry in family_names, store a callable producing that scalar for given wb (or None)
        # For brevity & correctness, redo with a small dispatcher:

        plan: list[tuple[str, dict[str, Any]]] = []
        # Rebuild plan consistent with family_names construction:
        for cc in contrasts:
            treat_arm = cc["treat_arm"]
            control_arms = cc.get("control_arms", None)
            by_stratum = bool(cc.get("by_stratum", False))
            estimators = [str(s).lower() for s in cc.get("estimators", ["ra"])]
            targets = [str(t).lower() for t in cc.get("targets", ("ate", "att"))]
            # strata subset: use numeric strata_codes (not original label strings) so masking
            # remains correct after any trimming/drop operations that changed lengths
            if cc.get("strata_subset", None) is None:
                subset_mask = np.ones(n, dtype=bool)
                strata_levels = (
                    np.unique(strata_codes) if strata_codes is not None else None
                )
            else:
                if strata_codes is None:
                    raise ValueError("strata_subset specified but no strata provided.")
                # Map allowed labels to numeric codes using strata_map and keep only codes present
                allowed = set(cc["strata_subset"])
                allowed_codes = [strata_map[s] for s in allowed if s in strata_map]
                subset_mask = np.isin(strata_codes, allowed_codes)
                strata_levels = np.array(allowed_codes) if by_stratum else None
            # probabilities for this contrast
            if isinstance(p, str) and p == "estimate":
                # use the precomputed estimated_P and estimated_levels
                pi_t, Pi0 = _resolve_estimated_pi_for_contrast(
                    av, estimated_P, estimated_levels, treat_arm, control_arms,
                )
                use_mask_all = np.isin(
                    av,
                    [treat_arm]
                    + (
                        control_arms
                        if control_arms is not None
                        else [x for x in np.unique(av) if x != treat_arm]
                    ),
                )
            else:
                pi_t, Pi0, use_mask_all = _resolve_pi_for_contrast(
                    av, strata_codes, p, treat_arm, control_arms,
                )
            overall_mask = subset_mask & use_mask_all
            ctrl_list = (
                control_arms
                if control_arms is not None
                else [x for x in np.unique(av) if x != treat_arm]
            )
            # Overall
            if "ra" in estimators:
                plan.append(
                    (
                        "ra_overall",
                        {
                            "treat": treat_arm,
                            "ctrl": ctrl_list,
                            "mask": overall_mask,
                            "targets": targets,
                        },
                    ),
                )
            if "ipw" in estimators:
                plan.append(
                    (
                        "ipw_overall",
                        {
                            "treat": treat_arm,
                            "ctrl": ctrl_list,
                            "mask": overall_mask,
                            "targets": targets,
                            "pi_t": pi_t,
                            "Pi0": Pi0,
                        },
                    ),
                )
            # By-stratum
            if by_stratum and strata_levels is not None and strata_levels.size > 0:
                for s_code in strata_levels:
                    m_s = (strata_codes == s_code) & use_mask_all
                    if "ra" in estimators:
                        plan.append(
                            (
                                "ra_stratum",
                                {
                                    "treat": treat_arm,
                                    "ctrl": ctrl_list,
                                    "s_code": int(s_code),
                                    "targets": targets,
                                },
                            ),
                        )
                    if "ipw" in estimators:
                        plan.append(
                            (
                                "ipw_stratum",
                                {
                                    "treat": treat_arm,
                                    "ctrl": ctrl_list,
                                    "mask": m_s,
                                    "targets": targets,
                                    "pi_t": pi_t,
                                    "Pi0": Pi0,
                                },
                            ),
                        )

        # Execute plan for each bootstrap draw
        K_total = len(family_names)
        thetastar = np.empty((K_total, self.B), dtype=np.float64)
        for b in range(self.B):
            wb = W[:, b]
            pos = 0
            for kind, info in plan:
                if kind == "ra_overall":
                    # Compute RA on the restricted sample defined by info['mask'] to avoid
                    # index-space mismatches between diff vectors and the global sample.
                    ate_b, att_b, atc_b, _ = _ra_pair_effects(
                        y=yv,
                        a=av,
                        treat_arm=info["treat"],
                        control_arms=info["ctrl"],
                        X=Xv,
                        S=Sv,
                        weights=wb,
                        rank_policy=self.rank_policy,
                        use_mask=info["mask"],
                    )
                    if "ate" in info["targets"]:
                        thetastar[pos, b] = ate_b
                        pos += 1
                    if "att" in info["targets"]:
                        thetastar[pos, b] = att_b
                        pos += 1
                    if "atc" in info["targets"]:
                        thetastar[pos, b] = atc_b
                        pos += 1

                elif kind == "ipw_overall":
                    ate_b, att_b, atc_b = _ipw_pair_effects(
                        y=yv,
                        a=av,
                        treat_arm=info["treat"],
                        control_arms=info["ctrl"],
                        pi_t=info["pi_t"],
                        Pi0=info["Pi0"],
                        use_mask=info["mask"],
                        w=wb,
                    )
                    if "ate" in info["targets"]:
                        thetastar[pos, b] = ate_b
                        pos += 1
                    if "att" in info["targets"]:
                        thetastar[pos, b] = att_b
                        pos += 1
                    if "atc" in info["targets"]:
                        thetastar[pos, b] = atc_b
                        pos += 1

                elif kind == "ra_stratum":
                    sel = strata_codes == info["s_code"]
                    ate_b, att_b, atc_b, _ = _ra_pair_effects(
                        y=yv[sel],
                        a=av[sel],
                        treat_arm=info["treat"],
                        control_arms=[
                            x for x in np.unique(av[sel]) if x != info["treat"]
                        ],
                        X=Xv[sel, :],
                        S=Sv[sel, :],
                        weights=wb[sel],
                        rank_policy=self.rank_policy,
                    )
                    if "ate" in info["targets"]:
                        thetastar[pos, b] = ate_b
                        pos += 1
                    if "att" in info["targets"]:
                        thetastar[pos, b] = att_b
                        pos += 1
                    if "atc" in info["targets"]:
                        thetastar[pos, b] = atc_b
                        pos += 1

                elif kind == "ipw_stratum":
                    ate_b, att_b, atc_b = _ipw_pair_effects(
                        y=yv,
                        a=av,
                        treat_arm=info["treat"],
                        control_arms=info["ctrl"],
                        pi_t=info["pi_t"],
                        Pi0=info["Pi0"],
                        use_mask=info["mask"],
                        w=wb,
                    )
                    if "ate" in info["targets"]:
                        thetastar[pos, b] = ate_b
                        pos += 1
                    if "att" in info["targets"]:
                        thetastar[pos, b] = att_b
                        pos += 1
                    if "atc" in info["targets"]:
                        thetastar[pos, b] = atc_b
                        pos += 1

            if pos != K_total:
                # internal consistency guard
                raise RuntimeError(
                    "Internal error: family size mismatch when filling thetastar.",
                )

        # Simultaneous (family-wise) uniform band (sup-t; B+1 correction). No p-values / criticals returned.
        lo, hi, se = _uniform_band_from_bootstrap(theta, thetastar, alpha=self.alpha)

        # Package results (values-only; no analytic SE, no p/critical values)
        boot_meta = {
            "origin": "multiplier",
            "dist": "Exp(1)",
            "by_stratum": bool(self.bootstrap_by_stratum),
        }
        model_info = {
            "Estimator": "RCT (multi-contrast) RA & IPW",
            "Alpha": self.alpha,
            "B": self.B,
            "Bootstrap": "positive multipliers (Exp(1))",
            "RankPolicy": self.rank_policy,
            "BootstrapByStratum": self.bootstrap_by_stratum,
            "NoAnalyticPValues": True,
            "NoAnalyticSE": True,
            "PairBootstrapProhibited": True,
            "FamilySize": int(theta.shape[0]),
            "ParamNames": list(family_names),
            "SE_Origin": "bootstrap",
        }

        extra = {
            "names": list(family_names),
            "theta_star_shape": thetastar.shape,
            "se_source": "bootstrap",
            "boot_meta": boot_meta,
        }
        # Include estimated PS artifact if present
        if estimated_P is not None:
            extra["ps_estimated_P_shape"] = estimated_P.shape
            extra["ps_levels"] = list(estimated_levels)
        if balance_payloads:
            sanitized: list[dict[str, Any]] = [
                {
                    "name": payload.get("name"),
                    "X": np.asarray(payload.get("X"), dtype=np.float64),
                    "group": np.asarray(payload.get("group"), dtype=int),
                    "w_before": np.asarray(payload.get("w_before"), dtype=np.float64),
                    "w_after": np.asarray(payload.get("w_after"), dtype=np.float64),
                    "covariate_names": list(payload.get("covariate_names", [])),
                    "metadata": payload.get("metadata", {}),
                }
                for payload in balance_payloads
            ]
            extra["balance_plot"] = {
                "payloads": sanitized,
                "default": sanitized[0]["name"] if sanitized else None,
            }

        # Convert arrays to Series with names as index
        import pandas as pd

        params_series = pd.Series(theta, index=family_names, name="params")
        se_series = pd.Series(se, index=family_names, name="se")
        lo_series = pd.Series(lo, index=family_names, name="lower")
        hi_series = pd.Series(hi, index=family_names, name="upper")

        self._results = EstimationResult(
            params=params_series,
            se=se_series,  # bootstrap-only
            bands={
                "uniform": {"lower": lo_series, "upper": hi_series, "alpha": self.alpha},
            },
            n_obs=int(n),
            model_info=model_info,
            extra=extra,
        )
        return self._results

    # Convenience wrapper to produce balance plots compatible with cobalt::love.plot
    def plot_balance(self, X, a, *, w_ipw=None, w_base=None, **kw):
        """Wrapper around the package balance_plot to match cobalt::love.plot style.

        Parameters
        ----------
        X : array-like or DataFrame
            Covariate matrix (n x k)
        a : array-like
            Treatment assignment vector
        w_ipw : array-like or None
            Adjusted weights (after IPW) to visualize as 'after' balance
        w_base : array-like or None
            Base weights to visualize as 'before' balance (defaults to equal weights)
        Other keyword args are forwarded to the underlying balance_plot.

        """
        from lineareg.output.plots import balance_plot

        return balance_plot(X, a, w_after=w_ipw, w_before=w_base, **kw)

    def fit_formula(  # noqa: PLR0913
        self,
        *,
        data,
        y_formula: str,
        a_col: str,
        ps_formula: str | None = None,
        contrasts: list | None = None,
        p: object = None,
    ):
        """Fit using formula syntax. y_formula is for outcome regression (RA);
        ps_formula is optional and used when p='estimate' (propensity score).

        Parameters
        ----------
        data : pandas.DataFrame
            Input dataset with a unique index.
        y_formula : str
            Outcome formula, e.g. 'y ~ x1 + x2 + fe(strata)'.
        ps_formula : optional str
            Propensity score formula, e.g. 'arm ~ z1 + z2 + fe(strata)'.
        contrasts : optional
            Contrast specification forwarded to fit().
        p : optional
            Propensity handling forwarded to fit() (e.g., 'estimate').

        """
        from lineareg.utils.formula import FormulaParser

        # Parse outcome formula and build RA design
        fp_y = FormulaParser(data=data)
        out_y = fp_y.parse(y_formula, enforce_drop=True)
        X_ra = np.asarray(out_y["X"], dtype=np.float64)
        y_vec = np.asarray(out_y["y"], dtype=np.float64).reshape(-1)
        fe_list = out_y.get("fe_codes_list", None)
        strata_codes = (
            fe_list[0] if (fe_list is not None and len(fe_list) > 0) else None
        )
        var_names = list(
            out_y.get("var_names", [f"x{j}" for j in range(X_ra.shape[1])]),
        )

        # Handle propensity formula if provided: set self.ps_X temporarily
        old_psX = getattr(self, "ps_X", None)
        try:
            if ps_formula is not None:
                fp_ps = FormulaParser(data=data)
                out_ps = fp_ps.parse(ps_formula, enforce_drop=True)
                X_ps = np.asarray(out_ps["X"], dtype=np.float64)
                self.ps_X = X_ps
            else:
                self.ps_X = None

            a_vec = np.asarray(data[a_col]).reshape(-1)
            return self.fit(
                y=y_vec,
                a=a_vec,
                X=X_ra,
                strata=strata_codes,
                var_names=var_names,
                p=p,
                contrasts=contrasts,
            )
        finally:
            # restore original ps_X
            self.ps_X = old_psX
