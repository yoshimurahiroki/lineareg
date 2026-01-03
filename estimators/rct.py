"""RCT estimators.

Implements Regression Adjustment (RA) and Hájek-IPW for randomized experiments
with stratification and simultaneous inference.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Union
from collections.abc import Sequence

import numpy as np
import scipy.linalg as sla

# Project-internal linear algebra and result container
from lineareg.core import (
    linalg as la,  # must provide la.solve(..., method="qr", rank_policy={'stata','r'})
    bootstrap as bt,
)
from lineareg.estimators.base import (
    EstimationResult,  # value-only container (no analytic p, criticals, etc.)
    BootConfig,
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
    """Coerce to a 1D array without forcing numeric dtype.

    In this module, treatment arms may be numeric or non-numeric (e.g., strings).
    We still enforce the absence of missing values.
    """
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D.")

    # Missingness / finiteness checks
    if a.dtype.kind in {"f"}:
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite.")
    elif a.dtype.kind in {"O"}:
        for v in a.tolist():
            if v is None:
                raise ValueError(f"{name} must not contain None.")
            if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                raise ValueError(f"{name} must not contain NaN/Inf.")
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
            if len(control_arms_set) != 1:
                raise ValueError(
                    "scalar p is only valid for a binary contrast with exactly one control arm; use dict probabilities for multi-arm or pooled-control contrasts.",
                )
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
                missing = [c for c in control_arms_set if c not in pdict]
                if missing:
                    raise ValueError(
                        f"p[{strata_codes[i]}] missing control arm probabilities for {missing!r}.",
                    )
                Pi0[i] = float(sum(float(pdict[c]) for c in control_arms_set))
            else:
                # global dict arm->prob
                if treat_arm not in p:
                    raise ValueError(f"p must include key for treat_arm={treat_arm!r}.")
                pi_t[i] = float(p[treat_arm])
                missing = [c for c in control_arms_set if c not in p]
                if missing:
                    raise ValueError(
                        f"p missing control arm probabilities for {missing!r}.",
                    )
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
            and pi_t[i] <= 1.0000000001
            and Pi0[i] <= 1.0000000001
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
            raise ValueError("sampling weights must be positive & finite.")
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
        classes = getattr(clf, "classes_", None)
        if classes is None:
            raise ValueError("ps_learner must expose classes_ after fit().")
        if set(classes.tolist()) != set(levels):
            raise ValueError(
                "ps_learner classes_ do not match observed treatment levels; cannot align predict_proba columns safely.",
            )
        if proba.shape[1] != len(classes):
            raise ValueError("predict_proba column count does not match classes_.")
        # Align columns to first-appearance level order.
        col_map = {cls: j for j, cls in enumerate(classes.tolist())}
        cols = [col_map[lv] for lv in levels]
        proba_aligned = proba[:, cols]
        if proba_aligned.shape[1] != G:
            raise ValueError(
                "predict_proba column count does not match number of treatment levels.",
            )
        P[:] = proba_aligned
        return P, levels

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.arange(n), A):
        clf = clone(ps_learner)
        clf.fit(X[tr, :], A[tr])
        proba = clf.predict_proba(X[te, :])
        classes = getattr(clf, "classes_", None)
        if classes is None:
            raise ValueError("ps_learner must expose classes_ after fit().")
        # Strict support: each training fold must contain all levels.
        if set(classes.tolist()) != set(levels):
            raise ValueError(
                "Cross-fit fold is missing at least one treatment level in training data; reduce k_folds or ensure each arm has enough observations.",
            )
        if proba.shape[1] != len(classes):
            raise ValueError("predict_proba column count does not match classes_.")
        col_map = {cls: j for j, cls in enumerate(classes.tolist())}
        cols = [col_map[lv] for lv in levels]
        proba_aligned = proba[:, cols]
        if proba_aligned.shape[1] != G:
            raise ValueError(
                "predict_proba column count does not match number of treatment levels.",
            )
        P[te, :] = proba_aligned
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
) -> dict[str, Any]:
    """Compute ATE, ATT, ATC for pair (treat_arm vs control_arms) using Hájek IPW on subset m=use_mask.
    π_t and Π0 are per-observation assign probs for treat and control mixture in this contrast.
    Returns a dictionary with point estimates and their corresponding Influence Functions (IFs).
    """
    m = use_mask
    n_obs = y.shape[0] # Total number of observations in the original sample

    # Initialize IFs to zeros for all observations, then fill for relevant ones
    psi_ate = np.zeros(n_obs, dtype=np.float64)
    psi_att = np.zeros(n_obs, dtype=np.float64)
    psi_atc = np.zeros(n_obs, dtype=np.float64)

    if m.sum() == 0:
        return {
            "ate": np.nan, "psi_ate": psi_ate,
            "att": np.nan, "psi_att": psi_att,
            "atc": np.nan, "psi_atc": psi_atc,
        }

    # w is sampling weights (default 1), not bootstrap multipliers
    if w is None:
        w_samp = np.ones_like(y)
    else:
        w_samp = np.asarray(w, dtype=np.float64)

    yv = y
    av = a

    # Treatment indicators for the full sample, masked later
    treat = av == treat_arm
    control = np.isin(av, list(control_arms))

    # Propensity weights, clipped to avoid division by zero
    pi_t_safe = np.clip(pi_t, 1e-12, np.inf)
    Pi0_safe = np.clip(Pi0, 1e-12, np.inf)

    # ATE
    # D1 and D0 are indicators for being in the treated/control group AND in the use_mask
    D1 = (treat & m).astype(float)
    D0 = (control & m).astype(float)

    # Denominators for Hajek estimator (sum of inverse probability weights)
    denom1 = np.sum(w_samp * D1 / pi_t_safe)
    denom0 = np.sum(w_samp * D0 / Pi0_safe)

    mu1 = np.sum(w_samp * D1 * yv / pi_t_safe) / denom1 if denom1 > 0 else np.nan
    mu0 = np.sum(w_samp * D0 * yv / Pi0_safe) / denom0 if denom0 > 0 else np.nan

    ate = mu1 - mu0 if (np.isfinite(mu1) and np.isfinite(mu0)) else np.nan

    # Influence Functions for ATE
    # psi_i(mu1) = (w_samp_i * D1_i / pi_t_i) * (Y_i - mu1) / (denom1 / n_obs)
    # The (denom / n_obs) term normalizes the sum of weights to average weight.
    # This ensures the IF sums to 0 and its variance is correctly scaled for the mean.

    # Only compute IF for observations within the use_mask
    if denom1 > 0:
        scale1 = denom1 / n_obs
        psi_mu1 = (w_samp * D1 / pi_t_safe) * (yv - mu1) / scale1
        psi_ate[m] += psi_mu1[m] # Add to the full-length IF vector
    else:
        psi_mu1 = np.zeros(n_obs) # No contribution if denom is zero

    if denom0 > 0:
        scale0 = denom0 / n_obs
        psi_mu0 = (w_samp * D0 / Pi0_safe) * (yv - mu0) / scale0
        psi_ate[m] -= psi_mu0[m] # Subtract from the full-length IF vector
    else:
        psi_mu0 = np.zeros(n_obs) # No contribution if denom is zero

    # ATT (Average Treatment effect on the Treated)
    # Y1 part: mean of treated outcomes
    denom1_att = np.sum(w_samp * D1)
    mu1_att = np.sum(w_samp * D1 * yv) / denom1_att if denom1_att > 0 else np.nan

    if denom1_att > 0:
        scale1_att = denom1_att / n_obs
        psi_mu1_att = (w_samp * D1) * (yv - mu1_att) / scale1_att
        psi_att[m] += psi_mu1_att[m]
    else:
        psi_mu1_att = np.zeros(n_obs)

    # Y0 part: reweighted control outcomes
    odds = pi_t_safe / Pi0_safe
    w0_att = w_samp * D0 * odds
    denom0_att = np.sum(w0_att)
    mu0_att = np.sum(w0_att * yv) / denom0_att if denom0_att > 0 else np.nan

    if denom0_att > 0:
        scale0_att = denom0_att / n_obs
        psi_mu0_att = w0_att * (yv - mu0_att) / scale0_att
        psi_att[m] -= psi_mu0_att[m]
    else:
        psi_mu0_att = np.zeros(n_obs)

    att = mu1_att - mu0_att if (np.isfinite(mu1_att) and np.isfinite(mu0_att)) else np.nan

    # ATC (Average Treatment effect on the Control)
    # Y0 part: mean of control outcomes
    denom0_atc = np.sum(w_samp * D0)
    mu0_atc = np.sum(w_samp * D0 * yv) / denom0_atc if denom0_atc > 0 else np.nan

    if denom0_atc > 0:
        scale0_atc = denom0_atc / n_obs
        psi_mu0_atc = (w_samp * D0) * (yv - mu0_atc) / scale0_atc
        psi_atc[m] -= psi_mu0_atc[m]
    else:
        psi_mu0_atc = np.zeros(n_obs)

    # Y1 part: reweighted treated outcomes
    inv_odds = Pi0_safe / pi_t_safe
    w1_atc = w_samp * D1 * inv_odds
    denom1_atc = np.sum(w1_atc)
    mu1_atc = np.sum(w1_atc * yv) / denom1_atc if denom1_atc > 0 else np.nan

    if denom1_atc > 0:
        scale1_atc = denom1_atc / n_obs
        psi_mu1_atc = w1_atc * (yv - mu1_atc) / scale1_atc
        psi_atc[m] += psi_mu1_atc[m]
    else:
        psi_mu1_atc = np.zeros(n_obs)

    atc = mu1_atc - mu0_atc if (np.isfinite(mu1_atc) and np.isfinite(mu0_atc)) else np.nan

    return {
        "ate": ate, "psi_ate": psi_ate,
        "att": att, "psi_att": psi_att,
        "atc": atc, "psi_atc": psi_atc,
    }


def _ipw_unit_weights(
    a: np.ndarray,
    treat_arm: Any,
    control_arms: Sequence[Any],
    pi_t: np.ndarray,
    Pi0: np.ndarray,
    use_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Construct per-unit IPW weights for balance diagnostics.

    Uses ATE-style Hájek weights (treated: 1/pi_t, control: 1/Pi0) on the eligible
    sample; zeros elsewhere.
    """
    n = int(a.shape[0])
    m = np.ones(n, dtype=bool) if use_mask is None else np.asarray(use_mask, dtype=bool)
    is_t = (a == treat_arm) & m
    is_c = np.isin(a, list(control_arms)) & m
    pi_t_safe = np.clip(np.asarray(pi_t, dtype=np.float64), 1e-12, np.inf)
    Pi0_safe = np.clip(np.asarray(Pi0, dtype=np.float64), 1e-12, np.inf)
    w = np.zeros(n, dtype=np.float64)
    w[is_t] = 1.0 / pi_t_safe[is_t]
    w[is_c] = 1.0 / Pi0_safe[is_c]
    return w


def _hajek_mean_and_bootstrap(
    y: np.ndarray,
    w: np.ndarray,
    W: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return (mu, mu_star) for Hájek mean with multiplier bootstrap.

    mu = sum(w*y)/sum(w)
    mu_star = mu + sum(w*(y-mu)*W_b)/sum(w)
    where W has shape (n,B) with mean-zero multipliers.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    denom = float(np.sum(w))
    if not np.isfinite(denom) or denom <= 0.0:
        return np.nan, np.full(W.shape[1], np.nan, dtype=np.float64)
    mu = float(np.dot(w, y) / denom)
    term = w * (y - mu)
    mu_star = mu + (term @ W) / denom
    return mu, mu_star


def _ipw_point_and_bootstrap(
    *,
    y: np.ndarray,
    a: np.ndarray,
    treat_arm: Any,
    control_arms: Sequence[Any],
    pi_t: np.ndarray,
    Pi0: np.ndarray,
    use_mask: np.ndarray,
    W: np.ndarray,
) -> dict[str, tuple[float, np.ndarray]]:
    """Compute (point, bootstrap) for ATE/ATT/ATC using Hájek IPW.

    Bootstrap is multiplier/wild: plug-in perturbation using mean-zero multipliers W.
    """
    m = np.asarray(use_mask, dtype=bool)
    is_t = (a == treat_arm) & m
    is_c = np.isin(a, list(control_arms)) & m
    pi_t_safe = np.clip(np.asarray(pi_t, dtype=np.float64), 1e-12, np.inf)
    Pi0_safe = np.clip(np.asarray(Pi0, dtype=np.float64), 1e-12, np.inf)

    # ATE: treated vs pooled control mixture
    wt = np.zeros_like(y, dtype=np.float64)
    wc = np.zeros_like(y, dtype=np.float64)
    wt[is_t] = 1.0 / pi_t_safe[is_t]
    wc[is_c] = 1.0 / Pi0_safe[is_c]
    mu1, mu1_star = _hajek_mean_and_bootstrap(y, wt, W)
    mu0, mu0_star = _hajek_mean_and_bootstrap(y, wc, W)
    ate = mu1 - mu0 if (np.isfinite(mu1) and np.isfinite(mu0)) else np.nan
    ate_star = mu1_star - mu0_star

    # ATT
    wt_att = is_t.astype(np.float64)
    mu1_att, mu1_att_star = _hajek_mean_and_bootstrap(y, wt_att, W)
    odds = pi_t_safe / Pi0_safe
    wc_att = np.zeros_like(y, dtype=np.float64)
    wc_att[is_c] = odds[is_c]
    mu0_att, mu0_att_star = _hajek_mean_and_bootstrap(y, wc_att, W)
    att = mu1_att - mu0_att if (np.isfinite(mu1_att) and np.isfinite(mu0_att)) else np.nan
    att_star = mu1_att_star - mu0_att_star

    # ATC
    wc_atc = is_c.astype(np.float64)
    mu0_atc, mu0_atc_star = _hajek_mean_and_bootstrap(y, wc_atc, W)
    inv_odds = Pi0_safe / pi_t_safe
    wt_atc = np.zeros_like(y, dtype=np.float64)
    wt_atc[is_t] = inv_odds[is_t]
    mu1_atc, mu1_atc_star = _hajek_mean_and_bootstrap(y, wt_atc, W)
    atc = mu1_atc - mu0_atc if (np.isfinite(mu1_atc) and np.isfinite(mu0_atc)) else np.nan
    atc_star = mu1_atc_star - mu0_atc_star

    return {
        "ate": (ate, ate_star),
        "att": (att, att_star),
        "atc": (atc, atc_star),
    }


def _ra_point_and_bootstrap(
    *,
    y: np.ndarray,
    a: np.ndarray,
    treat_arm: Any,
    control_arms: Sequence[Any],
    X: np.ndarray,
    S: np.ndarray,
    strata_codes: np.ndarray | None,
    use_mask: np.ndarray,
    W: np.ndarray,
    rank_policy: str,
) -> dict[str, tuple[float, np.ndarray]]:
    """Regression adjustment with no-extrapolation by-stratum and multiplier bootstrap.

    Implementation uses two separate OLS fits (treated and control) on Z=[1,S,X]
    restricted to strata with overlap (both treated and control observed).
    Bootstrap draws reuse the same design matrices and apply wild multipliers to
    residuals: y* = yhat + u ⊙ W.
    """
    control_arms_set = set(control_arms)
    m0 = (a == treat_arm) | np.isin(a, list(control_arms_set))
    m = np.asarray(use_mask, dtype=bool) & m0
    if not np.any(m):
        nanB = np.full(W.shape[1], np.nan, dtype=np.float64)
        return {"ate": (np.nan, nanB), "att": (np.nan, nanB), "atc": (np.nan, nanB)}

    # Enforce overlap by strata: drop strata without both groups.
    if strata_codes is not None:
        sc = np.asarray(strata_codes)
        codes_m = sc[m]
        d_m = (a[m] == treat_arm)
        uniq, inv = np.unique(codes_m, return_inverse=True)
        G = int(uniq.size)
        has_t = np.zeros(G, dtype=bool)
        has_c = np.zeros(G, dtype=bool)
        for g in range(G):
            hit = inv == g
            has_t[g] = bool(np.any(d_m & hit))
            has_c[g] = bool(np.any((~d_m) & hit))
        ok = has_t & has_c
        ok_local = ok[inv]
        idx_m = np.where(m)[0]
        ok_mask = np.zeros_like(m)
        ok_mask[idx_m[ok_local]] = True
    else:
        ok_mask = m

    if not np.any(ok_mask):
        nanB = np.full(W.shape[1], np.nan, dtype=np.float64)
        return {"ate": (np.nan, nanB), "att": (np.nan, nanB), "atc": (np.nan, nanB)}

    y_ok = y[ok_mask]
    a_ok = a[ok_mask]
    X_ok = X[ok_mask, :]
    S_ok = S[ok_mask, :]
    W_ok = W[ok_mask, :]

    d_ok = (a_ok == treat_arm)
    c_ok = np.isin(a_ok, list(control_arms_set))
    if int(np.sum(d_ok)) < 2 or int(np.sum(c_ok)) < 2:
        nanB = np.full(W.shape[1], np.nan, dtype=np.float64)
        return {"ate": (np.nan, nanB), "att": (np.nan, nanB), "atc": (np.nan, nanB)}

    const = np.ones((y_ok.shape[0], 1), dtype=np.float64)
    Z_ok = np.column_stack([const, S_ok, X_ok])

    # group-specific designs
    Z1 = Z_ok[d_ok, :]
    y1 = y_ok[d_ok].reshape(-1, 1)
    Z0 = Z_ok[c_ok, :]
    y0 = y_ok[c_ok].reshape(-1, 1)

    # base fits
    beta1 = la.solve(Z1, y1, method="qr", rank_policy=str(rank_policy).lower())
    beta0 = la.solve(Z0, y0, method="qr", rank_policy=str(rank_policy).lower())
    yhat1 = la.dot(Z1, beta1)
    yhat0 = la.dot(Z0, beta0)
    u1 = (y1 - yhat1).astype(np.float64, copy=False)
    u0 = (y0 - yhat0).astype(np.float64, copy=False)

    # bootstrap refits (vectorized)
    W1 = W_ok[d_ok, :]
    W0 = W_ok[c_ok, :]
    Y1_star = yhat1 + u1 * W1
    Y0_star = yhat0 + u0 * W0
    beta1_star = la.solve(Z1, Y1_star, method="qr", rank_policy=str(rank_policy).lower())
    beta0_star = la.solve(Z0, Y0_star, method="qr", rank_policy=str(rank_policy).lower())

    mu1 = la.dot(Z_ok, beta1)
    mu0 = la.dot(Z_ok, beta0)
    mu1_star = la.dot(Z_ok, beta1_star)
    mu0_star = la.dot(Z_ok, beta0_star)
    diff = (mu1 - mu0).reshape(-1)
    diff_star = (mu1_star - mu0_star)

    ate = float(np.mean(diff))
    ate_star = np.mean(diff_star, axis=0)
    att = float(np.mean(diff[d_ok])) if bool(np.any(d_ok)) else np.nan
    att_star = np.mean(diff_star[d_ok, :], axis=0) if bool(np.any(d_ok)) else np.full(W.shape[1], np.nan)
    atc = float(np.mean(diff[c_ok])) if bool(np.any(c_ok)) else np.nan
    atc_star = np.mean(diff_star[c_ok, :], axis=0) if bool(np.any(c_ok)) else np.full(W.shape[1], np.nan)

    return {
        "ate": (ate, np.asarray(ate_star, dtype=np.float64).reshape(-1)),
        "att": (att, np.asarray(att_star, dtype=np.float64).reshape(-1)),
        "atc": (atc, np.asarray(atc_star, dtype=np.float64).reshape(-1)),
    }


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
    """Stratified RCT estimator with uniform bands.

    Supports Regression Adjustment and IPW. Inference via wild/multiplier bootstrap.

    Parameters
    ----------
    alpha : float, default 0.05
        Family-wise noncoverage level (uniform band).
    rank_policy : {'stata', 'r'}, default 'stata'
        Rank decision policy forwarded to QR solver.
    seed : int, optional
        Random seed.
    """

    def __init__(  # noqa: PLR0913
        self,
        alpha: float = 0.05,
        rank_policy: str = "stata",
        seed: int | None = None,
        *,
        ps_learner: object | None = None,
        ps_X: MatrixLike | None = None, # Restored
        n_folds: int = 5,
        fold_seed: int | None = None,
        trim_ps: float = 0.0,
        trim_mode: str = "clip",
    ) -> None:
        self.alpha = float(alpha)
        # B and bootstrap_by_stratum are now handled by BootConfig
        self.rank_policy = str(rank_policy).lower()
        if self.rank_policy not in {"stata", "r"}:
            raise ValueError("rank_policy must be 'stata' or 'r'.")
        self.seed = seed
        # Propensity score estimation settings
        self.ps_learner = ps_learner
        self.ps_X = ps_X # Restored
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
        var_names: Sequence[str] | None = None,
        p: float | dict[Any, Any] | str | None = None,
        contrasts: list[dict[str, Any]] | None = None,
        boot: BootConfig | None = None,
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

            # Only auto-define the default contrast for canonical numeric/boolean two-arm designs.
            arms_list = arms.tolist()
            is_canonical_binary = set(arms_list) in ({0, 1}, {False, True})
            if not is_canonical_binary:
                raise ValueError(
                    "Default 'contrasts' is only defined for binary arms {0,1} (or booleans). "
                    "Provide 'contrasts' explicitly for non-numeric or non-canonical arm labels.",
                )

            treat_default = 1
            control_default = [0]
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

        # BootConfig setup
        if boot is None:
            boot = BootConfig(n_boot=999, dist="rademacher", seed=self.seed)
        elif getattr(boot, "seed", None) is None and self.seed is not None:
            boot = replace(boot, seed=self.seed)

        self.B = int(getattr(boot, "n_boot", 0))
        if self.B < 2:
            raise ValueError("BootConfig.n_boot must be >= 2 for inference.")

        # If strata are provided and caller did not specify clustering, default
        # to clustering by strata for finite-sample validity under stratification.
        if (
            (strata_codes is not None)
            and getattr(boot, "cluster_ids", None) is None
            and getattr(boot, "multiway_ids", None) is None
            and getattr(boot, "space_ids", None) is None
            and getattr(boot, "time_ids", None) is None
        ):
            boot = replace(boot, cluster_ids=strata_codes)

        # If p == 'estimate', perform cross-fit PS estimation once up-front and
        # apply trimming/clipping rules. If rows are dropped, we must also subset
        # any boot IDs and re-generate multipliers on the final sample.
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
                    # After clipping, renormalize rows to sum to 1.
                    rs = np.sum(P, axis=1, keepdims=True)
                    if np.any(~np.isfinite(rs)) or np.any(rs <= 0.0):
                        raise ValueError("Invalid propensity probabilities after clipping (row sums non-positive).")
                    P = P / rs
                else:
                    keep = np.all(
                        (self.trim_ps <= P) & (1.0 - self.trim_ps >= P), axis=1,
                    )
                    if not np.any(keep):
                        raise ValueError(
                            "All observations trimmed by trim_ps/drop settings.",
                        )

                    def _subset_ids(ids):
                        if ids is None:
                            return None
                        if isinstance(ids, (list, tuple)):
                            return [np.asarray(v)[keep] for v in ids]
                        return np.asarray(ids)[keep]

                    # synchronize primary arrays and boot IDs to kept rows
                    av = av[keep]
                    yv = yv[keep]
                    Xv = Xv[keep, :]
                    Sv = Sv[keep, :]
                    strata_codes = (
                        strata_codes[keep] if strata_codes is not None else None
                    )
                    P = P[keep, :]
                    # keep n consistent after trimming
                    n = av.shape[0]

                    # Subset any existing bootstrap IDs to match trimmed sample.
                    if getattr(boot, "cluster_ids", None) is not None:
                        boot = replace(boot, cluster_ids=_subset_ids(boot.cluster_ids))
                    if getattr(boot, "multiway_ids", None) is not None:
                        boot = replace(boot, multiway_ids=_subset_ids(boot.multiway_ids))
                    if getattr(boot, "space_ids", None) is not None:
                        boot = replace(boot, space_ids=_subset_ids(boot.space_ids))
                    if getattr(boot, "time_ids", None) is not None:
                        boot = replace(boot, time_ids=_subset_ids(boot.time_ids))
            estimated_P = P
            estimated_levels = levels

        # Generate multipliers on the final sample (after any trimming/drop).
        W_df, boot_log = boot.make_multipliers(n_obs=n)
        W = W_df.to_numpy()  # (n, B)

        # Build the family vector (names, theta) and the bootstrap replicates theta*
        family_names: list[str] = []
        theta_list: list[float] = []
        theta_star_rows: list[np.ndarray] = []
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
                unknown = [s for s in allowed if (s not in strata_map and not isinstance(s, (int, np.integer)))]
                if unknown:
                    raise ValueError(
                        f"Unknown strata labels in strata_subset: {unknown!r}."
                    )
                allowed_codes_list: list[int] = []
                for s in allowed:
                    if s in strata_map:
                        allowed_codes_list.append(int(strata_map[s]))
                    elif isinstance(s, (int, np.integer)):
                        allowed_codes_list.append(int(s))
                allowed_codes = np.array(allowed_codes_list, dtype=int)
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
                p_use = p
                # If by-stratum probabilities are keyed by original strata labels,
                # translate those keys to the internal integer codes.
                if (
                    isinstance(p, dict)
                    and strata_codes is not None
                    and any((k in strata_map) and isinstance(p[k], dict) for k in p)
                ):
                    p_by_code: dict[Any, Any] = {}
                    for k, v in p.items():
                        if k in strata_map and isinstance(v, dict):
                            p_by_code[int(strata_map[k])] = v
                        else:
                            p_by_code[k] = v
                    p_use = p_by_code

                pi_t, Pi0, use_mask_all = _resolve_pi_for_contrast(
                    av, strata_codes, p_use, treat_arm, control_arms,
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
                ra_out = _ra_point_and_bootstrap(
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
                    strata_codes=strata_codes,
                    use_mask=overall_mask,
                    W=W,
                    rank_policy=self.rank_policy,
                )
                if "ate" in targets:
                    val, star = ra_out["ate"]
                    push_one("ra", "ate", val)
                    theta_star_rows.append(star)
                if "att" in targets:
                    val, star = ra_out["att"]
                    push_one("ra", "att", val)
                    theta_star_rows.append(star)
                if "atc" in targets:
                    val, star = ra_out["atc"]
                    push_one("ra", "atc", val)
                    theta_star_rows.append(star)

            # IPW
            if "ipw" in estimators:
                if p is None:
                    raise ValueError(
                        "IPW requested but 'p' (assignment probabilities) is missing.",
                    )
                ipw_out = _ipw_point_and_bootstrap(
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
                    W=W,
                )
                if "ate" in targets:
                    val, star = ipw_out["ate"]
                    push_one("ipw", "ate", val)
                    theta_star_rows.append(star)
                if "att" in targets:
                    val, star = ipw_out["att"]
                    push_one("ipw", "att", val)
                    theta_star_rows.append(star)
                if "atc" in targets:
                    val, star = ipw_out["atc"]
                    push_one("ipw", "atc", val)
                    theta_star_rows.append(star)
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
                        ra_out_s = _ra_point_and_bootstrap(
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
                            strata_codes=strata_codes,
                            use_mask=m_s,
                            W=W,
                            rank_policy=self.rank_policy,
                        )
                        if "ate" in targets:
                            val, star = ra_out_s["ate"]
                            push_one("ra", "ate", val, tag)
                            theta_star_rows.append(star)
                        if "att" in targets:
                            val, star = ra_out_s["att"]
                            push_one("ra", "att", val, tag)
                            theta_star_rows.append(star)
                        if "atc" in targets:
                            val, star = ra_out_s["atc"]
                            push_one("ra", "atc", val, tag)
                            theta_star_rows.append(star)
                    if "ipw" in estimators:
                        if p is None:
                            raise ValueError("IPW requested but 'p' is missing.")
                        ipw_out_s = _ipw_point_and_bootstrap(
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
                            W=W,
                        )
                        if "ate" in targets:
                            val, star = ipw_out_s["ate"]
                            push_one("ipw", "ate", val, tag)
                            theta_star_rows.append(star)
                        if "att" in targets:
                            val, star = ipw_out_s["att"]
                            push_one("ipw", "att", val, tag)
                            theta_star_rows.append(star)
                        if "atc" in targets:
                            val, star = ipw_out_s["atc"]
                            push_one("ipw", "atc", val, tag)
                            theta_star_rows.append(star)
        theta = np.array(theta_list, dtype=np.float64)
        if len(theta_star_rows) != len(theta_list):
            raise RuntimeError(
                "Internal error: bootstrap replicate rows do not match theta length.",
            )
        thetastar = np.vstack([row.reshape(1, -1) for row in theta_star_rows])

        # Simultaneous (family-wise) uniform band (sup-t; B+1 correction). No p-values / criticals returned.
        lo, hi, se = _uniform_band_from_bootstrap(theta, thetastar, alpha=self.alpha)

        # Package results (values-only; no analytic SE, no p/critical values)
        boot_meta = {
            "origin": "multipliers",
            "dist": getattr(boot, "dist", None),
            "by_stratum": bool(strata_codes is not None),
            "multipliers_log": boot_log,
        }
        model_info = {
            "Estimator": "RCT (multi-contrast) RA & IPW",
            "Alpha": self.alpha,
            "B": self.B,
            "Bootstrap": f"Wild Bootstrap ({boot.dist})",
            "RankPolicy": self.rank_policy,
            "BootstrapByStratum": (strata_codes is not None),

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
        extra["boot_config"] = boot
        extra["W_multipliers_inference"] = W_df
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


def prefix_from_task(task: str, kw: dict, s_tag: str = "") -> str:
    """Helper to generate consistent parameter names from task info."""
    t = kw["treat"]
    c = kw.get("ctrl", [])
    c_str = "+".join(map(str, c)) if c else "rest"
    mask_prefix = f"{t}_vs_{c_str}"
    return f"{mask_prefix}{s_tag}"
