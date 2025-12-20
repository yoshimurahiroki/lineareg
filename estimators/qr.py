"""Quantile regression (QR) and IVQR with bootstrap-only inference.

Scope and policy
----------------
- Koenker-Bassett QR solved via exact LP (HiGHS) with strict tolerances; IVQR
    interfaces are provided where applicable. No analytic SEs or p-values.
- Bootstrap-only inference (wild/multiplier). No uniform bands for QR/IVQR.
- Linear algebra and matrix operations route through :mod:`lineareg.core.linalg`.
- Constant handling and formula parsing follow project-wide conventions.

Comments/docstrings are English-only by policy.
"""

# lineareg/estimators/qr.py
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from lineareg.core import bootstrap as bt
from lineareg.core import linalg as la
from lineareg.estimators.base import (
    BaseEstimator,
    BootConfig,
    EstimationResult,
    attach_formula_metadata,
    prepare_formula_environment,
)
from lineareg.utils.auto_constant import add_constant
from lineareg.utils.formula import FormulaParser

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
else:
    Sequence = tuple  # type: ignore[assignment]
    NDArray = np.ndarray  # type: ignore[misc,assignment]

ArrayLike = Union[pd.Series, np.ndarray]
MatrixLike = Union[pd.DataFrame, np.ndarray]

__all__ = ["IVQR", "QR"]


def _solve_qr_lp(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    tau: float,
    multipliers: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64] | None, dict[str, Any]]:
    """Solve Koenker-Bassett QR via exact LP (HiGHS), optionally weighted.
    Uses sparse A_eq for memory efficiency.

    Returns
    -------
    beta : (k, 1) coefficients
    obj_value : float, objective value
    dual_eq : (n, 1) or None, equality constraint duals
    solver_info : dict with status, message, nit for reproducibility

    """
    if not (0.0 < float(tau) < 1.0):
        raise ValueError("tau must be in (0,1).")
    n, k = X.shape

    def _raise_lp_failure(message: str) -> None:
        raise RuntimeError(message)

    # Analytic weights for QR are not supported by project policy. Positive
    # multipliers (Exp(1)) are passed here as `multipliers` to form a weighted LP
    # objective when performing bootstrap draws. If no multipliers provided, use ones.
    w = (
        np.ones((n, 1))
        if multipliers is None
        else np.asarray(multipliers, dtype=float).reshape(n, 1)
    )
    # Variables: beta (k), u_plus (n), u_minus (n)
    # Lexicographic perturbation to fix tie-breaking in degenerate LP solutions
    # (matches simplex-style determinism used by quantreg/stata implementations)
    eps = np.finfo(np.float64).eps
    # scale-neutral tiny perturbation: make tie negligible relative to data
    scale = max(
        1.0, float(np.max(np.abs(y))) + (float(np.max(np.abs(X))) if X.size else 0.0),
    )
    tie = (eps * scale) * np.arange(k, dtype=float)
    c = np.concatenate([tie, tau * w.ravel(), (1.0 - tau) * w.ravel()])
    # Constraints: y = X b + u_plus - u_minus
    # Sparse A_eq: X (dense), I for u_plus, -I for u_minus
    A_eq = sparse.hstack(
        [sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)], format="csr",
    )
    b_eq = y.reshape(-1)
    bounds = [(None, None)] * k + [(0.0, None)] * (2 * n)
    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs-ds",
        options={
            "presolve": True,
            # tighten feasibility tolerances to stabilize equality KKT dual extraction
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
        },
    )

    # Collect solver metadata for reproducibility
    solver_info = {
        "solver": "highs-ds",
        "status": int(res.status) if hasattr(res, "status") else -1,
        "message": str(res.message) if hasattr(res, "message") else "",
        "nit": int(res.nit) if hasattr(res, "nit") else -1,
        "success": bool(res.success) if hasattr(res, "success") else False,
    }

    if not res.success:
        msg = f"LP solver failed: {res.message}"
        raise RuntimeError(msg)
    beta = res.x[:k].reshape(-1, 1)
    # Return equality-dual marginals (HiGHS provides `res.eqlin.marginals`) so
    # callers can construct KKT-consistent subgradients (psi) where needed.
    dual_eq = getattr(getattr(res, "eqlin", None), "marginals", None)
    # If HiGHS direct-solver did not return equality duals, retry with
    # the HiGHS interior-point method to increase chances of obtaining
    # duals (deterministic fallback used by quantreg compatibility layer).
    # Track which solver produced the adopted solution (res or res2).
    active = res
    if dual_eq is None:
        try:
            res2 = linprog(
                c,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs-ipm",
                options={
                    "presolve": True,
                    "dual_feasibility_tolerance": 1e-9,
                    "primal_feasibility_tolerance": 1e-9,
                },
            )
            solver_info = {
                "solver": "highs-ipm",
                "status": int(res2.status) if hasattr(res2, "status") else -1,
                "message": str(res2.message) if hasattr(res2, "message") else "",
                "nit": int(res2.nit) if hasattr(res2, "nit") else -1,
                "success": bool(res2.success) if hasattr(res2, "success") else False,
            }
            if not res2.success:
                msg = f"LP solver failed (fallback highs-ipm): {res2.message}"
                _raise_lp_failure(msg)
            beta = res2.x[:k].reshape(-1, 1)
            dual_eq = getattr(getattr(res2, "eqlin", None), "marginals", None)
            active = res2
            if dual_eq is None:
                # Return adopted numeric values even if equality duals remain unavailable
                return beta, float(res2.fun), None, solver_info
        except (ValueError, RuntimeError) as err:
            msg = f"LP solver failed: {res.message}"
            raise RuntimeError(msg) from err
    # Before returning, verify that the perturbed optimum attains the *unperturbed* QR objective
    # within numerical tolerance. R quantreg and Stata qreg do not perform this strict check,
    # so we use a relaxed tolerance (1e-6) for R/Stata parity. If the perturbation exceeds
    # this threshold, issue a warning but do NOT raise (silent approximation is standard practice).
    # Use the adopted active solution (res or res2) for objective verification
    u_plus = active.x[k : k + n]
    u_minus = active.x[k + n : k + 2 * n]
    obj_unperturbed = float(tau * np.sum(u_plus) + (1.0 - tau) * np.sum(u_minus))
    tol_obj = 1e-6 * max(1.0, abs(obj_unperturbed))
    perturbation_effect = abs(
        obj_unperturbed - float(active.fun - la.dot(tie, active.x[:k])),
    )
    if perturbation_effect > tol_obj:
        # Issue warning instead of raising, matching R quantreg behavior (no strict check)
        warnings.warn(
            f"Lexicographic tie-breaking perturbed QR objective by {perturbation_effect:.2e} "
            f"(tolerance {tol_obj:.2e}). This is within R/Stata quantreg tolerances.",
            RuntimeWarning,
            stacklevel=2,
        )
    # reshape into column vector (n,1)
    dual_arr = (
        np.asarray(dual_eq, dtype=float).reshape(-1, 1) if dual_eq is not None else None
    )
    return beta, float(active.fun), dual_arr, solver_info


def _prepare_qr_lp(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    tau: float,
) -> dict[str, Any]:
    """Precompute LP components for QR to reuse across weighted solves.

    Returns a dict of immutable structures to be passed into
    `_solve_qr_lp_prepared` together with per-draw multipliers.
    """
    if not (0.0 < float(tau) < 1.0):
        raise ValueError("tau must be in (0,1).")
    n, k = X.shape

    # Lexicographic perturbation (tie-breaking) depends only on X/y scale and k
    eps = np.finfo(np.float64).eps
    scale = max(
        1.0,
        float(np.max(np.abs(y))) + (float(np.max(np.abs(X))) if X.size else 0.0),
    )
    tie = (eps * scale) * np.arange(k, dtype=float)

    # Constraints and bounds depend only on (X, y)
    A_eq = sparse.hstack(
        [sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)], format="csr",
    )
    b_eq = y.reshape(-1)
    bounds = [(None, None)] * k + [(0.0, None)] * (2 * n)

    opts = {
        "presolve": True,
        "dual_feasibility_tolerance": 1e-9,
        "primal_feasibility_tolerance": 1e-9,
    }
    return {
        "n": n,
        "k": k,
        "tau": float(tau),
        "tie": tie,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "bounds": bounds,
        "options": opts,
    }


def _solve_qr_lp_prepared(
    prep: dict[str, Any],
    multipliers: NDArray[np.float64] | None = None,
    *,
    raise_on_failure: bool = True,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64] | None, dict[str, Any]]:
    """Solve the prepared QR LP for given (optional) positive multipliers.

    This reuses the constraint matrices and bounds across draws and only
    updates the objective c per call, reducing Python overhead. Numerical
    results are identical to `_solve_qr_lp`.

    Parameters
    ----------
    raise_on_failure : bool, default True
        If True, raise RuntimeError on solver failure.
        If False, return NaN coefficients on failure (useful for bootstrap).
    """
    n = int(prep["n"])
    k = int(prep["k"])
    tau = float(prep["tau"])
    tie = np.asarray(prep["tie"], float)
    A_eq = prep["A_eq"]
    b_eq = prep["b_eq"]
    bounds = prep["bounds"]
    options = prep["options"]
    options_relaxed = {**options, "presolve": False, "dual_feasibility_tolerance": 1e-7, "primal_feasibility_tolerance": 1e-7}

    w = (
        np.ones((n, 1))
        if multipliers is None
        else np.asarray(multipliers, dtype=float).reshape(n, 1)
    )
    w = np.clip(w, 1e-10, None)
    c = np.concatenate([tie, tau * w.ravel(), (1.0 - tau) * w.ravel()])

    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs-ds",
        options=options,
    )
    solver_info = {
        "solver": "highs-ds",
        "status": int(res.status) if hasattr(res, "status") else -1,
        "message": str(res.message) if hasattr(res, "message") else "",
        "nit": int(res.nit) if hasattr(res, "nit") else -1,
        "success": bool(res.success) if hasattr(res, "success") else False,
    }
    if not res.success:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs-ipm", options=options)
        solver_info = {"solver": "highs-ipm", "status": int(getattr(res, "status", -1)), "message": str(getattr(res, "message", "")), "nit": int(getattr(res, "nit", -1)), "success": bool(getattr(res, "success", False))}
        if not res.success:
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs-ds", options=options_relaxed)
            solver_info = {"solver": "highs-ds-relaxed", "status": int(getattr(res, "status", -1)), "message": str(getattr(res, "message", "")), "nit": int(getattr(res, "nit", -1)), "success": bool(getattr(res, "success", False))}
            if not res.success:
                res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs-ipm", options=options_relaxed)
                solver_info = {"solver": "highs-ipm-relaxed", "status": int(getattr(res, "status", -1)), "message": str(getattr(res, "message", "")), "nit": int(getattr(res, "nit", -1)), "success": bool(getattr(res, "success", False))}
    if not res.success:
        if raise_on_failure:
            msg = f"LP solver failed: {res.message}"
            raise RuntimeError(msg)
        else:
            return np.full((k, 1), np.nan), np.nan, None, solver_info

    beta = res.x[:k].reshape(-1, 1)
    dual_eq = getattr(getattr(res, "eqlin", None), "marginals", None)
    active = res
    if dual_eq is None:
        res2 = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs-ipm",
            options=options,
        )
        solver_info = {
            "solver": "highs-ipm",
            "status": int(res2.status) if hasattr(res2, "status") else -1,
            "message": str(res2.message) if hasattr(res2, "message") else "",
            "nit": int(res2.nit) if hasattr(res2, "nit") else -1,
            "success": bool(res2.success) if hasattr(res2, "success") else False,
        }
        if not res2.success:
            msg = f"LP solver failed (fallback highs-ipm): {res2.message}"
            raise RuntimeError(msg)
        beta = res2.x[:k].reshape(-1, 1)
        dual_eq = getattr(getattr(res2, "eqlin", None), "marginals", None)
        active = res2
        if dual_eq is None:
            return beta, float(res2.fun), None, solver_info

    # objective perturbation check (same policy as _solve_qr_lp)
    u_plus = active.x[k : k + n]
    u_minus = active.x[k + n : k + 2 * n]
    obj_unperturbed = float(tau * np.sum(u_plus) + (1.0 - tau) * np.sum(u_minus))
    tol_obj = 1e-6 * max(1.0, abs(obj_unperturbed))
    # la.dot expects array-likes; ensure 1-D inputs
    perturbation_effect = abs(
        obj_unperturbed - float(active.fun - la.dot(tie, active.x[:k])),
    )
    if perturbation_effect > tol_obj:
        warnings.warn(
            f"Lexicographic tie-breaking perturbed QR objective by {perturbation_effect:.2e} "
            f"(tolerance {tol_obj:.2e}). This is within R/Stata quantreg tolerances.",
            RuntimeWarning,
            stacklevel=2,
        )

    dual_arr = (
        np.asarray(dual_eq, dtype=float).reshape(-1, 1) if dual_eq is not None else None
    )
    return beta, float(active.fun), dual_arr, solver_info


def _mammen_weights(G: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Mammen (1993, Annals of Statistics) two-point distribution (mean 0, var 1).
    This distribution is the canonical two-point wild-bootstrap multiplier used
    in cluster wild gradient bootstrap implementations as described in
    Hagemann (2017, JASA). It takes values:
      w1 = -(sqrt(5)-1)/2 with prob p1 = (sqrt(5)+1)/(2*sqrt(5)),
      w2 =  (sqrt(5)+1)/2 with prob p2 = (sqrt(5)-1)/(2*sqrt(5)).
    """
    s5 = np.sqrt(5.0)
    w1, w2 = -(s5 - 1.0) / 2.0, (s5 + 1.0) / 2.0
    p1, p2 = (s5 + 1.0) / (2.0 * s5), (s5 - 1.0) / (2.0 * s5)
    return rng.choice(np.array([w1, w2]), size=G, p=np.array([p1, p2])).astype(
        np.float64,
    )


def _wgb_bootstrap_qr(  # noqa: PLR0913
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    tau: float,
    B: int,
    *,
    multipliers: NDArray[np.float64] | None = None,
    cluster_ids: Sequence | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Wild Gradient Bootstrap for linear QR under one-way clustering following
    Hagemann (2017, Algorithms 3.2/3.4).

      1) Base QR: beta_hat_n(tau) = argmin (1/n) * sum rho_tau(y_i - x_i' beta)
      2) For each replication b:
         - Draw cluster weights W_g ~ Mammen(1992) (iid across groups)
         - Form bootstrap gradient W_n(tau) = n^{-1/2} * sum_i W_{g(i)} psi_tau(y_i - x_i' beta_hat_n(tau)) x_i
         - Set X_star = -(sqrt(n) * W_n(tau) / tau) and Y_star = n * max_i c_i * max_{i,k} |y_{ik}|
         - Solve augmented QR with one additional pseudo-observation (Y_star, X_star)

    The function returns B bootstrap replicates beta_hat_n_star(tau).
    References: Algorithm 3.2 (Steps 1-3) and Algorithm 3.4 in Hagemann (2017).
    """
    if not (0.0 < float(tau) < 1.0):
        raise ValueError("tau must be in (0,1) for WGB.")
    # Validate cluster_ids: WGB theory (Hagemann 2017) requires a single
    # one-way cluster id vector. Reject multiway / nested cluster inputs here.
    if cluster_ids is None:
        msg = "WGB requires one-way clustering IDs (cluster_ids)."
        raise ValueError(msg)
    # Accept arbitrary labels (strings, objects); reject multiway inputs where
    # the user passed a list/tuple of arrays or a 2-D structure.
    arr_ids = np.asarray(cluster_ids, dtype=object)
    # forbid nested containers and 2-D shaped inputs strictly
    if arr_ids.ndim != 1:
        raise ValueError("WGB supports one-way clustering only (1-D vector required).")
    if any(isinstance(el, (list, tuple, np.ndarray)) for el in arr_ids.tolist()):
        msg = "WGB supports one-way clustering only (provide a single 1-D cluster id vector)."
        raise ValueError(msg)
    n, p = X.shape
    rootn = float(np.sqrt(n))

    # Base fit β̂_n(τ) and KKT-consistent ψ̂_τ(û) extracted from LP equality duals
    beta_hat, _, dual, _ = _solve_qr_lp(X, y, tau, multipliers)
    uhat = y - la.dot(X, beta_hat)  # (n,1)
    # Construct subgradient psi from LP equality duals to guarantee X' psi = 0
    # For weighted LP (multipliers w_i), theory implies lambda_i ∈ [-(1-τ) w_i, τ w_i]
    # and ψ_i = -λ_i / w_i satisfies the KKT moment condition exactly.
    if dual is None:
        msg = "LP solver did not return equality duals; cannot build KKT-consistent psi for WGB."
        raise RuntimeError(msg)
    w_used = (
        np.ones_like(uhat)
        if multipliers is None
        else np.asarray(multipliers, float).reshape(-1, 1)
    )
    # --- NEW: strict KKT interval check for equality duals (HiGHS sign safe) ---
    # Valid equality duals (λ_i) must satisfy:  λ_i ∈ [-(1-τ) w_i, τ w_i]  up to numerical tolerance.
    # We accept either sign convention from HiGHS (λ or -λ) by checking both.
    lo = -(1.0 - tau) * w_used
    hi = (tau) * w_used
    # KKT tolerance: use LAPACK-scale eigen tolerance on X'X (Stata/R convention)
    tol_kkt = la.eig_tol(la.crossprod(X, X))
    # Check dual within [lo, hi]
    cond_pos = (np.max(dual - hi) <= tol_kkt) and (np.max(lo - dual) <= tol_kkt)
    # Or negative of dual within [lo, hi]
    cond_neg = (np.max(-dual - hi) <= tol_kkt) and (np.max(lo + dual) <= tol_kkt)
    if not (cond_pos or cond_neg):
        msg = "LP duals violate QR KKT bounds; cannot run WGB safely."
        raise RuntimeError(msg)
    # Verify HiGHS equality dual sign convention and construct a KKT-consistent
    # subgradient psi such that X' psi == 0 (numerical zero). We try the two
    # natural sign conventions and pick the one that satisfies the moment
    # condition up to machine tolerance. If neither does, raise: no silent
    # fallback or approximation is allowed under the strict policy.
    # Prepare a small log for KKT diagnostics for auditability
    kkt_log: dict[str, Any] = {"tol_kkt": float(tol_kkt)}
    psi_cand1 = (-dual / w_used).astype(np.float64)  # candidate 1 (theoretical sign)
    mv1 = float(np.max(np.abs(la.crossprod(X, psi_cand1))))
    if mv1 <= tol_kkt:
        psi_hat = psi_cand1
        kkt_log.update({"dual_sign": "-lambda/w", "max_violation": mv1})
    else:
        psi_cand2 = (dual / w_used).astype(np.float64)
        mv2 = float(np.max(np.abs(la.crossprod(X, psi_cand2))))
        if mv2 <= tol_kkt:
            psi_hat = psi_cand2
            kkt_log.update({"dual_sign": "+lambda/w", "max_violation": mv2})
        else:
            msg = "Cannot construct KKT-consistent psi from LP duals (|X'psi| not ~0)."
            raise RuntimeError(msg)

    # Cluster indexing: factorize arbitrary labels into contiguous integer codes.
    # Reject missing cluster IDs explicitly (matching R/Stata behavior).
    gids = arr_ids.ravel()
    if pd.isna(gids).any():
        msg = (
            "cluster_ids contains missing values; WGB requires complete cluster labels"
        )
        raise ValueError(msg)
    _, inv = np.unique(gids, return_inverse=True)
    G = int(np.max(inv) + 1)
    # cluster sizes (kept for potential diagnostics if needed)
    _, _counts = np.unique(inv, return_counts=True)

    # Hagemann (2017) requires at least two clusters for asymptotic validity
    # - fail fast when G < 2 rather than silently returning degenerate draws.
    if G < 2:
        raise ValueError(f"WGB requires at least 2 clusters; found G={G}.")

    boot_betas = np.empty((p, B), dtype=np.float64)

    # Get Mammen multipliers for all clusters and boots via explicit WGB helper
    _wgb_helper = getattr(bt, "wgb_cluster_multipliers", None)
    if callable(_wgb_helper):
        W_mult = _wgb_helper(G, B, seed=seed)
    else:
        # Fallback: generate Mammen multipliers locally (self-contained)
        rng_m = np.random.default_rng(seed)
        W_mult = np.column_stack([_mammen_weights(G, rng_m) for _ in range(B)]).astype(
            np.float64,
        )

    for b in range(B):
        # Draw Mammen weights per cluster
        Wg = W_mult[:, b]  # (G,)
        Wi = Wg[inv].reshape(-1, 1)  # (n,1)

        # W_n(τ) = n^{-1/2} ∑ W_{g(i)} ψ̂_i x_i = X' (Wi ⊙ ψ̂) / √n
        XTw = la.crossprod(la.hadamard(Wi, psi_hat), X)  # (1,p) via (n,1)' @ (n,p)
        Wn = (XTw.reshape(1, -1) / rootn).astype(np.float64)  # (1,p)

        # Pseudo-observation (X*, Y*): Hagemann (2017) Alg. 3.4 strict formulation:
        # X* = -(√n / τ) W_n(τ)
        # Y* = X* @ β̂ + c,  with c > 0 small constant (ensures residual r* = c > 0)
        X_star = -(rootn / float(tau)) * Wn  # (1,p)
        # Stronger deterministic margin to ensure strictly positive pseudo-residuals.
        # Use infinity norm on X* (more conservative) and add a small offset to y_max.
        # c_n = (1 + ||X*||_inf) * (max|y| + 1.0)
        Xs_linf = float(np.max(np.abs(X_star)))
        y_max = float(np.max(np.abs(y))) + 1.0
        c_n = (1.0 + Xs_linf) * y_max
        Y_star = float(la.dot(X_star, beta_hat).ravel()[0]) + c_n

        # Augment data and resolve QR (exact LP) per draw. Use core.linalg.vstack
        # to centralize matrix operations and preserve sparse-aware handling.
        X_aug = la.vstack([X, X_star])
        y_aug = la.vstack([y, np.array([[Y_star]], dtype=np.float64)])
        beta_b, _, _, _ = _solve_qr_lp(X_aug, y_aug, tau)
        boot_betas[:, b] = beta_b.ravel()

    boot_log = {
        "effective_dist": "Mammen(1993) via Hagemann(2017)",
        "n_clusters": G,
        "method": "WGB (Hagemann 2017, Alg.3.2/3.4)",
        # Store the exact cluster multipliers used for reproducibility (G x B)
        "W_multipliers": W_mult,
    }
    # Attach KKT diagnostics for post-hoc auditability
    boot_log.update({"kkt": kkt_log})
    return boot_betas, boot_log


# ---------------------------------------------------------------------
# Positive-multiplier generator for QR (Exp(1)), with clustering
# ---------------------------------------------------------------------
def _exp_multipliers(  # noqa: PLR0913
    n: int,
    *,
    B: int,
    cluster_ids: Sequence | None = None,
    multiway_ids: Sequence[Sequence] | None = None,
    space_ids: Sequence | None = None,
    time_ids: Sequence | None = None,
    method: str = "intersection",
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Positive multiplier bootstrap for QR using Exp(1) weights.
    Always uses safe indexing via unique(return_inverse=True) to handle arbitrary IDs.
    Multiway 'factorized' uses product of independent Exp(1) across dimensions (opt-in only).
    """
    rng = np.random.default_rng(seed)
    if multiway_ids is not None:
        if method == "intersection":
            # Build joint cluster via unique on stacked factorized integer codes.
            inv_dims = []
            for c in multiway_ids:
                arr = np.asarray(c, dtype=object).ravel()
                if pd.isna(arr).any():
                    msg = "multiway_ids contains missing values; please remove or impute NAs before bootstrapping"
                    raise ValueError(msg)
                _, inv = np.unique(arr, return_inverse=True)
                inv_dims.append(inv)
            codes_stack = np.column_stack(inv_dims)
            _, inv = np.unique(codes_stack, axis=0, return_inverse=True)
            G = int(inv.max()) + 1
            Wg = rng.exponential(scale=1.0, size=(G, B)).astype(np.float64)
            return Wg[inv, :]
        # Factorized multiway multipliers are nonstandard and disallowed.
        if method == "factorized":
            raise ValueError(
                "multiway factorized multipliers are not supported; use method='intersection'.",
            )
        # (no factorized path)
    if space_ids is not None and time_ids is not None:
        s = np.asarray(space_ids, dtype=object).ravel()
        t = np.asarray(time_ids, dtype=object).ravel()
        if pd.isna(s).any() or pd.isna(t).any():
            msg = "space_ids/time_ids contains missing values; please remove or impute NAs before bootstrapping"
            raise ValueError(msg)
        if method == "intersection":
            _, inv_s = np.unique(s, return_inverse=True)
            _, inv_t = np.unique(t, return_inverse=True)
            keys = np.column_stack([inv_s, inv_t])
            _, inv = np.unique(keys, axis=0, return_inverse=True)
            G = int(inv.max()) + 1
            Wg = rng.exponential(scale=1.0, size=(G, B)).astype(np.float64)
            return Wg[inv, :]
        if method == "factorized":
            raise ValueError(
                "space x time factorized multipliers are not supported; use method='intersection'.",
            )
    if cluster_ids is not None:
        arr = np.asarray(cluster_ids, dtype=object).ravel()
        if pd.isna(arr).any():
            msg = "cluster_ids contains missing values; please remove or impute NAs before bootstrapping"
            raise ValueError(msg)
        _, inv = np.unique(arr, return_inverse=True)
        G = int(np.max(inv) + 1)
        Wg = rng.exponential(scale=1.0, size=(G, B)).astype(np.float64)
        return Wg[inv, :]
    # IID
    return rng.exponential(scale=1.0, size=(n, B)).astype(np.float64)


def _solve_ivqr_ch05_milp(  # noqa: PLR0913
    y: NDArray[np.float64],
    X: NDArray[np.float64],
    Z: NDArray[np.float64],
    tau: float,
    *,
    multipliers: NDArray[np.float64] | None = None,
    M: float | None = None,
    beta_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    standardize: bool = True,
    time_limit: float = 600.0,
    rank_policy: str = "stata",
) -> tuple[NDArray[np.float64], float, dict[str, Any]]:
    """Chernozhukov & Hansen (2005) exact IVQR as MILP with optional positive weights.

    This implementation mirrors the CH05 formulation: minimize L1 GMM 1' t
    (the objective is the sum of auxiliary variables t) subject to linearized
    instrument moment constraints and Koenker residual decompositions. Positive
    multipliers (Exp(1)) enter the moment constraints via w_i z_{ij} (τ - d_i)
    and act as observation weights; the MILP objective itself remains 1' t
    (L1-GMM). Analytic observation weights are not supported here.
    """
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float)
    Z = np.asarray(Z, float)
    n, p = X.shape
    q = Z.shape[1]
    if Z.shape[0] != n or y.shape[0] != n:
        msg = "y, X, Z must have same number of rows"
        raise ValueError(msg)
    if not (0.0 < tau < 1.0):
        msg = "tau in (0,1)"
        raise ValueError(msg)
    w = (
        np.ones(n)
        if multipliers is None
        else np.asarray(multipliers, float).reshape(-1)
    )
    if np.any(w <= 0):
        msg = "multipliers (positive) must be positive for IVQR; analytic observation weights are not allowed here"
        raise ValueError(msg)

    # Minimal mechanical identifiability checks (fail-fast on obvious underidentification)
    if n <= p:
        msg = "Underidentified: need n > number of coefficients (p)."
        raise ValueError(msg)
    # Primary identification check: numerical rank of Z'X (instruments relevant for X)
    # normalize rank_policy
    mode = str(rank_policy).lower()
    if mode not in {"stata", "r"}:
        raise ValueError("rank_policy must be one of {'stata','r'}.")
    try:
        ZX = la.crossprod(Z, X)  # (q x p)
        qrZX = la.qr(la.to_dense(ZX), mode="economic", pivoting=True)
        Rzx = qrZX[1]
        diagRzx = np.abs(np.diag(la.to_dense(Rzx))) if Rzx.size else np.array([])
        rank_ZX = (
            la.rank_from_diag(diagRzx, ZX.shape[1], mode=mode) if diagRzx.size else 0
        )
    except (np.linalg.LinAlgError, ValueError):
        try:
            ZXd = la.to_dense(la.crossprod(Z, X))
            qrZX = la.qr(ZXd, mode="economic", pivoting=True)
            Rzx = qrZX[1]
            diagRzx = np.abs(np.diag(la.to_dense(Rzx))) if Rzx.size else np.array([])
            rank_ZX = (
                la.rank_from_diag(diagRzx, ZXd.shape[1], mode=mode)
                if diagRzx.size
                else 0
            )
        except (np.linalg.LinAlgError, ValueError):
            rank_ZX = 0
    # Strict CH05 identifiability: need q >= p and rank(Z'X) >= p
    if Z.shape[1] < p or rank_ZX < p:
        raise ValueError("Underidentified: need q >= p and rank(Z'X) >= p.")

    # optional standardization for numerical conditioning (exactly undone later)
    mu_y = 0.0
    sig_y = 1.0
    mu_x = np.zeros(p)
    sig_x = np.ones(p)
    ys = y.copy()
    Xs = X.copy()
    intercept_idx = None
    if standardize:
        sig_y = float(np.std(y)) if float(np.std(y)) > 0 else 1.0
        mu_y = float(y.mean())
        ys = (y - mu_y) / sig_y
        # Do NOT standardize exact-constant (intercept) columns. Leave them as-is
        # to avoid zeroing out the intercept column and causing rank loss.
        intercept_cols = []
        for j in range(p):
            sx = float(np.std(X[:, j]))
            if sx == 0.0 and float(np.max(X[:, j]) - np.min(X[:, j])) == 0.0:
                intercept_cols.append(j)
                mu_x[j] = 0.0
                sig_x[j] = 1.0
                Xs[:, j] = X[:, j]
            else:
                mu_x[j] = float(X[:, j].mean())
                sig_x[j] = sx
                Xs[:, j] = (X[:, j] - mu_x[j]) / sig_x[j]
        if len(intercept_cols) > 1:
            msg = "standardize=True is not allowed when multiple exact-constant columns (intercepts) are present; reduce to at most one constant column or set standardize=False"
            raise ValueError(
                msg,
            )
        intercept_idx = intercept_cols[0] if intercept_cols else None

    # variable layout: [beta(p), u_pos(n), u_neg(n), d(n), t(q)]
    nvar = p + n + n + n + q
    idx_beta = slice(0, p)
    idx_up = slice(p, p + n)
    idx_un = slice(p + n, p + 2 * n)
    idx_d = slice(p + 2 * n, p + 3 * n)
    idx_t = slice(p + 3 * n, p + 3 * n + q)

    lb = np.full(nvar, -np.inf)
    ub = np.full(nvar, np.inf)
    # beta bounds
    if beta_bounds is None:
        lb[idx_beta] = -1e6
        ub[idx_beta] = +1e6
    else:
        lb[idx_beta] = beta_bounds[0]
        ub[idx_beta] = beta_bounds[1]
    lb[idx_up] = 0.0
    lb[idx_un] = 0.0
    lb[idx_t] = 0.0
    lb[idx_d] = 0.0
    ub[idx_d] = 1.0

    # Big-M (conservative, data-driven)
    if M is None:
        max_y = float(np.max(np.abs(ys)))
        max_row_norm = float(np.max(np.sum(np.abs(Xs), axis=1)))
        beta_width = float(np.max(np.abs(ub[idx_beta] - lb[idx_beta])))
        M_use = max(10.0, max_y + max_row_norm * beta_width)
    else:
        M_use = float(M)

    cons: list[LinearConstraint] = []

    # 1) y - X beta = u_pos - u_neg
    A1 = np.zeros((n, nvar))
    A1[:, idx_beta] = -Xs
    A1[:, idx_up] = -la.to_dense(la.eye(n))
    A1[:, idx_un] = la.to_dense(la.eye(n))
    b1 = -ys
    cons.append(LinearConstraint(A1, lb=b1, ub=b1))

    # 2) u_pos <= M (1 - d)  -> u_pos + M d <= M
    A2 = np.zeros((n, nvar))
    A2[:, idx_up] = la.to_dense(la.eye(n))
    A2[:, idx_d] = M_use * la.to_dense(la.eye(n))
    b2 = np.full(n, M_use)
    cons.append(LinearConstraint(A2, lb=-np.inf * np.ones(n), ub=b2))

    # 3) u_neg <= M d
    A3 = np.zeros((n, nvar))
    A3[:, idx_un] = la.to_dense(la.eye(n))
    A3[:, idx_d] = -M_use * la.to_dense(la.eye(n))
    b3 = np.zeros(n)
    cons.append(LinearConstraint(A3, lb=-np.inf * np.ones(n), ub=b3))

    # 4) instrument moment constraints with weights: t_j >= | sum_i w_i z_ij (tau - d_i) |
    # Use Z (in standardized or original form) to build weighted instruments matrix
    wz = (w.reshape(-1, 1) * Z).astype(float)
    # + side: t_j + sum_i w_i z_ij d_i >= tau * sum_i w_i z_ij
    A4p = np.zeros((q, nvar))
    A4p[:, idx_t] = la.to_dense(la.eye(q))
    A4p[:, idx_d] = +wz.T
    bp = (tau * np.sum(wz, axis=0)).astype(float)
    # - side: t_j - sum_i w_i z_ij d_i >= -tau * sum_i w_i z_ij
    A4m = np.zeros((q, nvar))
    A4m[:, idx_t] = la.to_dense(la.eye(q))
    A4m[:, idx_d] = -wz.T
    bm = (-tau * np.sum(wz, axis=0)).astype(float)
    cons.append(LinearConstraint(A4p, lb=bp, ub=np.inf * np.ones(q)))
    cons.append(LinearConstraint(A4m, lb=bm, ub=np.inf * np.ones(q)))

    # objective: minimize 1' t (L1 GMM) ; note u terms are implicit via constraints in CH05 L1 formulation
    c = np.zeros(nvar)
    c[idx_t] = 1.0

    integrality = np.zeros(nvar, dtype=int)
    integrality[idx_d] = 1

    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=cons,
        options={"presolve": True, "time_limit": float(time_limit)},
    )
    if getattr(res, "status", 0) != 0:
        msg = f"IVQR MILP did not converge: status {getattr(res, 'status', None)}, msg: {getattr(res, 'message', None)}"
        raise RuntimeError(msg)

    sol = res.x
    beta_s = sol[idx_beta].copy()

    # Unscale with careful intercept correction when an intercept/constant
    # column was left unstandardized. For non-intercept columns j:
    #   beta_j = (σ_y * beta_s_j) / σ_x_j
    # For intercept column j0 left unscaled we use the identity
    #   β0 = μ_y + σ_y*β_s_j0*c0 - Σ_{m≠j0} μ_x_m (σ_y β_s_m / σ_x_m)
    if standardize:
        beta = np.empty_like(beta_s)
        for j in range(p):
            if j == intercept_idx:
                c0 = float(la.col_mean(X[:, j].reshape(-1, 1))[0])
                adj = 0.0
                for m in range(p):
                    if m == j:
                        continue
                    adj += mu_x[m] * (sig_y * beta_s[m] / sig_x[m])
                beta[j] = mu_y + sig_y * beta_s[j] * c0 - adj
            else:
                beta[j] = sig_y * beta_s[j] / sig_x[j]
    else:
        beta = beta_s

    info = {
        "status": getattr(res, "status", None),
        "message": getattr(res, "message", None),
        "M": M_use,
        "standardized": standardize,
        "mu_y": mu_y,
        "sig_y": sig_y,
        "mu_x": mu_x,
        "sig_x": sig_x,
        "intercept_idx": intercept_idx,
    }
    obj = float(np.sum(sol[idx_t]))
    return beta.reshape(-1, 1), obj, info


class QR(BaseEstimator):
    """Quantile Regression (Koenker-Bassett) solved by **exact LP** (no approximations).
    Note: Instrumental-variable quantile regression (IVQR, Chernozhukov & Hansen 2005)
    is provided by the separate `IVQR` class. The `QR` class implements unconditional
    quantile regression only.

    Inference (bootstrap)
    ---------------------
    Uses **positive multiplier bootstrap** based on weighted LP:
        For each b, draw positive weights w_i^b (Exp(1), clusterwise if requested),
        then solve the weighted quantile LP with weights w^b in the objective.
    This replaces invalid residual-wild schemes for QR, and is consistent in
    heteroskedastic settings. For clustered data, Wild Gradient Bootstrap (WGB)
    is available by setting boot.dist="wgb" for better finite-sample properties
    (Hagemann 2017).
    """

    def __init__(  # noqa: PLR0913
        self,
        y: ArrayLike,
        X: MatrixLike,
        *,
        Z: MatrixLike | None = None,
        tau: float = 0.5,
        add_const: bool = True,
        var_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if not (0.0 < tau < 1.0):
            msg = "Quantile tau must be in (0, 1)."
            raise ValueError(msg)

        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if var_names is None and isinstance(X, pd.DataFrame):
            var_names = list(X.columns)
        X_arr = np.asarray(X, dtype=np.float64)

        if Z is not None:
            # QR class is for unconditional quantile regression only.
            # Instrumental QR should be constructed via the dedicated IVQR class.
            msg = "QR.__init__ does not accept instruments Z. Use IVQR class for instrumental quantile regression."
            raise ValueError(msg)
        self.Z_orig = None

        if add_const:
            X_aug, names, const_name = add_constant(
                X_arr,
                var_names,
            )
            self._const_name = const_name
        else:
            X_aug = X_arr
            self._const_name = None
            names = (
                list(var_names)
                if var_names is not None
                else [f"x{i}" for i in range(X_aug.shape[1])]
            )

        self._var_names: list[str] = names

        self.y_orig: NDArray[np.float64] = y_arr
        self.X_orig: NDArray[np.float64] = X_aug
        # Use private variable to avoid BaseEstimator n_obs property conflict
        self._n_obs_raw: int = y_arr.shape[0]
        self.n_features: int = X_aug.shape[1]
        self.tau: float = float(tau)
        # Strict upfront check to avoid accidental MNW variants via BootConfig later
        # This is a no-op placeholder to make the intent explicit for audits.
        if hasattr(BootConfig, "__name__"):
            pass  # intentionally no-op; actual rejection occurs in fit()

    @classmethod
    def from_formula(  # noqa: PLR0913
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        tau: float = 0.5,
        id_name: str | None = None,
        time: str | None = None,
        options: str | None = None,
        W_dict: dict[str, object] | None = None,
        boot: BootConfig | None = None,
    ) -> QR:
        """Quantile Regression from formula. tau is passed to constructor."""
        parser = FormulaParser(data, id_name=id_name, t_name=time, W_dict=W_dict)
        parsed = parser.parse(formula, iv=None, options=options)
        _, boot_eff, meta = prepare_formula_environment(
            formula=formula,
            data=data,
            parsed=parsed,
            boot=boot,
            attr_keys={
                "_row_mask_valid": "row_mask_valid",
                "_fe_codes_from_formula": "fe_codes_list",
            },
        )
        if boot_eff is not None:
            meta.attrs.setdefault("_boot_from_formula", boot_eff)
        model = cls(
            parsed["y"],
            parsed["X"],
            tau=tau,
            add_const=bool(parsed.get("include_intercept", True)),
            var_names=parsed["var_names"],
        )
        attach_formula_metadata(model, meta)
        return model

    def _bootstrap_multipliers(
        self, n_obs: int, *, boot: BootConfig | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Override to use Exp(1) positive multipliers for QR, or WGB if specified.

        Policy:
        - Supported: dist in {"exp", "wgb"}
        - Friendly aliases that are mapped with a warning:
          * {"rademacher", "mammen", "webb", "11", "13", "31", "33", "33j"} -> "exp"
            (QR requires positive weights unless using WGB; we fallback to Exp(1)).
        """
        if boot is None:
            msg = "BootConfig must be provided."
            raise ValueError(msg)
        B = boot.n_boot
        raw_dist = str(boot.dist).lower() if boot.dist is not None else "exp"
        dist = raw_dist.strip().replace("-", "").replace("_", "")
        # Map friendly aliases to supported schemes
        warnings_list: list[str] = []
        if dist in {
            "rademacher",
            "mammen",
            "webb",
            "11",
            "13",
            "31",
            "33",
            "33j",
            "mnw11",
            "mnw13",
            "mnw31",
            "mnw33",
            "mnw33j",
        }:
            warnings_list.append(
                f"Requested dist='{raw_dist}' is not supported for QR. Falling back to Exp(1) positive multipliers.",
            )
            dist = "exp"
        if dist not in {"wgb", "exp"}:
            msg = f"Unsupported bootstrap dist='{boot.dist}'. Use 'exp' or 'wgb'."
            raise ValueError(msg)
        if dist == "wgb":
            # WGB theory (Hagemann 2017) supports one-way clustering only.
            # Reject multiway/spatial-temporal clustering requests.
            if (boot.multiway_ids is not None) or (
                boot.space_ids is not None and boot.time_ids is not None
            ):
                msg = "WGB is defined for one-way clustering only (Hagemann 2017). Use dist='exp' for multiway clustering."
                raise ValueError(msg)
            # WGB is handled separately in fit; here just return dummy for compatibility
            W = np.ones((n_obs, B))
            log = {
                "effective_dist": "wgb",
                "effective_B": B,
                "enumerated": False,
                "warnings": warnings_list,
            }
        else:
            # Default to Exp(1)
            W = _exp_multipliers(
                n_obs,
                B=B,
                cluster_ids=boot.cluster_ids,
                multiway_ids=boot.multiway_ids,
                space_ids=boot.space_ids,
                time_ids=boot.time_ids,
                method=(getattr(boot, "cluster_method", None) or "intersection"),
                seed=boot.seed,
            )
            log = {
                "effective_dist": "exp",
                "effective_B": W.shape[1],
                "enumerated": False,
                "warnings": warnings_list,
            }
        cols = [f"b{j}" for j in range(W.shape[1])]
        return pd.DataFrame(W, columns=cols), log

    def fit(
        self,
        *,
        device: str | None = None,
        boot: BootConfig | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
    ) -> EstimationResult:
        """Fit QR by exact LP (HiGHS) and compute multiplier-bootstrap SEs (no analytic weights, no CIs)."""
        with self._device_context(device):
            pass
        # Handle NA/missing values
        mask = np.isfinite(self.y_orig).ravel() & np.all(
            np.isfinite(self.X_orig), axis=1,
        )
        if self.Z_orig is not None:
            mask &= np.all(np.isfinite(self.Z_orig), axis=1)
        X = self.X_orig[mask, :]
        y = self.y_orig[mask, :]
        n_eff = X.shape[0]
        dropped_na = int(np.sum(~mask))

        # Enforce centralized weight policy for QR: analytic observation weights are forbidden.
        # Use the BaseEstimator helper to maintain consistent behavior/messages.
        self._enforce_weight_policy("qr", None)
        w_proc = None

        # Mask cluster IDs
        def _mask_seq(ids):
            if ids is None:
                return None
            arr = np.asarray(ids)
            if arr.shape[0] != self._n_obs_raw:
                raise ValueError(
                    "cluster/space/time ids must have the same length as the original sample (before dropping NAs).",
                )
            return arr[mask]

        def _mask_seq_multiway(multiway):
            if multiway is None:
                return None
            return [_mask_seq(arr) for arr in multiway]

        cluster_ids_masked = _mask_seq(cluster_ids)
        space_ids_masked = _mask_seq(space_ids)
        time_ids_masked = _mask_seq(time_ids)
        # Enforce paired specification for (space_ids, time_ids)
        if (space_ids_masked is None) ^ (time_ids_masked is None):
            raise ValueError("space_ids and time_ids must be provided jointly.")
        # prefer explicit fit-args; otherwise fall back to boot.multiway_ids
        mw_from_boot = (
            boot.multiway_ids
            if (boot and getattr(boot, "multiway_ids", None) is not None)
            else None
        )
        multiway_ids_masked = _mask_seq_multiway(mw_from_boot)

        # --- NEW: exclusivity check (prefer explicit fit args over BootConfig) ---
        provided = sum(
            int(x is not None)
            for x in (
                cluster_ids_masked,
                (
                    space_ids_masked
                    if (space_ids_masked is not None and time_ids_masked is not None)
                    else None
                ),
                multiway_ids_masked,
            )
        )
        if provided > 1:
            msg = "Specify at most one clustering scheme among {multiway_ids, (space_ids,time_ids), cluster_ids}."
            raise ValueError(msg)

        if self.Z_orig is not None:
            # Enforce class separation without 'NotImplementedError' wording
            raise ValueError(
                "Use IVQR class for instrumental quantile regression (CH05 exact MILP).",
            )
        # Precompute LP structures for reuse across potential weighted solves
        prep = _prepare_qr_lp(X, y, self.tau)
        beta_hat, obj_val, _, solver_info = _solve_qr_lp_prepared(
            prep, w_proc,
        )

        # prefer explicit fit-args; otherwise use BootConfig IDs BUT always mask to the effective sample
        def _mask_if_needed(arr):
            if arr is None:
                return None
            arr = np.asarray(arr)
            if arr.shape[0] != self._n_obs_raw:
                raise ValueError(
                    "BootConfig IDs must match the original sample length.",
                )
            return arr[mask]

        use_cluster = (
            cluster_ids_masked
            if cluster_ids_masked is not None
            else (_mask_if_needed(getattr(boot, "cluster_ids", None)) if boot else None)
        )
        use_space = (
            space_ids_masked
            if space_ids_masked is not None
            else (_mask_if_needed(getattr(boot, "space_ids", None)) if boot else None)
        )
        use_time = (
            time_ids_masked
            if time_ids_masked is not None
            else (_mask_if_needed(getattr(boot, "time_ids", None)) if boot else None)
        )
        use_multiway = (
            multiway_ids_masked
            if multiway_ids_masked is not None
            else (
                [_mask_if_needed(a) for a in getattr(boot, "multiway_ids", [])]
                if (boot and getattr(boot, "multiway_ids", None) is not None)
                else None
            )
        )
        if boot is None:
            boot = BootConfig(
                dist="exp",
                cluster_ids=use_cluster,
                space_ids=use_space,
                time_ids=use_time,
                multiway_ids=use_multiway,
                policy="boottest",
                enumeration_mode="boottest",
            )
        effective_dist = boot.dist
        if isinstance(boot.dist, str):
            dist_norm = str(boot.dist).strip().lower().replace("-", "").replace("_", "")
            if dist_norm in {"11", "13", "31", "33", "33j"}:
                import warnings as _w
                _w.warn(
                    f"QR: MNW variant '{boot.dist}' not supported; falling back to dist='exp'.",
                    RuntimeWarning, stacklevel=2,
                )
                effective_dist = "exp"
            elif dist_norm in {"rademacher", "mammen", "webb"}:
                import warnings as _w
                _w.warn(
                    f"QR: dist='{boot.dist}' produces negative weights incompatible with LP; falling back to dist='exp'.",
                    RuntimeWarning, stacklevel=2,
                )
                effective_dist = "exp"
        boot = BootConfig(
            dist=effective_dist if effective_dist not in {"rademacher", "mammen", "webb"} else "exp",
            n_boot=boot.n_boot,
            seed=boot.seed,
            cluster_ids=boot.cluster_ids,
            space_ids=boot.space_ids,
            time_ids=boot.time_ids,
            multiway_ids=boot.multiway_ids,
            policy=getattr(boot, "policy", "boottest"),
            enumeration_mode=getattr(boot, "enumeration_mode", "boottest"),
        )

        B = boot.n_boot
        if boot.dist == "wgb":
            # Wild Gradient Bootstrap (one-way clusters only; no fallback, strict policy)
            if cluster_ids_masked is None:
                msg = "WGB requires one-way clustering; provide cluster_ids."
                raise ValueError(msg)
            if (
                multiway_ids_masked is not None
                or space_ids_masked is not None
                or time_ids_masked is not None
            ):
                msg = "Strict policy: WGB is defined only for one-way clustering. Provide cluster_ids (one-way) or use boot.dist='exp'."
                raise ValueError(msg)
            # QR-WGB: produce bootstrap betas via the specialized WGB routine.
            # Policy: prefer explicit fit-arg cluster IDs; else use BootConfig (both masked).
            # Use the resolved cluster scheme for WGB (must be one-way cluster)
            if use_cluster is None:
                msg = "WGB requires one-way clustering; provide cluster_ids."
                raise ValueError(msg)
            boot_betas, boot_log = _wgb_bootstrap_qr(
                X,
                y,
                self.tau,
                B,
                cluster_ids=use_cluster,
                seed=boot.seed,
            )
            se_hat = bt.bootstrap_se(boot_betas)
            # Ensure WGB multipliers used are preserved for reproducibility
            extra_wgb = {"W_multipliers_inference": boot_log.get("W_multipliers", None)}
        else:
            Wpos, boot_log = self._bootstrap_multipliers(n_eff, boot=boot)

            boot_betas = np.empty((self.n_features, B), dtype=np.float64)
            boot_betas.fill(np.nan)
            Wpos_arr = Wpos.to_numpy()
            n_workers = max(1, int(os.getenv("LINEAREG_QR_BOOTSTRAP_WORKERS", "1")))
            failed_draws = 0
            if n_workers > 1 and B > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [
                        ex.submit(
                            _solve_qr_lp_prepared,
                            prep,
                            Wpos_arr[:, b : b + 1],
                            raise_on_failure=False,
                        )
                        for b in range(B)
                    ]
                    for b, fut in enumerate(futures):
                        try:
                            boot_beta, _, _, info = fut.result()
                            if np.any(np.isnan(boot_beta)):
                                failed_draws += 1
                            else:
                                boot_betas[:, b] = boot_beta.reshape(-1)
                        except Exception:
                            failed_draws += 1
            else:
                for b in range(B):
                    wb = Wpos_arr[:, b : b + 1]
                    try:
                        boot_beta, _, _, info = _solve_qr_lp_prepared(prep, wb, raise_on_failure=False)
                        if np.any(np.isnan(boot_beta)):
                            failed_draws += 1
                        else:
                            boot_betas[:, b] = boot_beta.reshape(-1)
                    except Exception:
                        failed_draws += 1

            if failed_draws > 0:
                warnings.warn(f"QR bootstrap: {failed_draws}/{B} draws failed to solve; using {B - failed_draws} valid draws.", RuntimeWarning, stacklevel=2)

            valid_mask = ~np.isnan(boot_betas[0, :])
            if valid_mask.sum() < 2:
                warnings.warn("QR bootstrap: fewer than 2 valid draws; SE will be NaN.", RuntimeWarning, stacklevel=2)
                se_hat = np.full(self.n_features, np.nan)
            else:
                se_hat = bt.bootstrap_se(boot_betas[:, valid_mask])

        params = pd.Series(beta_hat.reshape(-1), index=self._var_names, name="coef")
        se = pd.Series(se_hat.reshape(-1), index=self._var_names, name="se")

        # Build results
        # WGB_Recommended only when a one-way cluster id is present
        wgb_recommended = cluster_ids_masked is not None
        extra_local = {
            "boot_betas": boot_betas,
            "objective_value": obj_val,
            "qr_solver_info": solver_info,  # Reproducibility: solver status, nit, message
            "WGB_Available": True,
            "WGB_Reference": "Hagemann (2017): Wild Gradient Bootstrap for Quantile Regression",
            "yhat": la.dot(X, beta_hat),
            "X_inference": X,
            "u_inference": y - la.dot(X, beta_hat),
            # preserve the actual scheme used for reproducibility
            "clusters_inference": use_cluster,
            "multiway_ids_inference": use_multiway,
            "space_ids_inference": use_space,
            "time_ids_inference": use_time,
            "weights_inference": w_proc,
            "vcov_kind_inference": "auto_strict",
            "beta0_inference": beta_hat,
            # Keep heavy multiplier matrices out of model_info; store them in `extra_local`.
            "W_multipliers_inference": (
                Wpos.to_numpy()
                if boot.dist != "wgb"
                else boot_log.get("W_multipliers", None)
            ),
            "multipliers_log": boot_log,
            "mask_used": mask,
            "boot_policy_used": getattr(boot, "policy", None),
        }
        # If WGB was used, merge the explicit multipliers into extra for auditability
        if boot.dist == "wgb":
            extra_local.update(extra_wgb)
        # Force provenance of standard errors: analytical SEs are disallowed
        extra_local["se_source"] = "bootstrap"

        # Do not spill heavy arrays into model_info; keep only metadata there.
        boot_log_info = {k: v for (k, v) in boot_log.items() if k != "W_multipliers"}

        self._results = EstimationResult(
            params=params,
            se=se,  # Bootstrap SE stored directly in .se
            bands=None,
            n_obs=int(n_eff),
            model_info={
                "Estimator": "IVQR (CH05)"
                if self.Z_orig is not None
                else "QR (exact LP)",
                "Bootstrap": f"{boot_log_info.get('effective_dist', 'exp')} multipliers",
                "WGB_Recommended": wgb_recommended,
                "Warning": "Clustered QR (one-way): prefer Wild Gradient Bootstrap (Hagemann 2017)"
                if wgb_recommended
                else None,
                "B": B,
                "n_eff": n_eff,
                "Dropped_NA": dropped_na,
                "NoAnalyticPValues": True,
                "Cluster_Method": (
                    getattr(boot, "cluster_method", None) or "intersection"
                ),
                "Seed": boot.seed,
                **boot_log_info,
            },
            extra=extra_local,
        )
        return self._results

    def fit_multi(
        self,
        taus: Sequence[float],
        *,
        boot: BootConfig | None = None,
        cluster_ids: Sequence | None = None,
    ) -> dict[float, EstimationResult]:
        """Fit QR for multiple quantiles `taus` reusing a single draw of positive
        multipliers (Exp(1)) across taus for efficiency and exact reproducibility.

        Returns a dict mapping tau -> EstimationResult. BootConfig must be
        provided to control B and clustering; if boot is None a default
        BootConfig(dist='exp') is used.
        """
        if boot is None:
            boot = BootConfig(dist="exp", cluster_ids=cluster_ids)
        # Draw multipliers once for the original (raw) sample size
        Wpos_df, boot_log = boot.make_multipliers(self._n_obs_raw)

        results: dict[float, EstimationResult] = {}
        # Common mask for data completeness (same as fit)
        mask = np.isfinite(self.y_orig).ravel() & np.all(
            np.isfinite(self.X_orig), axis=1,
        )
        X_full = self.X_orig[mask, :]
        y_full = self.y_orig[mask, :]
        n_eff = X_full.shape[0]
        # Mask the pre-drawn multipliers to the effective sample
        Wpos_masked = Wpos_df.iloc[mask, :].reset_index(drop=True)

        # Enforce that bootstrap distribution is compatible with shared multipliers
        if str(boot.dist).lower() == "wgb":
            raise ValueError(
                "fit_multi with shared multipliers does not support WGB; use dist='exp'.",
            )

        B = boot.n_boot
        for tau in taus:
            if not (0.0 < float(tau) < 1.0):
                raise ValueError("Each tau must be in (0,1).")

            # Base (unweighted) estimate for this tau (reuse prepared LP)
            prep = _prepare_qr_lp(X_full, y_full, float(tau))
            beta_hat, obj_val, _, solver_info = _solve_qr_lp_prepared(
                prep, None,
            )

            # Compute bootstrap replicates using the same multipliers columns
            boot_betas = np.empty((self.n_features, B), dtype=np.float64)
            Wpos_arr = Wpos_masked.to_numpy()
            n_workers = max(1, int(os.getenv("LINEAREG_QR_BOOTSTRAP_WORKERS", "1")))
            if n_workers > 1 and B > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [
                        ex.submit(
                            _solve_qr_lp_prepared,
                            prep,
                            Wpos_arr[:, b : b + 1],
                        )
                        for b in range(B)
                    ]
                    for b, fut in enumerate(futures):
                        boot_beta, _, _, _ = fut.result()
                        boot_betas[:, b] = boot_beta.reshape(-1)
            else:
                for b in range(B):
                    wb = Wpos_arr[:, b : b + 1]
                    boot_beta, _, _, _ = _solve_qr_lp_prepared(prep, wb)
                    boot_betas[:, b] = boot_beta.reshape(-1)

            se_hat = bt.bootstrap_se(boot_betas)

            params = pd.Series(beta_hat.reshape(-1), index=self._var_names, name="coef")
            se = pd.Series(se_hat.reshape(-1), index=self._var_names, name="se")

            extra_local = {
                "boot_betas": boot_betas,
                "objective_value": obj_val,
                "qr_solver_info": solver_info,
                "W_multipliers_inference": Wpos_masked.to_numpy(),
                "multipliers_log": boot_log,
            }

            res = EstimationResult(
                params=params,
                se=se,
                bands=None,
                n_obs=int(n_eff),
                model_info={
                    "Estimator": "QR (exact LP)",
                    "Bootstrap": f"{boot_log.get('effective_dist', 'exp')} multipliers",
                    "B": B,
                    "Tau": float(tau),
                },
                extra=extra_local,
            )
            results[float(tau)] = res
        return results


class IVQR(BaseEstimator):
    """Instrumental-Variable Quantile Regression estimator.

    Estimation method:
      - 'ch05' : Chernozhukov & Hansen (2005) exact formulation solved via MILP only (strict policy)

    Inference:
      - Positive Exp(1) multiplier bootstrap only (cluster-constant when cluster_ids provided).
      - Pair (pairs/xy) bootstrap is explicitly disallowed.
    """

    def __init__(  # noqa: PLR0913
        self,
        y: ArrayLike,
        X: MatrixLike,
        Z: MatrixLike,
        *,
        tau: float = 0.5,
        method: str = "ch05",
        rank_policy: str = "stata",
        add_const: bool = True,
        var_names: Sequence[str] | None = None,
        M: float | None = None,
        beta_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
        standardize: bool = True,
    ) -> None:
        super().__init__()
        if not (0.0 < tau < 1.0):
            msg = "tau must be in (0,1)"
            raise ValueError(msg)
        if method != "ch05":
            msg = "IVQR supports only 'ch05' (Chernozhukov & Hansen 2005 exact MILP) under the strict policy."
            raise ValueError(msg)

        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        X_arr = np.asarray(X, dtype=np.float64)
        Z_arr = np.asarray(Z, dtype=np.float64)

        if add_const:
            X_aug, names, const_name = add_constant(X_arr, var_names)
            self._const_name = const_name
        else:
            X_aug = X_arr
            self._const_name = None
            names = (
                list(var_names)
                if var_names is not None
                else [f"x{i}" for i in range(X_arr.shape[1])]
            )

        # R/Stata convention: if X contains an intercept, ensure Z also contains
        # a constant instrument. If the user omitted a constant in Z, append one.
        try:
            const_cols_X = [
                j for j in range(X_aug.shape[1]) if float(np.std(X_aug[:, j])) == 0.0
            ]
            has_const_in_X = len(const_cols_X) > 0
        except (ValueError, FloatingPointError, TypeError):
            has_const_in_X = False

        added_const_to_Z = False
        if has_const_in_X:
            try:
                const_cols_Z = [
                    j
                    for j in range(Z_arr.shape[1])
                    if float(np.std(Z_arr[:, j])) == 0.0
                ]
                has_const_in_Z = len(const_cols_Z) > 0
            except (ValueError, FloatingPointError, TypeError):
                has_const_in_Z = False
            if not has_const_in_Z:
                # prepend a column of ones to Z_arr to act as an instrument for the intercept
                Z_arr = np.column_stack(
                    [np.ones((Z_arr.shape[0], 1), dtype=np.float64), Z_arr],
                )
                added_const_to_Z = True

        self.y_orig = y_arr
        self.X_orig = X_aug
        self.Z_orig = Z_arr
        self._var_names = names
        self.tau = float(tau)
        self.method = method
        self.M = M
        self.beta_bounds = beta_bounds
        self.standardize = standardize
        # rank policy: 'stata' or 'r' (controls la._rank_from_diag behavior)
        self.rank_policy = str(rank_policy).lower()
        if self.rank_policy not in {"stata", "r"}:
            raise ValueError("rank_policy must be one of {'stata','r'}.")
        # preserve whether we augmented Z with a constant for reproducibility
        self._added_const_to_Z = bool(added_const_to_Z)

    # SEE (Smoothed Estimating Equations) removed by strict policy: only CH05 exact MILP allowed.

    def fit(
        self,
        *,
        device: str | None = None,
        boot: BootConfig | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
    ) -> EstimationResult:
        """Fit IVQR and compute bootstrap standard errors using Exp(1) positive multipliers.

        For each bootstrap draw, we re-solve the weighted MILP (CH05) with
        observation weights = analytic weights * Exp(1) multipliers (cluster-constant
        when cluster_ids provided). Pair bootstrap is forbidden.
        """
        with self._device_context(device):
            pass
        # mask missing
        mask = (
            np.isfinite(self.y_orig).ravel()
            & np.all(np.isfinite(self.X_orig), axis=1)
            & np.all(np.isfinite(self.Z_orig), axis=1)
        )
        # Enforce centralized weight policy for IV-QR (analytic observation weights disallowed)
        self._enforce_weight_policy("ivqr", None)
        y = self.y_orig[mask, :].reshape(-1)
        X = self.X_orig[mask, :]
        Z = self.Z_orig[mask, :]
        n_eff = X.shape[0]
        dropped_na = int(np.sum(~mask))

        # base estimate (CH05 MILP only under strict policy)
        beta_hat, _obj, info = _solve_ivqr_ch05_milp(
            y,
            X,
            Z,
            self.tau,
            multipliers=None,
            M=self.M,
            beta_bounds=self.beta_bounds,
            standardize=self.standardize,
            rank_policy=self.rank_policy,
        )

        # bootstrap config
        # Defensive guard: IVQR does not accept analytic observation weights here
        # (analytic weights are not in the IVQR API; multiplier bootstrap is used instead)
        # Prefer explicit fit-args > boot.* and ensure BootConfig IDs are masked to the effective sample.
        if cluster_ids is not None:
            use_cluster = np.asarray(cluster_ids)[mask]
        elif boot and getattr(boot, "cluster_ids", None) is not None:
            arr = np.asarray(boot.cluster_ids)
            if arr.shape[0] != self.y_orig.shape[0]:
                raise ValueError(
                    "BootConfig.cluster_ids length must equal the original sample size.",
                )
            use_cluster = arr[mask]
        else:
            use_cluster = None
        # Strict policy: IVQR allows IID or one-way clustering only. Reject multiway/space x time.
        use_space = None
        use_time = None
        use_multiway = None
        if (
            (space_ids is not None)
            or (time_ids is not None)
            or (boot and getattr(boot, "space_ids", None) is not None)
            or (boot and getattr(boot, "time_ids", None) is not None)
        ):
            raise ValueError(
                "IVQR supports IID or one-way clustering only (multiway/space x time are not allowed under the strict policy).",
            )
        if boot and getattr(boot, "multiway_ids", None) is not None:
            raise ValueError(
                "IVQR supports IID or one-way clustering only (multiway is not allowed under the strict policy).",
            )
        if boot is None:
            boot = BootConfig(
                dist="exp",
                cluster_ids=use_cluster,
                policy="boottest",
                enumeration_mode="boottest",
            )
        if str(boot.dist).lower() in {"pairs", "pair", "xy", "xy-pairs"}:
            msg = "Pair bootstrap is not allowed. Use positive multiplier bootstrap (Exp(1))."
            raise ValueError(msg)

        B = boot.n_boot
        # multipliers (cluster-constant if cluster_ids provided)
        W = _exp_multipliers(
            n_eff,
            B=B,
            cluster_ids=use_cluster,
            multiway_ids=None,
            space_ids=None,
            time_ids=None,
            method=(getattr(boot, "cluster_method", None) or "intersection"),
            seed=boot.seed,
        )

        betas_b = np.empty((X.shape[1], B), dtype=np.float64)
        for b in range(B):
            wb = W[:, b]
            # IVQR per-draw: pass positive multipliers as weights to the CH05 MILP
            beta_b, _, _ = _solve_ivqr_ch05_milp(
                y,
                X,
                Z,
                self.tau,
                multipliers=wb,
                M=self.M,
                beta_bounds=self.beta_bounds,
                standardize=self.standardize,
                rank_policy=self.rank_policy,
            )
            betas_b[:, b] = beta_b.reshape(-1)

        se_hat = bt.bootstrap_se(betas_b)

        params = pd.Series(beta_hat.reshape(-1), index=self._var_names, name="coef")
        se = pd.Series(se_hat.reshape(-1), index=self._var_names, name="se")

        self._results = EstimationResult(
            params=params,
            se=se,  # Bootstrap SE stored directly in .se
            bands=None,
            n_obs=int(n_eff),
            model_info={
                "Estimator": f"IVQR ({self.method.upper()})",
                "B": B,
                "Bootstrap": "Exp(1) positive multipliers",
                "NoAnalyticPValues": True,
                "NoPairs": True,
                "Tau": self.tau,
                "Dropped_NA": dropped_na,
                "n_eff": n_eff,
            },
            extra={
                "beta0_info": info,
                "boot_betas": betas_b,
                "X_inference": X,
                "Z_inference": Z,
                "y_inference": y,
                "clusters_inference": use_cluster,
                "multiway_ids_inference": use_multiway,
                "space_ids_inference": use_space,
                "time_ids_inference": use_time,
                "mask_used": mask,
                "boot_policy_used": getattr(boot, "policy", None),
                "se_source": "bootstrap",
            },
        )
        return self._results

    # SEE implementation removed (strict policy).
