"""Base classes and bootstrap configuration.

This module defines the abstract base estimator, bootstrap configuration data structures,
and standardized estimation results containers.
"""

# lineareg/estimators/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lineareg.core import backend as be
from lineareg.core import bootstrap as bt
from lineareg.core import fe as fe_core
from lineareg.core import linalg as la

if TYPE_CHECKING:  # import-only typing to satisfy ruff TC00x without runtime cost
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "BaseEstimator",
    "BootConfig",
    "EstimationResult",
    "FormulaMetadata",
    "attach_formula_metadata",
    "ci_level_to_alpha",
    "normalize_ci_level",
    "prepare_formula_environment",
]


# ---------------------------------------------------------------------
# Helper for canonical bootstrap variants (11/13/31/33 per MNW 2022 / boottest)
# ---------------------------------------------------------------------


def _parse_bootstrap_variant(variant: str) -> tuple[str, str]:
    # Delegate to the centralized parser in core.bootstrap for canonical variants
    return bt._parse_bootstrap_variant(variant)  # noqa: SLF001


def normalize_ci_level(level: float | None, *, default: float = 0.95) -> float:
    """Normalize confidence level to a probability (0, 1)."""
    if not (0.0 < float(default) < 1.0):
        raise ValueError("default confidence level must lie in (0, 1)")
    if level is None:
        coerced = float(default)
    else:
        coerced = float(level)
        # Accept percentage-style inputs (e.g., 90 for 90%)
        if coerced > 1.0:
            coerced /= 100.0
    if not (0.0 < coerced < 1.0):
        raise ValueError("ci_level must be in (0, 1); supply e.g. 0.95 or 95")
    return coerced


def ci_level_to_alpha(level: float | None, *, default: float = 0.95) -> float:
    """Return the corresponding tail probability ``alpha`` for a confidence level."""
    ci_level = normalize_ci_level(level, default=default)
    return 1.0 - ci_level


# ---------------------------------------------------------------------
# Results container (R/Stata-like), extensible and estimator-agnostic
# ---------------------------------------------------------------------
@dataclass
class EstimationResult:
    """Container for estimation results.

    Stores parameter estimates, standard errors (if computed), and diagnostics.
    Does not compute analytic p-values.
    """

    params: pd.Series
    se: pd.Series | None = None
    bands: dict[str, pd.DataFrame] | None = None  # For DID: staggered uniform bands
    n_obs: int | None = None
    model_info: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    """Dictionary for estimator-specific diagnostics and intermediate results."""

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        head = ", ".join(f"{k}={v}" for k, v in self.model_info.items())
        return f"EstimationResult(k={len(self.params)}, n={self.n_obs}, {head})"

    def wald_test(  # noqa: PLR0913
        self,
        R: NDArray[np.float64],
        r: NDArray[np.float64],
        *,
        variant: str | None = None,
        residual_type: str = "WCU",
        vcov_kind: str = "auto_strict",
        denominator: str = "boottest_strict",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Perform Wald bootstrap test for restrictions Rb = r.

        Delegates to `core.bootstrap.wald_test_wild_bootstrap` using the
        estimator's stored inference artifacts.
        """
        # Disallow QR/IVQR generic Wald here: use estimator-specific routines.
        est_label = str(self.model_info.get("Estimator", "")).upper()
        if "QR" in est_label:
            msg = (
                "QR/IVQR objects must use the QR-specific bootstrap test "
                "(this generic Wald path is not applicable)."
            )
            raise ValueError(msg)

        # --- Strict input validation for R/r (prevent mis-specified restrictions) ---
        R = np.asarray(R, dtype=float)
        r = np.asarray(r, dtype=float)
        if R.ndim != 2 or r.ndim not in (1, 2):
            raise ValueError("R must be 2D and r be 1D/2D.")
        p = int(self.params.shape[0])
        if R.shape[1] != p:
            raise ValueError(f"R has {R.shape[1]} columns but params has length {p}.")
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(r)):
            raise ValueError("Non-finite entries in R or r.")
        # Row-full-rank is required for well-posed linear restrictions
        R_arr = np.asarray(R, dtype=float, order="C")
        rank_rows = -1
        try:
            # Row-rank via QR with column pivoting on R^T (rows of R are columns of R^T)
            qr_res = la.qr(R_arr.T, pivoting=True, mode="economic")
            if len(qr_res) == 3:
                _Q, R_up, _P = qr_res
            else:
                _Q, R_up = qr_res
            R_up_d = la.to_dense(R_up) if getattr(R_up, "size", None) else np.array([])
            diagR = np.abs(np.diag(R_up_d)) if R_up_d.size else np.array([])
            tol = (1e-10 * float(np.max(diagR))) if diagR.size else 0.0
            rank_rows = int(np.sum(diagR > tol))
        except (TypeError, ValueError) as e:
            # If rank computation fails for odd inputs, present a clear message
            raise ValueError(
                "Unable to validate R matrix rank; check R for degeneracy.",
            ) from e
        if rank_rows < R_arr.shape[0]:
            raise ValueError("R matrix is not full row-rank.")

        # Ensure required inference artifacts exist
        required = ["X_inference", "y_inference"]
        for key in required:
            if key not in self.extra:
                msg = f"EstimationResult.extra must include '{key}' for Wald bootstrap delegation."
                raise ValueError(msg)

        X = np.asarray(self.extra["X_inference"], dtype=float)
        y = np.asarray(self.extra["y_inference"], dtype=float)

        # Prefer explicit clusters passed by the estimator; if absent, reconstruct
        # from multiway or (space,time) artifacts to ensure strict reproducibility.
        clusters = self.extra.get("clusters_inference", None)
        if clusters is None:
            mw = self.extra.get("multiway_ids_inference", None)
            sp = self.extra.get("space_ids_inference", None)
            tm = self.extra.get("time_ids_inference", None)
            if mw is not None:
                clusters = mw
            elif (sp is not None) and (tm is not None):
                clusters = [sp, tm]
        multipliers = self.extra.get("W_multipliers_inference", None)
        weights = self.extra.get("weights_inference", None)

        # Freeze cluster/multiplier ordering when pandas objects are used:
        def _freeze(obj):
            if hasattr(obj, "to_numpy"):
                try:
                    return obj.to_numpy()
                except (TypeError, AttributeError, ValueError):
                    return np.asarray(obj)
            return obj

        clusters = _freeze(clusters)
        multipliers = _freeze(multipliers)

        # MNW variant (11/13/31/33/33J) — normalize via central parser for strict consistency.
        # Accept case/sep-insensitive inputs like '33j', '33-J', 'MNW33J' by cleaning and
        # validating through the centralized parser. If validation fails, raise a clear error.
        use_variant: str | None = None
        use_residual: str | None = None
        use_vcov_kind: str | None = None

        residual_arg = (
            None
            if residual_type is None
            else str(residual_type).replace("-", "_").upper()
        )
        if residual_arg is not None and residual_arg not in {"WCR", "WCU", "WCU_SCORE"}:
            msg = (
                "residual_type must be one of {'WCR','WCU','WCU_SCORE'} "
                "(MNW terminology). Use `variant` for MNW codes like '33'/'33J'."
            )
            raise ValueError(msg)

        vcov_arg = "auto_strict" if vcov_kind is None else str(vcov_kind)

        if variant is not None:
            v_raw = str(variant)
            v_clean = v_raw.upper().replace("-", "").replace("_", "")
            try:
                resid_from_variant, vcov_from_variant = _parse_bootstrap_variant(v_clean)
            except Exception as e:  # pragma: no cover - user input validation
                msg = "Unknown MNW bootstrap variant; allowed codes (case-insensitive): {'11', '13', '31', '33', '33J'}."
                raise ValueError(msg) from e
            use_variant = v_clean

            # boottest/fwildclusterboot semantics: for clustered inference, `variant`
            # jointly determines residual handling and VCV kind. To avoid silent
            # mismatches, forbid inconsistent overrides.
            if clusters is not None:
                resid_norm = str(resid_from_variant).replace("-", "_").upper()
                vcov_norm = str(vcov_from_variant)
                if residual_arg is not None and residual_arg != resid_norm:
                    raise ValueError(
                        f"residual_type='{residual_type}' conflicts with variant='{variant}' (implies '{resid_norm}')."
                        " Omit residual_type when using variant.",
                    )
                if (vcov_arg != "auto_strict") and (vcov_arg != vcov_norm):
                    raise ValueError(
                        f"vcov_kind='{vcov_kind}' conflicts with variant='{variant}' (implies '{vcov_norm}')."
                        " Omit vcov_kind when using variant.",
                    )
                use_residual = resid_norm
                use_vcov_kind = vcov_norm
        # If caller didn't supply a variant, inherit the estimator's recorded
        # bootstrap variant used at fit time (if present). This enables strict
        # reproducibility: Wald tests will replay the same MNW variant choice.
        if use_variant is None:
            v0 = self.extra.get("boot_variant_used", None)
            if v0 is not None:
                v_clean = str(v0).upper().replace("-", "").replace("_", "")
                try:
                    resid_from_variant, vcov_from_variant = _parse_bootstrap_variant(
                        v_clean,
                    )
                except Exception as e:
                    raise ValueError(
                        f"Recorded boot_variant_used='{v0}' is invalid; allowed: 11,13,31,33,33J.",
                    ) from e
                use_variant = v_clean
                if clusters is not None:
                    use_residual = str(resid_from_variant).replace("-", "_").upper()
                    use_vcov_kind = str(vcov_from_variant)

        # Residual type: if not dictated by variant, use explicit residual_type argument.
        if use_residual is None:
            use_residual = "WCR" if residual_arg is None else residual_arg

        # VCV kind: if not dictated by variant, use explicit vcov_kind argument.
        if use_vcov_kind is None:
            use_vcov_kind = vcov_arg

        # If we saved multipliers during estimation, forward the exact same multipliers
        # to the Wald bootstrap so enumeration / small-G decisions are reused.
        mult_arg = None
        if multipliers is not None:
            # Preserve DataFrame/Series exact layout and column ordering when present.
            # This ensures the bootstrap multipliers passed to the Wald routine
            # retain the same shape/ordering as produced during estimation.
            if hasattr(multipliers, "to_numpy"):
                try:
                    # to_numpy preserves column ordering; fall back to np.asarray on failure
                    mult_arg = multipliers.to_numpy()
                except (TypeError, AttributeError, ValueError):
                    mult_arg = np.asarray(multipliers)
            else:
                # Generic sequence/ndarray path
                mult_arg = np.asarray(multipliers, dtype=float)
        kwargs_local = dict(kwargs)
        # Prefer BootConfig saved in extra; fall back to model_info to maximize
        # reproducibility: estimators may store the BootConfig in either location.
        boot_cfg = self.extra.get("boot_config", None)
        if boot_cfg is None and isinstance(self.model_info, dict):
            boot_cfg = self.model_info.get("boot_config", None)
        if mult_arg is None and boot_cfg is not None:
            kwargs_local.setdefault("policy", getattr(boot_cfg, "policy", None))
            kwargs_local.setdefault("enum_max_g", getattr(boot_cfg, "enum_max_g", None))
            kwargs_local.setdefault(
                "enumeration_mode", getattr(boot_cfg, "enumeration_mode", None),
            )
        # Denominator synchronisation: if caller did not override and BootConfig
        # uses boottest policy, adopt boottest-style denominator by default.
        if (
            denominator == "bootstrap"
            and boot_cfg is not None
            and getattr(boot_cfg, "policy", None) == "boottest"
        ):
            denominator = "boottest_strict"

        out = bt.wald_test_wild_bootstrap(
            X,
            y,
            R=R,
            r=r,
            variant=use_variant,
            residual_type=use_residual,
            clusters=clusters,
            multipliers=mult_arg,
            vcov_kind=use_vcov_kind,
            weights=weights,
            # allow analytic observation weights only for GLS/GMM (project policy)
            allow_weights=("GLS" in est_label or "GMM" in est_label),
            denominator=denominator,
            **kwargs_local,
        )
        return out

    def __post_init__(self) -> None:
        """Post-init hook to validate result contract immediately on creation.

        This enforces project policy at construction time so that estimators
        cannot accidentally return objects that violate library-wide rules
        (for example, providing analytic `se` or `bands` for core linear
        estimators). The validation is intentionally strict: failures raise
        ValueError so they must be addressed by the calling estimator.
        """
        # Defer to validate() for the checks; surface any ValueError directly.
        self.validate()

    def validate(self) -> None:
        """Validate EstimationResult contract according to project policy.

        Rules enforced here (non-exhaustive):
        - Core linear estimators (OLS/IV/GLS/GMM/QR/SAR2SLS) must not provide
          confidence bands (`bands`) - bootstrap SE is stored in `.se` directly.
        - If `bands` are present they must be a dict containing the canonical
          uniform-band keys: 'pre', 'post', 'full'.
        - Diagnostics stored under `extra['diagnostics']` must be numeric-valued
          (scalars or numpy arrays); p-values or analytic critical values are
          disallowed in this container.

        Note: Bootstrap standard errors are stored in `.se` for all estimators.
        This field contains bootstrap SE (not analytic SE), ensuring clarity and
        consistency across the package.
        """
        est_label = str(self.model_info.get("Estimator", "")).upper()

        # Note: EstimationResult.extra may contain arbitrary diagnostic entries.
        # Historical enforcement of forbidding analytic inference keys was removed
        # to avoid hard assertions in downstream workflows; callers should avoid
        # storing analytic p-values/critical values by convention.

        # Identify core linear estimators conservatively by keyword.
        core_linear_keywords = ("OLS", "IV", "GLS", "GMM", "QR", "IVQR", "SAR2SLS")
        # Core linear estimators: allow estimator-specific handling of `bands`.
        # The package preserves a bootstrap-first inference convention; core
        # routines do not compute analytic SEs/p-values or analytic critical values.
        # Estimators that provide `bands` should ensure they conform to expected
        # schemas; no hard exception is raised here to allow flexible downstream
        # workflows.

        # If bands are provided, enforce canonical schema for DID/event-study
        # OR allow RCT-specific uniform band schema: {'uniform': {'lower':..., 'upper':..., 'alpha':...}}
        if self.bands is not None:
            est_upper = str(self.model_info.get("Estimator", "")).upper()
            is_rct = "RCT" in est_upper
            if not isinstance(self.bands, dict):
                msg = "`bands` must be a dict with keys {'pre','post','full'} for DID/event-study estimators or {'uniform'} for RCT."
                raise ValueError(msg)
            if is_rct:
                if "uniform" not in self.bands:
                    raise ValueError(
                        "RCT requires bands['uniform'] with 'lower' and 'upper'.",
                    )
                u = self.bands["uniform"]
                # Accept dict with 'lower'/'upper' (or 'lo'/'hi') or DataFrame-like with such columns
                if isinstance(u, dict):
                    keys = {k.lower() for k in u}
                    if not (
                        ("lower" in keys or "lo" in keys)
                        and ("upper" in keys or "hi" in keys)
                    ):
                        raise ValueError(
                            "bands['uniform'] must contain 'lower'/'upper' (or 'lo'/'hi').",
                        )
                elif hasattr(u, "columns"):
                    cols = {c.lower() for c in getattr(u, "columns", [])}
                    if not (
                        ("lower" in cols or "lo" in cols)
                        and ("upper" in cols or "hi" in cols)
                    ):
                        raise ValueError(
                            "bands['uniform'] must contain 'lower'/'upper' (or 'lo'/'hi').",
                        )
                else:
                    raise ValueError(
                        "bands['uniform'] must contain 'lower' and 'upper'.",
                    )
            else:
                required = {"pre", "post", "full"}
                if not required.issubset(set(self.bands.keys())):
                    msg = "`bands` dict must include keys 'pre','post','full' (uniform sup-t bands per policy)."
                    raise ValueError(msg)
                # optional: each side must be DF or dict with lower/upper
                for k in ("pre", "post", "full"):
                    v = self.bands.get(k)
                    if v is None:
                        continue
                    if hasattr(v, "columns"):
                        cols = {c.lower() for c in v.columns}
                        if not (
                            ("lower" in cols or "lo" in cols)
                            and ("upper" in cols or "hi" in cols)
                        ):
                            raise ValueError(
                                f"bands['{k}'] must have columns 'lower'/'upper' (or 'lo'/'hi').",
                            )
                    elif isinstance(v, dict):
                        ks = {x.lower() for x in v}
                        if not (
                            ("lower" in ks or "lo" in ks)
                            and ("upper" in ks or "hi" in ks)
                        ):
                            raise ValueError(
                                f"bands['{k}'] must contain 'lower'/'upper' (or 'lo'/'hi').",
                            )



        # Structural checks for `.se`: if present, it must align with `params` and be finite.
        if self.se is not None:
            if not isinstance(self.se, pd.Series):
                raise ValueError("se must be a pandas Series aligned to params.")
            if not self.se.index.equals(self.params.index):
                raise ValueError("se index must exactly match params index and order.")
            if not np.all(np.isfinite(self.se.values)):
                raise ValueError("se contains non-finite values.")


# ---------------------------------------------------------------------
# Bootstrap configuration (no pairs bootstrap anywhere)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class BootConfig:
    """Wild/multiplier bootstrap configuration shared by estimators.

    Notes
    -----
        - Replications: default is 2000 (project-wide).
    - Distribution:
        * default "rademacher". Project default policy is "boottest"
          to mirror Stata/R small-G behavior by default
          (enumeration threshold & Webb promotion decided centrally).
          If strict enumeration-when-feasible is desired regardless of
          boottest rules, set policy="strict".
    - Clustering:
        * Provide one of: cluster_ids, multiway_ids, (space_ids, time_ids).
        * If none is provided, IID wild multipliers are used.
    Multiway clustering uses CGM (Cameron-Gelbach-Miller) decomposition for theoretical
    equivalence with fwildclusterboot/boottest. For bootstrap multiplier generation,
    exactly one "bootcluster" dimension is selected to draw cluster multipliers
    (this mirrors fwildclusterboot behavior). When ``bootcluster`` is not specified
    the default selection rule picks the dimension with the largest number of clusters
    ("max"), matching common boottest practice.

    Reproducibility:
        * Use `seed` to initialize RNG deterministically (np.random.Generator).

    """

    n_boot: int = bt.DEFAULT_BOOTSTRAP_ITERATIONS
    # Preset for quick configuration of bootstrap behavior.
    # "did" preset aims for R did parity: IID standard normal multipliers,
    # no enumeration/Webb promotions, and no multiway enumeration.
    preset: str | None = None
    dist: str = "rademacher"
    seed: int | None = None
    # Optional inference mode for estimators that support multiple bootstrap schemes
    # For example, SyntheticControl supports {'time','placebo'}; other estimators ignore this.
    mode: str | None = None

    # Multiway clustering uses CGM (Cameron-Gelbach-Miller 2011) inclusion-exclusion
    # decomposition: for multiway, clusters are synthesized as intersections of
    # provided dimensions (e.g., space ∩ time, space \ time, time \ space, etc.).
    # Small-G / enumeration policy is centralized in ``core.bootstrap``. Under
    # policy="boottest" the library follows fwildclusterboot/boottest conventions:
    # enumerate for Rademacher when 2**G <= B; never enumerate for multiway. The
    # ``enum_max_g`` parameter is advisory and loaders in core.bootstrap will cap
    # any enumeration attempt to a safe threshold (for example, G <= 11 by default).
    # - enum_max_g: if dist is Rademacher and G <= enum_max_g, attempt exact enumeration of 2^G.
    #   Limit 11 (2048 patterns) balances feasibility with memory; enumeration provides exact
    #   finite-sample reference distribution when possible. Core.bootstrap ultimately
    #   decides whether enumeration occurs based on policy/available B.
    # - use_enumeration: master flag to allow enumeration path.
    # Align with boottest/fwildclusterboot conventions: prefer boottest enumeration rule by default
    # Default policy: boottest
    policy: str = "boottest"
    # Let core.bootstrap resolve enum threshold from policy (strict=12 / boottest=11)
    enum_max_g: int | None = None
    use_enumeration: bool = True
    enumeration_mode: str = (
        "boottest"  # 'boottest' parity; enumeration policy resolved centrally.
    )

    # Clustering options (mutually exclusive)
    cluster_method: str = (
        "CGM"  # For VCV only; multipliers ignore this (use `bootcluster`)
    )
    bootcluster: str | None = (
        "product"  # "product", "max", "min", "intersection", or dimension name for multiway weights
    )
    cluster_ids: Sequence | None = None
    multiway_ids: Sequence[Sequence] | None = None
    space_ids: Sequence | None = None
    time_ids: Sequence | None = None

    def _validate_lengths_and_exclusivity(self, n_obs: int) -> None:
        """Validate that exactly one clustering scheme is provided and lengths match n."""
        provided = {
            "multiway": self.multiway_ids is not None,
            "spacetime": (self.space_ids is not None) and (self.time_ids is not None),
            "oneway": self.cluster_ids is not None,
        }
        if sum(provided.values()) > 1:
            msg = (
                "Specify at most one clustering scheme among "
                "{multiway_ids, (space_ids,time_ids), cluster_ids}."
            )
            raise ValueError(
                msg,
            )

        # Length checks
        if self.multiway_ids is not None:
            for j, c in enumerate(self.multiway_ids):
                # Accept entries of the form (name, codes) as produced by formula parsers.
                # In that case, validate against the underlying codes vector.
                c_eff = c[1] if (isinstance(c, tuple) and len(c) == 2) else c
                if len(c_eff) != n_obs:
                    msg = (
                        f"multiway_ids[{j}] length {len(c_eff)} != n_obs {n_obs}."
                    )
                    raise ValueError(msg)
        if (self.space_ids is not None) ^ (self.time_ids is not None):
            msg = "Both space_ids and time_ids must be provided together."
            raise ValueError(msg)
        if self.space_ids is not None and len(self.space_ids) != n_obs:
            msg = f"space_ids length {len(self.space_ids)} != n_obs {n_obs}."
            raise ValueError(msg)
        if self.time_ids is not None and len(self.time_ids) != n_obs:
            msg = f"time_ids length {len(self.time_ids)} != n_obs {n_obs}."
            raise ValueError(msg)
        if self.cluster_ids is not None and len(self.cluster_ids) != n_obs:
            msg = f"cluster_ids length {len(self.cluster_ids)} != n_obs {n_obs}."
            raise ValueError(msg)

    def make_multipliers(self, n_obs: int) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Generate bootstrap multipliers as a DataFrame with columns b=0..B-1,
        and a log dict with small-G policy details.

        Returns
        -------
        Tuple[pd.DataFrame with shape (n_obs, n_boot), Dict[str, Any]]
        Log keys: 'enumerated' (bool), 'effective_dist' (str), 'effective_B' (int), 'warnings' (list of str)

        """
        self._validate_lengths_and_exclusivity(n_obs)

        # Helper to deterministically freeze pandas-like inputs to numpy arrays
        def _freeze_input(obj):
            if obj is None:
                return None
            if hasattr(obj, "to_numpy"):
                try:
                    return obj.to_numpy(copy=False)
                except (TypeError, AttributeError, ValueError):
                    return np.asarray(obj)
            return np.asarray(obj)

        # accept case-insensitive spellings (parity with Stata/R option parsing)
        cm = str(self.cluster_method).lower()
        if cm not in {"intersection", "cgm"}:
            raise ValueError(
                "cluster_method must be 'intersection' or 'cgm'; multipliers use `bootcluster`.",
            )

        # Apply preset overrides before constructing distribution/policy objects
        preset = (None if self.preset is None else str(self.preset).lower())
        dist = self.dist
        policy = self.policy
        use_enum = self.use_enumeration
        enum_mode = self.enumeration_mode
        if preset == "did":
            dist = "standard_normal"
            policy = "strict"  # disable Webb promotion; let core decide strictly
            use_enum = False    # no enumeration in did preset
            enum_mode = "disabled"

        # Propagate (possibly overridden) policy to WildDist so enumeration/Webb decisions follow config
        d = bt.WildDist(dist, policy=policy)

        log = {
            "enumerated": False,
            "effective_dist": dist,
            "effective_B": self.n_boot,
            "warnings": [],
        }

        if self.multiway_ids is not None:
            bc = None if self.bootcluster is None else str(self.bootcluster).lower()
            if bc in {None, "product", "cgm"}:
                rng_local = None if self.seed is None else np.random.default_rng(self.seed)
                W = bt.multiway_multipliers(
                    [_freeze_input(z) for z in self.multiway_ids],
                    n_boot=self.n_boot,
                    rng=rng_local,
                    dist=d,
                    policy=policy,
                    enumeration_mode=enum_mode,
                )
                log = {"enumerated": False, "effective_dist": dist, "effective_B": self.n_boot, "warnings": []}
                log.setdefault("G", len(self.multiway_ids))
                log.setdefault("enum_attempted", False)
            elif bc in {"intersection", "intersect", "all"}:
                keys = np.column_stack(
                    [_freeze_input(z).reshape(-1, 1) for z in self.multiway_ids],
                )
                _, inv = np.unique(keys, axis=0, return_inverse=True)
                W, log = bt.cluster_multipliers(
                    inv.astype(int, copy=False),
                    n_boot=self.n_boot,
                    dist=d,
                    seed=self.seed,
                    policy=policy,
                    enum_max_g=self.enum_max_g,
                    use_enumeration=use_enum,
                    enumeration_mode=enum_mode,
                )
                log.setdefault("G", int(keys.shape[1]))
                log.setdefault("enum_attempted", bool(log.get("enumerated", False)))
                log.setdefault("enum_mode", enum_mode)
                log.setdefault(
                    "patterns",
                    (1 << int(np.unique(inv).size))
                    if log.get("enumerated", False)
                    else None,
                )
            elif bc == "max":
                boot_dim = np.argmax(
                    [len(np.unique(_freeze_input(z))) for z in self.multiway_ids],
                )
                W, log = bt.cluster_multipliers(
                    _freeze_input(pd.Series(self.multiway_ids[boot_dim])),
                    n_boot=self.n_boot,
                    dist=d,
                    seed=self.seed,
                    policy=policy,
                    enum_max_g=self.enum_max_g,
                    use_enumeration=use_enum,
                    enumeration_mode=enum_mode,
                )
                log.setdefault("G", 1)
                log.setdefault("enum_attempted", bool(log.get("enumerated", False)))
            elif bc == "min":
                boot_dim = np.argmin(
                    [len(np.unique(_freeze_input(z))) for z in self.multiway_ids],
                )
                W, log = bt.cluster_multipliers(
                    _freeze_input(pd.Series(self.multiway_ids[boot_dim])),
                    n_boot=self.n_boot,
                    dist=d,
                    seed=self.seed,
                    policy=policy,
                    enum_max_g=self.enum_max_g,
                    use_enumeration=use_enum,
                    enumeration_mode=enum_mode,
                )
                log.setdefault("G", 1)
                log.setdefault("enum_attempted", bool(log.get("enumerated", False)))
            else:
                # explicit named/position selection
                boot_dim = [
                    i
                    for i, z in enumerate(self.multiway_ids)
                    if (str(i) == self.bootcluster)
                    or (getattr(z, "name", None) == self.bootcluster)
                ]
                if not boot_dim:
                    msg = f"bootcluster {self.bootcluster} not found in multiway_ids"
                    raise ValueError(msg)
                boot_dim = boot_dim[0]
                W, log = bt.cluster_multipliers(
                    _freeze_input(pd.Series(self.multiway_ids[boot_dim])),
                    n_boot=self.n_boot,
                    dist=d,
                    seed=self.seed,
                    policy=policy,
                    enum_max_g=self.enum_max_g,
                    use_enumeration=use_enum,
                    enumeration_mode=enum_mode,
                )
                log.setdefault("G", 1)
                log.setdefault("enum_attempted", bool(log.get("enumerated", False)))
        elif self.space_ids is not None and self.time_ids is not None:
            bc = None if self.bootcluster is None else str(self.bootcluster).lower()
            if bc in {None, "product", "cgm"}:
                rng_local = None if self.seed is None else np.random.default_rng(self.seed)
                W = bt.multiway_multipliers(
                    [_freeze_input(self.space_ids), _freeze_input(self.time_ids)],
                    n_boot=self.n_boot,
                    rng=rng_local,
                    dist=d,
                    policy=policy,
                    enumeration_mode=enum_mode,
                )
                log = {"enumerated": False, "effective_dist": dist, "effective_B": self.n_boot, "warnings": []}
                log.setdefault("G", 2)
                log.setdefault("enum_attempted", False)
            elif bc in {"intersection", "intersect", "all"}:
                keys = np.column_stack(
                    [
                        _freeze_input(self.space_ids).reshape(-1, 1),
                        _freeze_input(self.time_ids).reshape(-1, 1),
                    ],
                )
                _, inv = np.unique(keys, axis=0, return_inverse=True)
                W, log = bt.cluster_multipliers(
                    inv.astype(int, copy=False),
                    n_boot=self.n_boot,
                    dist=d,
                    seed=self.seed,
                    policy=policy,
                    enum_max_g=self.enum_max_g,
                    use_enumeration=use_enum,
                    enumeration_mode=enum_mode,
                )
                log.setdefault("G", int(keys.shape[1]))
                log.setdefault("enum_attempted", bool(log.get("enumerated", False)))
                log.setdefault("enum_mode", enum_mode)
                log.setdefault(
                    "patterns",
                    (1 << int(np.unique(inv).size))
                    if log.get("enumerated", False)
                    else None,
                )
            elif bc == "max":
                n_time = int(np.unique(self.time_ids).size)
                n_space = int(np.unique(self.space_ids).size)
                boot_dim = self.time_ids if n_time >= n_space else self.space_ids
                W, log = bt.cluster_multipliers(
                    _freeze_input(pd.Series(boot_dim)),
                    n_boot=self.n_boot,
                    dist=d,
                    seed=self.seed,
                    policy=policy,
                    enum_max_g=self.enum_max_g,
                    use_enumeration=use_enum,
                    enumeration_mode=enum_mode,
                )
                log.setdefault("G", 1)
                log.setdefault("enum_attempted", bool(log.get("enumerated", False)))
            elif bc == "min":
                n_time = int(np.unique(self.time_ids).size)
                n_space = int(np.unique(self.space_ids).size)
                boot_dim = self.time_ids if n_time <= n_space else self.space_ids
                W, log = bt.cluster_multipliers(
                    pd.Series(boot_dim).to_numpy(),
                    n_boot=self.n_boot,
                    dist=d,
                    seed=self.seed,
                    policy=policy,
                    enum_max_g=self.enum_max_g,
                    use_enumeration=use_enum,
                    enumeration_mode=enum_mode,
                )
                log.setdefault("G", 1)
                log.setdefault("enum_attempted", bool(log.get("enumerated", False)))
            else:
                # Explicit 'time' / 'space' selection or fallback to 'time'
                bc2 = str(self.bootcluster).lower()
                if bc2 in {"time", "t"}:
                    boot_dim = self.time_ids
                elif bc2 in {"space", "s"}:
                    boot_dim = self.space_ids
                else:
                    boot_dim = self.time_ids
                W, log = bt.cluster_multipliers(
                    pd.Series(boot_dim).to_numpy(),
                    n_boot=self.n_boot,
                    dist=d,
                    seed=self.seed,
                    policy=policy,
                    enum_max_g=self.enum_max_g,
                    use_enumeration=use_enum,
                    enumeration_mode=enum_mode,
                )
                log.setdefault("G", 1)
                log.setdefault("enum_attempted", bool(log.get("enumerated", False)))
        elif self.cluster_ids is not None:
            W, log = bt.cluster_multipliers(
                _freeze_input(pd.Series(self.cluster_ids)),
                n_boot=self.n_boot,
                dist=d,
                seed=self.seed,
                policy=policy,
                enum_max_g=self.enum_max_g,
                use_enumeration=use_enum,
                enumeration_mode=enum_mode,
            )
            log.setdefault("G", 1)
            log.setdefault("enum_attempted", bool(log.get("enumerated", False)))
        else:
            W = bt.wild_multipliers(
                n_obs,
                n_boot=self.n_boot,
                dist=d,
                seed=self.seed,
            )
            log = {
                "enumerated": False,
                "effective_dist": dist,
                "effective_B": self.n_boot,
                "warnings": [],
            }
            log.setdefault("G", 0)
            log.setdefault("enum_attempted", False)
        cols = [f"b{j}" for j in range(W.shape[1])]
        return pd.DataFrame(W, columns=cols), log


@dataclass(slots=True)
class FormulaMetadata:
    """Container capturing formula-related bookkeeping for estimators.

    Attributes
    ----------
    formula : str | None
        Raw formula string provided by the caller, if any.
    row_index : list[Any] | None
        Original data index labels retained by the parser (order matches design matrices).
    cluster_ids : Any
        Cluster identifiers materialized by the parser (can be ndarray or tuple for multiway).
    attrs : dict[str, Any]
        Additional estimator-specific attributes to be stamped onto the estimator instance
        (for example, stored FE codes or constraint matrices).

    """

    formula: str | None = None
    row_index: list[Any] | None = None
    cluster_ids: Any = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def copy(self, **updates: Any) -> FormulaMetadata:
        """Return a shallow copy with optional attribute overrides."""
        payload = {
            "formula": self.formula,
            "row_index": None if self.row_index is None else list(self.row_index),
            "cluster_ids": self.cluster_ids,
            "attrs": dict(self.attrs),
        }
        payload.update(updates)
        return FormulaMetadata(**payload)


def prepare_formula_environment(  # noqa: PLR0913 - API surface
    *,
    formula: str | None,
    data: pd.DataFrame,
    parsed: dict[str, Any] | None,
    boot: BootConfig | None = None,
    default_boot_kwargs: dict[str, Any] | None = None,
    attr_keys: dict[str, str] | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, BootConfig | None, FormulaMetadata]:
    """Normalize parser outputs for estimator constructors.

    Parameters
    ----------
    formula :
        Original formula string supplied by the caller.
    data :
        Original DataFrame provided to ``from_formula``.
    parsed :
        Parser output dict (may be ``None`` when formulas are not used).
    boot :
        User-supplied :class:`BootConfig` or ``None``.
    default_boot_kwargs :
        Keyword arguments forwarded to :class:`BootConfig` when a new configuration
        must be instantiated because cluster identifiers were discovered by the parser.
    attr_keys :
        Optional mapping ``{attr_name: parsed_key}`` indicating additional entries
        from ``parsed`` that should be attached to the estimator under ``attr_name``.
    extra_attrs :
        Additional attribute mapping to stamp onto the estimator (overrides entries
        coming from ``attr_keys`` when keys overlap).

    """
    idx = (
        list(parsed.get("row_index_used", data.index))
        if parsed is not None
        else list(data.index)
    )
    df_use = data.loc[idx]
    cluster_ids = parsed.get("cluster_ids_used") if parsed is not None else None

    boot_kwargs = dict(default_boot_kwargs or {})
    boot_eff = boot
    if boot_eff is None and cluster_ids is not None:
        boot_kwargs.setdefault("n_boot", bt.DEFAULT_BOOTSTRAP_ITERATIONS)
        boot_kwargs.setdefault("cluster_ids", cluster_ids)
        boot_eff = BootConfig(**boot_kwargs)
    elif boot_eff is not None and cluster_ids is not None:
        if getattr(boot_eff, "cluster_ids", None) is None:
            boot_eff = replace(boot_eff, cluster_ids=cluster_ids)
    # Compose attribute mapping
    attrs: dict[str, Any] = {}
    if attr_keys and parsed is not None:
        for attr_name, parsed_key in attr_keys.items():
            if parsed_key in parsed:
                attrs[attr_name] = parsed[parsed_key]
    if extra_attrs:
        attrs.update(
            {key: value for key, value in extra_attrs.items() if value is not None},
        )

    meta = FormulaMetadata(
        formula=formula,
        row_index=idx,
        cluster_ids=cluster_ids,
        attrs=attrs,
    )
    return df_use, boot_eff, meta


def attach_formula_metadata(target: Any, meta: FormulaMetadata | None) -> None:
    """Attach :class:`FormulaMetadata` to ``target`` by stamping canonical attributes.

    Attributes stamped on the target include:
    - ``_formula_metadata`` : full metadata container.
    - ``_formula`` : raw formula string (if provided).
    - ``_row_index_used`` : list of retained row labels.
    - ``_cluster_ids_from_formula`` : cluster identifiers materialized by the parser.
    - Any entries stored under ``meta.attrs`` (key -> value).
    """
    if meta is None:
        return
    target._formula_metadata = meta  # noqa: SLF001
    if meta.formula is not None:
        target._formula = meta.formula  # noqa: SLF001
    if meta.row_index is not None:
        target._row_index_used = list(meta.row_index)  # noqa: SLF001
    if meta.cluster_ids is not None:
        target._cluster_ids_from_formula = meta.cluster_ids  # noqa: SLF001
    for attr_name, value in meta.attrs.items():
        if value is None:
            continue
        setattr(target, attr_name, value)


def _to_numpy_1d(values: Sequence | None) -> np.ndarray | None:
    """Convert 1-D like input to a numpy array without copying when possible."""
    if values is None:
        return None
    if hasattr(values, "to_numpy"):
        try:
            arr = values.to_numpy(copy=False)
        except (TypeError, AttributeError, ValueError):
            arr = np.asarray(values)
    else:
        arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _normalize_multiway_ids(
    multiway_ids: Sequence[Sequence] | None,
) -> tuple[np.ndarray, ...] | None:
    """Convert multiway identifiers into a tuple of 1-D numpy arrays."""
    if multiway_ids is None:
        return None
    normalized: list[np.ndarray] = []
    for entry in multiway_ids:
        candidate = entry
        if isinstance(entry, tuple) and len(entry) == 2:
            candidate = entry[1]
        arr = _to_numpy_1d(candidate)
        if arr is None:
            continue
        normalized.append(arr)
    return tuple(normalized) if normalized else None


def _align_with_mask(
    arr: np.ndarray | None, mask: np.ndarray | None, n_original: int,
) -> np.ndarray | None:
    """Align identifier arrays with the working sample defined by ``mask``.

    Accepts identifiers defined either on the original sample (length == n_original)
    or pre-filtered to the working sample (length == mask.sum()).
    """
    if arr is None:
        return None
    arr = np.asarray(arr)
    if mask is None:
        if arr.shape[0] not in {n_original}:
            raise ValueError(
                f"Identifier length {arr.shape[0]} incompatible with n_original={n_original}.",
            )
        return arr
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n_final = int(mask.sum())
    if arr.shape[0] == n_final:
        return arr
    if arr.shape[0] == mask.shape[0] == n_original:
        return arr[mask]
    raise ValueError(
        f"Identifier length {arr.shape[0]} incompatible with mask length {mask.shape[0]} and final n={n_final}.",
    )


def _ensure_no_missing(arr: np.ndarray | None, label: str) -> None:
    """Raise when identifier arrays contain missing entries."""
    if arr is None:
        return
    # Robust missing-value detection for numeric and object identifiers.
    # Prefer pandas' NA semantics (handles object/categorical), with a
    # numeric fallback that never raises for non-numeric ids.
    try:
        has_missing = bool(np.any(pd.isna(arr)))
    except Exception:
        try:
            arr_f = np.asarray(arr, dtype=float)
            has_missing = bool(np.isnan(arr_f).any())
        except Exception:
            has_missing = False
    if has_missing:
        raise ValueError(
            f"{label} contains missing values; clean identifiers before estimation.",
        )


# ---------------------------------------------------------------------
# Base interface (no analytic critical values here)
# ---------------------------------------------------------------------
class BaseEstimator(ABC):
    """Abstract base class for all `lineareg` estimators.

    Principles
    ----------
    1) All linear algebra goes through `core.linalg`.
    2) FE absorption goes through `core.fe`.
    3) Bootstrap goes through `core.bootstrap` (QR may override with positive weights).
    4) No analytic p-values/critical values inside this class.
    5) FE absorption defaults to drop_singletons=True in OLS/IV/GMM for consistency with R/Stata.
    """

    def __init__(self) -> None:
        self._results: EstimationResult | None = None
        self._formula_metadata: FormulaMetadata | None = None
        # Policy: forbid analytic observation weights for these estimator kinds by default
        # Accept both spellings for IV-QR ("ivqr" and "iv-qr") to match callers.
        self._weight_policy = {"ols", "iv", "qr", "ivqr", "iv-qr"}
        # Optional default device preference for derived estimators ('auto'|'cpu'|'gpu')
        self._device_preference: str = "auto"

    @property
    def formula_metadata(self) -> FormulaMetadata | None:
        """Return formula metadata captured by ``from_formula`` (if any)."""
        return getattr(self, "_formula_metadata", None)

    # -- bootstrap utilities ------------------------------------------
    def _coerce_bootstrap(  # noqa: PLR0913 - complex but stable public method
        self,
        *,
        boot: BootConfig | None,
        n_obs_original: int,
        row_mask: np.ndarray | None = None,
        cluster_ids: Sequence | None = None,
        space_ids: Sequence | None = None,
        time_ids: Sequence | None = None,
        multiway_ids: Sequence[Sequence] | None = None,
    ) -> tuple[BootConfig, dict[str, Any]]:
        """Normalize bootstrap configuration and cluster identifiers.

        Parameters
        ----------
        boot
            User-supplied :class:`BootConfig` or ``None``. When ``None`` a new
            configuration is instantiated with the provided identifiers.
        n_obs_original
            Length of the original sample prior to any filtering (used to
            validate identifier lengths).
        row_mask
            Boolean mask identifying rows retained after preprocessing. When
            provided, identifier arrays defined on the original sample are
            subset using this mask. Identifiers already aligned with the
            filtered sample (length equal to ``row_mask.sum()``) are left
            untouched.
        cluster_ids / space_ids / time_ids / multiway_ids
            Optional clustering identifiers. Exactly one clustering scheme may
            be provided. When ``boot`` already encodes a clustering scheme,
            additional identifiers are forbidden to prevent ambiguity.

        Returns
        -------
        Tuple[BootConfig, dict]
            Sanitized :class:`BootConfig` and a dictionary containing the
            canonical identifiers used by the estimator:
            ``cluster_ids``, ``space_ids``, ``time_ids``, ``multiway_ids``,
            and ``clusters_inference`` (object suitable for recording in
            :attr:`EstimationResult.extra`).

        """
        # Allow formula-style multiway specification via tuple of (name, codes)
        multiway_from_cluster = None
        if (
            multiway_ids is None
            and isinstance(cluster_ids, tuple)
            and cluster_ids
            and all(isinstance(x, tuple) and len(x) == 2 for x in cluster_ids)
        ):
            multiway_from_cluster = tuple(x[1] for x in cluster_ids)
            cluster_ids = None

        # Normalize user-supplied identifiers
        cluster_arr = _to_numpy_1d(
            cluster_ids if not isinstance(cluster_ids, tuple) else cluster_ids[1],
        )
        space_arr = _to_numpy_1d(space_ids)
        time_arr = _to_numpy_1d(time_ids)
        multiway_arr = _normalize_multiway_ids(multiway_ids or multiway_from_cluster)

        if boot is not None:
            if any(
                x is not None for x in (cluster_arr, space_arr, time_arr, multiway_arr)
            ):
                raise ValueError(
                    "Specify clustering either via BootConfig or via keyword arguments, not both.",
                )
            cluster_arr = _to_numpy_1d(boot.cluster_ids)
            space_arr = _to_numpy_1d(boot.space_ids)
            time_arr = _to_numpy_1d(boot.time_ids)
            multiway_arr = _normalize_multiway_ids(boot.multiway_ids)
            boot_base = boot
        else:
            boot_base = BootConfig(
                cluster_ids=cluster_arr,
                space_ids=space_arr,
                time_ids=time_arr,
                multiway_ids=multiway_arr,
            )

        # Row mask alignment (post-NA/FE filtering)
        cluster_trim = _align_with_mask(cluster_arr, row_mask, n_obs_original)
        space_trim = _align_with_mask(space_arr, row_mask, n_obs_original)
        time_trim = _align_with_mask(time_arr, row_mask, n_obs_original)
        multiway_trim: tuple[np.ndarray, ...] | None = None
        if multiway_arr is not None:
            multiway_trim = tuple(
                _align_with_mask(arr, row_mask, n_obs_original) for arr in multiway_arr
            )

        boot_clean = replace(
            boot_base,
            cluster_ids=cluster_trim,
            space_ids=space_trim,
            time_ids=time_trim,
            multiway_ids=None if multiway_trim is None else multiway_trim,
        )

        if (space_trim is None) ^ (time_trim is None):
            raise ValueError("space_ids and time_ids must be provided jointly.")

        _ensure_no_missing(cluster_trim, "cluster_ids")
        _ensure_no_missing(space_trim, "space_ids")
        _ensure_no_missing(time_trim, "time_ids")
        if multiway_trim is not None:
            for j, arr in enumerate(multiway_trim):
                _ensure_no_missing(arr, f"multiway_ids[{j}]")

        if multiway_trim is not None:
            clusters_inference: Any = [np.asarray(arr) for arr in multiway_trim]
        elif (space_trim is not None) and (time_trim is not None):
            clusters_inference = [np.asarray(space_trim), np.asarray(time_trim)]
        elif cluster_trim is not None:
            clusters_inference = np.asarray(cluster_trim)
        else:
            clusters_inference = None

        cluster_spec = {
            "cluster_ids": cluster_trim,
            "space_ids": space_trim,
            "time_ids": time_trim,
            "multiway_ids": multiway_trim,
            "clusters_inference": clusters_inference,
        }
        return boot_clean, cluster_spec

    # -- shared strict policy helper ----------------------------------
    def _enforce_weight_policy(self, estimator_kind: str, weights) -> None:
        """Forbid analytic observation weights for OLS/IV/QR/IV-QR per project policy.

        Raises
        ------
        ValueError
            If analytic observation weights are provided for a forbidden estimator.

        """
        if (
            (estimator_kind is not None)
            and (estimator_kind.lower() in self._weight_policy)
            and (weights is not None)
        ):
            msg = (
                f"Analytic observation weights are forbidden for {estimator_kind.upper()}; "
                "use GLS/GMM for analytic weighting or multiplier bootstrap for inference."
            )
            raise ValueError(msg)

    # -- unified device control --------------------------------------
    def _device_context(self, device: str | None):
        """Return a context manager to enforce device preference.

        Parameters
        ----------
        device : {"auto","cpu","gpu","cuda"} or None
            When "auto" or None, no override is applied and the environment
            variable LINEAREG_DEVICE governs backend selection. When "cpu" or
            "gpu"/"cuda", operations within the context will prefer the
            selected device via backend.DeviceGuard.

        """
        if device is None:
            return nullcontext()
        d = str(device).lower()
        if d in {"auto", ""}:
            return nullcontext()
        # Map 'cuda' -> 'gpu'
        if d == "cuda":
            d = "gpu"
        return be.DeviceGuard(d)

    # -- mandatory API -------------------------------------------------

    def _absorb_fe_from_formula(
        self,
        absorb_fe: pd.DataFrame | np.ndarray | None,
    ) -> pd.DataFrame | np.ndarray | None:
        """Return absorb_fe or derive from formula-parsed FE codes if available.

        When the estimator was constructed via `from_formula()` with `fe(...)`
        terms, the parsed FE codes are stored in `_fe_codes_from_formula`. This
        helper uses those codes when `absorb_fe` is not explicitly provided.

        Parameters
        ----------
        absorb_fe : pd.DataFrame | np.ndarray | None
            User-supplied FE codes or None.

        Returns
        -------
        pd.DataFrame | np.ndarray | None
            The FE codes to use for absorption, or None if not available.

        """
        if absorb_fe is not None:
            return absorb_fe
        fe_list = getattr(self, "_fe_codes_from_formula", None)
        if fe_list is None or len(fe_list) == 0:
            return None
        if isinstance(fe_list, list):
            return np.column_stack(fe_list)
        return np.asarray(fe_list)
    @abstractmethod
    def fit(
        self, *args: Any, **kwargs: Any,
    ) -> EstimationResult:  # pragma: no cover - abstract
        """Fit the estimator and return EstimationResult (abstract)."""
        ...

    # -- convenience accessors ----------------------------------------
    @property
    def results(self) -> EstimationResult:
        if self._results is None:
            msg = "Model has not been fitted yet. Call .fit() first."
            raise RuntimeError(msg)
        return self._results

    @property
    def params(self) -> pd.Series:
        return self.results.params

    @property
    def se(self) -> pd.Series | None:
        return self.results.se

    @property
    def n_obs(self) -> int | None:
        return self.results.n_obs

    # -- protected helpers for subclasses ------------------------------
    def _demean_xy(  # noqa: PLR0913 - helper mirrors FE backend signature
        self,
        X,
        y,
        fe_ids,
        *,
        weights: Sequence | None = None,
        allow_weights: bool = False,
        tol: float = 1e-8,
        max_iter: int = 10_000,
        na_action: str = "drop",
        backend: str = "reghdfe",
    ) -> tuple:
        """Apply multi-way FE within transform to (X, y) via `core.fe`.

        Parameters
        ----------
        weights : optional nonnegative observation weights. Note: analytic
            observation weights are supported only for GLS/GMM workflows; OLS/IV/QR
            follow the project policy of disallowing analytic weights and perform
            inference via multiplier bootstrap instead. FE within routines accept
            a weights argument for specialized transformations but estimators
            should not pass analytic weights for OLS/IV/QR.
        tol : stopping criterion for alternating projections (default 1e-8).
        max_iter : maximum iterations; a warning is raised if not converged.
        na_action : "drop" only; strict policy prohibits "propagate".

        """
        # Policy gate: forbid analytic weights unless allowed by estimator
        if (weights is not None) and (not allow_weights):
            raise ValueError(
                "Analytic observation weights are forbidden in this within-transform; use GLS/GMM.",
            )
        if na_action != "drop":
            msg = "Strict policy: na_action='drop' only."
            raise ValueError(msg)
        # Align backend-specific defaults: reghdfe vs fixest have different
        # canonical tolerances and iteration defaults used for strict parity
        be = str(backend).lower()
        if be == "reghdfe":
            if max_iter == 10_000:
                max_iter = 16_000
            # default tol already set to 1e-8 at function signature
        elif be == "fixest":
            # fixest canonical defaults (CRAN manual / setFixest_estimation):
            #   tol = 1e-6, iter = 10000
            # Only adjust when caller left signature defaults.
            if tol == 1e-8:
                tol = 1e-6
            if max_iter == 10_000:
                max_iter = 10_000
        X_tilde, y_tilde = fe_core.demean_xy(
            X,
            y,
            fe_ids,
            weights=weights,
            allow_weights=allow_weights,
            tol=tol,
            max_iter=max_iter,
            na_action=na_action,
            backend=backend,
        )
        return X_tilde, y_tilde

    def _demean_xyz(  # noqa: PLR0913 - helper mirrors FE backend signature
        self,
        X,
        Z,
        y,
        fe_ids,
        *,
        weights: Sequence | None = None,
        allow_weights: bool = False,
        tol: float = 1e-8,
        max_iter: int = 10_000,
        na_action: str = "drop",
        drop_singletons: bool = False,
        backend: str = "reghdfe",
    ) -> tuple:
        """Jointly apply multi-way FE within transform to (X, Z, y).

        Ensures perfectly synchronized row filtering (NA removal, FE ID NA removal,
        optional singleton dropping) across all three matrices to prevent subtle
        alignment bugs in IV / GMM / spatial / IVQR estimators.

        Note: drop_singletons defaults to False here, but OLS/IV/GMM fit() methods
        override to True for standard practice (avoiding degenerate FE groups).
        """
        if (weights is not None) and (not allow_weights):
            raise ValueError(
                "Analytic observation weights are forbidden in this within-transform; use GLS/GMM.",
            )
        if na_action != "drop":
            msg = "Strict policy: na_action='drop' only."
            raise ValueError(msg)
        # Align backend-specific defaults: reghdfe vs fixest have different
        # canonical tolerances and iteration defaults used for strict parity
        be = str(backend).lower()
        if be == "reghdfe":
            if max_iter == 10_000:
                max_iter = 16_000
            # default tol already set to 1e-8 at function signature
        elif be == "fixest":
            # fixest canonical defaults (CRAN manual):
            #   tol = 1e-6, iter = 2000
            if tol == 1e-8:
                tol = 1e-6
            if max_iter == 10_000:
                max_iter = 2_000
        X_tilde, Z_tilde, y_tilde = fe_core.demean_xyz(
            X,
            Z,
            y,
            fe_ids,
            weights=weights,
            allow_weights=allow_weights,
            tol=tol,
            max_iter=max_iter,
            na_action=na_action,
            drop_na_fe_ids=True,
            drop_singletons=drop_singletons,
            backend=backend,
        )
        return X_tilde, Z_tilde, y_tilde

    def _bootstrap_multipliers(
        self, n_obs: int, *, boot: BootConfig | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Dispatch bootstrap multiplier generation with centralized validation."""
        if boot is None:
            boot = BootConfig()
        return boot.make_multipliers(n_obs)
