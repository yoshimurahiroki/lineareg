"""Demonstration of the lineareg package estimators.

This module illustrates the usage of OLS, GLS, IV, GMM, Quantile Regression,
spatial models, and various event-study estimators using simulated data.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .core import linalg as la
from .estimators import (
    GLS,
    GMM,
    IV2SLS,
    OLS,
    QR,
    SDID,
    BootConfig,
    DDDEventStudy,
    DREventStudy,
    EventStudyCS,
    SpatialDID,
    SyntheticControl,
)
from .output import event_study_auto_plot, event_study_plot, modelsummary
from .sim.montecarlo import (
    simulate_cs_data,
    simulate_ddd_data,
    simulate_dr_data,
    simulate_iv_data,
    simulate_qr_data,
    simulate_sar_data,
    simulate_sc_data,
    simulate_sdid_data,
    simulate_spatial_did_data,
)
from .spatial import SAR2SLS, moran_i

DEMO_FIG_DIR = Path(__file__).resolve().parent / "demo_output"
_LOGGER = logging.getLogger(__name__)
SOFT_FAILURE_EXCEPTIONS: tuple[type[Exception], ...] = (
    RuntimeError,
    ValueError,
    np.linalg.LinAlgError,
    KeyError,
    TypeError,
    AttributeError,
    ImportError,
    OSError,
    NotImplementedError,
)


def _save_demo_figure(fig, filename: str) -> None:
    """Save the figure to the output directory and close the handle."""
    try:
        DEMO_FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = DEMO_FIG_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:  # pragma: no cover - best effort log
        _LOGGER.debug("Figure save failed for %s: %s", filename, exc)
        print(f"  [Figure save failed: {exc}]")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        _LOGGER.debug("matplotlib unavailable while closing figure: %s", exc)
    else:
        plt.close(fig)
    print(f"  [Figure saved to {path}]")


def _run_demo_block(label: str, func: Callable[[], None]) -> None:
    """Execute a demonstration function, logging any soft failures."""
    try:
        func()
    except SOFT_FAILURE_EXCEPTIONS as exc:
        _LOGGER.debug("%s demo failed: %s", label, exc)
        print(f"\n[{label} demo failed: {exc}]")


def demo_ols_gls():
    """Demonstrate Ordinary Least Squares (OLS) and Generalized Least Squares (GLS)."""
    print("\n" + "=" * 70)
    print(" 1. OLS AND GLS DEMONSTRATION")
    print("=" * 70)

    # Simulate data (n=100 for demonstration)
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 3))
    beta_true = np.array([2.0, -1.5, 0.8])
    y = la.dot(X, beta_true.reshape(-1, 1)) + rng.standard_normal((n, 1))

    # (a) Unconstrained OLS with cluster-robust standard errors
    print("\n(a) Unconstrained OLS with 10 clusters")
    cluster_ids = np.repeat(np.arange(10), n // 10)
    ols = OLS(y, X, add_const=True)
    ols_res = ols.fit(
        boot=BootConfig(
            n_boot=100,
            policy="strict",
            cluster_ids=cluster_ids,
        ),
    )

    # (b) Constrained OLS: beta_1 + beta_2 = 0
    print("\n(b) Constrained OLS: beta_x0 + beta_x1 = 0")
    R = np.array([[0, 1, 1, 0]])  # Constraint matrix
    q = np.array([0])
    ols_const = OLS(y, X, add_const=True)
    ols_const_res = ols_const.fit(constraints=R, constraint_vals=q)

    # (c) Weighted Least Squares (WLS)
    print("\n(c) Weighted Least Squares (WLS)")
    # Simulate heteroskedastic data
    w = np.exp(-X[:, 0])
    weights = w / w.mean()
    gls = GLS(y, X, add_const=True)
    gls_res = gls.fit(weights=weights, boot=BootConfig(n_boot=2000))

    # Output summary table
    print("\n" + "-" * 70)
    print(
        modelsummary(
            [ols_res, ols_const_res, gls_res],
            model_names=["OLS (Clustered)", "OLS (Constrained)", "WLS"],
        ),
    )


def demo_iv_gmm():
    """Demonstrate IV-2SLS and GMM estimators."""
    print("\n" + "=" * 70)
    print(" 2. IV-2SLS AND GMM DEMONSTRATION")
    print("=" * 70)

    # Simulate IV data
    y, X, Z, _ = simulate_iv_data(n_obs=500)

    # (a) IV-2SLS with weak IV diagnostics
    print("\n(a) IV-2SLS with weak IV tests")
    iv = IV2SLS(y, X, Z, endog_idx=[0], z_excluded_idx=[0], add_const=True)
    iv_res = iv.fit(boot=BootConfig(n_boot=2000))

    print(f"  Cragg-Donald min eigenvalue: {iv_res.extra.get('cd_min_eig', 0.0):.4f}")
    print(f"  Kleibergen-Paap rk statistic (LM): {iv_res.extra.get('kp_rk_LM', 0.0):.4f}")
    sw_f = iv_res.extra.get("SW_F", {})
    print(f"  Sanderson-Windmeijer F: {sw_f if sw_f else 'N/A'}")

    # (b) GMM (1-step and 2-step with optimal weighting)
    print("\n(b) GMM 1-step and 2-step")
    gmm1 = GMM(y, X, Z, endog_idx=[0], add_const=True)
    gmm1_res = gmm1.fit()

    gmm2 = GMM(y, X, Z, endog_idx=[0], add_const=True)
    gmm2_res = gmm2.fit(steps=2)

    hj = gmm2_res.extra.get("hansen_j", None)
    sg = gmm1_res.extra.get("sargan", None)
    if hj is None or (isinstance(hj, float) and not np.isfinite(hj)):
        print("  Hansen J statistic (2-step): N/A")
    else:
        print(f"  Hansen J statistic (2-step): {float(hj):.4f}")
    if sg is None or (isinstance(sg, float) and not np.isfinite(sg)):
        print("  Sargan statistic (1-step): N/A")
    else:
        print(f"  Sargan statistic (1-step): {float(sg):.4f}")

    # Summary table
    print("\n" + "-" * 70)
    print(
        modelsummary(
            [iv_res, gmm1_res, gmm2_res],
            model_names=["IV-2SLS", "GMM 1-step", "GMM 2-step"],
        ),
    )


def demo_qr():
    """Demonstrate Quantile Regression (QR) estimators."""
    print("\n" + "=" * 70)
    print(" 3. QUANTILE REGRESSION DEMONSTRATION")
    print("=" * 70)

    # Simulate heteroskedastic data
    y, X, _ = simulate_qr_data(n_obs=400, seed=456)
    cluster_ids = np.repeat(np.arange(40), 10)

    # (a) Median regression (tau=0.5) with Exponential bootstrap
    print("\n(a) Median regression (tau=0.5) with Exp(1) bootstrap")
    qr_median = QR(y, X, tau=0.5, add_const=True)
    qr_median_res = qr_median.fit(
        cluster_ids=cluster_ids,
        boot=BootConfig(dist="exp", n_boot=2000),
    )

    # (b) 90th percentile regression (tau=0.9) with Wild Gradient Bootstrap
    print("\n(b) 90th percentile regression with WGB")
    qr_90 = QR(y, X, tau=0.9, add_const=True)
    qr_90_res = qr_90.fit(
        cluster_ids=cluster_ids,
        boot=BootConfig(dist="wgb", n_boot=2000),
    )

    # Summary table
    print("\n" + "-" * 70)
    print(
        modelsummary(
            [qr_median_res, qr_90_res],
            model_names=["QR tau=0.5", "QR tau=0.9 (WGB)"],
        ),
    )


def demo_spatial():
    """Demonstrate Spatial Autoregressive (SAR) models."""
    print("\n" + "=" * 70)
    print(" 4. SPATIAL AUTOREGRESSIVE (SAR) DEMONSTRATION")
    print("=" * 70)

    # Simulate SAR data
    y, X, W, _ = simulate_sar_data(n_obs=200, seed=789)

    print("\n(a) SAR-2SLS with row-normalized weight matrix W")
    sar = SAR2SLS(y, X, W, add_const=True, include_W2=True)
    sar_res = sar.fit(boot=BootConfig(n_boot=2000))

    # Compute Moran's I on residuals
    resid = sar_res.extra.get("resid")
    moran_stat = moran_i(resid, W) if resid is not None else None
    print(f"  Estimated rho: {sar_res.params.iloc[0]:.4f}")
    print(
        f"  Moran's I (residuals): {moran_stat:.4f}"
        if (moran_stat is not None and np.isfinite(moran_stat))
        else "  Moran's I: N/A",
    )

    print("\n" + "-" * 70)
    print(modelsummary([sar_res], model_names=["SAR-2SLS"]))


def demo_callaway_santanna():
    """Demonstrate the Callaway & Sant'Anna event-study estimator."""
    print("\n" + "=" * 70)
    print(" 5. CALLAWAY-SANT'ANNA EVENT STUDY DEMONSTRATION")
    print("=" * 70)

    # Simulate staggered adoption data
    df = simulate_cs_data(n_units=200, n_periods=20, seed=111)

    print("\n(a) Group-time ATT estimation with 'not-yet-treated' controls")
    cs = EventStudyCS(
        id_name="id",
        t_name="time",
        cohort_name="g",
        y_name="y",
        control_group="notyet",
        center_at=-1,
        base_period="varying",
        boot=BootConfig(n_boot=2000),
    )
    cs_res = cs.fit(df)

    # Event-time aggregation
    att_tau = cs_res.extra.get("att_tau", pd.DataFrame())
    if not att_tau.empty:
        print("\n  Event-time ATT (tau):")
        print(att_tau[["tau", "att", "se"]].head(10))

    # Event study plot
    try:
        event_study_plot(cs_res, title="Callaway-Sant'Anna Event Study")
        print("\n  [Event study plot generated]")
    except SOFT_FAILURE_EXCEPTIONS as exc:
        _LOGGER.debug("Event study plot failed: %s", exc)
        print(f"\n  [Event study plot failed: {exc}]")


def demo_doubly_robust():
    """Demonstrate the Doubly-Robust (DR) event-study estimator."""
    print("\n" + "=" * 70)
    print(" 6. DOUBLY-ROBUST EVENT STUDY DEMONSTRATION")
    print("=" * 70)

    # Simulate panel data with covariates
    df = simulate_dr_data(n_units=300, n_periods=15, seed=222)

    print("\n(a) DR-DID with propensity score weighting")
    dr = DREventStudy(
        id_name="id",
        t_name="time",
        cohort_name="g",
        y_name="y",
        treat_name="treat",
        method="dr",  # Doubly-robust method
        boot=BootConfig(n_boot=2000),
    )
    dr_res = dr.fit(df)

    att_tau = dr_res.extra.get("att_tau", pd.DataFrame())
    if not att_tau.empty:
        print("\n  Event-time ATT (tau):")
        print(att_tau.head(10))


def demo_triple_difference():
    """Demonstrate the Triple-Difference (DDD) estimator."""
    print("\n" + "=" * 70)
    print(" 7. TRIPLE-DIFFERENCE (DDD) DEMONSTRATION")
    print("=" * 70)

    # Simulate DDD data
    df = simulate_ddd_data(n_units=250, n_periods=12, seed=333)

    print("\n(a) DDD with group-specific trends")
    ddd = DDDEventStudy(
        group_name="group",
        group_A_value="A",
        group_B_value="B",
        id_name="id",
        t_name="time",
        cohort_name="g",
        y_name="y",
        boot=BootConfig(n_boot=2000),
    )
    ddd_res = ddd.fit(df)

    print(f"\n  DDD ATT estimate: {ddd_res.params.get('DDD_ATT', 'N/A')}")


def demo_spatial_did():
    """Demonstrate Spatial Difference-in-Differences (SDID)."""
    print("\n" + "=" * 70)
    print(" 8. SPATIAL DID DEMONSTRATION")
    print("=" * 70)

    # Simulate spatial DID data
    df, W = simulate_spatial_did_data(n_units=150, n_periods=10, seed=444)

    print("\n(a) Spatial DID with spillover effects")
    sdid_model = SpatialDID(
        W=W,
        id_name="id",
        t_name="time",
        treat_name="treat",
        y_name="y",
        cohort_name="g",
        boot=BootConfig(n_boot=2000),
    )
    sdid_res = sdid_model.fit(df)

    print(f"\n  Direct effect: {sdid_res.extra.get('direct_effect', 'N/A')}")
    print(
        f"  Indirect effect (spillover): {sdid_res.extra.get('indirect_effect', 'N/A')}",
    )
    try:
        fig, _axes = event_study_auto_plot(
            sdid_res,
            which="both",
            which_band="post",
            title="Spatial DID: Direct vs Spillover",
        )
        _save_demo_figure(fig, "spatial_did_direct_spill.png")
    except SOFT_FAILURE_EXCEPTIONS as exc:
        _LOGGER.debug("Spatial DID plotting failed: %s", exc)
        print(f"  [Spatial DID plot failed: {exc}]")


def demo_synthetic_control():
    """Demonstrate the Synthetic Control (SC) method."""
    print("\n" + "=" * 70)
    print(" 9. SYNTHETIC CONTROL DEMONSTRATION")
    print("=" * 70)

    # Simulate synthetic control data (single treated unit)
    df = simulate_sc_data(
        n_units=41, n_periods=30, treated_unit=1, treatment_period=20, seed=555,
    )

    print("\n(a) Synthetic control with pre-treatment weight learning")
    sc = SyntheticControl(
        y_name="y",
        id_name="id",
        t_name="time",
        cohort_name="g",
        tau_weight="group",
    )
    boot_cfg = BootConfig(n_boot=1000, seed=777)
    sc_res = sc.fit(df, boot=boot_cfg)

    post_att = sc_res.model_info.get("PostATT", "N/A")
    post_ci = sc_res.extra.get("post_scalar", None)
    post_band = ("N/A", "N/A")
    if isinstance(post_ci, pd.DataFrame) and (not post_ci.empty):
        post_band = (float(post_ci["lower"].iloc[0]), float(post_ci["upper"].iloc[0]))
    print(f"\n  Post-period ATT (weighted): {post_att}")
    print(f"  95% uniform band (post): {post_band}")
    try:
        fig, _ax = event_study_auto_plot(
            sc_res,
            which_band="full",
            title="Synthetic Control Event Study",
            hide_base=False,
        )
        _save_demo_figure(fig, "synthetic_control_event_study.png")
    except SOFT_FAILURE_EXCEPTIONS as exc:
        _LOGGER.debug("Synthetic control plotting failed: %s", exc)
        print(f"  [Synthetic control plot failed: {exc}]")


def demo_sdid():
    """Demonstrate Synthetic Difference-in-Differences (SDID)."""
    print("\n" + "=" * 70)
    print(" 10. SYNTHETIC DID DEMONSTRATION")
    print("=" * 70)

    # Simulate SDID data
    df = simulate_sdid_data(
        n_units=150, n_periods=25, treated_units=50, treatment_start=15, seed=666,
    )

    print("\n(a) SDID with unit and time weights")
    sdid = SDID(
        id_name="id",
        t_name="time",
        treat_name="treat",
        y_name="y",
        boot=BootConfig(n_boot=1000, seed=888),
    )
    sdid_res = sdid.fit(df)

    print(f"\n  SDID post_ATT: {sdid_res.params.get('post_ATT', 'N/A')}")
    try:
        fig, _ax = event_study_auto_plot(
            sdid_res,
            which_band="full",
            title="Synthetic DID Event Study",
        )
        _save_demo_figure(fig, "synthetic_did_event_study.png")
    except SOFT_FAILURE_EXCEPTIONS as exc:
        _LOGGER.debug("SDID plotting failed: %s", exc)
        print(f"  [SDID plot failed: {exc}]")


def run_all_demos():
    """Run all demonstrations sequentially."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 15 + "LINEAREG PACKAGE DEMONSTRATION" + " " * 23 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print(
        "\nDemonstration of 13 estimators with bootstrap inference.",
    )
    print("Intended as an illustrative demo; results depend on RNG/seeds.")
    print("\n")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        demo_tasks: list[tuple[str, Callable[[], None]]] = [
            ("OLS/GLS", demo_ols_gls),
            ("IV/GMM", demo_iv_gmm),
            ("QR", demo_qr),
            ("Spatial", demo_spatial),
            ("CS", demo_callaway_santanna),
            ("DR", demo_doubly_robust),
            ("DDD", demo_triple_difference),
            ("Spatial DID", demo_spatial_did),
            ("SC", demo_synthetic_control),
            ("SDID", demo_sdid),
        ]
        for label, func in demo_tasks:
            _run_demo_block(label, func)

    print("\n" + "*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 20 + "DEMO COMPLETE" + " " * 35 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\nEstimators demonstrated (n=100, B=100 for testing).")
    print("Production defaults: B=2000.")
    print("Inference: Bootstrap-only (wild/multiplier).")
    print("No analytic SE/p-values or pairs bootstrap.")
    print("\nSee README.md for details.")
    print("\n")


if __name__ == "__main__":
    run_all_demos()
