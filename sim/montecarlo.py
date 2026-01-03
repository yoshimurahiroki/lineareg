"""Monte Carlo simulations and smoke tests.

Provides small-sample data generation and testing for estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lineareg.core import linalg as la
from lineareg.estimators import (
    GLS,
    GMM,
    IV2SLS,
    OLS,
    QR,
    SDID,
    DDDEventStudy,
    DREventStudy,
    EventStudyCS,
    SpatialDID,
    SyntheticControl,
)
from lineareg.estimators.base import BootConfig
from lineareg.estimators.qr import IVQR
from lineareg.spatial import SAR2SLS
from lineareg.spatial.spatial import moran_i


def simulate_ols_data(n_obs=100, n_features=3):
    """Simulates simple data for OLS (n=100)."""
    rng = np.random.default_rng(42)
    X = rng.random((n_obs, n_features))
    beta_true = np.arange(1, n_features + 1)
    epsilon = rng.standard_normal(n_obs)
    y = la.dot(X, beta_true.reshape(-1, 1)).reshape(-1) + epsilon
    # Return plain numpy arrays to match estimator constructors
    return y, X, beta_true


def test_ols():
    """Tests OLS estimator."""
    y, X, beta_true = simulate_ols_data(n_obs=100, n_features=3)
    model = OLS(y, X, add_const=True)
    boot_cfg = BootConfig(n_boot=100, dist="rademacher", seed=42)  # n_boot=100 for testing speed
    results = model.fit(boot=boot_cfg)
    print("--- OLS Monte Carlo Test ---")
    print(f"True Beta: {beta_true}")
    coeffs = results.params.to_numpy()
    print(f"Estimated Beta (excluding constant): {coeffs[:-1]}")
    print(f"Constant: {coeffs[-1]}")
    assert np.allclose(coeffs[:-1], beta_true, atol=0.5), \
        f"OLS parameter recovery failed: {coeffs[:-1]} vs {beta_true}"
    # Bootstrap is run for internal diagnostics but SE/CI are not exposed
    # Verify boot was run by checking for boot_betas in extra
    assert "boot_betas" in results.extra, "OLS bootstrap betas missing from extra"
    assert results.extra["boot_betas"] is not None, "OLS bootstrap betas is None"
    print(f"Bootstrap run: {results.extra['boot_betas'].shape[0]} iterations")
    print("✓ OLS test passed.\n")


def simulate_iv_data(n_obs=100, seed: int | None = 123):
    """Simulates data with one endogenous regressor and two excluded instruments.

    The two excluded instruments (z1, z2) ensure over-identification so that
    weak-IV diagnostics (SW F, CD, KP) can be validly computed.
    """
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_obs)
    z2 = rng.standard_normal(n_obs)  # Second excluded instrument for overidentification
    v = rng.standard_normal(n_obs)
    x_exog = rng.random(n_obs)

    # Endogeneity: cov(x_endog, u) != 0
    u = 0.5 * v + rng.standard_normal(n_obs) * 0.5

    # First stage: x_endog is correlated with z1 and z2
    x_endog = 0.8 * z1 + 0.6 * z2 + 0.5 * x_exog + 0.5 * v

    y = 2 * x_endog + 3 * x_exog + u

    X = pd.DataFrame({"x_endog": x_endog, "x_exog": x_exog})
    # z_excluded_idx=[0,1] refers to z1 and z2 columns in Z_df
    Z_df = pd.DataFrame({"z1": z1, "z2": z2, "x_exog": x_exog})
    return pd.Series(y, name="y"), X, Z_df, {"x_endog": 2, "x_exog": 3}


def test_iv():
    """Tests the IV2SLS estimator."""
    y, X, Z, beta_true = simulate_iv_data()
    # z_excluded_idx=[0,1] for z1 and z2 columns in Z (first two columns)
    model = IV2SLS(y, X, Z, endog_idx=[0], z_excluded_idx=[0, 1], add_const=True)
    results = model.fit()
    print("--- IV2SLS Monte Carlo Test ---")
    print("True Beta:", beta_true)
    print("Estimated Beta:", results.params.to_dict())
    assert np.allclose(results.params["x_endog"], beta_true["x_endog"], atol=0.5), \
        f"IV parameter recovery failed: {results.params['x_endog']} vs {beta_true['x_endog']}"
    assert np.allclose(results.params["x_exog"], beta_true["x_exog"], atol=0.5), \
        f"IV parameter recovery failed: {results.params['x_exog']} vs {beta_true['x_exog']}"
    fs_stats = results.extra.get("first_stage_stats", {})
    sw_f = fs_stats.get("F_SW_min", None)
    cd_eig = fs_stats.get("cd_min_eig", None)
    kp_eig = fs_stats.get("kp_min_eig", None)
    print(f"Weak IV SW F: {sw_f}")
    print(f"Weak IV CD min eig: {cd_eig}")
    print(f"Weak IV KP min eig: {kp_eig}")
    assert sw_f is not None and np.isfinite(sw_f), "SW F-statistic should be computed"
    assert cd_eig is not None and np.isfinite(cd_eig), "Cragg-Donald should be computed"
    assert kp_eig is not None and np.isfinite(kp_eig), "Kleibergen-Paap should be computed"
    print("✓ IV2SLS test passed.\n")


def simulate_gls_data(n_obs=100, seed: int | None = 321):
    """Simulates data with AR(1) errors for GLS (n=100 for testing speed)."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_obs, 3))
    beta_true = np.array([1, 2, 3])
    y_mean = la.dot(X, beta_true.reshape(-1, 1)).reshape(-1)
    rho = 0.5
    epsilon = rng.standard_normal(n_obs)
    for i in range(1, n_obs):
        epsilon[i] += rho * epsilon[i-1]
    y = y_mean + epsilon
    return pd.Series(y, name="y"), pd.DataFrame(X, columns=[f"x{i}" for i in range(3)]), beta_true


def test_gls():
    """Tests the GLS estimator."""
    y, X, beta_true = simulate_gls_data()
    model = GLS(y, X, add_const=True)
    time_ids = list(range(len(y)))
    results = model.fit(omega="AR1", time_ids=time_ids)
    print("--- GLS Monte Carlo Test ---")
    print("True Beta:", beta_true)
    coeffs = results.params.to_numpy()
    print(
        "Estimated Beta:",
        coeffs[:-1] if coeffs.shape[0] > len(beta_true) else coeffs,
    )
    assert np.allclose(coeffs[:-1], beta_true, atol=0.6), \
        f"GLS parameter recovery failed: {coeffs[:-1]} vs {beta_true}"
    print("✓ GLS test passed.\n")


def test_gmm():
    """Tests the GMM estimator."""
    y, X, Z, beta_true = simulate_iv_data()
    endog_idx = [0]
    model = GMM(y, X, Z, endog_idx=endog_idx)
    results = model.fit()
    print("--- GMM Monte Carlo Test ---")
    print("True Beta:", beta_true)
    print("Estimated Beta:", results.params.to_dict())
    assert np.allclose(results.params["x_endog"], beta_true["x_endog"], atol=0.5)
    assert np.allclose(results.params["x_exog"], beta_true["x_exog"], atol=0.5)
    j_stat = results.extra.get("J_stat", None)
    print(f"J-stat: {j_stat}")
    assert j_stat is not None and np.isfinite(j_stat), "Over-identification J-statistic should be computed"
    fs_stats = results.extra.get("first_stage_stats", {})
    sw_f = fs_stats.get("F_SW_min", None)
    print(f"Weak IV SW F: {sw_f}")
    print("✓ GMM test passed.\n")


def simulate_qr_data(n_obs=100, seed: int | None = 456):
    """Simulates data for QR (n=100 for testing speed)."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_obs, 3))
    beta_true = np.array([1, 2, 3])
    y_mean = la.dot(X, beta_true.reshape(-1, 1)).reshape(-1)
    epsilon = rng.standard_normal(n_obs)
    y = y_mean + epsilon
    return pd.Series(y, name="y"), pd.DataFrame(X, columns=[f"x{i}" for i in range(3)]), beta_true


def test_qr():
    """Tests the QR estimator with improved data generation for LP solver stability."""
    # Use larger sample and more stable data for LP solver
    n_obs = 200
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_obs, 3))
    beta_true = np.array([1, 2, 3])
    # Use less extreme error distribution for LP feasibility
    epsilon = rng.standard_normal(n_obs) * 0.5
    y = la.dot(X, beta_true.reshape(-1, 1)).reshape(-1) + epsilon

    model = QR(pd.Series(y), pd.DataFrame(X, columns=["x0", "x1", "x2"]), tau=0.5)
    # Create meaningful clusters (20 clusters of 10 observations each)
    cluster_ids = np.repeat(np.arange(20), 10)
    boot_cfg = BootConfig(n_boot=50, dist="wgb", seed=42)
    results = model.fit(boot=boot_cfg, cluster_ids=cluster_ids)
    print("--- QR Monte Carlo Test ---")
    print("True Beta:", beta_true)
    coeffs = results.params.to_numpy()
    print("Estimated Beta:", coeffs[:-1])
    assert np.all(np.isfinite(coeffs)), "QR returned non-finite coefficients"
    print("✓ QR test passed.\n")


def test_ivqr():
    """Tests the IVQR estimator."""
    y, X, Z, beta_true = simulate_iv_data()
    model = IVQR(y, X, Z, tau=0.5, method="ch05")
    results = model.fit()
    print("--- IVQR Monte Carlo Test ---")
    print("True Beta:", beta_true)
    print("Estimated Beta:", results.params.to_dict())
    if "x_endog" in results.params.index:
        assert abs(results.params["x_endog"] - beta_true["x_endog"]) < 1.5
    if "x_exog" in results.params.index:
        assert abs(results.params["x_exog"] - beta_true["x_exog"]) < 1.5
    print("✓ IVQR test passed.\n")


def simulate_sar_data(n_obs=100, seed: int | None = 789):
    """Simulates spatial data for SAR2SLS (n=100 for testing speed)."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_obs, 2))
    beta_true = np.array([1, 2])
    W = np.zeros((n_obs, n_obs), dtype=float)
    for i in range(n_obs):
        nbrs = {(i - 1) % n_obs, (i + 1) % n_obs}
        for j in nbrs:
            W[i, j] = 0.5
    np.fill_diagonal(W, 0.0)
    rho_true = 0.3
    A = np.eye(n_obs) - rho_true * W
    y_mean = la.dot(X, beta_true.reshape(-1, 1)).reshape(-1)
    epsilon = rng.standard_normal(n_obs)
    y = la.solve(A, (y_mean + epsilon).reshape(-1, 1)).reshape(-1)
    return pd.Series(y, name="y"), pd.DataFrame(X, columns=[f"x{i}" for i in range(2)]), W, beta_true


def test_sar():
    """Tests the SAR2SLS estimator."""
    # Use a slightly larger sample and a deterministic seed to stabilize
    # rho recovery in a small-sample 2SLS setting.
    y, X, W, beta_true = simulate_sar_data(n_obs=150, seed=4)
    model = SAR2SLS(y, X, W)
    results = model.fit()
    print("--- SAR2SLS Monte Carlo Test ---")
    print("True Beta:", beta_true)
    coeffs = results.params.to_numpy()
    print("Estimated coeffs:", coeffs)
    assert np.isfinite(results.params["rho"]), "SAR2SLS rho should be finite"
    assert abs(results.params["rho"] - 0.3) < 0.3, f"SAR2SLS rho recovery failed: {results.params['rho']}"
    assert abs(results.params["x0"] - beta_true[0]) < 1.0, f"SAR2SLS beta recovery failed: {results.params['x0']}"
    assert abs(results.params["x1"] - beta_true[1]) < 1.0, f"SAR2SLS beta recovery failed: {results.params['x1']}"
    moran_stat = moran_i(y.to_numpy(), W)
    print(f"Moran I: {moran_stat}")
    print("✓ SAR2SLS test passed.\n")


def simulate_cs_data(n_units=100, n_periods=10, seed: int | None = 101):
    """Simulates panel data with staggered treatment adoption."""
    rng = np.random.default_rng(seed)
    unit_ids = np.arange(n_units)
    time_ids = np.arange(n_periods)

    df = pd.DataFrame([(i, t) for i in unit_ids for t in time_ids], columns=["id", "time"])

    # Assign treatment cohorts
    cohorts = rng.integers(2, n_periods, size=n_units)
    cohorts[rng.random(n_units) < 0.2] = 0 # Never treated
    df["g"] = df["id"].map(pd.Series(cohorts))

    df["treat"] = ((df["time"] >= df["g"]) & (df["g"] > 0)).astype(int)

    # Generate outcome
    unit_fe = rng.standard_normal(n_units)
    time_fe = rng.standard_normal(n_periods)
    df["y"] = df["id"].map(pd.Series(unit_fe)) + df["time"].map(pd.Series(time_fe))
    df["y"] += 2 * df["treat"] # Constant treatment effect
    df["y"] += rng.standard_normal(len(df))

    return df


def test_cs():
    """Tests the EventStudyCS estimator."""
    data = simulate_cs_data()
    model = EventStudyCS(
        id_name="id",
        t_name="time",
        cohort_name="g",
        y_name="y",
    )
    results = model.fit(data)
    print("--- EventStudyCS Monte Carlo Test ---")
    print("Estimated Event-Time ATTs:")
    att_tau = results.extra.get("att_tau", results.params)
    print(att_tau.head() if hasattr(att_tau, "head") else att_tau[:5])
    post_att = results.model_info.get("PostATT", None)
    assert post_att is not None and np.isfinite(post_att)
    assert abs(post_att - 2.0) < 1.0, f"EventStudyCS PostATT recovery failed: {post_att}"
    print("✓ EventStudyCS test passed.\n")


def simulate_ddd_data(n_units=100, n_periods=10, seed: int | None = 202):
    """Simulates data for DDD event study."""
    rng = np.random.default_rng(seed)
    unit_ids = np.arange(n_units)
    time_ids = np.arange(n_periods)

    df = pd.DataFrame([(i, t) for i in unit_ids for t in time_ids], columns=["id", "time"])

    # Assign groups and cohorts
    groups = rng.choice(["A", "B"], size=n_units)
    cohorts = rng.integers(2, n_periods, size=n_units)
    cohorts[rng.random(n_units) < 0.2] = 0  # Never treated
    df["group"] = df["id"].map(pd.Series(groups))
    df["g"] = df["id"].map(pd.Series(cohorts))

    df["treat"] = ((df["time"] >= df["g"]) & (df["g"] > 0)).astype(int)

    # Generate outcome with group-specific effects
    unit_fe = rng.standard_normal(n_units)
    time_fe = rng.standard_normal(n_periods)
    group_fe = {"A": 1.0, "B": -1.0}
    df["y"] = df["id"].map(pd.Series(unit_fe)) + df["time"].map(pd.Series(time_fe)) + df["group"].map(group_fe)
    df["y"] += 3 * df["treat"] * (df["group"] == "A")  # Treatment effect only in group A
    df["y"] += rng.standard_normal(len(df))

    return df


def test_ddd():
    """Tests the DDDEventStudy estimator."""
    data = simulate_ddd_data()
    model = DDDEventStudy(
        group_name="group",
        group_A_value="A",
        group_B_value="B",
        id_name="id",
        t_name="time",
        cohort_name="g",
        y_name="y",
    )
    results = model.fit(data)
    print("--- DDDEventStudy Monte Carlo Test ---")
    print("Estimated DDD ATTs:")
    att_tau = results.extra.get("att_tau", results.params)
    print(att_tau.head() if hasattr(att_tau, "head") else att_tau[:5])
    post_att = results.model_info.get("PostATT", None)
    assert post_att is not None and np.isfinite(post_att)
    assert abs(post_att - 3.0) < 1.5, f"DDDEventStudy PostATT recovery failed: {post_att}"
    print("✓ DDDEventStudy test passed.\n")


def simulate_dr_data(n_units=100, n_periods=10, seed: int | None = 303):
    """Simulates data for DR event study."""
    rng = np.random.default_rng(seed)
    unit_ids = np.arange(n_units)
    time_ids = np.arange(n_periods)

    df = pd.DataFrame([(i, t) for i in unit_ids for t in time_ids], columns=["id", "time"])

    # Assign cohorts
    cohorts = rng.integers(2, n_periods, size=n_units)
    cohorts[rng.random(n_units) < 0.2] = 0  # Never treated
    df["g"] = df["id"].map(pd.Series(cohorts))

    df["treat"] = ((df["time"] >= df["g"]) & (df["g"] > 0)).astype(int)

    # Generate outcome
    unit_fe = rng.standard_normal(n_units)
    time_fe = rng.standard_normal(n_periods)
    df["y"] = df["id"].map(pd.Series(unit_fe)) + df["time"].map(pd.Series(time_fe))
    df["y"] += 2.5 * df["treat"]
    df["y"] += rng.standard_normal(len(df))

    return df


def test_dr():
    """Tests the DR estimator."""
    data = simulate_dr_data()
    model = DREventStudy(
        id_name="id",
        t_name="time",
        cohort_name="g",
        treat_name="treat",
        y_name="y",
    )
    results = model.fit(data)
    print("--- DR Monte Carlo Test ---")
    print("Estimated DR ATT:")
    print(results.model_info.get("PostATT", "NA"))
    assert abs(results.model_info.get("PostATT", 0) - 2.5) < 0.5
    print("DR test passed.\n")


def simulate_sdid_data(
    n_units: int = 50,
    n_periods: int = 10,
    *,
    treated_units: int = 1,
    treatment_start: int = 5,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulates data for SDID with reproducible random components."""
    rng = np.random.default_rng(seed)
    unit_ids = np.arange(n_units)
    time_ids = np.arange(n_periods)

    df = pd.DataFrame([(i, t) for i in unit_ids for t in time_ids], columns=["id", "time"])

    treated_set = set(unit_ids[:treated_units])
    treated_mask = df["id"].isin(treated_set)
    df["treat"] = (treated_mask & (df["time"] >= treatment_start)).astype(int)
    df["g"] = np.where(treated_mask, treatment_start, 0)

    unit_fe = rng.standard_normal(n_units)
    time_fe = rng.standard_normal(n_periods)
    df["y"] = df["id"].map(pd.Series(unit_fe)) + df["time"].map(pd.Series(time_fe))
    df["y"] += 3 * df["treat"]
    df["y"] += rng.standard_normal(len(df))

    return df


def test_sdid():
    """Tests the SDID estimator with reduced bootstrap (B=100) for speed."""
    data = simulate_sdid_data(n_units=50, n_periods=10, seed=202)
    model = SDID(
        id_name="id",
        t_name="time",
        treat_name="treat",
        y_name="y",
        boot=BootConfig(n_boot=100, seed=404),
    )
    results = model.fit(data)
    print("--- SDID Monte Carlo Test ---")
    post_att = results.model_info.get("PostATT", results.params.get("post_ATT", None))
    print(f"Estimated SDID post_ATT: {post_att}")
    assert post_att is not None and abs(post_att - 3) < 1.5, f"SDID ATT recovery failed: {post_att}"
    print("✓ SDID test passed.\n")


def simulate_sc_data(
    n_units: int = 30,
    n_periods: int = 15,
    *,
    treated_unit: int = 0,
    treatment_period: int = 10,

    seed: int | None = None,
) -> pd.DataFrame:
    """Simulates data for Synthetic Control with explicit adoption times."""
    rng = np.random.default_rng(seed)
    unit_ids = np.arange(n_units)
    time_ids = np.arange(n_periods)

    df = pd.DataFrame([(i, t) for i in unit_ids for t in time_ids], columns=["id", "time"])

    df["treat"] = ((df["id"] == treated_unit) & (df["time"] >= treatment_period)).astype(int)
    df["g"] = np.where(df["id"] == treated_unit, treatment_period, 0)

    unit_fe = rng.standard_normal(n_units)
    time_fe = rng.standard_normal(n_periods)
    df["y"] = df["id"].map(pd.Series(unit_fe)) + df["time"].map(pd.Series(time_fe))
    df["y"] += 4 * df["treat"]
    df["y"] += rng.standard_normal(len(df))

    return df

def test_sc():
    """Tests the SyntheticControl estimator with reduced bootstrap (B=100) for speed."""
    data = simulate_sc_data(n_units=30, n_periods=15, seed=303)
    model = SyntheticControl(
        y_name="y",
        id_name="id",
        t_name="time",
        treat_name="treat",
        cohort_name="g",
    )
    results = model.fit(data, boot=BootConfig(n_boot=100, seed=606))
    print("--- SyntheticControl Monte Carlo Test ---")
    post_att = results.model_info.get("PostATT", results.params.get("post_ATT", None))
    print(f"Estimated SC post_ATT: {post_att}")
    assert post_att is not None and np.isfinite(post_att), "SC ATT should be computed and finite"
    assert post_att > 0, f"SC PostATT should be positive in simulation: {post_att}"
    print("✓ SyntheticControl test passed.\n")


def simulate_spatial_did_data(n_units=40, n_periods=10, *, treated_units=5, g_time=5, seed=123):
    """Simulates data and a simple row-normalized W for Spatial DID.

    Returns (df, W) where W is an (n_units x n_units) dense row-normalized matrix
    with zero diagonal.
    """
    rng = np.random.default_rng(seed)
    unit_ids = np.arange(n_units)
    time_ids = np.arange(n_periods)

    df = pd.DataFrame([(i, t) for i in unit_ids for t in time_ids], columns=["id", "time"])

    # Cohorts: first treated_units adopt at g_time, others never (g=0)
    treated_mask_units = df["id"].isin(unit_ids[:treated_units])
    df["g"] = np.where(treated_mask_units, g_time, 0)
    df["treat"] = ((df["g"] > 0) & (df["time"] >= df["g"])).astype(int)

    # Outcome with unit/time FE and constant treatment effect
    unit_fe = rng.standard_normal(n_units)
    time_fe = rng.standard_normal(n_periods)
    df["y"] = df["id"].map(pd.Series(unit_fe)) + df["time"].map(pd.Series(time_fe))
    df["y"] += 2.5 * df["treat"]
    df["y"] += rng.standard_normal(len(df))

    # Build a simple spatial weights matrix W (k-nearest ring-like), row-normalized
    W = np.zeros((n_units, n_units), dtype=float)
    for i in range(n_units):
        # connect to neighbors i-1 and i+1 (ring) and one random extra neighbor
        nbrs = { (i - 1) % n_units, (i + 1) % n_units, int(rng.integers(0, n_units)) }
        nbrs.discard(i)
        for j in nbrs:
            W[i, j] = 1.0
    # zero diagonal and row-normalize
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1)
    nz = rs > 0
    W[nz, :] = W[nz, :] / rs[nz].reshape(-1, 1)

    return df, W


def test_spatial_did():
    """Tests the SpatialDID estimator with BootConfig and explicit W/cohort."""
    data, W = simulate_spatial_did_data()
    model = SpatialDID(
        id_name="id",
        t_name="time",
        cohort_name="g",
        treat_name="treat",
        y_name="y",
        W=W,
        row_normalized=True,
        boot=BootConfig(n_boot=50, seed=999),
        tau_weight="group",
    )
    results = model.fit(data)
    print("--- SpatialDID Monte Carlo Test ---")
    post_att = results.model_info.get("PostATT", None)
    print(f"Estimated Spatial DID PostATT: {post_att}")
    assert post_att is not None and np.isfinite(post_att), "SpatialDID PostATT should be computed and finite"
    assert abs(post_att - 2.5) < 2.0, f"SpatialDID PostATT recovery failed: {post_att}"
    print("✓ SpatialDID test passed.\n")


if __name__ == "__main__":
    test_ols()
    test_iv()
    test_gls()
    test_gmm()
    test_qr()
    test_ivqr()
    test_sar()
    test_cs()
    test_ddd()
    test_dr()
    test_sdid()
    test_sc()
    test_spatial_did()
