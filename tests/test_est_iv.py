
import pytest
import numpy as np
import pandas as pd
from lineareg.estimators.iv import IV2SLS
from lineareg.core import linalg as la
from lineareg.output.summary import weakiv_table
from lineareg.estimators.base import BootConfig

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def data_iv(seed=42):
    rng = np.random.default_rng(seed)
    N = 200
    
    # Instrument Z
    Z = rng.standard_normal((N, 2))
    # Endogenous X (correlated with u)
    u = rng.standard_normal(N)
    # First stage: X = Z @ pi + v
    v = 0.5 * u + rng.standard_normal(N) * 0.5
    X_endog = Z[:, 0] * 1.5 + Z[:, 1] * 0.5 + v
    X_endog = X_endog.reshape(-1, 1)
    
    # Exogenous W
    W = rng.standard_normal((N, 1))
    
    # Structural equation: y = X_endog * beta + W * gamma + u
    beta = 2.0
    gamma = -1.0
    y = X_endog.flatten() * beta + W.flatten() * gamma + u
    
    df = pd.DataFrame({
        'y': y,
        'x_en': X_endog.flatten(),
        'w': W.flatten(),
        'z1': Z[:, 0],
        'z2': Z[:, 1]
    })
    return df

# ---------------------------------------------------------------------
# Unit Tests: Estimates
# ---------------------------------------------------------------------

def test_iv_2sls_exact(data_iv):
    df = data_iv
    
    # Manual 2SLS
    # X = [x_en, w, const]
    # Z = [z1, z2, w, const]
    X_mat = np.column_stack([df['x_en'], df['w'], np.ones(len(df))])
    Z_mat = np.column_stack([df['z1'], df['z2'], df['w'], np.ones(len(df))])
    y = df['y'].values
    
    # Pz = Z(Z'Z)^-1 Z'
    # Xhat = Pz X
    Z_inv = np.linalg.pinv(Z_mat.T @ Z_mat)
    Pz = Z_mat @ Z_inv @ Z_mat.T
    Xhat = Pz @ X_mat
    
    # beta = (Xhat' X)^-1 Xhat' y
    # or (Xhat' Xhat)^-1 Xhat' y (equivalent usually)
    beta_manual = np.linalg.inv(Xhat.T @ X_mat) @ Xhat.T @ y
    
    # Estimator
    # IV2SLS(y, X_exog, Z_inst, endog, excluded)
    # API: y, X, Z, endog_idx, z_excluded_idx
    # This is low-level API.
    # Cleaner to use formula if supported? 
    # IV2SLS.from_formula("y ~ w + [x_en ~ z1 + z2]", df) ... 
    # Wait, IV2SLS.from_formula API: formula, data, iv="x_en ~ z1 + z2"
    # Actually from outline: from_formula(cls, formula, data, iv, ...)
    # Example: IV2SLS.from_formula("y ~ x1 + x2", df, iv="(x2 ~ z1 + z2)")
    
    # Let's use array API for exactness test
    # X input should contain ALL regressors (exog + endog).
    # Z input should contain ALL instruments (exog included + excluded).
    # But init says:
    # y: ArrayLike
    # X: MatrixLike (all regressors)
    # Z: MatrixLike (all instruments)
    # endog_idx: Sequence[int]
    # z_excluded_idx: Sequence[int]
    
    # Regressors order: x_en, w. (plus const)
    # Instruments order: z1, z2, w. (plus const)
    
    # If add_const=True, X and Z should NOT have const?
    X_in = df[['x_en', 'w']]
    Z_in = df[['z1', 'z2', 'w']] # Excluded are z1, z2. Included is w.
    
    # endog_idx = [0] (x_en)
    # z_excluded_idx = [0, 1] (z1, z2)
    
    model = IV2SLS(
        df['y'], X_in, Z_in, 
        endog_idx=[0], 
        z_excluded_idx=[0, 1],
        add_const=True
    )
    res = model.fit()
    
    # Params order: x_en, w, const
    assert np.allclose(res.params.values, beta_manual, rtol=1e-5)
    
    # Check SE existence (bootstrap default)
    assert res.se is not None
    assert (res.se > 0).all()

def test_wald_test(data_iv):
    # Test Wald hypothesis: beta_x_en = 2.0
    # R = [1, 0, 0], r = [2.0]
    X_in = data_iv[['x_en', 'w']]
    Z_in = data_iv[['z1', 'z2', 'w']]
    
    model = IV2SLS(
        data_iv['y'], X_in, Z_in, 
        endog_idx=[0], 
        z_excluded_idx=[0, 1],
        add_const=True
    )
    # Use small B for speed
    boot = BootConfig(n_boot=20, seed=123)
    res = model.fit(boot=boot)
    
    R = np.array([[1.0, 0.0, 0.0]])
    r = np.array([2.0])
    
    # wald_test likely returns a dict or object with 'stat', 'p_value' (bootstrap p-value)
    # Wait, no analytic p-value allowed. Bootstrap p-value is allowed?
    # Or just critical values?
    # "No analytic p-values or critical values" -> means no Chi2 lookup.
    # But bootstrap p-value is fine.
    
    test_res = res.wald_test(R, r)
    # Expect a dict (from core.bootstrap.wald_test_wild_bootstrap)
    assert isinstance(test_res, dict)
    assert 'wald_stat' in test_res
    assert test_res['wald_stat'] >= 0

# ---------------------------------------------------------------------
# Integration Tests: Diagnostics
# ---------------------------------------------------------------------

def test_weak_iv_diagnostics(data_iv):
    X_in = data_iv[['x_en', 'w']]
    Z_in = data_iv[['z1', 'z2', 'w']]
    
    model = IV2SLS(data_iv['y'], X_in, Z_in, endog_idx=[0], z_excluded_idx=[0,1])
    res = model.fit()
    
    # weakiv_table
    # Argument is 'res: EstimationResult' (single), unlike modelsummary
    tab = weakiv_table(res)
    # Check for presence of key stats (KP = Kleibergen-Paap)
    # Convert to string to check content, not just column names
    tab_str = str(tab)
    assert "KP" in tab_str or "Kleibergen-Paap" in tab_str
    # Check effective F logic presence
    # usually reports F stats
    
# ---------------------------------------------------------------------
# Edge Cases: Weights
# ---------------------------------------------------------------------

def test_iv_weights_forbidden(data_iv):
    X_in = data_iv[['x_en', 'w']]
    Z_in = data_iv[['z1', 'z2', 'w']]
    w = np.ones(len(data_iv))
    
    model = IV2SLS(
        data_iv['y'], X_in, Z_in, 
        endog_idx=[0], 
        z_excluded_idx=[0, 1]
    )
    with pytest.raises(ValueError, match="forbidden for IV"):
        # Estimator name might be IV or IV2SLS in error message
        model.fit(weights=w)
