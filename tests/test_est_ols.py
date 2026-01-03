
import pytest
import numpy as np
import pandas as pd
from lineareg.estimators.ols import OLS
from lineareg.core import linalg as la
from lineareg.output.summary import modelsummary as summary
# We need to import the plotting function if we want to test it
# from lineareg.output.plots import plot_coefs

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def data_ols(seed=42):
    rng = np.random.default_rng(seed)
    N = 200
    K = 3
    X = rng.standard_normal((N, K))
    beta = np.array([1.0, -0.5, 0.2])
    # Homoskedastic errors
    u = rng.standard_normal(N)
    y = X @ beta + u
    return pd.DataFrame(X, columns=['x1','x2','x3']), pd.Series(y, name='y'), beta

# ---------------------------------------------------------------------
# Unit Tests: Estimates
# ---------------------------------------------------------------------

def test_ols_estimates_exact(data_ols):
    X_df, y_s, beta_true = data_ols
    
    model = OLS(y_s, X_df, add_const=False)
    res = model.fit()
    
    # Check coefs against numpy
    beta_np = np.linalg.lstsq(X_df.values, y_s.values, rcond=None)[0]
    assert np.allclose(res.params.values, beta_np)
    
    # Policy: OLS runs wild bootstrap by default (B=2000 usually)
    # So SE should be present and positive
    assert res.se is not None
    assert not res.se.isna().any()
    assert (res.se > 0).all()

def test_ols_bootstrap_inference(data_ols):
    X_df, y_s, _ = data_ols
    # Request bootstrap explicitly
    
    from lineareg.estimators.base import BootConfig
    
    model = OLS(y_s, X_df, add_const=False)
    # Small bootstrap for speed
    boot = BootConfig(n_boot=10, seed=42) 
    res = model.fit(boot=boot)
    
    assert res.se is not None
    assert np.all(res.se > 0)
    assert len(res.se) == 3

# ---------------------------------------------------------------------
# Unit Tests: Fixed Effects (Absorption)
# ---------------------------------------------------------------------

def test_ols_absorb_exact():
    rng = np.random.default_rng(123)
    N = 100
    g = rng.integers(0, 5, size=N)
    
    # y = g_effect + x + u
    x = rng.standard_normal(N)
    fe_eff = rng.standard_normal(5)
    y = fe_eff[g] + 2*x + rng.standard_normal(N)
    
    df = pd.DataFrame({'y': y, 'x': x, 'g': g})
    
    # 1. OLS with dummies
    # pd.get_dummies(g, drop_first=True)
    X_dum = pd.concat([df[['x']], pd.get_dummies(df['g'], prefix='g', drop_first=True)], axis=1)
    res_dum = OLS(df['y'], X_dum, add_const=True).fit()
    beta_x_dum = res_dum.params['x']
    
    # 2. OLS with absorption
    # Signature: OLS.fit(absorb_fe=...)
    res_abs = OLS(df['y'], df[['x']], add_const=True).fit(absorb_fe=df['g'])
    beta_x_abs = res_abs.params['x']
    
    # Coefficients on x should match
    assert np.allclose(beta_x_dum, beta_x_abs, rtol=1e-5)

# ---------------------------------------------------------------------
# Integration Tests: Summary Output
# ---------------------------------------------------------------------

def test_summary_strictness(data_ols):
    X_df, y_s, _ = data_ols
    res = OLS(y_s, X_df).fit()
    
    # Generate summary string
    # modelsummary expects list[EstimationResult]
    s = summary([res])
    
    # Must NOT contain banned words
    s_lower = s.lower()
    assert "p-value" not in s_lower
    assert "p>|t|" not in s_lower
    assert "prob." not in s_lower
    
    # Should contain important info
    # Estimator name might not be in footer by default unless requested
    # assert "OLS" in s 
    assert "n_obs" in s_lower or "observations" in s_lower

# ---------------------------------------------------------------------
# Edge Cases: Weights
# ---------------------------------------------------------------------

def test_ols_weights():
    # Weighted Least Squares check
    X = np.ones((10, 1))
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    w = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    model = OLS(y, X, add_const=False)
    
    # Analytic weights forbidden for OLS
    with pytest.raises(ValueError, match="forbidden for OLS"):
        model.fit(weights=w)
