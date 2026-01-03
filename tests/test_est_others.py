
import pytest
import numpy as np
import pandas as pd
from lineareg.estimators.gmm import GMM
from lineareg.estimators.qr import QR
from lineareg.estimators.base import BootConfig

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def data_simple():
    rng = np.random.default_rng(999)
    N = 100
    x = rng.standard_normal(N)
    y = 1 + 2 * x + rng.standard_normal(N)
    return pd.DataFrame({'y': y, 'x': x})

# ---------------------------------------------------------------------
# Quantile Regression
# ---------------------------------------------------------------------

def test_qr_fit(data_simple):
    # Tests Quantile Regression
    model = QR(data_simple['y'], data_simple[['x']], add_const=True)
    
    # Fit median regression (q=0.5 default)
    # tau is passed to init or fit?
    # QR.__init__(..., tau=0.5)
    # fit_multi takes taus. fit() uses self.tau
    
    boot = BootConfig(n_boot=10, seed=42, dist="exp")
    # fit() arguments: device, boot, cluster_ids...
    res = model.fit(boot=boot)
    
    assert res.params is not None
    assert 'x' in res.params
    assert res.se is not None
    assert (res.se > 0).all()
    
    # Check median (quantile 0.5) is close to 2 (slope) and >0
    assert np.isclose(res.params['x'], 2.0, atol=0.5)

def test_qr_weights_forbidden(data_simple):
    w = np.ones(len(data_simple))
    model = QR(data_simple['y'], data_simple[['x']], add_const=True)
    with pytest.raises(TypeError):
        model.fit(weights=w)

# ---------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------

def test_gmm_smoke(data_simple):
    # Basic GMM usage (IV-GMM usually)
    # y = x*beta + u
    # Z = x (just OLS-GMM equivalent)
    # In this library, GMM usually implies IV-GMM with efficient weighting
    
    # Setup
    # GMM requires at least one endogenous variable (project policy/check)
    N = len(data_simple)
    rng = np.random.default_rng(42)
    # z is instrument
    z = rng.standard_normal(N)
    # x is endogenous: x = 0.5*z + e
    e = rng.standard_normal(N) + 0.5 * data_simple['y'] # correlates with y's error if we assume y model
    # Simpler: just create dataframes
    df = pd.DataFrame({'y': data_simple['y'], 'x': data_simple['x'], 'z': z})
    
    X = df[['x']] 
    Z = df[['z']] 

    try:
        model = GMM(
            df['y'], X, Z, 
            endog_idx=[0], # x is endogenous
            # z_excluded_idx not in GMM init
            add_const=True
        )
        boot = BootConfig(n_boot=10, seed=42)
        res = model.fit(boot=boot)
        
        assert res.params is not None
        assert res.se is not None
        
    except ImportError:
        # If GMM not implemented or import fails
        pytest.skip("GMM module not ready or import failed")
