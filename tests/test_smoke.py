
import pytest
import numpy as np
import pandas as pd
from lineareg.estimators.iv import IV2SLS
from lineareg.estimators.ols import OLS
from lineareg.estimators.base import BootConfig

def test_iv_wald_smoke():
    n = 100
    rng = np.random.default_rng(42)
    z = rng.standard_normal((n, 2))
    v = rng.standard_normal((n, 1))
    u = rng.standard_normal((n, 1)) + 0.5 * v
    x = 1.0 + z[:, 0:1] + z[:, 1:2] + v
    y = 1.0 + x * 2.0 + u
    
    df = pd.DataFrame({'y': y.flatten(), 'x': x.flatten(), 'z1': z[:,0], 'z2': z[:,1]})
    
    model = IV2SLS.from_formula("y ~ x", df, iv="(x ~ z1 + z2)")
    model.fit(boot=BootConfig(n_boot=19))
    
    # Test Wald
    R = np.array([[0, 1]]) # 0*const + 1*x
    r = np.array([2.0])
    
    tests = model.wald_test(R, r, B=19)
    assert len(tests) == 1
    assert "bootstrap_quantile" in tests[0]
    assert "reject" in tests[0]

def test_ols_smoke():
    n = 50
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n, 2))
    y = 1 + x[:, 0] + x[:, 1] + rng.standard_normal(n)
    model = OLS(y, x, add_const=True)
    res = model.fit(boot=BootConfig(n_boot=19))
    assert res.params.size == 3
