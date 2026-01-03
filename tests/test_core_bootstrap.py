
import pytest
import numpy as np
import pandas as pd
from lineareg.core import bootstrap as bs
# BootConfig is defined in estimators.base foundation
from lineareg.estimators.base import BootConfig

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(12345)

# ---------------------------------------------------------------------
# Unit Tests: Wild Distributions
# ---------------------------------------------------------------------

def test_wild_dist_rademacher(rng):
    D = bs.WildDist("rademacher")
    w = D.draw((1000, 1), rng=rng)
    # Mean 0, Variance 1
    assert np.all(np.isin(w, [-1, 1]))
    assert np.abs(w.mean()) < 0.1
    assert np.isclose(w.std(), 1.0, atol=0.1)

def test_wild_dist_mammen(rng):
    D = bs.WildDist("mammen")
    w = D.draw((2000, 1), rng=rng)
    # Check values
    # w1 = -(sqrt(5)-1)/2 = -0.618
    # w2 = (sqrt(5)+1)/2 = 1.618
    unique_vals = np.unique(np.round(w, 5))
    assert len(unique_vals) == 2
    assert -0.61803 in unique_vals
    assert 1.61803 in unique_vals
    
    # E[w] = 0, E[w^2] = 1, E[w^3] = 1
    assert np.abs(w.mean()) < 0.1
    assert np.isclose(w.std(), 1.0, atol=0.1)
    
def test_wild_dist_webb(rng):
    D = bs.WildDist("webb")
    w = D.draw((1000, 1), rng=rng)
    # 6 points: +/- sqrt(k/2) for k=1,2,3 -> +/- 0.707, 1, 1.22
    # Webb mean 0, var 1
    assert np.abs(w.mean()) < 0.1
    assert np.isclose(w.std(), 1.0, atol=0.1)

def test_wild_dist_normal(rng):
    D = bs.WildDist("normal")
    w = D.draw((5000, 1), rng=rng)
    assert np.abs(w.mean()) < 0.1
    assert np.isclose(w.std(), 1.0, atol=0.1)
    # Check normality via kurtosis approx or range
    # Just roughly
    assert w.max() > 2.0
    assert w.min() < -2.0

def test_wild_dist_disallowed():
    # Strict policy check
    D = bs.WildDist("gamma") # Constructor allows it (lazy check)
    with pytest.raises(ValueError, match="Unknown wild distribution"):
        D.draw((10, 1))

# ---------------------------------------------------------------------
# Unit Tests: Bootstrap SE / CI
# ---------------------------------------------------------------------

def test_bootstrap_se_input_validation():
    # Only 1 bootstrap draw
    t_boot = np.array([[1.0]])
    with pytest.raises(ValueError, match="requires at least 2 draws"):
        bs.bootstrap_se(t_boot)
    
    # NaNs in bootstrap
    # Must have >= 2 draws to pass the B check first
    t_nan = np.array([[1.0, 1.0], [np.nan, 2.0]])
    with pytest.raises(ValueError, match="Non-finite bootstrap draws detected"):
        bs.bootstrap_se(t_nan)

def test_bootstrap_se_computation():
    # Synthesize draws from N(beta, sigma^2)
    # True SE = 1
    rng = np.random.default_rng(42)
    t_boot = rng.standard_normal((5, 10000)) # 5 params, 10k draws
    
    se = bs.bootstrap_se(t_boot)
    # se should be close to 1
    assert np.allclose(se, 1.0, atol=0.1)
    
    # Check ddof=1 (n-1) vs ddof=0 (n)
    # Existing code uses ddof=1 by default (Stata behavior)
    # Var = sum(x-bar)^2 / (B-1)
    var_manual = np.var(t_boot, axis=1, ddof=1)
    se_manual = np.sqrt(var_manual)
    assert np.allclose(se, se_manual)

# ---------------------------------------------------------------------
# Unit Tests: Cluster Handling
# ---------------------------------------------------------------------

def test_cluster_check_stata_strict():
    # Check that it warns/errors on too few clusters depending on policy
    pass

def test_weights_shape(rng):
    # Test generation of weights
    # N=100, B=50
    # No clustering
    D = bs.WildDist("rademacher")
    # BootConfig takes: n_boot, seed, wild_dist
    # It does NOT take 'cluster_type'. Cluster type is inferred from data or separate args usually.
    boot = BootConfig(n_boot=50)
    
    # make_multipliers helper in BootConfig handles generation usually
    # But checking raw WildDist again:
    w = D.draw((100, 50), rng=rng)
    assert w.shape == (100, 50)

def test_boot_config_immutability():
    # Ideally config objects should be immutable/stable
    b = BootConfig(n_boot=100)
    assert b.n_boot == 100
    # Modifying it should require replace replacement if dataclass dependent
    # (Just a sanity check of the struct)
