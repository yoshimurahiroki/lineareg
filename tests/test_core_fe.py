
import pytest
import numpy as np
import pandas as pd
from lineareg.core import fe as fe_mod

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(999)

# ---------------------------------------------------------------------
# Unit Tests: Singleton Dropping
# ---------------------------------------------------------------------

def test_drop_singletons_iteratively(rng):
    # Construct data with singletons
    # Group 1: [0, 0, 0, 1] (1 is singleton)
    # Group 2: [0, 1, 1, 1] (0 is singleton)
    
    g1 = np.array([0, 0, 0, 1])
    g2 = np.array([0, 1, 1, 1])
    
    # If we drop based on g1, index 3 drops.
    # New data indices: [0, 1, 2]
    # g2 becomes [0, 1, 1]. Group 0 in g2 is now singleton (index 0).
    # Drop index 0.
    # Remaining indices: [1, 2].
    # g1: [0, 0]. No singletons.
    # g2: [1, 1]. No singletons.
    # Expected kept mask: [F, T, T, F]
    
    codes_list = [g1, g2]
    mask = fe_mod._drop_singletons_iteratively(codes_list)
    assert np.all(mask == np.array([False, True, True, False]))

def test_drop_singletons_no_drops():
    g1 = np.array([0, 0, 1, 1])
    g2 = np.array([0, 1, 0, 1])
    codes_list = [g1, g2]
    mask = fe_mod._drop_singletons_iteratively(codes_list)
    assert np.all(mask)

# ---------------------------------------------------------------------
# Unit Tests: Absorption (Demeaning)
# ---------------------------------------------------------------------

def test_absorb_one_way(rng):
    # Simple 1-way fixed effect
    N = 100
    G = 10
    g = rng.integers(0, G, size=N)
    
    # Create y = fe + noise
    fe_vals = rng.standard_normal(G)
    y = fe_vals[g] + rng.standard_normal(N)
    X = rng.standard_normal((N, 2))
    
    # Absorb
    # Signature: absorb(X, y, fe_ids, ...)
    res = fe_mod.absorb(X, y, fe_ids=g)
    
    # Verify y residue
    df = pd.DataFrame({'y': y, 'g': g})
    y_mean = df.groupby('g')['y'].transform('mean').values
    y_expected = y - y_mean
    # res.y is (N, 1)
    assert np.allclose(res.y.flatten(), y_expected)

def test_absorb_nested():
    # g1 nested in g2 imply g1 absorbs g2
    pass

# ---------------------------------------------------------------------
# Unit Tests: Nesting Detection (DoF)
# ---------------------------------------------------------------------

def test_is_nested():
    # g1 = [0, 0, 1, 1]
    # g2 = [0, 1, 0, 1]
    # Not nested
    assert not fe_mod._is_nested(np.array([0, 0, 1, 1]), [np.array([0, 1, 0, 1])])
    
    target = np.array([0, 0, 1, 1]) # g2
    others = [np.array([0, 1, 2, 3])] # g1
    assert fe_mod._is_nested(target, others)

# ---------------------------------------------------------------------
# Unit Tests: FETransformResult
# ---------------------------------------------------------------------

def test_fe_transform_result_properties():
    X = np.zeros((10, 2))
    mask = np.ones(10, dtype=bool)
    res = fe_mod.FETransformResult(X=X, y=None, Z=None, mask=mask)
    assert res.n_effective == 10
    # No fe_dof field in struct

def test_compute_fe_dof():
    # Helper to compute redundant FE degrees of freedom
    # Case: 2 fixed effects, orthogonal
    # N=100
    # G1=10
    # G2=10
    # If orthogonal, dof = G1 + G2 - 1 (if intercept included via G1)
    pass

