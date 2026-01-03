
import pytest
import numpy as np
import scipy.sparse as sp
from lineareg.core import linalg as la

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def data_dense(rng):
    X = rng.standard_normal((100, 5))
    y = X @ np.ones(5) + rng.standard_normal(100)
    return X, y

@pytest.fixture
def data_rank_deficient(rng):
    X = rng.standard_normal((100, 3))
    X = np.column_stack([X, X[:, 0] + X[:, 1]]) # 4th col is lin comb
    y = rng.standard_normal(100)
    return X, y

# ---------------------------------------------------------------------
# Unit Tests: Finite Checks
# ---------------------------------------------------------------------

def test_check_array_finiteness():
    x = np.array([1.0, 2.0, np.nan])
    with pytest.raises(ValueError, match="Input contains NA/NaN/Inf"):
        la._check_array_finiteness(x)
    
    x_inf = np.array([1.0, np.inf])
    with pytest.raises(ValueError, match="Input contains NA/NaN/Inf"):
        la._check_array_finiteness(x_inf)
    
    # Should pass
    la._check_array_finiteness(np.array([1.0, 2.0]))

def test_assert_all_finite_matrix():
    M = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError):
        la._assert_all_finite_matrix(M)
    
    # Sparse check
    S = sp.csc_matrix([[1.0, 0.0], [0.0, np.nan]])
    # _element_ check logic depends on sparse structure
    # If explicit nan is stored:
    with pytest.raises(ValueError):
        la._assert_all_finite_matrix(S)

# ---------------------------------------------------------------------
# Unit Tests: Decompositions and Rank
# ---------------------------------------------------------------------

def test_qr_outputs(data_dense):
    X, _ = data_dense
    # Test internal numpy fallback and scipy path if available
    # Force use of internal by mocking locally if needed, but for now strict testing
    
    # Standard decomposition
    Q, R = la.qr(X, pivoting=False)
    assert Q.shape == (100, 5)
    assert R.shape == (5, 5)
    # Reconstruct
    assert np.allclose(Q @ R, X)
    # Q orthogonality
    assert np.allclose(Q.T @ Q, np.eye(5))

def test_qr_pivoting(data_rank_deficient):
    X, _ = data_rank_deficient
    # X has shape (100, 4) but rank 3
    Q, R, P = la.qr(X, pivoting=True)
    
    # Permutation P must be valid
    assert len(P) == 4
    assert set(P) == {0, 1, 2, 3}
    
    # Reconstruction: A[:, P] = Q @ R
    assert np.allclose(X[:, P], Q @ R)
    
    # Rank check via rank_from_diag using Stata mode (strict)
    diagR = np.abs(np.diag(R))
    rank_stata = la.rank_from_diag(diagR, 4, mode='stata')
    assert rank_stata == 3

def test_rank_from_diag_modes():
    # Create diagonal that drops off
    diagR = np.array([1e2, 1e1, 1e-8, 1e-15])
    # Stata tolerance roughly 1e-13 * trace/rows. 
    # trace ~ 110, rows=4 => eta ~ 1e-13 * 27.5 ~ 2.7e-12.
    # 1e-8 > eta (rank includes it)
    # 1e-15 < eta (rank excludes it)
    # So rank should be 3
    r_stata = la.rank_from_diag(diagR, 4, mode='stata')
    assert r_stata == 3
    
    # R tolerance 1e-7 * max(|diag|) = 1e-5
    # 1e-8 < 1e-5, so it should be excluded in standard R tolerance? 
    # Actually standard R lm uses 1e-7 * max_abs in dqrdc2 usually, wait.
    # la.py says: tol = 1e-7 * float(np.max(np.abs(d)))
    # tol = 1e-5. 
    # 1e-8 < 1e-5. So 3rd element excluded ??
    # Let's verify 'R' behavior in the code.
    r_r = la.rank_from_diag(diagR, 4, mode='r')
    # 100 > 1e-5 (Keep)
    # 10 > 1e-5 (Keep)
    # 1e-8 < 1e-5 (Drop)
    # 1e-15 < 1e-5 (Drop)
    assert r_r == 2

# ---------------------------------------------------------------------
# Unit Tests: Solvers
# ---------------------------------------------------------------------

def test_solve_qr_rank_deficient(data_rank_deficient):
    X, y = data_rank_deficient
    # Exact collinearity
    
    # Stata mode: 0-fill for dropped vars
    beta_stata = la.qr_solve_stata(X, y) # Should handle rank deficiency
    assert beta_stata.shape == (4, 1)
    # One coef should be 0 exactly (dropped)
    # We don't know exactly which one is dropped due to pivoting, but one MUST be 0
    assert np.sum(beta_stata == 0.0) >= 1
    
    # R mode: NaN-fill for dropped vars
    beta_r = la.qr_coef_r(X, y)
    assert np.sum(np.isnan(beta_r)) >= 1

def test_crossprod(data_dense):
    X, y = data_dense
    XtX = la.crossprod(X, X)
    assert np.allclose(XtX, X.T @ X)
    
    XtY = la.crossprod(X, y)
    assert np.allclose(XtY.flatten(), X.T @ y)

def test_deprecated_crossprod_single_arg(data_dense):
    X, _ = data_dense
    # Expect deprecation warning
    with pytest.warns(DeprecationWarning, match="Calling crossprod\\(X\\) with a single argument is deprecated"):
        XtX = la.crossprod(X)
    assert np.allclose(XtX, X.T @ X)

# ---------------------------------------------------------------------
# Unit Tests: Leverage (Hat Matrix)
# ---------------------------------------------------------------------

def test_hat_diag(data_dense):
    X, _ = data_dense
    # Stata vs R definition
    
    # Unweighted: H = X(X'X)^-1 X'
    h_stata = la.hat_diag(X, convention='stata')
    # Manual H
    H_manual = X @ np.linalg.pinv(X.T @ X) @ X.T
    h_manual = np.diag(H_manual)
    assert np.allclose(h_stata, h_manual)
    
    h_r = la.hat_diag(X, convention='r')
    assert np.allclose(h_r, h_manual) # Should be same for full rank unweighted
    
    # Weighted
    weights = np.random.uniform(0.1, 1.0, size=100)
    
    # Stata: H_w = w * diag(X (X'WX)^-1 X')
    h_w_stata = la.hat_diag_stata(X, weights=weights)
    W = np.diag(weights)
    H_w_true = np.sqrt(W) @ X @ np.linalg.pinv(X.T @ W @ X) @ X.T @ np.sqrt(W)
    # The output of hat_diag_stata is leverage hi = H_ii.
    # Stata documentation says leverage is h_ii element of projection matrix.
    # WLS projection matrix P = X(X'WX)^-1 X'W ?? Or weighted hat matrix H* s.t. y_hat = H* y
    # y_hat = X beta = X (X'WX)^-1 X'Wy
    # So H* = X (X'WX)^-1 X'W. 
    # Diagonal limits of this are what we want? 
    # Stata formula: h_i = w_i * x_i (X'WX)^-1 x_i'
    # Our manual check:
    XtWX_inv = np.linalg.pinv(X.T @ W @ X)
    h_i_manual = []
    for i in range(100):
        val = weights[i] * X[i] @ XtWX_inv @ X[i].T
        h_i_manual.append(val)
    
    assert np.allclose(h_w_stata, h_i_manual)

# ---------------------------------------------------------------------
# Unit Tests: Effective F (MOP 2013)
# ---------------------------------------------------------------------

def test_effective_f_mop2013():
    # Simple setup
    # Z (n x 1)
    Z = np.ones((100, 1))
    pi = np.array([[2.0]])
    Sigma = np.array([[0.5]])
    
    # F_eff = (1/k) * pi_tilde' Sigma_tilde^+ pi_tilde
    # Qzz = Z'Z = 100
    # Qzz^1/2 = 10
    # pi_tilde = 10 * 2 = 20
    # Sigma_tilde = 10 * 0.5 * 10 = 50
    # F = (1/1) * 20 * (1/50) * 20 = 400 / 50 = 8.0
    
    f_val = la.effective_f_from_first_stage(pi, Sigma, Z)
    assert np.isclose(f_val, 8.0)

# ---------------------------------------------------------------------
# Unit Tests: Drop Duplicates
# ---------------------------------------------------------------------

def test_drop_duplicate_cols():
    rng = np.random.default_rng(42)
    A = rng.standard_normal((10, 3))
    # Add duplicate
    A = np.column_stack([A, A[:, 0]])
    assert A.shape == (10, 4)
    
    # Exact drop
    B = la.drop_duplicate_cols(A)
    assert B.shape == (10, 3)
    # Ensure preservation of order (leftmost kept)
    assert np.allclose(B[:, 0], A[:, 0])
