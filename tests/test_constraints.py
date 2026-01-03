import numpy as np

from lineareg.utils.constraints import build_rq_from_string, solve_constrained


def test_build_rq_from_string_prunes_labels_consistently() -> None:
    # Two constraints are identical; pruning should drop one and also drop its label.
    var_names = ["const", "x"]
    spec = "_b[x] = 0; x = 0"
    R, q, labels = build_rq_from_string(spec, var_names)

    assert R.shape == (1, 2)
    assert q.shape == (1, 1)
    assert len(labels) == 1


def test_solve_constrained_ridge_changes_solution() -> None:
    # Simple constrained problem where ridge>0 yields a different (but feasible) solution.
    rng = np.random.default_rng(0)
    n = 50
    X = rng.standard_normal((n, 2))
    y = rng.standard_normal((n, 1))

    # constrain beta_1 = 0
    R = np.array([[0.0, 1.0]])
    q = np.array([[0.0]])

    b0 = solve_constrained(X, y, R, q, prune_redundant=True, ridge=0.0)
    b1 = solve_constrained(X, y, R, q, prune_redundant=True, ridge=1e-3)

    assert b0.shape == (2,)
    assert b1.shape == (2,)
    # Feasibility holds exactly-ish for both
    r0 = float((R @ b0.reshape(-1, 1) - q).reshape(-1)[0])
    r1 = float((R @ b1.reshape(-1, 1) - q).reshape(-1)[0])
    assert abs(r0) < 1e-6
    assert abs(r1) < 1e-6
    # Ridge should generally alter the unconstrained coefficient
    assert not np.allclose(b0, b1)
