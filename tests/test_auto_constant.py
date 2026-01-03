import numpy as np
import pytest

from lineareg.utils.auto_constant import add_constant


def test_r_reserved_name_conflict_even_if_other_constant_exists() -> None:
    # First column is non-constant but uses R's reserved intercept name.
    # Second column is all-ones (a true constant).
    X = np.column_stack(
        [np.arange(5, dtype=float), np.ones(5, dtype=float)],
    )

    with pytest.raises(ValueError, match="reserved name '\\(Intercept\\)'|reserved name '\\(Intercept\\)'"):
        add_constant(X, var_names=["(Intercept)", "z"], dialect="r")


def test_stata_reserved_name_conflict_even_if_other_constant_exists() -> None:
    X = np.column_stack(
        [np.arange(5, dtype=float), np.ones(5, dtype=float)],
    )

    with pytest.raises(ValueError, match="reserved name '_cons'"):
        add_constant(X, var_names=["_cons", "z"], dialect="stata")


def test_statsmodels_reserved_name_conflict() -> None:
    X = np.column_stack(
        [np.arange(5, dtype=float), np.ones(5, dtype=float)],
    )

    with pytest.raises(ValueError, match="reserved intercept name 'const'"):
        add_constant(X, var_names=["const", "z"], dialect="statsmodels")


def test_r_allows_reserved_name_if_it_is_the_constant() -> None:
    X = np.column_stack(
        [np.ones(5, dtype=float), np.arange(5, dtype=float)],
    )

    Xo, names, const_name = add_constant(
        X,
        var_names=["(Intercept)", "x"],
        dialect="r",
    )

    assert const_name == "(Intercept)"
    assert names[0] == "(Intercept)"
    assert np.allclose(Xo[:, 0], 1.0)
