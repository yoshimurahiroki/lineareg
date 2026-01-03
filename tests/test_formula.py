import numpy as np
import pandas as pd

from lineareg.utils.formula import FormulaParser


def _toy_df() -> pd.DataFrame:
    n = 10
    return pd.DataFrame(
        {
            "y": np.arange(n, dtype=float),
            "x": np.arange(n, dtype=float) + 1.0,
        },
        index=pd.RangeIndex(n),
    )


def test_include_intercept_detects_0_plus_x() -> None:
    df = _toy_df()
    p = FormulaParser(df)
    out = p.parse("y ~ 0 + x")
    assert out["include_intercept"] is False


def test_include_intercept_detects_x_plus_0() -> None:
    df = _toy_df()
    p = FormulaParser(df)
    out = p.parse("y ~ x + 0")
    assert out["include_intercept"] is False


def test_include_intercept_detects_x_minus_1() -> None:
    df = _toy_df()
    p = FormulaParser(df)
    out = p.parse("y ~ x - 1")
    assert out["include_intercept"] is False


def test_include_intercept_default_true() -> None:
    df = _toy_df()
    p = FormulaParser(df)
    out = p.parse("y ~ x")
    assert out["include_intercept"] is True
