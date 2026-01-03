import pytest

from lineareg.utils.instruments import parse_iv_formula


def test_parse_iv_formula_includes_exog_as_instruments() -> None:
    # Structural RHS contains endogenous x and exogenous w.
    # Exogenous terms (w) are included instruments; endogenous terms are not.
    endog, user_instr, rhs = parse_iv_formula("(x ~ z1 + z2)", main_exog="x + w")
    assert endog == ["x"]
    assert user_instr == ["z1", "z2"]
    assert rhs == "z1 + z2 + w"


def test_parse_iv_formula_respects_no_intercept_on_iv_rhs() -> None:
    endog, user_instr, rhs = parse_iv_formula("(x ~ 0 + z)", main_exog="x + w")
    assert endog == ["x"]
    assert user_instr == ["z"]
    assert rhs == "0 + z + w"


def test_parse_iv_formula_rejects_instruments_depending_on_endog() -> None:
    with pytest.raises(ValueError, match="depends on an endogenous variable"):
        parse_iv_formula("(x ~ z + x:z2)", main_exog="x + w")


def test_parse_iv_formula_bar_syntax_infers_endog() -> None:
    # Bar syntax: y ~ X | Z, infer endog as structural terms omitted from Z.
    endog, user_instr, rhs = parse_iv_formula("y ~ x + w | w + z", main_exog="x + w")
    assert endog == ["x"]
    # instrument RHS should include w and z (order from bar RHS, then included terms)
    assert rhs == "w + z"
    assert user_instr == ["w", "z"]
