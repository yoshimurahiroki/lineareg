"""IV clause parsing utilities.

Parse endogenous ~ instrument clauses from formula text using Patsy AST.
No inference or diagnostics; estimators consume parsed structures.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import patsy

__all__ = ["parse_iv_formula", "parse_iv_formula_with_meta"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence
else:  # pragma: no cover
    Sequence = tuple  # type: ignore[assignment]


def _canonical_factor_text(s: str) -> str:
    """Canonicalize textual form of a factor/term for strict de-duplication:
    - strip leading/trailing spaces
    - collapse internal whitespace
    - remove spaces around '**', ':' and commas
    - trim spaces next to parentheses
    (No algebraic rewriting; purely lexical normalization.)
    """
    import re

    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"\s*\*\*\s*", "**", s)
    s = re.sub(r"\s*:\s*", ":", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\(\s*", "(", s)
    return re.sub(r"\s*\)", ")", s)


def _split_iv_clauses(txt: str) -> list[str]:
    """Split a string possibly containing multiple IV clauses:
    "(x1 ~ z1 + z2) + (x2 ~ z3)" -> ["x1 ~ z1 + z2", "x2 ~ z3"].
    Parentheses balancing is enforced; outer parens optional.
    NOTE: Segments are split only on top-level '+' operators. Terms such as
    '-1' used inside an IV RHS to remove the intercept are handled by Patsy
    and do not affect clause splitting here.
    """
    out: list[str] = []
    s = txt.strip()
    if not s:
        return out
    # Strip exactly one outermost pair *only if* it encloses the whole string
    # AND does not represent a single IV clause (e.g., "(endog ~ instr)" should NOT be stripped).
    # i.e., depth returns to 0 for the first time at the final character.
    if s and s[0] == "(" and s[-1] == ")":
        depth = 0
        encloses = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    encloses = False
                    break
                # if we hit depth==0 before the last char, then not a single wrapper
                if depth == 0 and i < len(s) - 1:
                    encloses = False
                    break
        # Only strip if it encloses AND the inner content is not a single IV clause
        # Check: if inner content has '~' at depth 0 (after stripping), it's a single clause
        if encloses and depth == 0:
            inner = s[1:-1].strip()
            # Count top-level '~' to detect single clause
            inner_depth = 0
            tilde_count_at_depth_0 = 0
            for ch in inner:
                if ch == "(":
                    inner_depth += 1
                elif ch == ")":
                    inner_depth -= 1
                elif ch == "~" and inner_depth == 0:
                    tilde_count_at_depth_0 += 1
            # If exactly one '~' at depth 0, it's a single clause -> don't strip
            if tilde_count_at_depth_0 == 1:
                # Single IV clause: keep outer parens to avoid incorrect splitting
                _no_strip = True  # Don't strip
            else:
                # Multiple clauses or no ~: safe to strip
                s = inner
    depth = 0
    in_s = False
    in_d = False
    start = 0
    for i, ch in enumerate(s):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif in_s or in_d:
            continue
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                msg = "Unbalanced parentheses in IV clause."
                raise ValueError(msg)
        elif ch == "+" and depth == 0:
            seg = s[start:i].strip()
            if seg:
                out.append(
                    seg[1:-1].strip() if seg[0] == "(" and seg[-1] == ")" else seg,
                )
            start = i + 1
    tail = s[start:].strip()
    if tail:
        out.append(tail[1:-1].strip() if tail[0] == "(" and tail[-1] == ")" else tail)
    # Final strict check: parentheses must be balanced at the end of scanning.
    if depth != 0:
        msg = "Unbalanced parentheses in IV clause."
        raise ValueError(msg)
    # require each segment to contain '~'
    for seg in out:
        if "~" not in seg:
            msg = "Each IV segment must be of the form '(endog ~ instr)'."
            raise ValueError(msg)
    return out


def _ast_symbols(expr: str) -> set[str]:
    """Extract variable symbols that appear in *non-function* positions
    from a Patsy EvalFactor expression string, e.g.:
        "I(x ** 2)" -> {"x"}
        "log(x)"    -> {"x"}
        "x"         -> {"x"}
        "x:z"       -> handled at Term-level (two factors "x" and "z")
        "np.log(x)" -> {"x"}
    Any names used as function callees (Name or Attribute in Call.func)
    are NOT included.
    """
    try:
        node = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, TypeError):
        # Be conservative: if parsing fails, fall back to raw-token heuristic
        return _tokens_fallback(expr)

    symbols: set[str] = set()
    # record names that appear *as callees* (functions) to avoid counting them as variables
    callees: set[str] = set()
    # Note: previously tracked common module roots (np, pandas, etc.) are not needed
    # now that we precisely exclude only names that appear in callee position.

    class _Visitor(ast.NodeVisitor):
        __slots__ = ("in_callee",)

        def __init__(self) -> None:
            self.in_callee = False

        def visit_Name(self, n: ast.Name) -> None:
            # If currently visiting a callee (function position), record in callees only.
            if self.in_callee:
                callees.add(n.id)
            else:
                symbols.add(n.id)

        def visit_Call(self, n: ast.Call) -> None:
            # traverse callee in a special mode so names there are recorded as callees
            prev = self.in_callee
            self.in_callee = True
            self.visit(n.func)
            self.in_callee = prev
            # traverse args/keywords (normal variable collection applies)
            for a in n.args:
                self.visit(a)
            for kw in n.keywords:
                if kw.value is not None:
                    self.visit(kw.value)

            # Special handling: Q("var") or C("var", ...) where first arg is a string
            def _func_name(node):
                import ast as _ast

                if isinstance(node, _ast.Name):
                    return node.id
                if isinstance(node, _ast.Attribute):
                    return node.attr
                return None

            fname = _func_name(n.func)
            if fname in {"Q", "C"} and n.args:
                # Special handling: treat Q(...) as opaque (quoted column names)
                # and do not tokenize the quoted string. For C(...), the first
                # argument is the variable (Name or string literal) that the
                # categorical transform depends on; record that symbol.
                import ast as _ast

                if fname == "C" and n.args:
                    arg0 = n.args[0]
                    if isinstance(arg0, _ast.Name):
                        # C(var) with unquoted identifier
                        symbols.add(arg0.id)
                    elif isinstance(arg0, (_ast.Constant, _ast.Str)):
                        # C("var") with string literal (Patsy R-style syntax)
                        # Extract string value: Constant.value (Py3.8+) or Str.s (legacy)
                        val = arg0.value if isinstance(arg0, _ast.Constant) else arg0.s
                        if isinstance(val, str):
                            symbols.add(val)

        def visit_Attribute(self, n: ast.Attribute) -> None:
            # In callee position, attributes like "np.log" should not contribute symbols.
            # We still need to traverse children to reach nested names when not in callee.
            if self.in_callee:
                # collect attribute chain tokens as callees (module/function identifiers)
                def _collect_attr_tokens(node):
                    import ast as _ast

                    if isinstance(node, _ast.Name):
                        callees.add(node.id)
                    elif isinstance(node, _ast.Attribute):
                        _collect_attr_tokens(node.value)
                        callees.add(node.attr)

                _collect_attr_tokens(n)
            else:
                self.visit(n.value)

    _Visitor().visit(node)
    # Remove only those names that actually appeared in callee (function) position.
    # Do NOT blanket-drop known module roots when they appear as plain variables.
    # This preserves variables named like common libraries (e.g., a column named 'np').
    symbols -= callees
    return symbols


def _tokens_fallback(expr: str) -> set[str]:
    """Fallback when AST parsing fails.
    Strict rule: treat a token as a 'function' only if it syntactically appears
    immediately before '(' (optionally with whitespace) or as a module root before '.'.
    Otherwise keep it as a variable token. No blanket blacklist is applied.
    """
    import re

    # Shield quoted strings so we do not extract inner tokens (e.g., Q("x:y"))
    _qstr = re.compile(r'(".*?"|\'.*?\')')
    placeholders: list = []

    def _shield(m: re.Match) -> str:
        placeholders.append(m.group(0))
        return f"__STR{len(placeholders) - 1}__"

    expr_ = _qstr.sub(_shield, expr)
    # all identifier-like tokens outside of quoted strings
    toks_all = set(re.findall(r"[A-Za-z_]\w*", expr_))
    # tokens that are followed by '(' -> functions
    fun_calls = set(re.findall(r"([A-Za-z_]\w*)\s*\(", expr_))
    # module roots before attribute access: np.log, numpy.exp, math.sin, pd.Categorical, scipy.stats, etc.
    # Keep this in sync with _ast_symbols.module_roots to avoid misclassifying common library names as variables.
    mod_roots = set(re.findall(r"\b(np|numpy|math|pd|pandas|scipy)\s*(?=[.])", expr_))
    # keep variables = all tokens minus (callee tokens ∪ module roots used as such)
    return toks_all - fun_calls - mod_roots


def parse_iv_formula_with_meta(
    iv_part: str, main_exog: str,
) -> tuple[list[str], list[str], str, dict]:
    """Extended parse_iv_formula that additionally returns metadata describing
    which instruments were implied from the main RHS and which were user-excluded.

    Returns (endogenous_vars, user_instr, rhs_instr, meta)
    where meta is a dict with keys 'implied_terms' and 'excluded_terms'.
    """
    endogenous_vars, user_instr, rhs_instr = parse_iv_formula(iv_part, main_exog)
    # Recompute implied terms from main_exog (cheap) to produce reproducible metadata
    main_desc = patsy.ModelDesc.from_formula(f"y ~ {main_exog}")
    endog_set = set(endogenous_vars)
    implied: list[str] = []
    for term in main_desc.rhs_termlist:
        if len(term.factors) == 0:
            continue
        if _term_depends_on_endog(term, endog_set):
            continue
        txt = _reconstruct_term(term)
        if txt and txt not in implied:
            implied.append(txt)
    # detect intercept presence in the final instrument RHS for auditing
    rhs_desc = patsy.ModelDesc.from_formula(f"y ~ {rhs_instr if rhs_instr else '0'}")
    rhs_has_intercept = any(len(t.factors) == 0 for t in rhs_desc.rhs_termlist)
    # also record whether main_exog had an intercept (for auditability)
    x_has_intercept = any(len(t.factors) == 0 for t in main_desc.rhs_termlist)
    meta = {
        "implied_terms": implied,
        "excluded_terms": list(user_instr),
        "rhs_has_intercept": bool(rhs_has_intercept),
        "x_has_intercept": bool(x_has_intercept),
    }
    return endogenous_vars, user_instr, rhs_instr, meta


def _term_depends_on_endog(term: patsy.desc.Term, endogenous: set[str]) -> bool:
    """Determine if a Patsy Term depends on at least one endogenous variable.
    For interactions, the Term has multiple EvalFactors, e.g., x:z -> [x, z].
    For transformed factors, inspect AST to recover underlying symbols.
    """
    for f in term.factors:
        name = f.name()
        # trivial variable case (e.g., "x")
        if name in endogenous:
            return True
        # function/transformation case (e.g., "I(x**2)", "log(x)")
        if _ast_symbols(name) & endogenous:
            return True
    return False


def _collect_endog_from_lhs(lhs: Sequence[patsy.desc.Term]) -> list[str]:
    """Collect plain names on LHS; disallow interactions on LHS."""
    out: list[str] = []
    for t in lhs:
        if len(t.factors) == 0:
            # intercept is not allowed on LHS
            continue
        if len(t.factors) > 1:
            msg = "LHS must list endogenous variables additively (no interactions)."
            raise ValueError(msg)
        name = t.factors[0].name()
        # For safety, forbid function-wrapped endog on LHS.
        if _ast_symbols(name) - {name}:
            msg = "Endogenous variables on LHS must be plain symbols (no transforms)."
            raise ValueError(msg)
        out.append(name)
    # keep order and drop duplicates
    seen: set[str] = set()
    uniq: list[str] = []
    for v in out:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def _reconstruct_term(term: patsy.desc.Term) -> str:
    """Reconstruct textual representation of a Term with strict canonicalization.

    Rules enforced:
    - Within a Term, factor order is treated as commutative (R/Stata semantics for
      interaction terms). We therefore sort factor *keys* by their lexical canonical
      form prior to joining with ':' so that algebraically identical Terms such as
      'x:z' and 'z:x' map to the same canonical string.
    - Whitespace and punctuation are normalized via _canonical_factor_text.

    Implementation note:
    - The returned text preserves the original factor text (e.g., C("g", Treatment)
      or Q("x:y")) so that categorical transforms and quoted names are not
      decomposed. Sorting uses a deterministic key derived from the canonicalized
      inner identifier for Q(...) and otherwise the canonicalized factor text. This
      keeps semantic meaning while providing a commutative deterministic ordering.
    """
    # collect raw factor texts (preserve exact textual tokens produced by Patsy)
    parts_raw = [f.name() for f in term.factors]

    import re as _re

    q_pat = _re.compile(r'^\s*Q\((?:"|\')(.+?)(?:"|\')\)\s*$')

    def _sort_key(s: str) -> str:
        m = q_pat.match(s)
        if m:
            # use inner quoted identifier canonicalized as the key
            return _canonical_factor_text(m.group(1))
        return _canonical_factor_text(s)

    parts_sorted = sorted(parts_raw, key=_sort_key)
    # Output is the original factor text normalized for whitespace/punctuation only
    return _canonical_factor_text(":".join(parts_sorted))


def parse_iv_formula(iv_part: str, main_exog: str) -> tuple[list[str], list[str], str]:
    """Parse an IV clause of the form:
        '(endog1 + endog2 ~ z1 + z2 + I(z3**2) + z4:x + ...)'
    together with the main regression RHS 'main_exog' (as a Patsy RHS string).

    Returns
    -------
    endogenous_vars : list[str]
        Endogenous variable names (order-preserving, unique).
    user_supplied_instr : list[str]
        Instruments explicitly listed on RHS of the IV clause (terms, no intercept).
    full_instrument_formula : str
        A Patsy-RHS string that equals:
            union(user_supplied_instr,
                  {Terms in main_exog that do NOT depend on any endogenous var})

    Notes
    -----
    - Interactions and any transformations that *involve* an endogenous variable
      are **excluded** from included instruments (R/Stata semantics).
    - We parse AST of EvalFactor('...') reliably to recover underlying symbols.
    - The final string is deduplicated (first occurrence wins) and has no intercept.

    References
    ----------
    - R AER::ivreg vignette: "Instrumental Variables Regression by 2SLS and GMM"
      Both structural and instrument equations include intercept by default;
      exogenous regressors that do not contain endogenous variables are
      automatically included as instruments.
    - Stata ivregress: "Included exogenous regressors are instruments."

    """
    if not iv_part:
        return [], [], ""
    txt0 = iv_part.strip()
    # --- NEW: R-style bars y ~ X | Z  (or y ~ X | endog | Z_endogonly) ---
    if "|" in txt0 and "~" in txt0:
        # Accept RHS bars. We infer endogenous regressors when only two parts are
        # provided, following AER::ivreg / typical R notation:
        #   y ~ X | Z
        # where X is the structural RHS and Z is the instrument RHS.
        rhs = txt0.split("~", 1)[1].strip()
        parts = [p.strip() for p in rhs.split("|")]
        if len(parts) == 2:
            x_bar, z_bar = parts[0], parts[1]

            # Parse structural RHS and instrument RHS into Patsy terms.
            x_desc = patsy.ModelDesc.from_formula(f"y ~ {x_bar}")
            z_desc = patsy.ModelDesc.from_formula(f"y ~ {z_bar}")

            # Canonical term texts (exclude intercept).
            x_terms: dict[str, patsy.desc.Term] = {}
            for t in x_desc.rhs_termlist:
                if len(t.factors) == 0:
                    continue
                x_terms[_reconstruct_term(t)] = t
            z_term_texts: set[str] = {
                _reconstruct_term(t)
                for t in z_desc.rhs_termlist
                if len(t.factors) != 0
            }

            # Infer endogenous regressors as those present in X but omitted from Z.
            # For theoretical clarity we only allow *plain* endogenous symbols here.
            inferred_endog: list[str] = []
            for txt, term in x_terms.items():
                if txt in z_term_texts:
                    continue
                if len(term.factors) != 1:
                    raise ValueError(
                        "Cannot infer endogenous regressors from bar-IV syntax when omitted structural terms include interactions. "
                        "Use explicit '(endog ~ instr)' or 'y ~ X | ENDOG | Z'.",
                    )
                fac_name = term.factors[0].name()
                if _ast_symbols(fac_name) - {fac_name}:
                    raise ValueError(
                        "Cannot infer endogenous regressors from bar-IV syntax when omitted structural terms include transforms (e.g., I(x**2), C(x)). "
                        "Use explicit '(endog ~ instr)' or 'y ~ X | ENDOG | Z'.",
                    )
                inferred_endog.append(fac_name)

            if not inferred_endog:
                raise ValueError(
                    "Bar-IV syntax 'y ~ X | Z' implies no endogenous regressors (Z contains all structural RHS terms). "
                    "If you intend IV, omit at least one endogenous regressor from Z or use explicit '(endog ~ instr)'.",
                )

            segments = [f"{' + '.join(inferred_endog)} ~ {z_bar}"]
        elif len(parts) == 3:
            # y ~ X | ENDOG | Z_endogonly  -> explicit endogenous+excluded instruments
            segments = [f"{parts[1]} ~ {parts[2]}"]
        else:
            raise ValueError("R-style IV bars allow two or three parts on RHS.")
    else:
        segments = _split_iv_clauses(txt0)
    # Support single segment input like "(x ~ z1 + z2)"
    if not segments:
        txt = iv_part.strip()
        if txt.startswith("(") and txt.endswith(")") and "~" in txt:
            segments = [txt[1:-1].strip()]
        else:
            msg = "IV clause must be '(endog ~ instr)' or a sum of such clauses."
            raise ValueError(msg)

    # First pass: collect all endogenous variables across segments
    endogenous_vars: list[str] = []
    for seg in segments:
        lhs_txt, _rhs_txt = [t.strip() for t in seg.split("~", 1)]
        iv_desc = patsy.ModelDesc.from_formula(f"{lhs_txt} ~ 1")
        endog_here = _collect_endog_from_lhs(iv_desc.lhs_termlist)
        for v in endog_here:
            if v not in endogenous_vars:
                endogenous_vars.append(v)
    endog_all = set(endogenous_vars)

    # Second pass: validate RHS terms against the full endogenous set and collect
    # user-supplied excluded instruments (no intercept). Any instrument term that
    # depends on any endogenous variable (across segments) is invalid.
    user_instr: list[str] = []
    for seg in segments:
        lhs_txt, rhs_txt = [t.strip() for t in seg.split("~", 1)]
        try:
            iv_desc = patsy.ModelDesc.from_formula(f"{lhs_txt} ~ {rhs_txt}")
        except Exception as e:
            # Make IV RHS parsing errors explicit and identify the offending segment
            raise ValueError(f"Invalid IV RHS in segment '{seg}': {e}") from e
        for term in iv_desc.rhs_termlist:
            if len(term.factors) == 0:
                continue
            if _term_depends_on_endog(term, endog_all):
                bad = _reconstruct_term(term)
                raise ValueError(
                    f"Invalid instrument term '{bad}': depends on an endogenous variable.",
                )
            term_txt = _reconstruct_term(term)
            if term_txt and term_txt not in user_instr:
                user_instr.append(term_txt)

    main_desc = patsy.ModelDesc.from_formula(f"y ~ {main_exog}")
    endog_set = set(endogenous_vars)
    included_terms: list[str] = []
    # Detect whether the main RHS includes an intercept (Patsy default is intercept on)
    # -> Instruments must mirror this choice (AER/Stata semantics).
    x_has_intercept = any(len(t.factors) == 0 for t in main_desc.rhs_termlist)
    for term in main_desc.rhs_termlist:
        if len(term.factors) == 0:
            continue  # intercept handled by x_has_intercept
        if _term_depends_on_endog(term, endog_set):
            continue
        term_txt = _reconstruct_term(term)
        if term_txt and term_txt not in included_terms:
            included_terms.append(term_txt)

    # Do *not* force endogenous vars to be written in main_exog.
    # Stata-style syntax allows y [exog] (endog = Z), where endog is implied in the structural RHS.

    # Build union of user_instr and included_terms (both already canonicalized via
    # _reconstruct_term, so no further canonicalization needed).
    union: list[str] = []
    for v in (*user_instr, *included_terms):
        if v and v not in {"Intercept", "1"} and v not in union:
            union.append(v)

    # IV RHS intercept rule (R ivreg/Stata ivregress exact semantics):
    # - If user_supplied_instr is non-empty (excluded instruments present),
    #   check IV clause RHS for intercept removal (e.g., '-1' or '0 + ...').
    # - If user_supplied_instr is empty (identification via functional-form exclusions),
    #   inherit intercept decision from main_exog (x_has_intercept).
    # This matches R ivreg() and Stata ivregress exactly when no excluded instruments.
    if user_instr:
        # Excluded instruments present: enforce *consistent* intercept rule across segments.
        intercept_flags: list[bool] = []
        for seg in segments:
            _lhs, rhs_txt = [t.strip() for t in seg.split("~", 1)]
            _desc = patsy.ModelDesc.from_formula(f"y ~ {rhs_txt}")
            has_icpt = any(len(t.factors) == 0 for t in _desc.rhs_termlist)
            intercept_flags.append(bool(has_icpt))
        # Strict: all segments must agree on intercept presence to avoid silent ambiguity.
        if len(set(intercept_flags)) != 1:
            raise ValueError(
                "Inconsistent intercept specification across IV clauses: "
                "either include an intercept in all clauses or remove it in all clauses (e.g., '-1').",
            )
        use_intercept = intercept_flags[0]
    else:
        # No excluded instruments: inherit from main RHS
        use_intercept = x_has_intercept

    # --- Identification sanity check (R/Stata semantics) ---
    if endogenous_vars:
        # No valid instruments apart from (optional) intercept -> reject
        if (not union) or (len(union) == 0):
            raise ValueError(
                "No valid instruments: the instrument set is empty (intercept-only or zero). "
                "Provide excluded instruments and/or include exogenous regressors on the main RHS.",
            )
        # OPTIONAL: order condition at term-level (disabled by default to match R/Stata staging)
        import os

        if bool(int(os.environ.get("IV_STRICT_ORDER_CONDITION", "0"))):
            if len(union) < len(endogenous_vars):
                raise ValueError(
                    "Insufficient number of instrument terms relative to endogenous count.",
                )
    if use_intercept:
        rhs_instr = " + ".join(union) if union else "1"
    else:
        rhs_instr = ("0 + " + " + ".join(union)) if union else "0"
    return endogenous_vars, user_instr, rhs_instr
