"""The relational rule compiler + its well-typedness gate."""

from __future__ import annotations

import pytest

from pipeline.schemas_relational import Predicate, RelationalRule
from verification.relational_compiler import (
    RelationalCompileError,
    RelationalRuleCompiler,
    RelationalUniverse,
)


def _rule(prem, conc, rid="R"):
    return RelationalRule(rule_id=rid, source_text="", premises=tuple(prem), conclusions=tuple(conc))


def _compiler():
    return RelationalRuleCompiler(RelationalUniverse())


def test_compiles_a_wellformed_rule():
    rule = _rule([Predicate("accepts", ("s", "u"), ("Session", "UE"))],
                 [Predicate("mac_verified", ("s", "u"), ("Session", "UE"))])
    expr = _compiler().compile_rule(rule)
    assert expr is not None


def test_gate_rejects_unknown_predicate():
    rule = _rule([Predicate("teleports", ("s", "u"), ("Session", "UE"))],
                 [Predicate("accepts", ("s", "u"), ("Session", "UE"))])
    with pytest.raises(RelationalCompileError):
        _compiler().compile_rule(rule)


def test_gate_rejects_sort_conflict():
    # 'x' used as both UE and SN
    rule = _rule([Predicate("accepts", ("s", "x"), ("Session", "UE"))],
                 [Predicate("authorized_by_hn", ("x", "h"), ("SN", "HN"))])
    with pytest.raises(RelationalCompileError):
        _compiler().compile_rule(rule)


def test_gate_rejects_arity_mismatch():
    rule = _rule([], [Predicate("accepts", ("s",), ("Session",))])  # accepts needs 2 args
    with pytest.raises(RelationalCompileError):
        _compiler().compile_rule(rule)


def test_compile_all_reports_rejections_without_crashing():
    good = _rule([], [Predicate("conceals_supi", ("s", "u"), ("Session", "UE"))], "GOOD")
    bad = _rule([], [Predicate("hallucinated", ("s",), ("Session",))], "BAD")
    _compiled, report = _compiler().compile_all([good, bad])
    assert report.compiled == 1
    assert len(report.rejected) == 1
    assert report.rejected[0][0] == "BAD"
