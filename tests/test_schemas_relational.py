"""Relational AST + controlled vocabulary."""

from __future__ import annotations

from pipeline.schemas_relational import (
    GLOSSARY,
    Predicate,
    RelationalRule,
    glossary_prompt_block,
    normalize_name,
)


def test_predicate_signature_ignores_variable_names():
    a = Predicate("key_bound_to", ("s", "k", "n"), ("Session", "Key", "SN"))
    b = Predicate("key_bound_to", ("x", "y", "z"), ("Session", "Key", "SN"))
    assert a.signature == b.signature  # alpha-renaming must not change the match key


def test_name_alias_normalizes_to_vocabulary():
    assert normalize_name("KSEAF_bound_to") == "key_bound_to"
    assert normalize_name("mac ok") == "mac_verified"
    assert normalize_name("authenticated_with") == "authenticated_with"


def test_negation_is_part_of_signature():
    pos = Predicate("accepts", ("s", "u"), ("Session", "UE"), negated=False)
    neg = Predicate("accepts", ("s", "u"), ("Session", "UE"), negated=True)
    assert pos.signature != neg.signature


def test_rule_json_round_trip():
    rule = RelationalRule(
        rule_id="R1",
        source_text="x",
        premises=(Predicate("accepts", ("s", "u"), ("Session", "UE")),),
        conclusions=(Predicate("mac_verified", ("s", "u"), ("Session", "UE")),),
    )
    back = RelationalRule.from_dict(rule.to_dict())
    assert back.premise_sigs() == rule.premise_sigs()
    assert back.conclusion_sigs() == rule.conclusion_sigs()
    assert back.rule_id == "R1"


def test_glossary_block_lists_every_predicate():
    block = glossary_prompt_block()
    for name in GLOSSARY:
        assert name in block
