"""The end-to-end relational discover -> fix -> re-verify loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.schemas_relational import load_gold, normalize_name
from verification.relational_closed_loop import (
    binding_property,
    decisive_predicates,
    key_confirmation_property,
    run_closed_loop,
    supi_concealment_property,
)
from verification.relational_compiler import RelationalUniverse

GOLD = "data/gold/relational_gold.jsonl"


def _gold_by_id():
    # The gold file is committed, but a git clean -xdf can remove it: the
    # annotator artifacts stay on the research machine and are not pushed, so a
    # fresh clone or CI checkout has no gold file. Skip rather than fail there --
    # the same convention tests/test_webapp_artifacts.py uses for data/generated*.
    if not Path(GOLD).is_file():
        pytest.skip(f"{GOLD} not present")
    return {r.rule_id: r for r in load_gold(GOLD)}


@pytest.mark.parametrize(("prop_factory", "expected"), [
    (binding_property, ("key_bound_to",)),
    (supi_concealment_property, ("encrypted_with_hn_key",)),
    (key_confirmation_property, ("res_matches_xres",)),
])
def test_decisive_predicates_auto_derived_from_ast(prop_factory, expected):
    """The exclude set is derived from the property AST (goal-referenced minus
    scenario-triggered = the guaranteed relation), not a hand-maintained list."""
    assert decisive_predicates(prop_factory()) == expected


def test_auto_derived_exclude_closes_each_anchor():
    """Using ONLY the auto-derived exclude set, the gold-control loop validates E2E for
    all three anchors (discover -> fix -> re-verify, non-vacuous)."""
    g = _gold_by_id()
    for prop_factory, mit in [(binding_property, "REL13"),
                              (supi_concealment_property, "REL02"),
                              (key_confirmation_property, "REL11")]:
        prop = prop_factory()
        base = _base_excluding(g, *decisive_predicates(prop))
        res = run_closed_loop(RelationalUniverse(), base, g[mit], prop, mitigation_source="gold")
        assert res.end_to_end_validated is True, prop.name


def _base_excluding(rules_by_id, *preds):
    """Base rules that do not already carry the anchor's decisive predicate(s), so
    DISCOVER is a fair test regardless of the extractor."""
    want = {normalize_name(p) for p in preds}
    return [r for r in rules_by_id.values()
            if not any(normalize_name(p.name) in want
                       for p in (*r.premises, *r.conclusions))]


def _binding_free_base(rules_by_id):
    return _base_excluding(rules_by_id, "key_bound_to")


def test_binding_rule_closes_discovered_vulnerability():
    g = _gold_by_id()
    res = run_closed_loop(RelationalUniverse(), _binding_free_base(g), g["REL13"],
                          binding_property(), mitigation_source="gold")
    assert res.discovered_vulnerability is True   # vuln exists without binding
    assert res.fixed is True                      # extracted/gold binding rule closes it
    assert res.secure_world_exists is True        # non-vacuous
    assert res.end_to_end_validated is True


def test_supi_concealment_second_anchor_closes_imsi_catcher():
    """Second anchor (privacy): the SUCI-encryption rule (extracted REL02) closes the
    IMSI-catcher witness — an identifying message sent in the clear."""
    g = _gold_by_id()
    base = _base_excluding(g, "identifies", "encrypted_with_hn_key")
    res = run_closed_loop(RelationalUniverse(), base, g["REL02"],
                          supi_concealment_property(), mitigation_source="gold")
    assert res.discovered_vulnerability is True   # IMSI-catcher exists without concealment
    assert res.fixed is True                      # SUCI-encryption rule closes it
    assert res.secure_world_exists is True        # non-vacuous
    assert res.end_to_end_validated is True


def test_supi_concealment_wrong_mitigation_does_not_fix():
    """Negative control: a MAC rule (REL05) must NOT close the privacy vulnerability."""
    g = _gold_by_id()
    base = _base_excluding(g, "identifies", "encrypted_with_hn_key")
    res = run_closed_loop(RelationalUniverse(), base, g["REL05"],
                          supi_concealment_property(), mitigation_source="gold")
    assert res.discovered_vulnerability is True
    assert res.fixed is False
    assert res.end_to_end_validated is False


def test_key_confirmation_third_anchor_closes_forged_accept():
    """Third anchor (entity authentication): the key-confirmation rule (extracted REL11,
    accepts -> res_matches_xres) closes the forged-accept witness — a UE that accepts
    without its RES* matching the expected XRES*."""
    g = _gold_by_id()
    base = _base_excluding(g, "res_matches_xres")
    res = run_closed_loop(RelationalUniverse(), base, g["REL11"],
                          key_confirmation_property(), mitigation_source="gold")
    assert res.discovered_vulnerability is True   # forged accept exists without confirmation
    assert res.fixed is True                      # key-confirmation rule closes it
    assert res.secure_world_exists is True        # non-vacuous
    assert res.end_to_end_validated is True


def test_key_confirmation_wrong_mitigation_does_not_fix():
    """Negative control: a MAC rule (REL05, accepts -> mac_verified) must NOT close the
    key-confirmation vulnerability — MAC verification is not challenge-response confirmation."""
    g = _gold_by_id()
    base = _base_excluding(g, "res_matches_xres")
    res = run_closed_loop(RelationalUniverse(), base, g["REL05"],
                          key_confirmation_property(), mitigation_source="gold")
    assert res.discovered_vulnerability is True
    assert res.fixed is False
    assert res.end_to_end_validated is False


def test_non_binding_mitigation_does_not_fix():
    """Negative control: a rule that doesn't constrain key_bound_to must NOT close the vuln."""
    g = _gold_by_id()
    res = run_closed_loop(RelationalUniverse(), _binding_free_base(g), g["REL05"],  # accepts->mac
                          binding_property(), mitigation_source="gold")
    assert res.discovered_vulnerability is True
    assert res.fixed is False
    assert res.end_to_end_validated is False


def test_inconsistent_extracted_base_is_repaired_and_named():
    """A mis-extracted rule that contradicts another (e.g. mac->computes vs mac->!computes)
    makes the base inconsistent; the loop must detect it, name the culprit, repair, and
    still drive the loop on the surviving rules."""
    from pipeline.schemas_relational import Predicate, RelationalRule
    g = _gold_by_id()
    base = _binding_free_base(g)
    mac = ("s", "u"), ("Session", "UE")
    p_ok = RelationalRule("BADP", "x",
                          premises=(Predicate("mac_verified", *mac),),
                          conclusions=(Predicate("computes_res", ("s", "u", "m"), ("Session", "UE", "Message")),))
    p_bad = RelationalRule("BADN", "x",
                           premises=(Predicate("mac_verified", *mac),),
                           conclusions=(Predicate("computes_res", ("s", "u", "m"),
                                                  ("Session", "UE", "Message"), negated=True),))
    res = run_closed_loop(RelationalUniverse(), [*base, p_ok, p_bad], g["REL13"], binding_property())
    assert res.base_consistent is False
    assert len(res.conflicting_rules) >= 1
    assert res.end_to_end_validated is True  # repaired base still drives discover->fix


def test_hallucinated_base_rule_is_skipped_not_crashed():
    from pipeline.schemas_relational import Predicate, RelationalRule
    g = _gold_by_id()
    base = _binding_free_base(g)
    base.append(RelationalRule("BADX", "x",
                               premises=(Predicate("bogus", ("s",), ("Session",)),),
                               conclusions=()))
    res = run_closed_loop(RelationalUniverse(), base, g["REL13"], binding_property())
    assert res.n_base_rules_rejected == 1
    assert res.end_to_end_validated is True  # the good rules still drive the loop
