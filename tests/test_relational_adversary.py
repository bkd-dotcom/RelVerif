"""The relational Dolev-Yao adversary capability library."""

from __future__ import annotations

from verification.relational_adversary import CAPABILITIES, RelationalAdversary, run_adversary_demos


def test_decompose_gated_by_key_knowledge():
    a = RelationalAdversary()
    assert a.decompose_attack(key_leaked=False).attack_possible is False   # secret safe
    assert a.decompose_attack(key_leaked=True).attack_possible is True     # decompose leaks it


def test_replay_blocked_by_freshness():
    a = RelationalAdversary()
    assert a.replay_attack(with_freshness=False).attack_possible is True   # replay works
    assert a.replay_attack(with_freshness=True).attack_possible is False   # freshness blocks it


def test_freshness_is_not_vacuous():
    # a legitimately-fresh message must still be acceptable under the freshness rule
    assert RelationalAdversary().legitimate_accept_holds() is True


def test_demos_validate_both_capabilities():
    d = run_adversary_demos()
    assert d["decompose_capability_validated"] is True
    assert d["replay_mitigation_validated"] is True


def test_capability_inventory_present():
    names = {name for name, _ in CAPABILITIES}
    assert {"intercept", "decompose", "compose", "key_secrecy", "replay"} <= names
