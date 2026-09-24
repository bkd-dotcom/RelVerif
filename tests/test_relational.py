"""The bounded relational vertical slice.

Pins the discover->fix evidence: the serving-network confusion pattern is a SAT
witness without the binding mitigation, and UNSAT (secure) with it.
"""

from __future__ import annotations

from verification.relational import RelationalAKASlice, run_slice


def test_confusion_is_sat_without_mitigation():
    r = RelationalAKASlice().check(with_mitigation=False)
    assert r.status == "SAT"
    assert r.secure is False


def test_binding_holds_with_mitigation():
    r = RelationalAKASlice().check(with_mitigation=True)
    assert r.status == "UNSAT"
    assert r.secure is True


def test_vertical_slice_validated():
    out = run_slice()
    assert out["without_mitigation"]["secure"] is False
    assert out["with_mitigation"]["secure"] is True
    assert out["vertical_slice_validated"] is True
