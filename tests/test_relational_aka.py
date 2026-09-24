"""The expanded (~15-rule) relational 5G-AKA slice.

Pins the discover->fix->re-verify evidence for TWO independently-published 5G-AKA
weaknesses reproduced as SAT witnesses and certified fixed (UNSAT):
  A. serving-network confusion (Cremers & Dehnel-Wild, NDSS'19)
  B. failure-message linkability (Basin et al., CCS'18)

Also pins the non-vacuity guard: each fix admits a consistent secure world, so the
UNSAT is a real certification, not an accidental contradiction.
"""

from __future__ import annotations

from z3 import Not, Solver, sat

from verification.relational_aka import RULE_SET, RelationalAKAModel, run_full_slice


# ---- Attack A: serving-network confusion (NDSS'19) --------------------------
def test_confusion_is_sat_without_mitigation():
    r = RelationalAKAModel().check_confusion(with_mitigation=False)
    assert r.status == "SAT"
    assert r.secure is False


def test_confusion_secure_with_mitigation():
    r = RelationalAKAModel().check_confusion(with_mitigation=True)
    assert r.status == "UNSAT"
    assert r.secure is True


# ---- Attack B: failure-message linkability (CCS'18) -------------------------
def test_linkability_is_sat_without_mitigation():
    r = RelationalAKAModel().check_linkability(with_mitigation=False)
    assert r.status == "SAT"
    assert r.secure is False


def test_linkability_secure_with_mitigation():
    r = RelationalAKAModel().check_linkability(with_mitigation=True)
    assert r.status == "UNSAT"
    assert r.secure is True


def test_linkability_witness_is_distinguishable_failure():
    """The SAT witness must be the real CCS'18 side channel: target sees SYNC_FAILURE,
    bystander sees MAC_FAILURE (not some degenerate model)."""
    m = RelationalAKAModel()
    s0, ue0, ue1 = m.sessions[0], m.ues[0], m.ues[1]
    solver = Solver()
    for r in m._core_rules():
        solver.add(r)
    for r in m._leaky_failure_rules():
        solver.add(r)
    solver.add(m.replayed(s0))
    solver.add(m.key_match(s0, ue0))
    solver.add(Not(m.key_match(s0, ue1)))
    solver.add(Not(m._unlinkability_property()))
    assert solver.check() == sat
    model = solver.model()
    assert str(model.eval(m.observed_failure(s0, ue0))) == "SYNC_FAILURE"
    assert str(model.eval(m.observed_failure(s0, ue1))) == "MAC_FAILURE"


# ---- Non-vacuity: each fix admits a consistent secure world -----------------
def test_linkability_fix_is_not_vacuous():
    """UNSAT-with-fix must be a real certification: a consistent secure world exists
    (scenario + core + unified failure is satisfiable without the negated property)."""
    m = RelationalAKAModel()
    s0, ue0, ue1 = m.sessions[0], m.ues[0], m.ues[1]
    solver = Solver()
    for r in m._core_rules():
        solver.add(r)
    for r in m._unified_failure_rule():
        solver.add(r)
    solver.add(m.replayed(s0))
    solver.add(m.key_match(s0, ue0))
    solver.add(Not(m.key_match(s0, ue1)))
    assert solver.check() == sat


def test_confusion_fix_is_not_vacuous():
    m = RelationalAKAModel()
    s0, ue0, sn0 = m.sessions[0], m.ues[0], m.sns[0]
    solver = Solver()
    for r in m._core_rules():
        solver.add(r)
    solver.add(m.authenticated_with(s0, ue0, sn0))
    solver.add(m.accepts(s0, ue0))
    solver.add(m._binding_mitigation())
    assert solver.check() == sat


# ---- The combined evidence artifact + rule inventory ------------------------
def test_full_slice_validates_both_attacks():
    out = run_full_slice()
    assert out["all_attacks_validated"] is True
    assert out["attacks"]["serving_network_confusion"]["validated"] is True
    assert out["attacks"]["failure_message_linkability"]["validated"] is True


def test_rule_set_inventory_is_hand_written_slice():
    # the slice is deliberately small ("~15 hand-written rules"); keep the inventory honest.
    assert len(RULE_SET) >= 15
    assert out_has_both_fix_and_leak_rules()


def out_has_both_fix_and_leak_rules() -> bool:
    joined = " ".join(name for name, _ in RULE_SET)
    return "R7 " in joined and "R7'" in joined and "R9" in joined
