"""Hand-written relational 5G-AKA rule set + relational adversary.

A *bounded relational* rendering of the 5G-AKA core (SUPI/SUCI, KSEAF binding, MAC, XRES, SQN,
failure signalling) as ~15 hand-written typed rules, with a relational Dolev-Yao
adversary (capture / replay / distinguish across sessions), that reproduces **two**
independently-published 5G-AKA weaknesses as SAT witnesses and certifies each fix as
UNSAT:

  A. Serving-network confusion  — Cremers & Dehnel-Wild, NDSS'19 (SN-binding class).
  B. Failure-message linkability — Basin, Dreier, Hirschi, Radomirovic, Sasse,
     Stettler, CCS'18 (the unlinkability attack; distinguishable MAC-failure vs
     SYNC-failure lets an attacker link a subscriber across sessions via replay).

The `relational.py` `RelationalAKASlice` is the smaller 1-pattern / 3-relation proof
of mechanism; this module is the fuller slice. Both are **bounded** model checks over
a hand-built slice — NOT message terms, unbounded sessions, or a Tamarin proof; the
unbounded leg lives in `tamarin_relational_compiler.py` and `ast_tamarin_compiler.py`.
What this module establishes is that the relational encoding, bounded Z3, and the
discover -> fix -> re-verify loop compose and work on *more than one* real, published
5G-AKA weakness — so the Basin CCS'18 linkability row has a relational witness at the
bounded level, not just a requirements-level claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from z3 import (
    And,
    BoolSort,
    Const,
    EnumSort,
    Exists,
    ForAll,
    Function,
    Implies,
    Not,
    Solver,
    sat,
)


@dataclass
class SliceResult:
    attack: str          # short attack id
    status: str          # "SAT" (attack witness) | "UNSAT" (secure)
    secure: bool         # True iff the property cannot be violated
    witness: str         # human-readable description


# The ~15 hand-written relational rules, documented as the model's rule set. Each
# entry names a clause and maps it to the 5G-AKA spec / attack literature. The load-
# bearing ones are encoded as Z3 constraints in RelationalAKAModel below; this list
# is the single readable inventory (kept in lock-step with the code, asserted by a
# test) so a reviewer can audit "which 15 rules".
RULE_SET: tuple[tuple[str, str], ...] = (
    ("R1  supi_concealed",     "SUCI conceals SUPI on every session (TS 33.501 6.12 identity privacy)."),
    ("R2  mac_iff_key",        "MAC verifies for a UE iff it holds the key matching the auth vector."),
    ("R3  accept_needs_mac",   "A UE accepts a session only if the MAC verified."),
    ("R4  accept_needs_fresh", "A UE accepts only if the SQN is in the acceptable range (freshness)."),
    ("R5  accept_no_failure",  "On acceptance no failure message is emitted (observed_failure = NONE)."),
    ("R6  accept_needs_xres",  "Acceptance requires the RES*/XRES* check to match (key confirmation)."),
    ("R7  wrong_key_mac_fail", "Wrong key => MAC-failure message  [LEAK: base protocol only]."),
    ("R8  stale_sqn_sync_fail","Right key + stale SQN => SYNC-failure message  [LEAK: base protocol only]."),
    ("R7' unified_failure",    "FIX: any non-acceptance emits one indistinguishable failure (Basin CCS'18 mitigation)."),
    ("R9  bind_to_auth_sn",    "FIX: key is bound to the SN the UE authenticated with (SNN in key derivation)."),
    ("R10 accept_binds_sn",    "Acceptance implies the UE authenticated with some serving network."),
    ("R11 replay_stale",       "A replayed auth request is stale for the UE that owns its vector (SQN not in range)."),
    ("R12 adv_capture_replay", "Dolev-Yao: adversary can capture an auth request and replay it to any UE/session."),
    ("R13 adv_distinguish",    "Dolev-Yao: adversary observes the failure-message type (the CCS'18 side channel)."),
    ("R14 adv_no_key_forge",   "Dolev-Yao: adversary cannot forge a MAC without the long-term key (crypto assumption)."),
    ("R15 adv_bind_confuse",   "Dolev-Yao: adversary can bind a session key to a serving network of its choice (NDSS'19)."),
)


class RelationalAKAModel:
    """Bounded relational 5G-AKA model over UE / ServingNetwork / Session.

    Relations (Z3 functions):
      authenticated_with(session, ue, sn) — the UE ran AKA with ``sn``.
      key_bound_to(session, sn)           — the derived KSEAF is bound to ``sn``.
      accepts(session, ue)                — the UE accepted the session.
      key_match(session, ue)              — UE holds the key of this session's vector.
      mac_verified(session, ue)           — the MAC verified for this UE.
      sqn_in_range(session, ue)           — the vector's SQN is fresh for this UE.
      xres_matches(session, ue)           — RES*/XRES* key-confirmation matched.
      supi_encrypted(session, ue)         — SUCI (encrypted SUPI) was used.
      replayed(session)                   — this session's request is an adversary replay.
      observed_failure(session, ue)       — the failure message the attacker observes.
    """

    _counter = 0  # unique sort-name suffix: EnumSorts share Z3's global context.

    def __init__(self, n_ue: int = 2, n_sn: int = 2, n_session: int = 2) -> None:
        RelationalAKAModel._counter += 1
        tag = RelationalAKAModel._counter
        self.UE, self.ues = EnumSort(f"UEr_{tag}", [f"ue{i}" for i in range(n_ue)])
        self.SN, self.sns = EnumSort(f"SNr_{tag}", [f"sn{i}" for i in range(n_sn)])
        self.Session, self.sessions = EnumSort(f"Sessr_{tag}", [f"s{i}" for i in range(n_session)])
        self.Failure, fvals = EnumSort(f"Failure_{tag}", ["MAC_FAILURE", "SYNC_FAILURE", "NONE"])
        self.MAC_FAILURE, self.SYNC_FAILURE, self.NONE = fvals

        sess, ue, sn, b = self.Session, self.UE, self.SN, BoolSort()
        self.authenticated_with = Function(f"authenticated_with_{tag}", sess, ue, sn, b)
        self.key_bound_to = Function(f"key_bound_to_{tag}", sess, sn, b)
        self.accepts = Function(f"accepts_{tag}", sess, ue, b)
        self.key_match = Function(f"key_match_{tag}", sess, ue, b)
        self.mac_verified = Function(f"mac_verified_{tag}", sess, ue, b)
        self.sqn_in_range = Function(f"sqn_in_range_{tag}", sess, ue, b)
        self.xres_matches = Function(f"xres_matches_{tag}", sess, ue, b)
        self.supi_encrypted = Function(f"supi_encrypted_{tag}", sess, ue, b)
        self.replayed = Function(f"replayed_{tag}", sess, b)
        self.observed_failure = Function(f"observed_failure_{tag}", sess, ue, self.Failure)

    # ---- shared protocol core (R1-R6, R10, R11) -----------------------------
    def _core_rules(self) -> list:
        s = Const("s_c", self.Session)
        u = Const("u_c", self.UE)
        n = Const("n_c", self.SN)
        rules = [
            ForAll([s, u], self.supi_encrypted(s, u)),                                        # R1
            ForAll([s, u], self.mac_verified(s, u) == self.key_match(s, u)),                  # R2
            ForAll([s, u], Implies(self.accepts(s, u), self.mac_verified(s, u))),             # R3
            ForAll([s, u], Implies(self.accepts(s, u), self.sqn_in_range(s, u))),             # R4
            ForAll([s, u], Implies(self.accepts(s, u), self.observed_failure(s, u) == self.NONE)),  # R5
            ForAll([s, u], Implies(self.accepts(s, u), self.xres_matches(s, u))),             # R6
            ForAll([s, u], Implies(self.accepts(s, u), Exists([n], self.authenticated_with(s, u, n)))),  # R10
            ForAll([s, u], Implies(And(self.replayed(s), self.key_match(s, u)), Not(self.sqn_in_range(s, u)))),  # R11
        ]
        return rules

    # ---- failure signalling: leaky (base) vs unified (fixed) ----------------
    def _leaky_failure_rules(self) -> list:
        s = Const("s_f", self.Session)
        u = Const("u_f", self.UE)
        return [
            ForAll([s, u], Implies(Not(self.key_match(s, u)),
                                   self.observed_failure(s, u) == self.MAC_FAILURE)),         # R7
            ForAll([s, u], Implies(And(self.key_match(s, u), Not(self.sqn_in_range(s, u))),
                                   self.observed_failure(s, u) == self.SYNC_FAILURE)),        # R8
        ]

    def _unified_failure_rule(self) -> list:
        s = Const("s_f", self.Session)
        u = Const("u_f", self.UE)
        # R7': every non-acceptance emits the same indistinguishable failure symbol.
        return [
            ForAll([s, u], Implies(Not(self.accepts(s, u)),
                                   self.observed_failure(s, u) == self.MAC_FAILURE)),         # R7'
        ]

    # ======================================================================
    # Attack A — serving-network confusion (Cremers & Dehnel-Wild NDSS'19)
    # ======================================================================
    def _binding_property(self):
        s = Const("s_b", self.Session)
        u = Const("u_b", self.UE)
        n = Const("n_b", self.SN)
        # P: if a UE accepts a session it authenticated with n, the key is bound to n.
        return ForAll([s, u, n],
                      Implies(And(self.accepts(s, u), self.authenticated_with(s, u, n)),
                              self.key_bound_to(s, n)))

    def _binding_mitigation(self):
        s = Const("s_m", self.Session)
        u = Const("u_m", self.UE)
        n = Const("n_m", self.SN)
        return ForAll([s, u, n],                                                              # R9
                      Implies(self.authenticated_with(s, u, n), self.key_bound_to(s, n)))

    def check_confusion(self, *, with_mitigation: bool) -> SliceResult:
        s0, ue0, sn0 = self.sessions[0], self.ues[0], self.sns[0]
        np = Const("np", self.SN)
        solver = Solver()
        for r in self._core_rules():
            solver.add(r)
        solver.add(self.authenticated_with(s0, ue0, sn0))
        solver.add(self.accepts(s0, ue0))
        # adversary (R15): the key is bound to some sn' != sn0 and NOT to sn0.
        solver.add(And(Exists([np], And(np != sn0, self.key_bound_to(s0, np))),
                       Not(self.key_bound_to(s0, sn0))))
        solver.add(Not(self._binding_property()))
        if with_mitigation:
            solver.add(self._binding_mitigation())
        if solver.check() == sat:
            return SliceResult(
                "serving_network_confusion", "SAT", secure=False,
                witness="confusion witness: ue0 authenticated with sn0 and accepted, but KSEAF "
                "is bound to a different serving network (NDSS'19 SN-binding class).")
        return SliceResult(
            "serving_network_confusion", "UNSAT", secure=True,
            witness="no confusion witness under the adversary — serving-network binding holds.")

    # ======================================================================
    # Attack B — failure-message linkability (Basin et al. CCS'18)
    # ======================================================================
    def _unlinkability_property(self):
        s = Const("s_u", self.Session)
        u1 = Const("u1_u", self.UE)
        u2 = Const("u2_u", self.UE)
        # P: for a replayed request the observable failure must not depend on which UE
        # received it (else the attacker links the subscriber). Indistinguishability.
        return ForAll([s, u1, u2],
                      Implies(self.replayed(s),
                              self.observed_failure(s, u1) == self.observed_failure(s, u2)))

    def check_linkability(self, *, with_mitigation: bool) -> SliceResult:
        s0, ue0, ue1 = self.sessions[0], self.ues[0], self.ues[1]
        solver = Solver()
        for r in self._core_rules():
            solver.add(r)
        # failure signalling: the leak (base) or the unified-message fix (mitigation).
        failure_rules = self._unified_failure_rule() if with_mitigation else self._leaky_failure_rules()
        for r in failure_rules:
            solver.add(r)
        # adversary (R12): capture ue0's vector, replay it to both ue0 and ue1.
        solver.add(self.replayed(s0))
        solver.add(self.key_match(s0, ue0))       # target owns the captured vector
        solver.add(Not(self.key_match(s0, ue1)))  # bystander does not
        # (R13) attacker searches for a distinguishing observation:
        solver.add(Not(self._unlinkability_property()))
        if solver.check() == sat:
            return SliceResult(
                "failure_message_linkability", "SAT", secure=False,
                witness="linkability witness: replaying ue0's vector yields SYNC_FAILURE from "
                "ue0 (right key, stale SQN) but MAC_FAILURE from ue1 (wrong key) — the "
                "distinguishable failure links the subscriber (Basin et al. CCS'18).")
        return SliceResult(
            "failure_message_linkability", "UNSAT", secure=True,
            witness="unified failure message: replay yields the same observation regardless of "
            "UE — the subscriber cannot be linked (CCS'18 mitigation certified).")


def run_full_slice() -> dict:
    """Run both attacks both ways; return the discover->fix->re-verify evidence."""
    model = RelationalAKAModel()
    attacks = {}
    validated = True
    for name, fn in (("serving_network_confusion", model.check_confusion),
                     ("failure_message_linkability", model.check_linkability)):
        before = fn(with_mitigation=False)
        after = fn(with_mitigation=True)
        attacks[name] = {
            "without_mitigation": {"status": before.status, "secure": before.secure, "witness": before.witness},
            "with_mitigation": {"status": after.status, "secure": after.secure, "witness": after.witness},
            "validated": (not before.secure) and after.secure,
        }
        validated = validated and attacks[name]["validated"]
    return {
        "rule_count": len(RULE_SET),
        "rules": [f"{name}: {desc}" for name, desc in RULE_SET],
        "attacks": attacks,
        "all_attacks_validated": validated,
    }
