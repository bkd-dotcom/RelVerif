"""Bounded relational encoder for the 5G-AKA vertical slice.

The propositional platform (rules over atoms) cannot express *who* talks to *whom*
in *which* session, so it cannot state protocol-grade properties like serving-network
binding. This is the minimal relational lift the crypto claims rest on:
a **typed, bounded** model over Party / ServingNetwork / Session with relations
encoded as Z3 functions, so the discover→fix→re-verify loop runs at the relational
level on a small finite universe.

Scope (honest): this is a *bounded* model checker over a hand-built vertical slice,
not a full protocol proof. It demonstrates the mechanism — a known 5G-AKA weakness
(serving-network *confusion*: the UE's session key ends up bound to a different SN
than the one it authenticated with) shows up as a **SAT witness**, and adding the
binding mitigation (SNN in key derivation) makes the violation **UNSAT**. Full
message-term / unbounded proofs are the Tamarin backend's job, not this module's
(see `tamarin_relational_compiler.py` and `ast_tamarin_compiler.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

from z3 import And, BoolSort, Const, EnumSort, Exists, ForAll, Function, Implies, Not, Solver, sat


@dataclass
class SliceResult:
    status: str          # "SAT" (attack witness) | "UNSAT" (secure)
    witness: str         # human-readable description
    secure: bool         # True iff the property cannot be violated


class RelationalAKASlice:
    """A bounded relational rendering of 5G-AKA serving-network binding.

    Universe: ``n_ue`` UEs, ``n_sn`` serving networks, ``n_session`` sessions.
    Relations (Z3 functions):
      - ``authenticated_with(session, ue, sn)`` — the UE ran AKA with ``sn``.
      - ``key_bound_to(session, sn)``           — the derived KSEAF is bound to ``sn``.
      - ``ue_accepts(ue, session)``             — the UE accepted the session.

    Security goal P (serving-network binding): whenever a UE accepts a session it
    authenticated with ``sn``, the session key is bound to that same ``sn``.
    """

    _counter = 0  # unique sort-name suffix: EnumSorts share Z3's global context,
    #               so re-declaring "UE" in a second instance would collide.

    def __init__(self, n_ue: int = 2, n_sn: int = 2, n_session: int = 1) -> None:
        RelationalAKASlice._counter += 1
        tag = RelationalAKASlice._counter
        self.UE, self.ues = EnumSort(f"UE_{tag}", [f"ue{i}" for i in range(n_ue)])
        self.SN, self.sns = EnumSort(f"SN_{tag}", [f"sn{i}" for i in range(n_sn)])
        self.Session, self.sessions = EnumSort(f"Session_{tag}", [f"s{i}" for i in range(n_session)])
        self.authenticated_with = Function(f"authenticated_with_{tag}", self.Session, self.UE, self.SN, BoolSort())
        self.key_bound_to = Function(f"key_bound_to_{tag}", self.Session, self.SN, BoolSort())
        self.ue_accepts = Function(f"ue_accepts_{tag}", self.UE, self.Session, BoolSort())

    # --- the security goal P (universally quantified over the bounded universe) ---
    def _binding_property(self):
        s = Const("s", self.Session)
        u = Const("u", self.UE)
        n = Const("n", self.SN)
        return ForAll(
            [s, u, n],
            Implies(And(self.ue_accepts(u, s), self.authenticated_with(s, u, n)), self.key_bound_to(s, n)),
        )

    # --- a concrete run: ue0 authenticates with sn0 and accepts s0 ---
    def _scenario(self):
        s0, ue0, sn0 = self.sessions[0], self.ues[0], self.sns[0]
        return And(self.authenticated_with(s0, ue0, sn0), self.ue_accepts(ue0, s0))

    # --- adversary (confusion/relay): the key is bound to a DIFFERENT sn ---
    def _adversary_confusion(self):
        s0, sn0 = self.sessions[0], self.sns[0]
        # there exists an sn' != sn0 the key is bound to, and it is NOT bound to sn0
        np = Const("np", self.SN)
        return And(
            Exists([np], And(np != sn0, self.key_bound_to(s0, np))),
            Not(self.key_bound_to(s0, sn0)),
        )

    # --- mitigation M: SNN in key derivation => key bound only to the auth'd sn ---
    def _binding_mitigation(self):
        s = Const("s", self.Session)
        u = Const("u", self.UE)
        n = Const("n", self.SN)
        return ForAll(
            [s, u, n],
            Implies(self.authenticated_with(s, u, n), self.key_bound_to(s, n)),
        )

    def check(self, *, with_mitigation: bool) -> SliceResult:
        """Try to VIOLATE the binding property under the confusion adversary.

        Without the mitigation the solver should find a confusion witness (SAT);
        with the mitigation the violation should be impossible (UNSAT = secure).
        """
        solver = Solver()
        solver.add(self._scenario())
        solver.add(self._adversary_confusion())
        solver.add(Not(self._binding_property()))  # we search for a property violation
        if with_mitigation:
            solver.add(self._binding_mitigation())

        if solver.check() == sat:
            return SliceResult(
                status="SAT",
                witness=(
                    "confusion witness: ue0 authenticated with sn0 and accepted s0, "
                    "but the session key is bound to a different serving network "
                    "(key_bound_to(s0, sn0) is False) — a serving-network confusion."
                ),
                secure=False,
            )
        return SliceResult(
            status="UNSAT",
            witness="no property violation exists under the adversary — serving-network binding holds.",
            secure=True,
        )


def run_slice() -> dict:
    """Run the vertical slice both ways; return the discover→fix evidence."""
    # one instance for both checks — check() uses a fresh Solver each call, and
    # re-declaring the EnumSorts in a second instance collides in Z3's global context.
    model = RelationalAKASlice()
    before = model.check(with_mitigation=False)
    after = model.check(with_mitigation=True)
    return {
        "without_mitigation": {"status": before.status, "secure": before.secure, "witness": before.witness},
        "with_mitigation": {"status": after.status, "secure": after.secure, "witness": after.witness},
        "vertical_slice_validated": (not before.secure) and after.secure,
    }
