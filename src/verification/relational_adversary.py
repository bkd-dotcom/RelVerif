"""A relational Dolev-Yao adversary (capability library) in bounded Z3.

The first adversary was two hardcoded violation constraints. This one models the *standard*
symbolic attacker as a composable capability set over an explicit knowledge relation
``knows(Message)`` and message structure ``encrypts(cipher, key, plain)``:

  intercept   — the attacker sees every message sent on the network
  decompose   — knowing a ciphertext AND its key yields the plaintext
  compose     — knowing a key AND a plaintext yields the ciphertext (re-encrypt)
  key-secrecy — the attacker cannot know a key it was not given (the crypto assumption)
  replay      — a known message can be re-injected into another session

**Soundness — why entailment, not satisfiability.** Dolev-Yao knowledge is a *least
fixpoint*: the attacker knows exactly what the capability rules derive, nothing more.
A free ``knows`` predicate in SMT could be set to ``true`` arbitrarily, which would
fake attacks. The capability rules are monotone Horn clauses, so every model is a
superset of the closure, and therefore:

    the attacker DERIVES X  ⟺  knows(X) holds in the closure
                            ⟺  (facts ∧ capabilities ∧ ¬knows(X)) is UNSAT.

So `derives()` checks UNSAT of the negation — this is the sound reading, and it needs
no explicit fixpoint. Bounded: small finite Message/Key/Session/Nonce universes.
"""

from __future__ import annotations

from dataclasses import dataclass

from z3 import And, BoolSort, Const, EnumSort, Exists, ForAll, Function, Implies, Not, Solver, sat, unsat

# capability inventory (documented, in lock-step with the rules encoded below)
CAPABILITIES: tuple[tuple[str, str], ...] = (
    ("intercept", "the attacker learns every message sent on the network (Dolev-Yao network control)"),
    ("decompose", "knows(cipher) ∧ encrypts(cipher,key,plain) ∧ knows_key(key) ⇒ knows(plain)"),
    ("compose",   "knows_key(key) ∧ knows(plain) ∧ encrypts(cipher,key,plain) ⇒ knows(cipher) (re-encrypt)"),
    ("key_secrecy", "knows_key(key) ⇒ given_key(key): keys are unforgeable (the crypto assumption)"),
    ("replay",    "a known message can be re-injected into any session (sent in a new session)"),
)


@dataclass
class AdversaryResult:
    scenario: str
    attack_possible: bool   # the attacker can achieve the target (derive secret / get replay accepted)
    detail: str


_ADV_COUNTER = [0]


class RelationalAdversary:
    """Bounded Dolev-Yao capability model over Message / Key / Session / Nonce."""

    def __init__(self, n_msg: int = 3, n_key: int = 2, n_session: int = 2, n_nonce: int = 2) -> None:
        _ADV_COUNTER[0] += 1
        tag = _ADV_COUNTER[0]
        p = f"adv{tag}_"
        self.Msg, self.msgs = EnumSort(f"{p}Msg", [f"{p}m{i}" for i in range(n_msg)])
        self.Key, self.keys = EnumSort(f"{p}Key", [f"{p}k{i}" for i in range(n_key)])
        self.Session, self.sessions = EnumSort(f"{p}S", [f"{p}s{i}" for i in range(n_session)])
        self.Nonce, self.nonces = EnumSort(f"{p}N", [f"{p}n{i}" for i in range(n_nonce)])
        b = BoolSort()
        self.knows = Function(f"{p}knows", self.Msg, b)
        self.knows_key = Function(f"{p}knows_key", self.Key, b)
        self.given_key = Function(f"{p}given_key", self.Key, b)
        self.encrypts = Function(f"{p}encrypts", self.Msg, self.Key, self.Msg, b)  # cipher=enc(key,plain)
        self.sent = Function(f"{p}sent", self.Session, self.Msg, b)
        self.accepted = Function(f"{p}accepted", self.Session, self.Msg, b)
        self.carries_nonce = Function(f"{p}carries_nonce", self.Msg, self.Nonce, b)
        self.fresh_in = Function(f"{p}fresh_in", self.Session, self.Nonce, b)

    # --- the Dolev-Yao capability rules (monotone Horn clauses) ---------------
    def capability_rules(self) -> list:
        s = Const("s", self.Session)
        m = Const("m", self.Msg)
        c = Const("c", self.Msg)
        pl = Const("pl", self.Msg)
        k = Const("k", self.Key)
        return [
            # intercept: everything sent is known
            ForAll([s, m], Implies(self.sent(s, m), self.knows(m))),
            # decompose: cipher + key -> plaintext
            ForAll([c, k, pl], Implies(And(self.knows(c), self.encrypts(c, k, pl), self.knows_key(k)),
                                       self.knows(pl))),
            # compose / re-encrypt: key + plaintext -> cipher
            ForAll([c, k, pl], Implies(And(self.knows_key(k), self.knows(pl), self.encrypts(c, k, pl)),
                                       self.knows(c))),
            # key secrecy: cannot know an un-given key
            ForAll([k], Implies(self.knows_key(k), self.given_key(k))),
        ]

    def derives(self, facts: list, target) -> bool:
        """Sound DY derivability: knows(target) is entailed iff negation is UNSAT."""
        solver = Solver()
        solver.add(*self.capability_rules())
        solver.add(*facts)
        solver.add(Not(target))
        return solver.check() == unsat

    # ======================================================================
    # Demo A — decompose: an encrypted secret leaks IFF the attacker has the key
    # ======================================================================
    def decompose_attack(self, *, key_leaked: bool) -> AdversaryResult:
        c0, secret = self.msgs[0], self.msgs[1]     # c0 = enc(k0, secret)
        k0 = self.keys[0]
        facts = [
            self.encrypts(c0, k0, secret),           # message structure
            self.sent(self.sessions[0], c0),         # c0 observed on the wire
            # a leaked key means the attacker KNOWS it; otherwise key-secrecy (a
            # capability rule) forces knows_key(k0) false because k0 was not given.
            self.knows_key(k0) if key_leaked else Not(self.given_key(k0)),
        ]
        leaked = self.derives(facts, self.knows(secret))
        return AdversaryResult(
            scenario=f"decompose (key_leaked={key_leaked})",
            attack_possible=leaked,
            detail=("attacker derives the plaintext secret by decomposing the intercepted "
                    "ciphertext with the leaked key" if leaked else
                    "secret NOT derivable — decompose is gated by key knowledge (secrecy holds)"),
        )

    # ======================================================================
    # Demo B — replay across sessions, blocked by freshness binding
    # ======================================================================
    def replay_attack(self, *, with_freshness: bool) -> AdversaryResult:
        s0, s1 = self.sessions[0], self.sessions[1]
        m0 = self.msgs[0]
        n0 = self.nonces[0]
        solver = Solver()
        solver.add(*self.capability_rules())
        # m0 carries EXACTLY nonce n0 (a fixed captured message — the attacker cannot
        # rewrite its nonce), fresh in its own session s0 but NOT in s1 (already used).
        solver.add(self.carries_nonce(m0, n0), self.fresh_in(s0, n0), Not(self.fresh_in(s1, n0)))
        nx = Const("nx", self.Nonce)
        solver.add(ForAll([nx], Implies(self.carries_nonce(m0, nx), nx == n0)))
        solver.add(self.sent(s0, m0))               # m0 legitimately sent in s0 -> attacker knows it
        # replay capability: the attacker injects the known m0 into s1
        solver.add(Implies(self.knows(m0), self.sent(s1, m0)))
        if with_freshness:
            # acceptance requires a nonce that is fresh in THIS session
            s = Const("s2", self.Session)
            m = Const("m2", self.Msg)
            nn = Const("nn2", self.Nonce)
            solver.add(ForAll([s, m], Implies(self.accepted(s, m),
                                              Exists([nn], And(self.carries_nonce(m, nn),
                                                               self.fresh_in(s, nn))))))
        # the attack goal: m0 is accepted in the fresh session s1
        solver.add(self.accepted(s1, m0))
        possible = solver.check() == sat
        return AdversaryResult(
            scenario=f"replay (with_freshness={with_freshness})",
            attack_possible=possible,
            detail=("replayed message accepted in a new session — no freshness binding"
                    if possible else
                    "replay rejected — acceptance requires a session-fresh nonce the replay lacks"),
        )


    def legitimate_accept_holds(self) -> bool:
        """Non-vacuity: under the freshness rule, a genuinely fresh message is still
        acceptable — so the mitigation blocks replays, not everything."""
        s1 = self.sessions[1]
        m1, n1 = self.msgs[1], self.nonces[1]
        solver = Solver()
        solver.add(*self.capability_rules())
        s = Const("s3", self.Session)
        m = Const("m3", self.Msg)
        nn = Const("nn3", self.Nonce)
        solver.add(ForAll([s, m], Implies(self.accepted(s, m),
                                          Exists([nn], And(self.carries_nonce(m, nn),
                                                           self.fresh_in(s, nn))))))
        solver.add(self.carries_nonce(m1, n1), self.fresh_in(s1, n1), self.accepted(s1, m1))
        return solver.check() == sat


def run_adversary_demos() -> dict:
    """Run the DY capability demonstrations; return the evidence artifact."""
    a = RelationalAdversary()
    no_key = a.decompose_attack(key_leaked=False)
    with_key = a.decompose_attack(key_leaked=True)
    replay_base = a.replay_attack(with_freshness=False)
    replay_fixed = a.replay_attack(with_freshness=True)
    return {
        "capabilities": [f"{name}: {desc}" for name, desc in CAPABILITIES],
        "decompose": {
            "secret_safe_without_key": (not no_key.attack_possible),
            "secret_leaks_with_key": with_key.attack_possible,
            "detail_no_key": no_key.detail,
            "detail_with_key": with_key.detail,
        },
        "replay": {
            "attack_without_freshness": replay_base.attack_possible,
            "blocked_with_freshness": (not replay_fixed.attack_possible),
            "legitimate_accept_still_holds": a.legitimate_accept_holds(),
            "detail_base": replay_base.detail,
            "detail_fixed": replay_fixed.detail,
        },
        "decompose_capability_validated": (not no_key.attack_possible) and with_key.attack_possible,
        "replay_mitigation_validated": (replay_base.attack_possible
                                        and (not replay_fixed.attack_possible)
                                        and a.legitimate_accept_holds()),
    }
