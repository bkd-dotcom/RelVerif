"""The relational discover -> fix -> re-verify loop over compiled rules.

Ties extraction to verification: given a set of `RelationalRule` ASTs (the
kind the LLM extractor emits), compile them (`relational_compiler`), then

  DISCOVER  — with the current rule set, can the security property be violated under
              the adversary?  (SAT = a vulnerability witness exists)
  FIX       — add the mitigation rule (itself a RelationalRule — in the end-to-end
              demo, an *LLM-extracted* one) and re-solve
  RE-VERIFY — the violation is now impossible  (UNSAT = secure)

A `RelationalProperty` bundles the safety goal + a concrete scenario + the adversary,
all expressed over one `RelationalUniverse` so they share the compiled function
symbols. `binding_property()` is the serving-network-binding instance used by the
end-to-end demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from z3 import And, Const, ExprRef, ForAll, Implies, Not, Solver, sat, unsat

from verification.relational_compiler import (
    RelationalCompileError,
    RelationalRuleCompiler,
    RelationalUniverse,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pipeline.schemas_relational import RelationalRule


@dataclass
class RelationalProperty:
    name: str
    goal: Callable[[RelationalUniverse], ExprRef]        # the safety property P (quantified)
    scenario: Callable[[RelationalUniverse], list]       # ground facts of a concrete run
    adversary: Callable[[RelationalUniverse], ExprRef]   # the attacker capability / violation attempt


class _PredicateRecorder:
    """Wraps a `RelationalUniverse`, recording (in first-seen order) every predicate
    name accessed via ``func()`` while a goal/scenario builder runs. Delegates every
    other method so the recorded builder produces an identical Z3 expression."""

    def __init__(self, universe: RelationalUniverse) -> None:
        self._u = universe
        self.seen: list[str] = []

    def sort(self, name: str):
        return self._u.sort(name)

    def const(self, sort_name: str, i: int = 0):
        return self._u.const(sort_name, i)

    def consts(self, sort_name: str) -> list:
        return self._u.consts(sort_name)

    def known_predicate(self, pred_name: str) -> bool:
        return self._u.known_predicate(pred_name)

    def func(self, pred_name: str):
        from pipeline.schemas_relational import normalize_name
        self.seen.append(normalize_name(pred_name))
        return self._u.func(pred_name)


def decisive_predicates(prop: RelationalProperty,
                        universe: RelationalUniverse | None = None) -> tuple[str, ...]:
    """Auto-derive an anchor's exclude set from the property's own AST — no hand list.

    The goal is a guarantee of the form ``ForAll(Implies(trigger.., guaranteed..))``.
    The *decisive* predicates are exactly those the **goal references but the triggering
    scenario does not** — i.e. the relation the property guarantees, not the facts that
    merely set the scenario up. An unconstrained base for a fair DISCOVER must exclude
    rules that already carry that guaranteed relation. Returns canonical predicate names
    in first-seen order.
    """
    u = universe or RelationalUniverse()
    grec = _PredicateRecorder(u)
    prop.goal(grec)
    srec = _PredicateRecorder(u)
    prop.scenario(srec)
    scenario_preds = set(srec.seen)
    # dedupe while preserving order, drop the scenario (trigger) predicates
    seen_goal = list(dict.fromkeys(grec.seen))
    return tuple(p for p in seen_goal if p not in scenario_preds)


def property_predicates(prop: RelationalProperty,
                        universe: RelationalUniverse | None = None) -> tuple[str, ...]:
    """Every predicate the property's goal and scenario mention, in first-seen order.

    Used to scope a reported counterexample: a witness over all nineteen vocabulary
    predicates is reproducible but unreadable, while a witness over the trigger and the
    guaranteed relation is exactly the pair a reader checks against the published attack.
    """
    u = universe or RelationalUniverse()
    rec = _PredicateRecorder(u)
    prop.goal(rec)
    prop.scenario(rec)
    return tuple(dict.fromkeys(rec.seen))


@dataclass
class LoopResult:
    property_name: str
    discovered_vulnerability: bool
    fixed: bool
    secure_world_exists: bool  # non-vacuity: base+mitigation+scenario is satisfiable
    mitigation_rule_id: str
    mitigation_source: str
    n_base_rules_compiled: int
    n_base_rules_rejected: int
    base_consistent: bool = True          # base+scenario satisfiable before repair
    conflicting_rules: list = field(default_factory=list)  # rule_ids dropped to restore consistency
    rejected: list = field(default_factory=list)

    @property
    def end_to_end_validated(self) -> bool:
        # a genuine end-to-end result: a vuln was found, the mitigation closes it, AND
        # the fix is non-vacuous (the secured system is still satisfiable, not a contradiction).
        return self.discovered_vulnerability and self.fixed and self.secure_world_exists

    def to_dict(self) -> dict:
        return {
            "property": self.property_name,
            "discovered_vulnerability": self.discovered_vulnerability,
            "fixed": self.fixed,
            "secure_world_exists": self.secure_world_exists,
            "end_to_end_validated": self.end_to_end_validated,
            "base_consistent": self.base_consistent,
            "conflicting_rules": self.conflicting_rules,
            "mitigation_rule_id": self.mitigation_rule_id,
            "mitigation_source": self.mitigation_source,
            "n_base_rules_compiled": self.n_base_rules_compiled,
            "n_base_rules_rejected": self.n_base_rules_rejected,
            "rejected": self.rejected,
        }


def _consistency_repair(compiled: list, scenario: list) -> tuple[list, list]:
    """Admit rules in canonical order, dropping any that contradict the admitted set.

    Extraction noise can make a rule set self-contradictory (e.g. a mis-extracted rule
    that inverts an implication), so the base must be repaired before it is used as the
    premise of a discovery run. Rules are considered in sorted ``rule_id`` order and a
    rule is dropped exactly when adding it to the already-admitted set plus the scenario
    is unsatisfiable. The result is a maximal consistent subset -- not a minimum-cardinality
    one, which would require solving an NP-hard optimisation for no benefit here.

    Canonical order is the point. The earlier implementation deleted whichever rule Z3's
    unsat core happened to name, and those cores depend on the generated symbol names,
    which depend in turn on how many universes the process had already constructed. The
    dropped set therefore varied with unrelated call order, which is not something an
    artifact should do. Deciding by ``rule_id`` makes the repair a function of the rule set
    alone, so a reported ``conflicting_rules`` list is reproducible.
    """
    solver = Solver()
    solver.add(*scenario)
    if solver.check() != sat:  # the scenario itself is unsat — no rule is to blame
        return list(compiled), []
    kept: list = []
    dropped: list = []
    for rid, expr in sorted(compiled, key=lambda p: p[0]):
        solver.push()
        solver.add(expr)
        if solver.check() == sat:
            kept.append((rid, expr))  # keep it asserted for the rules that follow
        else:
            solver.pop()
            dropped.append(rid)
    return kept, dropped


def run_closed_loop(
    universe: RelationalUniverse,
    base_rules: list[RelationalRule],
    mitigation_rule: RelationalRule,
    prop: RelationalProperty,
    *,
    mitigation_source: str = "extracted",
) -> LoopResult:
    compiler = RelationalRuleCompiler(universe)
    goal = prop.goal(universe)
    scenario = prop.scenario(universe)
    adversary = prop.adversary(universe)

    # compile per-rule (tracking rule_id) so we can name the conflicting extractions
    compiled_pairs: list = []
    rejected: list = []
    for r in base_rules:
        try:
            compiled_pairs.append((r.rule_id, compiler.compile_rule(r)))
        except RelationalCompileError as e:
            rejected.append((r.rule_id, str(e)))

    # CONSISTENCY — extraction noise can make the base self-contradictory; repair it
    # and report which rules were dropped (a downstream symptom of extraction errors).
    kept, conflicting = _consistency_repair(compiled_pairs, scenario)
    base_compiled = [e for _, e in kept]

    # DISCOVER — search for a property violation with the current (incomplete) rules
    disc = Solver()
    disc.add(*base_compiled, *scenario, adversary, Not(goal))
    discovered = disc.check() == sat

    # FIX + RE-VERIFY — add the mitigation rule; the violation must become impossible
    mit = compiler.compile_rule(mitigation_rule)
    reverify = Solver()
    reverify.add(*base_compiled, mit, *scenario, adversary, Not(goal))
    fixed = reverify.check() == unsat

    # NON-VACUITY — the secured system (base + mitigation + scenario, no adversary) must
    # itself be satisfiable, so "fixed" means "violation impossible", not "contradiction".
    secure = Solver()
    secure.add(*base_compiled, mit, *scenario)
    secure_world_exists = secure.check() == sat

    return LoopResult(
        property_name=prop.name,
        discovered_vulnerability=discovered,
        fixed=fixed,
        secure_world_exists=secure_world_exists,
        mitigation_rule_id=mitigation_rule.rule_id,
        mitigation_source=mitigation_source,
        n_base_rules_compiled=len(kept),
        n_base_rules_rejected=len(rejected),
        base_consistent=not conflicting,
        conflicting_rules=conflicting,
        rejected=rejected,
    )


# --- the serving-network-binding property instance (used by the end-to-end demo) ----
def binding_property() -> RelationalProperty:
    def goal(u: RelationalUniverse) -> ExprRef:
        s = Const("s", u.sort("Session"))
        ue = Const("ue", u.sort("UE"))
        sn = Const("sn", u.sort("SN"))
        k = Const("k", u.sort("Key"))
        aw, ac, kb = u.func("authenticated_with"), u.func("accepts"), u.func("key_bound_to")
        # P: a UE that authenticated with sn and accepted has its key bound to that sn.
        return ForAll([s, ue, sn, k], Implies(And(aw(s, ue, sn), ac(s, ue)), kb(s, k, sn)))

    def scenario(u: RelationalUniverse) -> list:
        s0, ue0, sn0 = u.const("Session"), u.const("UE", 0), u.const("SN", 0)
        return [u.func("authenticated_with")(s0, ue0, sn0), u.func("accepts")(s0, ue0)]

    def adversary(u: RelationalUniverse) -> ExprRef:
        # confusion: the key is bound to a DIFFERENT serving network (sn1), not sn0.
        s0, sn0, sn1, k0 = u.const("Session"), u.const("SN", 0), u.const("SN", 1), u.const("Key")
        kb = u.func("key_bound_to")
        return And(kb(s0, k0, sn1), Not(kb(s0, k0, sn0)))

    return RelationalProperty("serving_network_binding", goal, scenario, adversary)


# --- SUPI concealment / IMSI-catcher property (a second, privacy-flavoured anchor) --
def supi_concealment_property() -> RelationalProperty:
    """Subscriber-identity privacy: any message that identifies the subscriber must be
    encrypted with the home-network key (SUCI concealment). The witness is the classic
    IMSI-catcher — an identifying message sent in the clear — which the SUCI-encryption
    rule (extracted REL02) closes. A second anchor distinct from serving-network
    binding: it exercises the *privacy* predicates (`identifies` / `encrypted_with_hn_key`)
    rather than the key-binding relation, so it shows the discover->fix->re-verify loop
    generalises beyond one property.
    """
    def goal(u: RelationalUniverse) -> ExprRef:
        m = Const("m", u.sort("Message"))
        ue = Const("ue", u.sort("UE"))
        hn = Const("hn", u.sort("HN"))
        idf, enc = u.func("identifies"), u.func("encrypted_with_hn_key")
        # P: an identifying message is encrypted with the home-network key (concealed).
        return ForAll([m, ue, hn], Implies(idf(m, ue), enc(m, hn)))

    def scenario(u: RelationalUniverse) -> list:
        m0, ue0 = u.const("Message", 0), u.const("UE", 0)
        return [u.func("identifies")(m0, ue0)]

    def adversary(u: RelationalUniverse) -> ExprRef:
        # IMSI-catcher: an identifying message is NOT encrypted with the HN key (cleartext).
        m0, ue0, hn0 = u.const("Message", 0), u.const("UE", 0), u.const("HN", 0)
        idf, enc = u.func("identifies"), u.func("encrypted_with_hn_key")
        return And(idf(m0, ue0), Not(enc(m0, hn0)))

    return RelationalProperty("supi_concealment", goal, scenario, adversary)


# --- key-confirmation property (a third anchor: entity authentication) --------------
def key_confirmation_property() -> RelationalProperty:
    """Key confirmation / entity authentication: a UE must not accept an authentication
    unless its response RES* matches the expected XRES* held by the network. The witness
    is a UE that accepts without a confirmed response (a forged/replayed accept with no
    challenge-response binding), which the extracted key-confirmation rule (REL11,
    ``accepts -> res_matches_xres``) closes. A third anchor distinct from serving-network
    binding and SUPI concealment: it exercises the *challenge-response* predicates
    (`accepts` / `res_matches_xres`), showing the discover->fix->re-verify loop generalises
    to a third, orthogonal property class.
    """
    def goal(u: RelationalUniverse) -> ExprRef:
        s = Const("s", u.sort("Session"))
        ue = Const("ue", u.sort("UE"))
        ac, rx = u.func("accepts"), u.func("res_matches_xres")
        # P: a UE that accepts has a confirmed response (RES* matched XRES*).
        return ForAll([s, ue], Implies(ac(s, ue), rx(s, ue)))

    def scenario(u: RelationalUniverse) -> list:
        s0, ue0 = u.const("Session"), u.const("UE", 0)
        return [u.func("accepts")(s0, ue0)]

    def adversary(u: RelationalUniverse) -> ExprRef:
        # forged accept: the UE accepts although RES* did not match XRES* (no key confirmation).
        s0, ue0 = u.const("Session"), u.const("UE", 0)
        ac, rx = u.func("accepts"), u.func("res_matches_xres")
        return And(ac(s0, ue0), Not(rx(s0, ue0)))

    return RelationalProperty("key_confirmation", goal, scenario, adversary)
