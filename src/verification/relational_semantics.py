"""Cross-backend semantics for the relational loop: one AST, two solvers, measured.

The loop has two independent backends over the same controlled vocabulary: a bounded
finite-domain Z3 theory (`relational_compiler`) and an unbounded Tamarin theory
(`ast_tamarin_compiler`). Until now the *property* each backend checked was wired up
twice — the Tamarin lemma was compiled from a `RelationalRule`, while the Z3 goal was a
hand-written Python lambda (`relational_closed_loop.binding_property`). Nothing checked
that the two encodings said the same thing; they agreed because the same person wrote
both. That is exactly the semantic gap a reviewer asks about when they observe that
passing a type and arity gate does not make a requirement *mean* the right thing.

This module closes it and then measures what survives the crossing:

  * :func:`goal_from_rule` derives the bounded Z3 goal from the same `RelationalRule`
    the Tamarin lemma is compiled from, so one AST drives both backends.
  * :func:`goals_equivalent` machine-checks a derived goal against a legacy hand-written
    goal over the bounded universe (Z3 decides the biconditional), so the migration is
    verified rather than asserted.
  * :func:`extract_witness` reads the actual Z3 model and returns the ground atoms that
    make the property fail, replacing a hardcoded prose string with a witness a reader
    can check against the published attack.
  * :func:`bounded_verdict` / :func:`unbounded_verdict` return the two backends'
    verdicts under one vocabulary -- ``falsified`` / ``verified`` -- so they are directly
    comparable, with wall-clock attached.
  * the ``mutate_*`` operators perturb a rule's meaning in typed, semantically named
    ways, which is what turns "do the backends agree?" into a measurement.

What agreement does and does not mean. The bounded theory quantifies over a small finite
universe; the Tamarin theory quantifies over unboundedly many sessions. Bounded
``verified`` therefore does **not** entail unbounded ``verified``, and the disagreement
class is the interesting one: a property that closes the bounded witness but is falsified
unbounded is precisely a rule whose consequent lacks an operational guard. Counting that
class is how the expert-intervention boundary stops being an anecdote.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING

from z3 import Not, Solver, sat, unsat

from pipeline.schemas_relational import GLOSSARY, Predicate, RelationalRule, normalize_name
from verification.ast_tamarin_compiler import (
    CompileError,
    ProtocolSchema,
    compile_to_tamarin,
    tamarin_fact_name,
)
from verification.relational_compiler import (
    RelationalCompileError,
    RelationalRuleCompiler,
    RelationalUniverse,
)
from verification.tamarin_relational_compiler import run_tamarin, tamarin_available

if TYPE_CHECKING:
    from pathlib import Path

    from z3 import ExprRef

__all__ = [
    "FALSIFIED",
    "VERIFIED",
    "GATE_REJECTED",
    "OUT_OF_SCHEMA",
    "BackendVerdict",
    "goal_from_rule",
    "goals_equivalent",
    "extract_witness",
    "bounded_verdict",
    "unbounded_verdict",
    "schema_action_facts",
    "rule_facts",
    "expressible_in_schema",
    "MUTATIONS",
    "mutate_converse",
    "mutate_drop_premise",
    "mutate_add_premise",
    "mutate_substitute_conclusion",
    "mutate_permute_args",
    "mutate_negate_conclusion",
    "mutants_of",
]

# One verdict vocabulary for both backends, so the two are comparable at all.
VERIFIED = "verified"        # the property holds in the theory
FALSIFIED = "falsified"      # a counterexample to the property exists
GATE_REJECTED = "gate_rejected"    # the rule never reached a solver
OUT_OF_SCHEMA = "out_of_schema"    # well-typed, but the operational model cannot witness it
INCOMPLETE = "incomplete"          # the backend gave up (Tamarin non-termination / timeout)


@dataclass
class BackendVerdict:
    """One backend's answer about one property, with the cost of getting it."""

    verdict: str
    elapsed_s: float
    detail: str = ""
    steps: int | None = None
    witness: dict[str, dict[str, list[list[str]]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"verdict": self.verdict, "elapsed_s": round(self.elapsed_s, 4)}
        if self.detail:
            d["detail"] = self.detail
        if self.steps is not None:
            d["steps"] = self.steps
        if self.witness:
            d["witness"] = self.witness
        return d


# ---------------------------------------------------------------------------
# One AST, one goal: derive the bounded Z3 property from the rule
# ---------------------------------------------------------------------------

def goal_from_rule(rule: RelationalRule, universe: RelationalUniverse) -> ExprRef:
    """The bounded Z3 reading of ``rule`` as a *property* (not as an axiom).

    This is the same compilation the rule would get as a theory axiom --
    ``ForAll(vars, Implies(premises, conclusions))`` -- which is what makes the
    identification sound: a relational requirement and the property that requirement
    asserts are the same closed sentence. Using `compile_rule` here (rather than a
    second, parallel encoder) is the point: there is one place where a rule acquires
    meaning in the bounded backend.
    """
    return RelationalRuleCompiler(universe).compile_rule(rule)


def goals_equivalent(goal_a: ExprRef, goal_b: ExprRef) -> tuple[bool, str]:
    """Decide whether two bounded goals are the same sentence over the finite universe.

    Returns ``(equivalent, detail)``. Both goals are closed sentences over enumerated
    sorts and uninterpreted predicate symbols, so Z3 decides ``a <=> b`` outright; a
    countermodel means the two encodings disagree somewhere in the bound and the
    migration would have silently changed the claim.
    """
    solver = Solver()
    solver.add(Not(goal_a == goal_b))
    status = solver.check()
    if status == unsat:
        return True, "no countermodel: the two encodings are the same bounded sentence"
    if status == sat:
        return False, "countermodel exists: the encodings disagree within the bound"
    return False, f"z3 returned {status}"


# ---------------------------------------------------------------------------
# Witnesses: read the model instead of describing it
# ---------------------------------------------------------------------------

def extract_witness(
    model,
    universe: RelationalUniverse,
    predicates: tuple[str, ...] | None = None,
) -> dict[str, dict[str, list[list[str]]]]:
    """The ground atoms a Z3 model decides, as ``{predicate: {"holds": [...], "fails": [...]}}``.

    Both polarities are reported because for most security properties the counterexample
    *is* a negative fact. The SUPI-concealment witness, for instance, is an identifying
    message for which ``encrypted_with_hn_key`` does **not** hold; listing only the true
    atoms would render that witness as an empty set and hide the very thing it shows.

    Enumerates each predicate over the bounded sorts and evaluates it under
    ``model_completion`` so unconstrained atoms get a definite value instead of being
    omitted. Sort-constant names are stripped of their per-universe prefix, so witnesses
    from two different runs of the same property are directly comparable.
    """
    names = predicates or tuple(GLOSSARY)
    out: dict[str, dict[str, list[list[str]]]] = {}
    for pname in names:
        canonical = normalize_name(pname)
        if canonical not in GLOSSARY:
            continue
        arg_sorts, _gloss = GLOSSARY[canonical]
        fn = universe.func(canonical)
        holds: list[list[str]] = []
        fails: list[list[str]] = []
        for combo in product(*[universe.consts(s) for s in arg_sorts]):
            args = [_strip_prefix(str(c)) for c in combo]
            value = str(model.eval(fn(*combo), model_completion=True))
            if value == "True":
                holds.append(args)
            elif value == "False":
                fails.append(args)
        if holds or fails:
            out[canonical] = {"holds": holds, "fails": fails}
    return out


def _strip_prefix(const_name: str) -> str:
    """``ru7_sn1`` -> ``sn1``: drop the per-universe uniquifying prefix."""
    return re.sub(r"^ru\d+_", "", const_name)


# ---------------------------------------------------------------------------
# The two backends, under one verdict vocabulary
# ---------------------------------------------------------------------------

def bounded_verdict(
    universe: RelationalUniverse,
    theory: list[ExprRef],
    goal: ExprRef,
    *,
    witness_predicates: tuple[str, ...] | None = None,
) -> BackendVerdict:
    """Is ``goal`` violable in the bounded ``theory``? SAT(theory & !goal) = falsified.

    This is the same SAT/UNSAT reading the closed loop uses for DISCOVER and RE-VERIFY,
    renamed into the shared verdict vocabulary so it can be compared against Tamarin.
    """
    solver = Solver()
    solver.add(*theory, Not(goal))
    start = time.perf_counter()
    status = solver.check()
    elapsed = time.perf_counter() - start
    if status == sat:
        witness = extract_witness(solver.model(), universe, witness_predicates)
        return BackendVerdict(FALSIFIED, elapsed, "counterexample within the bound", witness=witness)
    if status == unsat:
        return BackendVerdict(VERIFIED, elapsed, "no counterexample within the bound")
    return BackendVerdict(INCOMPLETE, elapsed, f"z3 returned {status}")


def unbounded_verdict(
    rule: RelationalRule,
    schema_preamble: str,
    rules_block: str,
    workdir: Path,
    name: str,
    *,
    executable_from: RelationalRule | None = None,
    timeout: int = 300,
) -> BackendVerdict:
    """Compile ``rule`` as the all-traces lemma of a Tamarin theory and prove it.

    ``executable_from`` supplies the non-vacuity (exists-trace) lemma. It defaults to
    the rule itself, but a caller comparing many candidate properties against one
    operational model should pass the model's own honest-run rule: otherwise a mutant
    that mentions facts the skeleton never emits would report its own unreachability as
    a vacuity failure, conflating "this property is inexpressible here" with "this
    protocol cannot run".
    """
    if not tamarin_available():
        return BackendVerdict(INCOMPLETE, 0.0, "tamarin-prover not on PATH")
    schema = ProtocolSchema(
        name=name,
        preamble=schema_preamble,
        rules_block=rules_block,
        property_rule=rule,
        property_name="candidate_property",
        executable_from=executable_from,
    )
    try:
        theory = compile_to_tamarin(schema)
    except CompileError as exc:
        return BackendVerdict(GATE_REJECTED, 0.0, f"tamarin gate: {exc}")
    start = time.perf_counter()
    try:
        results = {r.lemma: r for r in run_tamarin(theory, workdir, name, timeout)}
    except RuntimeError as exc:
        elapsed = time.perf_counter() - start
        return BackendVerdict(INCOMPLETE, elapsed, f"tamarin produced no summary: {exc}"[:300])
    except Exception as exc:  # noqa: BLE001 - a prover timeout must not abort a sweep
        elapsed = time.perf_counter() - start
        return BackendVerdict(INCOMPLETE, elapsed, f"{type(exc).__name__}: {exc}"[:300])
    elapsed = time.perf_counter() - start
    prop = results.get("candidate_property")
    if prop is None:
        return BackendVerdict(INCOMPLETE, elapsed, "candidate_property lemma missing from summary")
    executable = results.get("executable")
    detail = ""
    if executable is not None and executable.verdict != VERIFIED:
        detail = "honest run unreachable in this theory (vacuity risk)"
    return BackendVerdict(prop.verdict, elapsed, detail, steps=prop.steps)


# ---------------------------------------------------------------------------
# What the declared operational model can actually witness
# ---------------------------------------------------------------------------

_ACTION_BLOCK_RE = re.compile(r"--\[(?P<body>.*?)\]->", re.DOTALL)
_FACT_RE = re.compile(r"\b(?P<fact>[A-Z]\w*)\s*\(")


def schema_action_facts(rules_block: str) -> set[str]:
    """The action-fact symbols a multiset-rewrite skeleton actually emits.

    Parsed from the rewrite rules rather than hardcoded, so the coverage number below
    stays honest if the skeleton grows.
    """
    facts: set[str] = set()
    for block in _ACTION_BLOCK_RE.finditer(rules_block):
        facts.update(m.group("fact") for m in _FACT_RE.finditer(block.group("body")))
    return facts


def rule_facts(rule: RelationalRule) -> set[str]:
    """The Tamarin fact symbols a rule's predicates would refer to."""
    out: set[str] = set()
    for pred in (*rule.premises, *rule.conclusions):
        try:
            out.add(tamarin_fact_name(pred.name))
        except CompileError:
            continue
    return out


def expressible_in_schema(rule: RelationalRule, action_facts: set[str]) -> bool:
    """Can this rule's property even be witnessed by the declared operational model?

    A lemma over facts the skeleton never emits is not *false* -- it is vacuously true,
    because its antecedent can never hold. Reporting that as a proof would be the exact
    failure mode the loop exists to prevent, so such rules are separated out and counted
    as a limit of the declared schema instead.
    """
    return rule_facts(rule).issubset(action_facts)


# ---------------------------------------------------------------------------
# Mutation operators: perturb meaning in typed, named ways
# ---------------------------------------------------------------------------

def _renamed(rule: RelationalRule, suffix: str, note: str) -> RelationalRule:
    return RelationalRule(
        rule_id=f"{rule.rule_id}~{suffix}",
        source_text=rule.source_text,
        premises=rule.premises,
        conclusions=rule.conclusions,
        notes=note,
        meta={**rule.meta, "mutation": suffix, "mutant_of": rule.rule_id},
    )


def mutate_converse(rule: RelationalRule) -> RelationalRule:
    """Swap premises and conclusions: the converse implication.

    The canonical directionality error, and the one a type gate cannot see: the converse
    of a well-typed rule is well-typed.
    """
    out = _renamed(rule, "converse", "premises and conclusions exchanged")
    out.premises, out.conclusions = rule.conclusions, rule.premises
    return out


def mutate_drop_premise(rule: RelationalRule, index: int = 0) -> RelationalRule:
    """Drop a premise: a strictly stronger claim (the guard-removal mutation).

    This is the REL13 shape -- a requirement whose consequent is asserted without the
    acceptance condition that makes it operational.
    """
    if len(rule.premises) <= 1:
        msg = "cannot drop the only premise: the result is not an implication"
        raise ValueError(msg)
    out = _renamed(rule, f"drop_premise{index}", f"premise {index} removed (claim strengthened)")
    out.premises = tuple(p for i, p in enumerate(rule.premises) if i != index)
    return out


def mutate_add_premise(rule: RelationalRule, spurious: Predicate) -> RelationalRule:
    """Add a premise the source text does not license: a strictly weaker claim.

    The measured failure mode of a second extractor in the paper's RQ4 -- well-typed,
    still plausible, but it no longer closes the witness.
    """
    out = _renamed(rule, f"add_premise_{normalize_name(spurious.name)}",
                   "spurious premise added (claim weakened)")
    out.premises = (*rule.premises, spurious)
    return out


def mutate_substitute_conclusion(rule: RelationalRule, replacement: Predicate) -> RelationalRule:
    """Replace the consequent relation with a different, same-sorted one."""
    if not rule.conclusions:
        msg = "no conclusion to substitute"
        raise ValueError(msg)
    out = _renamed(rule, f"subst_{normalize_name(replacement.name)}",
                   "consequent relation replaced by a different relation")
    out.conclusions = (replacement, *rule.conclusions[1:])
    return out


def mutate_permute_args(rule: RelationalRule) -> RelationalRule:
    """Rotate the arguments of the first multi-argument premise (a role confusion).

    Usually ill-sorted, and therefore the mutation the gate is *supposed* to catch --
    included so the sweep measures the gate rather than assuming it.
    """
    for i, pred in enumerate(rule.premises):
        if len(pred.args) >= 2:
            rotated = (*pred.args[1:], pred.args[0])
            out = _renamed(rule, "permute_args", f"arguments of '{pred.name}' rotated (role confusion)")
            out.premises = (
                *rule.premises[:i],
                Predicate(name=pred.name, args=rotated, sorts=pred.sorts, negated=pred.negated),
                *rule.premises[i + 1:],
            )
            return out
    msg = "no multi-argument premise to permute"
    raise ValueError(msg)


def mutate_negate_conclusion(rule: RelationalRule) -> RelationalRule:
    """Negate the consequent: the polarity inversion."""
    if not rule.conclusions:
        msg = "no conclusion to negate"
        raise ValueError(msg)
    head = rule.conclusions[0]
    out = _renamed(rule, "negate_conclusion", "consequent polarity inverted")
    out.conclusions = (
        Predicate(name=head.name, args=head.args, sorts=head.sorts, negated=not head.negated),
        *rule.conclusions[1:],
    )
    return out


#: The mutation suite, as ``(name, builder)``. Builders that do not apply to a given
#: rule raise ``ValueError`` and are skipped with a recorded reason.
MUTATIONS: tuple[tuple[str, object], ...] = (
    ("converse", mutate_converse),
    ("drop_premise0", lambda r: mutate_drop_premise(r, 0)),
    ("drop_premise1", lambda r: mutate_drop_premise(r, 1)),
    ("permute_args", mutate_permute_args),
    ("negate_conclusion", mutate_negate_conclusion),
)


def mutants_of(rule: RelationalRule) -> list[tuple[str, RelationalRule | None, str]]:
    """Apply every operator to ``rule``; returns ``(name, mutant_or_None, reason)``."""
    out: list[tuple[str, RelationalRule | None, str]] = []
    for name, build in MUTATIONS:
        try:
            out.append((name, build(rule), ""))  # type: ignore[operator]
        except (ValueError, KeyError) as exc:
            out.append((name, None, str(exc)))
    return out


def well_typed(rule: RelationalRule, universe: RelationalUniverse) -> tuple[bool, str]:
    """Run the bounded backend's well-typedness gate without solving anything."""
    try:
        RelationalRuleCompiler(universe).compile_rule(rule)
    except RelationalCompileError as exc:
        return False, str(exc)
    return True, ""
