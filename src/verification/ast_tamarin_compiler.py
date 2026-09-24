"""General relational-AST -> Tamarin compiler.

Earlier revisions compiled the serving-network-binding property from
**hand-written string templates** (`tamarin_relational_compiler._THEORY_TEMPLATE`
and friends): the rule names R2/R3/R6/R9 were documented in a prose `RULE_MAPPING`
dict but never *parsed* from a structured representation. This module closes that
gap — it compiles the project's canonical typed relational AST into Tamarin
multiset-rewriting theories, so the proven property lemma is *derived from the rule
AST*, not from a bespoke string.

The AST is the one the extractor and the relational gold already use:
:class:`pipeline.schemas_relational.RelationalRule` — a conjunctive implication
``IF premise_1 AND ... THEN conclusion_1 AND ...`` over typed
:class:`~pipeline.schemas_relational.Predicate` atoms drawn from the controlled
``GLOSSARY`` vocabulary (each predicate carries its argument sorts). Per-atom
negation is supported (``Predicate.negated``); disjunction is not part of the
relational vocabulary and is rejected rather than silently mis-encoded.

What is GENERAL here (validated against the hand-written oracle, see
:func:`run_snbinding_from_ast`):
the rule -> Tamarin *lemma* / *restriction* compilation. Any quantified conjunctive
implication over the declared predicate vocabulary compiles to a well-formed
trace property with correct sort-typed quantification (premise-only variables
universal, conclusion-only variables existential) and per-predicate action facts.

What is a DECLARED INPUT (not synthesized from prose): the :class:`ProtocolSchema`
— the multiset-rewrite skeleton (which rules emit which action facts, and the term
structure the MAC / KDF cover). This is the operational model; a *new property*
needs a new schema. This module is therefore a faithful AST->Tamarin *rule*
compiler wired to a per-property protocol schema, not a universal protocol
synthesizer. The soundness check is that the AST-compiled serving-network-binding
theory reproduces the hand-written oracle exactly: base FALSIFIED (Tamarin rediscovers the
NDSS'19 SN-confusion attack, unbounded), R9-fixed VERIFIED, both non-vacuous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pipeline.schemas_relational import GLOSSARY, Predicate, RelationalRule, normalize_name

# Reuse the prover harness from the hand-template module rather than duplicating it.
from verification.tamarin_relational_compiler import (
    LemmaResult,
    run_tamarin,
    tamarin_available,
)

__all__ = [
    "CompileError",
    "ProtocolSchema",
    "tamarin_fact_name",
    "check_predicate",
    "compile_property_lemma",
    "compile_executable_lemma",
    "compile_restriction",
    "compile_to_tamarin",
    "snbinding_property_rule",
    "compile_snbinding_from_ast",
    "run_snbinding_from_ast",
    "tamarin_available",
]


class CompileError(ValueError):
    """Raised when a rule cannot be soundly compiled (unknown predicate, wrong arity,
    unsupported connective). We reject rather than emit a theory that quietly does not
    mean what the rule says."""


# ---------------------------------------------------------------------------
# Predicate -> Tamarin fact
# ---------------------------------------------------------------------------

def tamarin_fact_name(pred_name: str) -> str:
    """CamelCase a (normalized) predicate name into a valid Tamarin fact symbol.

    ``authenticated_with`` -> ``AuthenticatedWith``; ``accepts`` -> ``Accepts``.
    Tamarin facts must start with an uppercase letter; the mapping is total and
    injective over the controlled vocabulary.
    """
    parts = [p for p in re.split(r"[_\s]+", normalize_name(pred_name)) if p]
    if not parts:
        raise CompileError(f"empty predicate name: {pred_name!r}")
    return "".join(p[:1].upper() + p[1:] for p in parts)


def check_predicate(pred: Predicate) -> tuple[str, tuple[str, ...]]:
    """Type-check a predicate against the controlled vocabulary.

    Returns (canonical_name, expected_sorts). Raises CompileError if the predicate
    is not in GLOSSARY or its arity disagrees with the declared signature — no
    silent fabrication of unknown predicates.
    """
    name = normalize_name(pred.name)
    if name not in GLOSSARY:
        raise CompileError(
            f"predicate '{pred.name}' (normalized '{name}') is not in the controlled "
            f"vocabulary; cannot compile it to Tamarin faithfully"
        )
    expected_sorts, _gloss = GLOSSARY[name]
    if len(pred.args) != len(expected_sorts):
        raise CompileError(
            f"predicate '{name}' expects {len(expected_sorts)} args {expected_sorts}, "
            f"got {len(pred.args)}: {pred.args}"
        )
    return name, expected_sorts


def _sanitize_var(v: str) -> str:
    """A Tamarin message variable: lowercase alnum/underscore, non-empty."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(v)).lstrip("_") or "x"
    if s[0].isdigit():
        s = "v_" + s
    return s[0].lower() + s[1:]


def _collect_vars(preds: tuple[Predicate, ...]) -> dict[str, str]:
    """Map each predicate argument variable -> its sort, checking consistency.

    A variable used in two positions must carry the same sort (else the rule is
    ill-typed and we refuse it).
    """
    var_sort: dict[str, str] = {}
    for p in preds:
        _name, sorts = check_predicate(p)
        for arg, sort in zip(p.args, sorts, strict=True):
            v = _sanitize_var(arg)
            if v in var_sort and var_sort[v] != sort:
                raise CompileError(
                    f"variable '{v}' used at sorts {var_sort[v]} and {sort}; ill-typed rule"
                )
            var_sort[v] = sort
    return var_sort


def _atom(pred: Predicate, timepoint: str) -> str:
    """Render one predicate as a timed Tamarin action-fact atom ``Fact(a,b) @ #t``."""
    if pred.negated:
        # negation is handled by the caller (context decides sound placement)
        raise CompileError("negated atom rendered without context")
    fact = tamarin_fact_name(pred.name)
    args = ", ".join(_sanitize_var(a) for a in pred.args)
    return f"{fact}({args}) @ {timepoint}"


# ---------------------------------------------------------------------------
# Rule (RelationalRule AST) -> Tamarin trace formula
# ---------------------------------------------------------------------------

def _conj_atoms(preds: tuple[Predicate, ...], tp_prefix: str) -> tuple[list[str], list[str]]:
    """Render a conjunction of (positive) predicates to atoms + their timepoints.

    Negated premises/conclusions are rejected here; the two callers that can encode
    negation soundly (none in the current vocabulary's compiled properties) would
    handle it explicitly. Keeping this strict avoids emitting a lemma that silently
    drops a NOT.
    """
    atoms: list[str] = []
    tps: list[str] = []
    for i, p in enumerate(preds):
        if p.negated:
            raise CompileError(
                f"negated predicate '{p.name}' in a compiled trace property is not "
                f"supported (would change the property's meaning); refusing"
            )
        tp = f"#{tp_prefix}{i}"
        atoms.append(_atom(p, tp))
        tps.append(tp)
    return atoms, tps


def compile_property_lemma(rule: RelationalRule, name: str) -> str:
    """Compile ``IF premises THEN conclusions`` to an all-traces Tamarin lemma.

    Premise variables are universally quantified; variables that appear only in the
    conclusion are existentially quantified inside the consequent. Sorts come from
    GLOSSARY (single source of truth), so the quantification is correctly typed.
    """
    if not rule.premises or not rule.conclusions:
        raise CompileError("property rule needs at least one premise and one conclusion")
    prem_vars = _collect_vars(rule.premises)
    all_vars = {**prem_vars, **_collect_vars(rule.conclusions)}
    concl_only = [v for v in all_vars if v not in prem_vars]

    prem_atoms, prem_tps = _conj_atoms(rule.premises, "i")
    concl_atoms, concl_tps = _conj_atoms(rule.conclusions, "j")

    universals = " ".join(sorted(prem_vars)) + " " + " ".join(prem_tps)
    antecedent = " & ".join(prem_atoms)
    existentials = (" ".join(sorted(concl_only)) + " " if concl_only else "") + " ".join(concl_tps)
    consequent = " & ".join(concl_atoms)

    return (
        f"lemma {name}:\n"
        f'  "All {universals.strip()}.\n'
        f"     ( {antecedent} )\n"
        f"    ==>\n"
        f"     ( Ex {existentials.strip()}. {consequent} )\"\n"
    )


def compile_executable_lemma(rule: RelationalRule, name: str = "executable") -> str:
    """An exists-trace non-vacuity lemma: all of the rule's facts can co-occur.

    Guards against a vacuously-verified property (the honest protocol run must exist).
    """
    preds = rule.premises + rule.conclusions
    var_sort = _collect_vars(preds)
    atoms, tps = _conj_atoms(preds, "e")
    quant = " ".join(sorted(var_sort)) + " " + " ".join(tps)
    return (
        f"lemma {name}: exists-trace\n"
        f'  "Ex {quant.strip()}. {" & ".join(atoms)}"\n'
    )


def compile_restriction(rule: RelationalRule, name: str) -> str:
    """Compile an all-traces mitigation rule to a Tamarin ``restriction``.

    Same shape as a property lemma but emitted as a restriction so it constrains the
    trace set (used when a mitigation is enforced globally rather than via the
    protocol term structure). General over the vocabulary; provided for completeness.
    """
    lemma = compile_property_lemma(rule, name)
    return "restriction" + lemma[len("lemma"):]


# ---------------------------------------------------------------------------
# Protocol schema (declared operational model) + whole-theory assembly
# ---------------------------------------------------------------------------

@dataclass
class ProtocolSchema:
    """The declared operational model a property is proved against.

    ``rules_block`` is the multiset-rewrite skeleton whose action facts are exactly
    the CamelCased predicate facts the compiled lemmas refer to. ``preamble`` holds
    the function signature / builtins. ``property_rule`` is the RelationalRule proved;
    ``restrictions`` are all-traces mitigation rules; ``extra_lemmas`` is raw Tamarin
    appended verbatim (rarely needed).
    """

    name: str
    preamble: str
    rules_block: str
    property_rule: RelationalRule
    property_name: str = "property"
    restrictions: tuple[RelationalRule, ...] = ()
    extra_lemmas: tuple[str, ...] = ()
    executable_from: RelationalRule | None = None  # defaults to property_rule


def compile_to_tamarin(schema: ProtocolSchema) -> str:
    """Assemble a full .spthy theory from a protocol schema + its rule AST."""
    parts: list[str] = [f"theory {schema.name}", "begin", "", schema.preamble.strip(), ""]
    for i, r in enumerate(schema.restrictions):
        parts.append(compile_restriction(r, f"mitigation_{i}"))
        parts.append("")
    parts.append(schema.rules_block.strip())
    parts.append("")
    exec_rule = schema.executable_from or schema.property_rule
    parts.append(compile_executable_lemma(exec_rule))
    parts.append(compile_property_lemma(schema.property_rule, schema.property_name))
    for extra in schema.extra_lemmas:
        parts.append(extra.strip())
        parts.append("")
    parts.append("end")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# The SN-binding oracle, compiled FROM THE AST
# ---------------------------------------------------------------------------

def _P(name: str, *args: str, negated: bool = False) -> Predicate:
    """Build a glossary predicate, filling sorts from the controlled vocabulary."""
    sorts, _ = GLOSSARY[normalize_name(name)]
    return Predicate(name=name, args=tuple(args), sorts=sorts, negated=negated)


def snbinding_property_rule() -> RelationalRule:
    """The NDSS'19 serving-network-binding property, as the canonical relational AST.

    IF a UE accepts a session it authenticated with serving network ``n``
    THEN the session's key is bound to ``n``.

    This is the same property `relational_aka.RelationalAKAModel._binding_property`
    encodes in Z3 (accepts ∧ authenticated_with ⇒ key_bound_to), now expressed as a
    RelationalRule so the Tamarin lemma is *derived from the AST*.
    """
    return RelationalRule(
        rule_id="AKA_P04_binding",
        source_text="If a UE accepts a session it authenticated with a serving network, "
                    "the derived key is bound to that serving network (TS 33.501 6.1.1.3).",
        premises=(_P("accepts", "s", "u"), _P("authenticated_with", "s", "u", "n")),
        conclusions=(_P("key_bound_to", "s", "key", "n"),),
        notes="NDSS'19 (Cremers & Dehnel-Wild) serving-network confusion class.",
    )


# The protocol skeleton (declared operational model). Term structure controlled by the
# R9 mitigation flag: base leaves the SN name out of the MAC / KDF (the Dolev-Yao
# attacker rewrites it in transit -> confusion); fixed binds the SN name into both
# (TS 33.501 Annex A) -> the attacker can no longer forge an accepting message for a
# different SN. Action facts are exactly AuthenticatedWith / Accepts / KeyBoundTo, i.e.
# the CamelCased glossary predicates the compiled lemma quantifies over.
_PREAMBLE = "functions: mac/2, kdf/2"

_RULES_BASE = """// UE and home network share a long-term key (unbounded UEs).
rule Register_LTK:
  [ Fr(~k) ] --> [ !LTK($UE, ~k) ]

// Network issues a challenge for $UE via serving network $SN. Session identity is the
// derived key term. BASE: neither the MAC nor the KDF covers $SN.
rule SN_Challenge:
  [ !LTK($UE, ~k), Fr(~r) ]
  --[ AuthenticatedWith(kdf(~k, ~r), $UE, $SN) ]->
  [ Out(<$SN, ~r, mac(~r, ~k)>) ]

// UE verifies the MAC and accepts, binding the key to the (unauthenticated) SN name it
// received -- which a Dolev-Yao attacker may have rewritten.
rule UE_Accept:
  [ !LTK($UE, ~k), In(<sn, r, mac(r, ~k)>) ]
  --[ Accepts(kdf(~k, r), $UE), KeyBoundTo(kdf(~k, r), kdf(~k, r), sn) ]->
  [ ]"""

_RULES_FIXED = """// UE and home network share a long-term key (unbounded UEs).
rule Register_LTK:
  [ Fr(~k) ] --> [ !LTK($UE, ~k) ]

// FIXED (R9): the SN name is covered by the MAC and enters the key derivation
// kdf(k, <r, SN>) -- TS 33.501 Annex A.
rule SN_Challenge:
  [ !LTK($UE, ~k), Fr(~r) ]
  --[ AuthenticatedWith(kdf(~k, <~r, $SN>), $UE, $SN) ]->
  [ Out(<$SN, ~r, mac(<~r, $SN>, ~k)>) ]

// UE accepts only on a MAC that covers the SN name, so it cannot be rewritten.
rule UE_Accept:
  [ !LTK($UE, ~k), In(<sn, r, mac(<r, sn>, ~k)>) ]
  --[ Accepts(kdf(~k, <r, sn>), $UE), KeyBoundTo(kdf(~k, <r, sn>), kdf(~k, <r, sn>), sn) ]->
  [ ]"""


def compile_snbinding_from_ast(*, with_mitigation: bool) -> str:
    """Compile the SN-binding theory from the relational AST + protocol schema."""
    schema = ProtocolSchema(
        name="FiveG_AKA_SNBinding_AST_" + ("Fixed" if with_mitigation else "Base"),
        preamble=_PREAMBLE,
        rules_block=_RULES_FIXED if with_mitigation else _RULES_BASE,
        property_rule=snbinding_property_rule(),
        property_name="sn_binding",
    )
    return compile_to_tamarin(schema)


def run_snbinding_from_ast(workdir: Path, timeout: int = 300) -> dict:
    """Discover -> fix -> certify record for SN-binding, compiled from the AST.

    Same record shape as `tamarin_relational_compiler.run_snbinding`, so a reviewer can
    diff the two and confirm the AST-compiled theory reproduces the hand-template
    oracle: base `sn_binding` FALSIFIED (attack rediscovered unbounded) + executable
    verified; fixed `sn_binding` VERIFIED + executable verified.
    """
    base = {r.lemma: r for r in run_tamarin(
        compile_snbinding_from_ast(with_mitigation=False), workdir, "snbinding_generated_base", timeout)}
    fixed = {r.lemma: r for r in run_tamarin(
        compile_snbinding_from_ast(with_mitigation=True), workdir, "snbinding_generated_fixed", timeout)}

    def _v(d: dict, lemma: str) -> str:
        return d.get(lemma, LemmaResult(lemma, "missing", None)).verdict

    attack_found = _v(base, "sn_binding") == "falsified"
    base_executable = _v(base, "executable") == "verified"
    fix_proven = _v(fixed, "sn_binding") == "verified"
    fix_nonvacuous = _v(fixed, "executable") == "verified"

    return {
        "record": "serving-network binding, compiled from the relational AST",
        "property": "serving-network / KSEAF binding (Cremers & Dehnel-Wild NDSS'19 class)",
        "setting": "UNBOUNDED sessions/UEs/SNs, Dolev-Yao attacker (tamarin-prover)",
        "compiled_from": "pipeline.schemas_relational.RelationalRule AST via "
                         "ast_tamarin_compiler.compile_property_lemma (not a hand template)",
        "property_ast": snbinding_property_rule().to_dict(),
        "base": {lm: {"verdict": r.verdict, "steps": r.steps} for lm, r in base.items()},
        "fixed": {lm: {"verdict": r.verdict, "steps": r.steps} for lm, r in fixed.items()},
        "attack_rediscovered_unbounded": attack_found and base_executable,
        "fix_proven_unbounded": fix_proven and fix_nonvacuous,
        "validated": attack_found and base_executable and fix_proven and fix_nonvacuous,
        "scope_note": (
            "General over the rule->lemma compilation (any conjunctive typed implication "
            "over the controlled vocabulary); the protocol schema (rewrite skeleton + MAC/"
            "KDF term structure) is a declared per-property input, not synthesized from prose."
        ),
    }
