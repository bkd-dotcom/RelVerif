# ruff: noqa: INP001
"""Does meaning survive the bounded -> unbounded crossing? Measure it instead of asserting it.

A type-and-arity gate decides whether a requirement is *well-formed*, not whether it
*means* what the specification says. The pipeline has two backends over one controlled
vocabulary -- a bounded finite-domain Z3 theory and an unbounded Tamarin theory -- and
three separate things can go wrong between them, so this script measures all three.

1. ENCODING EQUIVALENCE. Each anchor's bounded goal used to be a hand-written Python
   lambda while its unbounded lemma was compiled from a `RelationalRule`. Nothing checked
   that the two said the same thing. Here the goal is re-derived from the property AST and
   Z3 decides the biconditional against the legacy lambda, so "one AST drives both
   backends" becomes a checked claim rather than a design intention.

2. SCHEMA EXPRESSIVENESS. The unbounded leg is only as wide as the declared multiset-rewrite
   skeleton: a lemma over action facts the skeleton never emits is not false, it is
   *vacuously true*, and reporting it as proved would be the exact failure the loop exists
   to prevent. Counting how many vocabulary predicates and gold rules the skeleton can
   witness puts a number on how much operational specification remains expert work.

3. CROSS-BACKEND AGREEMENT. For every property the skeleton can express, both backends are
   asked whether the protocol model satisfies it, under one verdict vocabulary. The two
   theories are different artifacts by construction -- Z3 reasons over the extracted
   relational rules, Tamarin over the operational skeleton -- so a disagreement is not
   noise but a localisation:

     bounded verified / unbounded falsified -> the rule set entails the property but the
         operational model does not: the consequent lacks an operational guard. This is the
         REL13 class, and its size is the rate at which expert intervention is required.
     bounded falsified / unbounded verified -> the operational model is stronger than the
         extracted rules: an under-extraction, invisible to the bounded leg alone.

The population is each anchor property plus its systematic mutants plus every expressible
gold rule and its mutants. Mutants matter because agreement measured only on rules that are
already correct measures nothing: it is the wrong-but-well-typed ones that reveal whether a
backend crossing preserves meaning.

    PYTHONPATH=src .venv/bin/python scripts/run_semantic_preservation.py

Writes data/gold/semantic_preservation.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.schemas_relational import (  # noqa: E402
    GLOSSARY,
    Predicate,
    RelationalRule,
    load_gold,
)
from verification.tamarin_relational_compiler import require_tamarin  # noqa: E402
from verification.ast_tamarin_compiler import (  # noqa: E402
    _PREAMBLE,  # noqa: PLC2701 - the declared skeleton's own preamble
    _RULES_FIXED,  # noqa: PLC2701 - the declared MSR skeleton under test
    snbinding_property_rule,
)
from verification.relational_closed_loop import (  # noqa: E402
    _consistency_repair,  # noqa: PLC2701 - the loop's own primitive, reused not reimplemented
    binding_property,
    key_confirmation_property,
    supi_concealment_property,
)
from verification.relational_compiler import (  # noqa: E402
    DEFAULT_SIZES,
    RelationalCompileError,
    RelationalRuleCompiler,
    RelationalUniverse,
)
from verification.relational_semantics import (  # noqa: E402
    FALSIFIED,
    GATE_REJECTED,
    VERIFIED,
    bounded_verdict,
    goal_from_rule,
    goals_equivalent,
    mutants_of,
    rule_facts,
    schema_action_facts,
    unbounded_verdict,
)

GOLD = ROOT / "data" / "gold" / "relational_gold.jsonl"
DEST = ROOT / "data" / "gold" / "semantic_preservation.json"
TAMARIN_TIMEOUT = 60


def _p(name: str, args: tuple[str, ...]) -> Predicate:
    """A vocabulary predicate with its sorts taken from the glossary, not retyped by hand."""
    sorts, _gloss = GLOSSARY[name]
    return Predicate(name=name, args=args, sorts=tuple(sorts))


# The three anchor properties as ASTs. These are the same sentences the hand-written
# lambdas in `relational_closed_loop` encode; step 1 proves that rather than trusting it.
ANCHOR_RULES = {
    "serving_network_binding": snbinding_property_rule(),
    "supi_concealment": RelationalRule(
        rule_id="AKA_P_supi",
        source_text=("A message that identifies the subscriber is concealed under the home "
                     "network public key (SUCI, TS 33.501 6.12)."),
        premises=(_p("identifies", ("m", "u")),),
        conclusions=(_p("encrypted_with_hn_key", ("m", "h")),),
    ),
    "key_confirmation": RelationalRule(
        rule_id="AKA_P_keyconf",
        source_text=("A UE accepts an authentication only if its response RES* matches the "
                     "expected XRES* (TS 33.501 6.1.3.2)."),
        premises=(_p("accepts", ("s", "u")),),
        conclusions=(_p("res_matches_xres", ("s", "u")),),
    ),
}

ANCHOR_LAMBDAS = {
    "serving_network_binding": binding_property,
    "supi_concealment": supi_concealment_property,
    "key_confirmation": key_confirmation_property,
}


def _encoding_equivalence() -> list[dict]:
    """Step 1: is each AST-derived bounded goal the same sentence as the legacy lambda?"""
    out: list[dict] = []
    for name, rule in ANCHOR_RULES.items():
        universe = RelationalUniverse()
        legacy = ANCHOR_LAMBDAS[name]().goal(universe)
        derived = goal_from_rule(rule, universe)
        equivalent, detail = goals_equivalent(legacy, derived)
        out.append({
            "property": name,
            "property_ast": rule.rule_id,
            "equivalent": equivalent,
            "detail": detail,
            "legacy_goal": str(legacy),
            "ast_derived_goal": str(derived),
        })
    return out


def _schema_expressiveness(gold: list[RelationalRule], action_facts: set[str]) -> dict:
    """Step 2: how much of the vocabulary can the declared skeleton actually witness?"""
    covered_predicates = sorted(
        p for p in GLOSSARY
        if rule_facts(RelationalRule("probe", "", premises=(_p(p, tuple("abcd"[:len(GLOSSARY[p][0])])),),
                                     conclusions=())).issubset(action_facts)
    )
    expressible = [r.rule_id for r in gold if rule_facts(r).issubset(action_facts)]
    return {
        "action_facts_emitted": sorted(action_facts),
        "n_vocabulary_predicates": len(GLOSSARY),
        "vocabulary_predicates_expressible": covered_predicates,
        "n_vocabulary_predicates_expressible": len(covered_predicates),
        "gold_rules_expressible": expressible,
        "n_gold_rules": len(gold),
        "n_gold_rules_expressible": len(expressible),
        "anchors_expressible": [n for n, r in ANCHOR_RULES.items()
                                if rule_facts(r).issubset(action_facts)],
        "note": ("The unbounded leg can only witness properties over the action facts the "
                 "declared skeleton emits. Everything outside that set is an expert "
                 "modelling task, not an automated one."),
    }


def _population(gold: list[RelationalRule], action_facts: set[str]) -> list[tuple[str, RelationalRule, str]]:
    """Expressible anchor properties and gold rules, each with its mutants."""
    seeds: list[tuple[str, RelationalRule]] = [
        (f"anchor:{n}", r) for n, r in ANCHOR_RULES.items() if rule_facts(r).issubset(action_facts)
    ]
    seeds += [(f"gold:{r.rule_id}", r) for r in gold if rule_facts(r).issubset(action_facts)]

    out: list[tuple[str, RelationalRule, str]] = []
    for label, rule in seeds:
        out.append((label, rule, "original"))
        for mname, mutant, _reason in mutants_of(rule):
            # a mutant that leaves the skeleton's fact set cannot be judged unbounded,
            # so it is excluded rather than silently scored as vacuously true
            if mutant is not None and rule_facts(mutant).issubset(action_facts):
                out.append((f"{label}~{mname}", mutant, f"mutant:{mname}"))
    return out


def _signature(rule: RelationalRule) -> tuple[frozenset[str], tuple[str, ...]]:
    """A rule's shape as (premise relations, guaranteed relations), ignoring variable names."""
    return (frozenset(p.name for p in rule.premises),
            tuple(sorted(c.name for c in rule.conclusions)))


def _guard_localisation(rows: list[dict], rules_by_label: dict[str, RelationalRule]) -> list[dict]:
    """For each rule needing an operational guard, is the guarded form already extracted?

    A bounded-verified / unbounded-falsified rule asserts its consequent without the
    premise the operational model needs. The repair is a strictly stronger rule: same
    guaranteed relation, more premises. If such a rule is *already in the extracted set*
    and both backends verify it, the missing guard is recoverable from extraction output
    and no expert is required -- so this is the check that separates "the pipeline found
    the gap and can close it" from "a human must now write a model". Without it, every
    disagreement would be reported as expert work, overstating the manual burden.
    """
    # The guard may only be sourced from genuine extraction output. The declared anchor
    # property is an expert input and a mutant is synthetic, so counting either as
    # "already extracted" would make the automation claim circular.
    verified_both = {
        label: rules_by_label[label]
        for label, r in ((r["candidate"], r) for r in rows)
        if r["agreement"] == "agree"
        and r["bounded"]["verdict"] == VERIFIED
        and label in rules_by_label
        and label.startswith("gold:")
        and "~" not in label
    }
    out: list[dict] = []
    for row in rows:
        if row["agreement"] != "bounded_verified_unbounded_falsified":
            continue
        label = row["candidate"]
        rule = rules_by_label.get(label)
        if rule is None:
            continue
        weak_prem, weak_conc = _signature(rule)
        candidates = [
            other_label for other_label, other in verified_both.items()
            if _signature(other)[1] == weak_conc and _signature(other)[0] > weak_prem
        ]
        out.append({
            "needs_guard": label,
            "rule": str(rule),
            "missing_premises": sorted(
                set().union(*[_signature(rules_by_label[c])[0] for c in candidates]) - weak_prem
            ) if candidates else [],
            "guarded_form_already_extracted": sorted(candidates),
            "resolution": "recoverable from extraction output" if candidates else "requires expert modelling",
        })
    return out


def _bounded_theory(gold: list[RelationalRule]) -> tuple[RelationalUniverse, list, list[str]]:
    """The extracted specification as a bounded Z3 theory: the Z3-side model under test."""
    universe = RelationalUniverse()
    compiler = RelationalRuleCompiler(universe)
    pairs: list[tuple[str, object]] = []
    rejected: list[str] = []
    for r in gold:
        try:
            pairs.append((r.rule_id, compiler.compile_rule(r)))
        except RelationalCompileError:
            rejected.append(r.rule_id)
    kept, dropped = _consistency_repair(pairs, [])
    return universe, [e for _, e in kept], dropped + rejected


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,  # noqa: S607
                              text=True, check=True, cwd=ROOT).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    # Every row below is classified by comparing the bounded and unbounded
    # verdicts, so without Tamarin every classification silently collapses and
    # the recorded output would be overwritten with a degraded one.
    require_tamarin("the bounded/unbounded agreement and guard localisation")
    started = time.perf_counter()
    gold = load_gold(str(GOLD))
    action_facts = schema_action_facts(_RULES_FIXED)

    equivalence = _encoding_equivalence()
    expressiveness = _schema_expressiveness(gold, action_facts)

    universe, theory, excluded = _bounded_theory(gold)
    honest_run = ANCHOR_RULES["serving_network_binding"]

    rows: list[dict] = []
    rules_by_label: dict[str, RelationalRule] = {}
    workdir = ROOT / "data" / "gold" / "semantic_preservation_tamarin"
    workdir.mkdir(parents=True, exist_ok=True)

    for i, (label, rule, kind) in enumerate(_population(gold, action_facts)):
        rules_by_label[label] = rule
        # The bounded gate runs first. A rejection here is not a missing measurement: it is
        # the Z3-side gate doing its job, and which gate fires is itself the result.
        try:
            goal = goal_from_rule(rule, universe)
        except RelationalCompileError as exc:
            rows.append({"candidate": label, "kind": kind,
                         "bounded": {"verdict": GATE_REJECTED, "detail": str(exc)},
                         "unbounded": None, "agreement": "z3_gate_rejected"})
            continue
        b = bounded_verdict(universe, theory, goal)
        u = unbounded_verdict(rule, _PREAMBLE, _RULES_FIXED, workdir, f"sp{i:03d}",
                              executable_from=honest_run, timeout=TAMARIN_TIMEOUT)

        if u.verdict == GATE_REJECTED:
            # well-typed for Z3 but outside the Tamarin compiler's fragment -- the second,
            # independent gate catching an error class the first one admits
            agreement = "tamarin_gate_rejected"
        elif b.verdict == u.verdict:
            agreement = "agree"
        elif b.verdict == VERIFIED and u.verdict == FALSIFIED:
            agreement = "bounded_verified_unbounded_falsified"   # missing operational guard
        elif b.verdict == FALSIFIED and u.verdict == VERIFIED:
            agreement = "bounded_falsified_unbounded_verified"   # under-extracted rule set
        else:
            agreement = f"incomparable:{b.verdict}/{u.verdict}"
        rows.append({"candidate": label, "kind": kind, "bounded": b.to_dict(),
                     "unbounded": u.to_dict(), "agreement": agreement})

    agreement_counts: dict[str, int] = {}
    for r in rows:
        agreement_counts[r["agreement"]] = agreement_counts.get(r["agreement"], 0) + 1
    comparable = [r for r in rows if r["agreement"] in
                  ("agree", "bounded_verified_unbounded_falsified",
                   "bounded_falsified_unbounded_verified")]
    n_agree = sum(1 for r in comparable if r["agreement"] == "agree")
    guard_needed = [r["candidate"] for r in rows
                    if r["agreement"] == "bounded_verified_unbounded_falsified"]

    guard_rows = _guard_localisation(rows, rules_by_label)

    bounded_times = [r["bounded"]["elapsed_s"] for r in rows if r.get("bounded", {}).get("elapsed_s")]
    unb_times = [r["unbounded"]["elapsed_s"] for r in rows if r.get("unbounded")]

    # Which gate catches which error class? The two gates are independent implementations
    # over the same vocabulary, so if they reject disjoint mutation kinds then neither is
    # redundant -- and a mutation kind both admit is residual risk that must be stated.
    gates_by_kind: dict[str, dict[str, int]] = {}
    for r in rows:
        bucket = gates_by_kind.setdefault(r["kind"], {"z3_gate": 0, "tamarin_gate": 0, "passed_both": 0})
        if r["agreement"] == "z3_gate_rejected":
            bucket["z3_gate"] += 1
        elif r["agreement"] == "tamarin_gate_rejected":
            bucket["tamarin_gate"] += 1
        else:
            bucket["passed_both"] += 1

    report = {
        "description": (
            "Three measurements of semantic preservation across the bounded Z3 / unbounded "
            "Tamarin transition: that both backends are driven by the same property AST, "
            "how much of the vocabulary the declared MSR skeleton can witness at all, and "
            "how often the two backends agree on properties it can witness -- with "
            "disagreements classified rather than discarded."
        ),
        "git_sha": _git_sha(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "universe_sizes": dict(DEFAULT_SIZES),
        "tamarin_timeout_s": TAMARIN_TIMEOUT,
        "encoding_equivalence": equivalence,
        "encoding_equivalence_all_hold": all(e["equivalent"] for e in equivalence),
        "schema_expressiveness": expressiveness,
        "bounded_theory": {
            "n_rules_in_theory": len(theory),
            "excluded_rule_ids": excluded,
            "note": "The extracted specification itself, as the bounded model under test.",
        },
        "n_candidates": len(rows),
        "gates_by_mutation_kind": gates_by_kind,
        "agreement_counts": agreement_counts,
        "agreement_rate_on_comparable": (round(n_agree / len(comparable), 4) if comparable else None),
        "expert_guard_required": {
            "candidates": guard_needed,
            "n": len(guard_needed),
            "meaning": ("bounded-verified but unbounded-falsified: the extracted rule entails "
                        "the property, yet the operational model admits a trace violating it. "
                        "Closing these needs an operational guard the bounded leg cannot see."),
        },
        "guard_localisation": guard_rows,
        "expert_intervention_rate": {
            "n_needing_guard": len(guard_rows),
            "n_recoverable_from_extraction": sum(
                1 for g in guard_rows if g["guarded_form_already_extracted"]),
            "n_requiring_expert_modelling": sum(
                1 for g in guard_rows if not g["guarded_form_already_extracted"]),
        },
        "runtime_s": {
            "bounded_median": round(sorted(bounded_times)[len(bounded_times) // 2], 4) if bounded_times else None,
            "bounded_max": round(max(bounded_times), 4) if bounded_times else None,
            "unbounded_median": round(sorted(unb_times)[len(unb_times) // 2], 3) if unb_times else None,
            "unbounded_max": round(max(unb_times), 3) if unb_times else None,
        },
        "candidates": rows,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    DEST.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("1. encoding equivalence (AST-derived goal vs legacy hand-written lambda):")
    for e in equivalence:
        print(f"   {e['property']:26s} equivalent={e['equivalent']}  {e['detail']}")
    print(f"\n2. schema expressiveness: {expressiveness['n_vocabulary_predicates_expressible']}"
          f"/{expressiveness['n_vocabulary_predicates']} predicates, "
          f"{expressiveness['n_gold_rules_expressible']}/{expressiveness['n_gold_rules']} gold rules, "
          f"anchors={expressiveness['anchors_expressible']}")
    print(f"\n3. which gate catches which error class ({len(rows)} candidates):")
    print(f"   {'mutation kind':26s}{'z3 gate':>9}{'tamarin gate':>14}{'passed both':>13}")
    for k in sorted(gates_by_kind):
        g = gates_by_kind[k]
        print(f"   {k:26s}{g['z3_gate']:>9}{g['tamarin_gate']:>14}{g['passed_both']:>13}")
    print(f"\n4. cross-backend agreement:")
    for k, v in sorted(agreement_counts.items(), key=lambda kv: -kv[1]):
        print(f"   {k:42s} {v:4d}")
    print(f"   agreement rate on comparable: {report['agreement_rate_on_comparable']}")
    print(f"\n5. operational guards ({len(guard_rows)} rules bounded-verified but unbounded-falsified):")
    for g in guard_rows:
        got = ",".join(g["guarded_form_already_extracted"]) or "-"
        print(f"   {g['needs_guard']:44s} missing={g['missing_premises']}  guarded form: {got}")
    print(f"   recoverable from extraction: {report['expert_intervention_rate']['n_recoverable_from_extraction']}"
          f"/{len(guard_rows)}; requiring expert modelling: "
          f"{report['expert_intervention_rate']['n_requiring_expert_modelling']}/{len(guard_rows)}")
    print(f"\nelapsed {report['elapsed_s']}s -> {DEST}")


if __name__ == "__main__":
    main()
