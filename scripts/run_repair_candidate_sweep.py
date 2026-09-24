# ruff: noqa: INP001
"""Which check earns its place? Enumerate every candidate repair and see what catches it.

The closed loop admits a repair only if it survives four checks in order: the
well-typedness gate, base consistency, witness closure (the bounded counterexample is
gone), and non-vacuity (the secured system still has an honest run). Reporting that those
checks exist is not evidence that they do anything. This sweep turns each one into a
number by giving the loop far more candidate repairs than the three a paper can narrate.

The candidate space is every extracted rule in a source *plus* its systematic mutants
(converse, premise drop, argument permutation, polarity inversion). That space is the
right one because it is what an untrusted extractor can plausibly emit: a converse is
well-typed, a dropped guard is well-typed, and both are the documented failure modes.
Each candidate is offered as the mitigation for each anchor, and we record the *first*
check that rejects it. A check that is never the first to reject anything is a check the
architecture does not need; one that rejects candidates nothing else catches is a check
whose removal would admit unsound repairs -- which is the quantified form of the
ablation reviewers ask for.

Accepted repairs then face a cross-anchor regression test: a repair for one property must
not destroy another property's honest run or falsify its goal. "Closes the witness" is
not "is a good fix", and this is where that difference becomes visible.

    PYTHONPATH=src .venv/bin/python scripts/run_repair_candidate_sweep.py

Writes data/gold/repair_candidate_sweep.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.relational_extractor import RelationalExtractor  # noqa: E402
from pipeline.schemas_relational import (  # noqa: E402
    RelationalRule,
    load_gold,
    normalize_name,
)
from verification.relational_closed_loop import (  # noqa: E402
    binding_property,
    decisive_predicates,
    key_confirmation_property,
    property_predicates,
    run_closed_loop,
    supi_concealment_property,
    _consistency_repair,  # noqa: PLC2701 - the loop's own primitive, reused not reimplemented
)
from verification.relational_compiler import (  # noqa: E402
    DEFAULT_SIZES,
    RelationalCompileError,
    RelationalRuleCompiler,
    RelationalUniverse,
)
from verification.relational_semantics import extract_witness, mutants_of  # noqa: E402
from z3 import Not, Solver, sat, unsat  # noqa: E402

GOLD = ROOT / "data" / "gold" / "relational_gold.jsonl"
CACHE_ROOT = ROOT / "data" / "gold" / "relational_extractor_cache"
DEST = ROOT / "data" / "gold" / "repair_candidate_sweep.json"

ANCHORS = [
    (binding_property, "REL13", "NDSS'19 serving-network confusion"),
    (supi_concealment_property, "REL02", "IMSI-catcher (cleartext subscriber identity)"),
    (key_confirmation_property, "REL11", "forged accept without key confirmation"),
]

# outcome labels: the FIRST check to reject a candidate, or acceptance
GATE = "rejected_by_welltypedness_gate"
NO_CLOSE = "rejected_by_witness_closure"
VACUOUS = "rejected_by_non_vacuity"
ACCEPTED = "accepted"


class _NoClient:
    def generate(self, prompt: str):  # noqa: ARG002
        msg = "cache miss"
        raise RuntimeError(msg)


def _model_id_for(cache_name: str) -> str:
    if cache_name.startswith("ica_"):
        return "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    if cache_name.startswith("ollama_"):
        return "gpt-oss:120b"
    return cache_name


def _load_extracted(cache_dir: Path, gold: list[RelationalRule]) -> dict[str, RelationalRule]:
    ex = RelationalExtractor(_NoClient(), _model_id_for(cache_dir.name), cache_dir=cache_dir)
    out: dict[str, RelationalRule] = {}
    for g in gold:
        try:
            r = ex.extract(g.source_text, g.rule_id)
        except RuntimeError:
            continue
        if r is not None:
            out[g.rule_id] = r
    return out


def _carries(rule: RelationalRule, preds: tuple[str, ...]) -> bool:
    want = {normalize_name(p) for p in preds}
    return any(normalize_name(p.name) in want for p in (*rule.premises, *rule.conclusions))


def _candidates(rules: dict[str, RelationalRule]) -> list[tuple[str, RelationalRule, str]]:
    """Every rule, plus every mutant of it: ``(candidate_id, rule, kind)``."""
    out: list[tuple[str, RelationalRule, str]] = []
    for rid, rule in sorted(rules.items()):
        out.append((rid, rule, "extracted"))
        for name, mutant, _reason in mutants_of(rule):
            if mutant is not None:
                out.append((f"{rid}~{name}", mutant, f"mutant:{name}"))
    return out


def _sweep_anchor(
    anchor: tuple,
    rules: dict[str, RelationalRule],
    source: str,
) -> dict:
    prop_fn, canonical_id, attack = anchor
    prop = prop_fn()
    universe = RelationalUniverse()
    compiler = RelationalRuleCompiler(universe)

    goal = prop.goal(universe)
    scenario = prop.scenario(universe)
    adversary = prop.adversary(universe)
    exclude = decisive_predicates(prop, universe)

    # the unconstrained base: every rule that does not already carry the anchor's relation
    base_rules = [r for r in rules.values() if not _carries(r, exclude)]
    compiled_pairs: list[tuple[str, object]] = []
    for r in base_rules:
        try:
            compiled_pairs.append((r.rule_id, compiler.compile_rule(r)))
        except RelationalCompileError:
            continue
    kept, conflicting = _consistency_repair(compiled_pairs, scenario)
    base = [e for _, e in kept]

    # DISCOVER once: the witness is a property of the base, not of any candidate
    disc = Solver()
    disc.add(*base, *scenario, adversary, Not(goal))
    discovered = disc.check() == sat
    # scope the witness to the trigger + the guaranteed relation, so it reads as the
    # published attack rather than as a dump of all nineteen predicates
    witness_scope = property_predicates(prop, universe)
    witness = extract_witness(disc.model(), universe, witness_scope) if discovered else {}

    counts = {GATE: 0, NO_CLOSE: 0, VACUOUS: 0, ACCEPTED: 0}
    accepted_ids: list[str] = []
    vacuous_ids: list[str] = []
    per_candidate: list[dict] = []

    for cand_id, cand, kind in _candidates(rules):
        try:
            mit = compiler.compile_rule(cand)
        except RelationalCompileError as exc:
            counts[GATE] += 1
            per_candidate.append({"candidate": cand_id, "kind": kind,
                                  "outcome": GATE, "reason": str(exc)})
            continue

        rev = Solver()
        rev.add(*base, mit, *scenario, adversary, Not(goal))
        closes = rev.check() == unsat

        sec = Solver()
        sec.add(*base, mit, *scenario)
        non_vacuous = sec.check() == sat

        if not closes:
            outcome = NO_CLOSE
        elif not non_vacuous:
            outcome = VACUOUS
            vacuous_ids.append(cand_id)
        else:
            outcome = ACCEPTED
            accepted_ids.append(cand_id)
        counts[outcome] += 1
        per_candidate.append({"candidate": cand_id, "kind": kind, "outcome": outcome})

    return {
        "property": prop.name,
        "published_attack": attack,
        "canonical_mitigation": canonical_id,
        "source": source,
        "n_base_rules": len(kept),
        "base_consistent": not conflicting,
        "conflicting_rules": conflicting,
        "exclude_auto_derived": list(exclude),
        "bounded_witness_found": discovered,
        "bounded_witness": witness,
        "n_candidates": len(per_candidate),
        "counts": counts,
        "accepted": accepted_ids,
        "vacuous_closers": vacuous_ids,
        "per_candidate": per_candidate,
    }


@dataclass
class _Secured:
    """An anchor already repaired by its canonical mitigation -- the regression baseline.

    Testing a candidate against a bare scenario would be meaningless: without anchor B's
    own mitigation present, B's goal is violable no matter what the candidate says, and
    every repair would look like a regression. The baseline therefore has to be the
    *secured* theory, where B is verified and B's honest run exists, so any change after
    adding the candidate is attributable to the candidate.
    """

    name: str
    universe: RelationalUniverse
    compiler: RelationalRuleCompiler
    theory: list
    scenario: list
    adversary: object
    goal: object
    baseline_verified: bool
    baseline_live: bool


def _secured_context(anchor: tuple, rules: dict[str, RelationalRule]) -> _Secured | None:
    prop_fn, canonical_id, _atk = anchor
    if canonical_id not in rules:
        return None
    prop = prop_fn()
    universe = RelationalUniverse()
    compiler = RelationalRuleCompiler(universe)
    goal, scenario, adversary = prop.goal(universe), prop.scenario(universe), prop.adversary(universe)
    exclude = decisive_predicates(prop, universe)

    pairs: list[tuple[str, object]] = []
    for r in rules.values():
        if _carries(r, exclude):
            continue
        try:
            pairs.append((r.rule_id, compiler.compile_rule(r)))
        except RelationalCompileError:
            continue
    kept, _dropped = _consistency_repair(pairs, scenario)
    theory = [e for _, e in kept]
    try:
        theory.append(compiler.compile_rule(rules[canonical_id]))
    except RelationalCompileError:
        return None

    verified = Solver()
    verified.add(*theory, *scenario, adversary, Not(goal))
    live = Solver()
    live.add(*theory, *scenario)
    return _Secured(
        name=prop.name, universe=universe, compiler=compiler, theory=theory,
        scenario=scenario, adversary=adversary, goal=goal,
        baseline_verified=verified.check() == unsat,
        baseline_live=live.check() == sat,
    )


def _regression(
    accepted: list[str],
    rules_by_candidate: dict[str, RelationalRule],
    anchor_index: int,
    secured: list[_Secured | None],
) -> list[dict]:
    """Does an accepted repair for one anchor damage the *other* already-secured anchors?

    A repair is a new global axiom, so it applies to every session, not just the one whose
    counterexample it closes. Two harms matter and are distinguished, because they are
    different failures: making another property's honest run unsatisfiable forbids
    legitimate protocol behaviour, while making an already-verified property violable
    again introduces a new weakness. Only anchors whose baseline is verified *and* live
    are used -- a baseline that is already broken cannot witness a regression.
    """
    out: list[dict] = []
    for cand_id in accepted:
        cand = rules_by_candidate[cand_id]
        damage: list[dict] = []
        for j, ctx in enumerate(secured):
            if j == anchor_index or ctx is None:
                continue
            if not (ctx.baseline_verified and ctx.baseline_live):
                continue
            try:
                mit = ctx.compiler.compile_rule(cand)
            except RelationalCompileError:
                continue

            live = Solver()
            live.add(*ctx.theory, mit, *ctx.scenario)
            if live.check() != sat:
                damage.append({"property": ctx.name,
                               "harm": "honest run of an already-secured property is destroyed"})
                continue
            viol = Solver()
            viol.add(*ctx.theory, mit, *ctx.scenario, ctx.adversary, Not(ctx.goal))
            if viol.check() == sat:
                damage.append({"property": ctx.name,
                               "harm": "already-verified property becomes violable again"})
        if damage:
            out.append({"candidate": cand_id, "regressions": damage})
    return out


def _cross_check(rules: dict[str, RelationalRule], source: str) -> list[dict]:
    """The sweep must agree with `run_closed_loop` on the canonical anchor/mitigation triples."""
    out: list[dict] = []
    for prop_fn, mid, _atk in ANCHORS:
        if mid not in rules:
            continue
        prop = prop_fn()
        exclude = decisive_predicates(prop)
        base = [r for r in rules.values() if not _carries(r, exclude)]
        ref = run_closed_loop(RelationalUniverse(), base, rules[mid], prop,
                              mitigation_source=source)
        out.append({
            "source": source,
            "property": prop.name,
            "mitigation": mid,
            "reference_fixed": ref.fixed,
            "reference_non_vacuous": ref.secure_world_exists,
            "reference_validated": ref.end_to_end_validated,
        })
    return out


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,  # noqa: S607
                              text=True, check=True, cwd=ROOT).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    started = time.perf_counter()
    gold = load_gold(str(GOLD))
    sources: dict[str, dict[str, RelationalRule]] = {}
    if CACHE_ROOT.exists():
        for cache_dir in sorted(CACHE_ROOT.iterdir()):
            if cache_dir.is_dir():
                extracted = _load_extracted(cache_dir, gold)
                if len(extracted) >= 10:
                    sources[cache_dir.name] = extracted
    sources["gold-control"] = {r.rule_id: r for r in gold}

    per_source: list[dict] = []
    totals = {GATE: 0, NO_CLOSE: 0, VACUOUS: 0, ACCEPTED: 0}
    all_regressions: list[dict] = []
    cross_checks: list[dict] = []

    for source, rules in sources.items():
        cand_map = {cid: rule for cid, rule, _k in _candidates(rules)}
        secured = [_secured_context(a, rules) for a in ANCHORS]
        anchor_reports: list[dict] = []
        for i, anchor in enumerate(ANCHORS):
            rep = _sweep_anchor(anchor, rules, source)
            regs = _regression(rep["accepted"], cand_map, i, secured)
            rep["n_accepted_with_regression"] = len(regs)
            rep["regressions"] = regs
            all_regressions.extend({**r, "source": source, "anchor": rep["property"]} for r in regs)
            for k, v in rep["counts"].items():
                totals[k] += v
            anchor_reports.append(rep)
        cross_checks.extend(_cross_check(rules, source))
        per_source.append({
            "source": source,
            "n_rules": len(rules),
            # the regression test is only as good as its baseline, so publish it
            "regression_baselines": [
                {"property": c.name, "verified": c.baseline_verified, "honest_run_exists": c.baseline_live}
                for c in secured if c is not None
            ],
            "anchors": anchor_reports,
        })

    n_candidates = sum(totals.values())
    # A check's value = how many candidates it is the FIRST to reject. Removing that check
    # would admit exactly those candidates into the next stage.
    check_value = {
        "welltypedness_gate": totals[GATE],
        "witness_closure": totals[NO_CLOSE],
        "non_vacuity": totals[VACUOUS],
        "cross_anchor_regression": len(all_regressions),
    }

    report = {
        "description": (
            "Every extracted rule and its systematic mutants offered as a candidate repair "
            "for every anchor, labelled by the FIRST check that rejects it. Gives each "
            "admission check a measured value instead of a qualitative claim, and "
            "regression-tests every accepted repair against the other anchors."
        ),
        "git_sha": _git_sha(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "universe_sizes": dict(DEFAULT_SIZES),
        "mutation_operators": ["converse", "drop_premise0", "drop_premise1",
                              "permute_args", "negate_conclusion"],
        "n_sources": len(sources),
        "n_candidate_evaluations": n_candidates,
        "totals": totals,
        "check_value_first_rejecter": check_value,
        "cross_anchor_regressions": all_regressions,
        "cross_check_vs_run_closed_loop": cross_checks,
        "per_source": per_source,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    DEST.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"candidate evaluations: {n_candidates}")
    for k, v in totals.items():
        print(f"  {k:36s} {v:5d}  ({100.0 * v / n_candidates:5.1f}%)")
    print(f"\ncross-anchor regressions among accepted repairs: {len(all_regressions)}")
    print(f"elapsed {report['elapsed_s']}s -> {DEST}")


if __name__ == "__main__":
    main()
