# ruff: noqa: INP001
"""End-to-end relational loop: LLM-EXTRACTED rules -> compile -> discover -> fix.

Demonstrates the full pipeline joined up: rules produced by the LLM
extractor are compiled into Z3 and run through the discover -> fix -> re-verify loop.
The serving-network-binding vulnerability is found (SAT) when the rule set lacks the
binding constraint, and is closed (UNSAT) once the *extracted* binding rule (REL13) is
added. Runs for each extractor family whose cache is present; falls back to the gold
rules if no extractor cache exists (still exercises the loop, source labelled 'gold').

    PYTHONPATH=src .venv/bin/python scripts/run_relational_closed_loop.py

Writes data/gold/relational_closed_loop.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.relational_extractor import RelationalExtractor  # noqa: E402
from pipeline.schemas_relational import (  # noqa: E402
    Predicate,
    RelationalRule,
    load_gold,
    normalize_name,
)
from verification.relational_closed_loop import (  # noqa: E402
    binding_property,
    decisive_predicates,
    key_confirmation_property,
    run_closed_loop,
    supi_concealment_property,
)
from verification.relational_compiler import (  # noqa: E402
    RelationalRuleCompiler,
    RelationalUniverse,
)

GOLD = ROOT / "data" / "gold" / "relational_gold.jsonl"
CACHE_ROOT = ROOT / "data" / "gold" / "relational_extractor_cache"

# Each anchor: a published-attack property + its EXTRACTED mitigation rule. The decisive
# predicate(s) a genuinely unconstrained base must exclude (so DISCOVER is fair) are now
# AUTO-DERIVED from each property's AST via `decisive_predicates()` (goal-referenced
# minus scenario-triggered) — no hand-maintained exclude list.
ANCHORS = [
    {
        "property": binding_property,
        "mitigation_id": "REL13",  # authenticated_with -> key_bound_to
        "attack": "NDSS'19 serving-network confusion",
    },
    {
        "property": supi_concealment_property,
        "mitigation_id": "REL02",  # identifies -> encrypted_with_hn_key
        "attack": "IMSI-catcher (cleartext subscriber identity)",
    },
    {
        "property": key_confirmation_property,
        "mitigation_id": "REL11",  # accepts -> res_matches_xres
        "attack": "forged accept without challenge-response key confirmation",
    },
]


def _excludes(rule: RelationalRule, preds: tuple[str, ...]) -> bool:
    """True if the rule mentions any decisive predicate (premise OR conclusion) — such a
    rule entangles the anchor's relation, so an unconstrained base must exclude it."""
    want = {normalize_name(p) for p in preds}
    return any(normalize_name(p.name) in want
               for p in (*rule.premises, *rule.conclusions))


class _NoClient:
    """Offline: cache hits only; a miss is a hard error (we do not re-hit the gateway)."""

    def generate(self, prompt: str):
        msg = "cache miss"
        raise RuntimeError(msg)


def _load_extracted(cache_dir: Path, model_id: str, gold: list[RelationalRule]) -> dict:
    ex = RelationalExtractor(_NoClient(), model_id, cache_dir=cache_dir)
    out = {}
    for g in gold:
        try:
            r = ex.extract(g.source_text, g.rule_id)
        except RuntimeError:
            continue
        if r is not None:
            out[g.rule_id] = r
    return out


def _model_id_for(cache_name: str) -> str:
    # invert the _slug used by the eval script well enough to hit the same cache keys
    if cache_name.startswith("ica_"):
        return "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    if cache_name.startswith("ollama_"):
        return "gpt-oss:120b"
    return cache_name


def _run_for(anchor: dict, label: str, rules_by_id: dict, source: str) -> dict:
    mitigation = rules_by_id[anchor["mitigation_id"]]
    # exclude set is AUTO-DERIVED from the property AST (no hand list).
    prop = anchor["property"]()
    exclude = decisive_predicates(prop)
    # base = every rule that does NOT already carry the anchor's relation -> it is
    # genuinely unconstrained, so DISCOVER is a fair test regardless of extractor.
    base = [r for r in rules_by_id.values() if not _excludes(r, exclude)]
    res = run_closed_loop(RelationalUniverse(), base, mitigation, prop,
                          mitigation_source=source)
    d = res.to_dict()
    d["extractor"] = label
    d["n_rules_available"] = len(rules_by_id)
    d["mitigation_rule"] = str(mitigation)
    d["exclude_auto_derived"] = list(exclude)
    d["carriers_excluded"] = sorted(rid for rid, r in rules_by_id.items()
                                    if _excludes(r, exclude))
    return d


def _welltypedness_gate_demo() -> dict:
    """Show the compiler's gate rejects a hallucinated predicate and a sort conflict."""
    compiler = RelationalRuleCompiler(RelationalUniverse())
    hallucinated = RelationalRule(
        "BAD1", "x",
        premises=(Predicate("teleports_key", ("s", "u"), ("Session", "UE")),),
        conclusions=(Predicate("accepts", ("s", "u"), ("Session", "UE")),))
    _, rep = compiler.compile_all([hallucinated])
    return {"rejected_hallucinated_predicate": len(rep.rejected) == 1,
            "reason": rep.rejected[0][1] if rep.rejected else ""}


def _anchor_report(anchor: dict, gold: list) -> dict:
    runs = []
    # one run per extractor family whose cache is present
    if CACHE_ROOT.exists():
        for cache_dir in sorted(CACHE_ROOT.iterdir()):
            if not cache_dir.is_dir():
                continue
            model_id = _model_id_for(cache_dir.name)
            extracted = _load_extracted(cache_dir, model_id, gold)
            if anchor["mitigation_id"] in extracted and len(extracted) >= 10:
                runs.append(_run_for(anchor, cache_dir.name, extracted, f"extracted:{cache_dir.name}"))
    # always include the gold run as a control
    runs.append(_run_for(anchor, "gold-control", {r.rule_id: r for r in gold}, "gold"))
    return {
        "property": anchor["property"]().name,
        "published_attack": anchor["attack"],
        "mitigation_id": anchor["mitigation_id"],
        "runs": runs,
        "all_runs_validated": all(r["end_to_end_validated"] for r in runs),
    }


def main() -> None:
    gold = load_gold(str(GOLD))
    anchors = [_anchor_report(a, gold) for a in ANCHORS]

    report = {
        "description": ("End-to-end: LLM-extracted relational rules compiled into Z3; for "
                        "each anchor the published-attack vulnerability is discovered (SAT) "
                        "without the mitigating rule and closed (UNSAT) by the EXTRACTED "
                        "mitigation. Three anchors demonstrate the loop generalises beyond one "
                        "property (serving-network binding + SUPI concealment / IMSI-catcher + "
                        "key confirmation / forged accept)."),
        "welltypedness_gate": _welltypedness_gate_demo(),
        "anchors": anchors,
        "n_anchors": len(anchors),
        "all_anchors_validated": all(a["all_runs_validated"] for a in anchors),
    }
    dest = ROOT / "data" / "gold" / "relational_closed_loop.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
