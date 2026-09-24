# ruff: noqa: INP001
"""The generated theory against an expert-written one: same protocol, same fix, two authors.

Reviewers ask what the pipeline is worth compared with a human writing the Tamarin model.
The repository already contains both halves of that comparison for the NDSS'19
serving-network-binding property -- a hand-written theory pair
(`snbinding_handwritten_*`) and a pair whose lemmas the AST compiler emitted
(`snbinding_generated_*`) -- so the comparison can be
made on artifacts that already exist rather than on a new hand-modelling exercise.

Read this as a comparison of *modelling effort and outcome*, not of logical identity. The
two theories state the property in different idioms and they are not interderivable:

    expert     Commit(ue, sn, key) ==> Ex #j. Running(ue, sn, key) @ j
    generated  Accepts(s, u) & AuthenticatedWith(s, u, n) ==> Ex key. KeyBoundTo(s, key, n)

The expert uses the standard two-fact agreement idiom; the generated theory must phrase
the same requirement in the controlled vocabulary the extractor can actually emit, which
factors one correspondence into three relations. Claiming the lemmas equivalent would be
false, so what is measured instead is whether both *find the same attack and validate the
same fix*, and what each costs.

The measurement that matters most here is the one that limits the paper's own claim. In
both theories the multiset-rewrite rules are a **declared expert input** -- the pipeline
generates the lemma, not the protocol model -- so the script reports the rules block and
the lemma block separately. The fraction of each theory that is expert-declared is the
honest size of "from prose to proof", and it is better stated here with a number than left
for a reviewer to notice.

    PYTHONPATH=src .venv/bin/python scripts/run_expert_baseline.py

Writes data/gold/expert_baseline.json.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verification.relational_semantics import schema_action_facts  # noqa: E402
from verification.tamarin_relational_compiler import (  # noqa: E402
    require_tamarin,
    run_tamarin,
)

THEORIES_DIR = ROOT / "data" / "gold" / "tamarin_theories"
DEST = ROOT / "data" / "gold" / "expert_baseline.json"
WORK = ROOT / "data" / "gold" / "expert_baseline_tamarin"

#: (authorship, variant, path). The expert pair is hand-written; in the generated pair the
#: lemmas are compiled from the `RelationalRule` AST.
THEORIES = (
    ("expert", "base", THEORIES_DIR / "snbinding_handwritten_base.spthy"),
    ("expert", "fixed", THEORIES_DIR / "snbinding_handwritten_fixed.spthy"),
    ("generated", "base", THEORIES_DIR / "snbinding_generated_base.spthy"),
    ("generated", "fixed", THEORIES_DIR / "snbinding_generated_fixed.spthy"),
)

TAMARIN_TIMEOUT = 120
REPEATS = 3  # tamarin wall-clock is noisy; report the median of a few runs

_RULE_RE = re.compile(r"^\s*rule\s+(\w+)\s*:", re.MULTILINE)
_LEMMA_RE = re.compile(r"^\s*lemma\s+(\w+)", re.MULTILINE)


def _blocks(text: str) -> tuple[str, str]:
    """Split a theory into (multiset-rewrite rules, lemmas).

    The split is where the provenance changes: everything up to the first lemma is the
    declared operational skeleton, and the lemmas are what the compiler emits.
    """
    m = _LEMMA_RE.search(text)
    if m is None:
        return text, ""
    return text[: m.start()], text[m.start():]


def _code_lines(text: str) -> int:
    """Lines that are neither blank nor a comment -- the part someone had to write."""
    return sum(
        1 for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("//")
    )


def _structure(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    rules_block, lemma_block = _blocks(text)
    facts = sorted(schema_action_facts(rules_block))
    lemmas = _LEMMA_RE.findall(text)
    return {
        "file": str(path.relative_to(ROOT)),
        "bytes": len(text.encode("utf-8")),
        "lines_total": len(text.splitlines()),
        "code_lines_total": _code_lines(text),
        "code_lines_rules_block": _code_lines(rules_block),
        "code_lines_lemma_block": _code_lines(lemma_block),
        # the pipeline generates lemmas; the rewrite rules are a declared expert input,
        # so this ratio is the honest scope of the automation claim
        "expert_declared_fraction_of_code": round(
            _code_lines(rules_block) / max(_code_lines(text), 1), 3),
        "n_rewrite_rules": len(_RULE_RE.findall(text)),
        "n_lemmas": len(lemmas),
        "lemmas": lemmas,
        "action_facts": facts,
        "n_action_facts": len(facts),
    }


def _prove(path: Path, tag: str) -> dict:
    """Prove every lemma, `REPEATS` times, reporting median wall-clock."""
    text = path.read_text(encoding="utf-8")
    WORK.mkdir(parents=True, exist_ok=True)
    times: list[float] = []
    results: dict[str, dict] = {}
    for i in range(REPEATS):
        start = time.perf_counter()
        try:
            got = run_tamarin(text, WORK, f"{tag}_r{i}", TAMARIN_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - a prover failure must not abort the comparison
            return {"error": f"{type(exc).__name__}: {exc}"[:300]}
        times.append(time.perf_counter() - start)
        results = {r.lemma: {"verdict": r.verdict, "steps": r.steps} for r in got}
    return {
        "lemmas": results,
        "wall_clock_s": {
            "median": round(statistics.median(times), 3),
            "min": round(min(times), 3),
            "max": round(max(times), 3),
            "repeats": REPEATS,
        },
        "total_steps": sum(v["steps"] or 0 for v in results.values()),
    }


def _property_lemma(proof: dict) -> dict | None:
    """The security lemma, i.e. the one that is not the non-vacuity sanity check."""
    for name, v in proof.get("lemmas", {}).items():
        if name != "executable":
            return {"lemma": name, **v}
    return None


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,  # noqa: S607
                              text=True, check=True, cwd=ROOT).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    require_tamarin("the expert-written vs generated theory comparison")
    started = time.perf_counter()

    rows: list[dict] = []
    for authorship, variant, path in THEORIES:
        struct = _structure(path)
        proof = _prove(path, f"{authorship}_{variant}")
        rows.append({
            "authorship": authorship,
            "variant": variant,
            "structure": struct,
            "proof": proof,
            "property_lemma": _property_lemma(proof),
            "executable": proof.get("lemmas", {}).get("executable"),
        })

    by = {(r["authorship"], r["variant"]): r for r in rows}

    # The outcome the paper actually claims: the base admits the attack, the fix removes
    # it, and the fixed theory is still non-vacuous. Agreement means both authorships
    # reach that same outcome -- not that their lemmas are logically equivalent.
    def _outcome(authorship: str) -> dict:
        base, fixed = by[(authorship, "base")], by[(authorship, "fixed")]
        bp, fp = base["property_lemma"], fixed["property_lemma"]
        return {
            "base_property": bp,
            "fixed_property": fp,
            "base_falsified": bool(bp and bp["verdict"] == "falsified"),
            "fixed_verified": bool(fp and fp["verdict"] == "verified"),
            "fixed_non_vacuous": (fixed["executable"] or {}).get("verdict") == "verified",
            "attack_found_and_fix_validated": bool(
                bp and fp and bp["verdict"] == "falsified" and fp["verdict"] == "verified"
                and (fixed["executable"] or {}).get("verdict") == "verified"),
        }

    expert_outcome, generated_outcome = _outcome("expert"), _outcome("generated")

    def _cost(authorship: str) -> dict:
        base, fixed = by[(authorship, "base")], by[(authorship, "fixed")]
        return {
            "code_lines": base["structure"]["code_lines_total"] + fixed["structure"]["code_lines_total"],
            "code_lines_expert_declared": (base["structure"]["code_lines_rules_block"]
                                           + fixed["structure"]["code_lines_rules_block"]),
            "code_lines_lemmas": (base["structure"]["code_lines_lemma_block"]
                                  + fixed["structure"]["code_lines_lemma_block"]),
            "n_action_facts": base["structure"]["n_action_facts"],
            "proof_steps": base["proof"].get("total_steps", 0) + fixed["proof"].get("total_steps", 0),
            "prove_s_median": round(base["proof"]["wall_clock_s"]["median"]
                                    + fixed["proof"]["wall_clock_s"]["median"], 3),
        }

    expert_cost, generated_cost = _cost("expert"), _cost("generated")

    report = {
        "description": (
            "A hand-written Tamarin theory pair and an AST-compiled one for the same "
            "NDSS'19 serving-network-binding property and the same protocol-level fix, "
            "compared on outcome (is the attack found and the fix validated?) and on cost "
            "(code lines, action facts, proof steps, prover wall-clock). The two lemmas "
            "are NOT interderivable -- the expert uses the standard agreement idiom and "
            "the generated theory must phrase the requirement in the controlled "
            "vocabulary -- so this measures modelling effort and outcome, not logical "
            "equivalence."
        ),
        "git_sha": _git_sha(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tamarin_timeout_s": TAMARIN_TIMEOUT,
        "repeats_per_theory": REPEATS,
        "outcome": {
            "expert": expert_outcome,
            "generated": generated_outcome,
            "same_outcome": (expert_outcome["attack_found_and_fix_validated"]
                             == generated_outcome["attack_found_and_fix_validated"]),
            "both_succeed": (expert_outcome["attack_found_and_fix_validated"]
                             and generated_outcome["attack_found_and_fix_validated"]),
        },
        "cost": {
            "expert": expert_cost,
            "generated": generated_cost,
            "generated_vs_expert_code_lines": round(
                generated_cost["code_lines"] / max(expert_cost["code_lines"], 1), 2),
            "generated_vs_expert_proof_steps": round(
                generated_cost["proof_steps"] / max(expert_cost["proof_steps"], 1), 2),
            "generated_vs_expert_prove_s": round(
                generated_cost["prove_s_median"] / max(expert_cost["prove_s_median"], 1e-6), 2),
        },
        "automation_scope": {
            "note": (
                "In both authorships the multiset-rewrite rules are a declared expert "
                "input: the pipeline emits the lemma, not the protocol model. This "
                "fraction is therefore the part of the unbounded artifact that automation "
                "does not produce, and it bounds what 'from prose to proof' can mean here."
            ),
            "expert_declared_fraction_of_generated_theory": by[("generated", "base")]["structure"][
                "expert_declared_fraction_of_code"],
            "generated_lemma_code_lines": generated_cost["code_lines_lemmas"],
            "declared_skeleton_code_lines": generated_cost["code_lines_expert_declared"],
        },
        "theories": rows,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    DEST.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'authorship':11s}{'variant':9s}{'property lemma':>18}{'verdict':>12}"
          f"{'steps':>7}{'median s':>10}{'code':>6}")
    for r in rows:
        pl = r["property_lemma"] or {}
        wc = r["proof"].get("wall_clock_s", {}).get("median", float("nan"))
        print(f"{r['authorship']:11s}{r['variant']:9s}{pl.get('lemma', '-'):>18}"
              f"{pl.get('verdict', '-'):>12}{str(pl.get('steps', '-')):>7}{wc:>10.3f}"
              f"{r['structure']['code_lines_total']:>6}")
    print(f"\nattack found AND fix validated:  expert="
          f"{expert_outcome['attack_found_and_fix_validated']}  "
          f"generated={generated_outcome['attack_found_and_fix_validated']}")
    print(f"cost (generated / expert): code x{report['cost']['generated_vs_expert_code_lines']}, "
          f"steps x{report['cost']['generated_vs_expert_proof_steps']}, "
          f"prove x{report['cost']['generated_vs_expert_prove_s']}")
    print(f"expert-declared fraction of the generated theory: "
          f"{report['automation_scope']['expert_declared_fraction_of_generated_theory']}")
    print(f"\nelapsed {report['elapsed_s']}s -> {DEST}")


if __name__ == "__main__":
    main()
