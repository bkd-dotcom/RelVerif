"""Compile the relational 5G-AKA slice to Tamarin for UNBOUNDED proofs.

The bounded relational encoder (`relational.py`) reproduces the NDSS'19 serving-network-confusion weakness
as a Z3 SAT witness over a small finite universe (2 UE / 2 SN / 2 sessions) —
sound *to the bound* only. This module compiles the binding-relevant fragment of
that slice (rules R2/R3/R6 + the R9 mitigation) into Tamarin multiset-rewriting
theories and runs `tamarin-prover`, upgrading the result to the unbounded
setting:

  base theory  (no R9): `sn_binding` is FALSIFIED — Tamarin rediscovers the
                        serving-network confusion attack as a counterexample
                        trace, for unboundedly many UEs / SNs / sessions.
  fixed theory (with R9, SNN in the MAC + key derivation per TS 33.501 Annex A):
                        `sn_binding` is VERIFIED for all traces.

Both variants also carry an `executable` exists-trace sanity lemma so a
"verified" fix is never vacuous (the honest protocol run must still exist).

Honest scope: the SN/KSEAF binding property is compiled from an
abstracted AKA core (mac/kdf over a shared long-term key; no SQN sequence
numbers, no full 5G key schedule). The CCS'18 failure-message *linkability*
result is compiled as a minimal observational-equivalence (`--diff`)
witness — a type-revealing vs uniform failure observable — but NOT yet as the
full SQN-desync protocol; see :func:`compile_linkability_theory`. The adversary
is Tamarin's built-in Dolev-Yao network attacker.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TAMARIN_BINARY = "tamarin-prover"

# ---------------------------------------------------------------------------
# Compilation: relational-slice rules -> Tamarin multiset rewriting fragments
# ---------------------------------------------------------------------------

# Mapping from relational_aka.RULE_SET entries to the Tamarin encoding. The
# compiler is template-based: each relational rule contributes either a rewrite
# rule, a term-structure decision (what the MAC covers / what enters the KDF),
# or a lemma.
RULE_MAPPING: dict[str, str] = {
    "R2  mac_iff_key": "UE accepts only on a MAC valid under its long-term key "
                       "(pattern match on mac(...) under !LTK in UE_Accept).",
    "R3  accept_needs_mac": "UE_Accept consumes the challenge only when the MAC verifies.",
    "R6  accept_needs_xres": "abstracted into the single accept step (key confirmation "
                             "is part of the Commit action).",
    "R9  bind_to_auth_sn": "FIX variant only: the SN name is covered by the MAC and "
                           "enters the key derivation kdf(k, <r, SN>) — TS 33.501 Annex A.",
    "R15 adv_bind_confuse": "subsumed by Tamarin's Dolev-Yao network attacker, which can "
                            "rewrite the (unauthenticated) SN name in transit.",
}

_THEORY_TEMPLATE = """theory {name}
begin

functions: mac/2, kdf/2

// R-LTK: UE and home network share a long-term key (unbounded UEs).
rule Register_LTK:
  [ Fr(~k) ] --> [ !LTK($UE, ~k) ]

// Network side issues a challenge for $UE via serving network $SN.
{challenge_rule}

// UE side verifies the MAC and accepts (R2/R3/R6), committing to the SN
// named in the message.
{accept_rule}

// Sanity: an honest run exists (non-vacuity of the fix).
lemma executable: exists-trace
  "Ex ue sn key #i #j. Running(ue, sn, key) @ i & Commit(ue, sn, key) @ j"

// NDSS'19 SN-binding property: if a UE commits to a key with sn, the
// network side ran a session with that UE, that sn, and that key.
lemma sn_binding:
  "All ue sn key #i. Commit(ue, sn, key) @ i
     ==> Ex #j. Running(ue, sn, key) @ j"

end
"""

_CHALLENGE_BASE = """rule SN_Challenge:
  [ !LTK($UE, ~k), Fr(~r) ]
  --[ Running($UE, $SN, kdf(~k, ~r)) ]->
  [ Out(<$SN, ~r, mac(~r, ~k)>) ]"""

_ACCEPT_BASE = """rule UE_Accept:
  [ !LTK($UE, ~k), In(<sn, r, mac(r, ~k)>) ]
  --[ Commit($UE, sn, kdf(~k, r)) ]->
  []"""

# R9: SNN covered by the MAC and bound into the key derivation.
_CHALLENGE_FIXED = """rule SN_Challenge:
  [ !LTK($UE, ~k), Fr(~r) ]
  --[ Running($UE, $SN, kdf(~k, <~r, $SN>)) ]->
  [ Out(<$SN, ~r, mac(<~r, $SN>, ~k)>) ]"""

_ACCEPT_FIXED = """rule UE_Accept:
  [ !LTK($UE, ~k), In(<sn, r, mac(<r, sn>, ~k)>) ]
  --[ Commit($UE, sn, kdf(~k, <r, sn>)) ]->
  []"""


def compile_theory(*, with_mitigation: bool) -> str:
    """Compile the binding fragment of the relational slice to a .spthy theory."""
    if with_mitigation:
        return _THEORY_TEMPLATE.format(
            name="FiveG_AKA_SNBinding_Fixed",
            challenge_rule=_CHALLENGE_FIXED,
            accept_rule=_ACCEPT_FIXED,
        )
    return _THEORY_TEMPLATE.format(
        name="FiveG_AKA_SNBinding_Base",
        challenge_rule=_CHALLENGE_BASE,
        accept_rule=_ACCEPT_BASE,
    )


# ---------------------------------------------------------------------------
# CCS'18 failure-message linkability via observational equivalence
# ---------------------------------------------------------------------------
#
# Borgaonkar et al. (CCS'18-class) linkability: a UE that fails an authentication
# challenge leaks *which* check failed (SQN de-sync vs MAC mismatch) through a
# distinguishable failure message, so a passive attacker who replays a recorded
# challenge can tell whether the responding UE is the same subscriber. This is a
# privacy (indistinguishability) property, not a trace property, so it needs
# Tamarin's observational-equivalence mode (`--diff`): the `diff(l, r)` operator
# builds two systems that differ only in the observable failure term, and Tamarin
# proves/refutes `Observational_equivalence`.
#
# HONEST SCOPE (alpha): this is a *minimal* obs-equiv witness, not the full
# SQN-desync protocol. It isolates the security-relevant observable — a
# type-revealing failure cause vs a uniform (encrypted) one — and exercises the
# diff pipeline end to end:
#   base  : Out(diff('sync_failure', 'mac_failure')) — two distinct public
#           constants, so the attacker distinguishes the branches and
#           `Observational_equivalence` is FALSIFIED (== linkable).
#   fixed : Out(diff(senc('failure', k), senc('failure', k))) — an identical
#           uniform encrypted failure on both sides, so equivalence HOLDS
#           (== unlinkable). This is the "uniform/opaque failure" mitigation.
# A faithful SQN-window model (real ordered counter, freshness window, AUTS re-sync)
# is shipped as the theories data/gold/tamarin_theories/sqn_replay.spthy and
# sqn_resync.spthy, where the linkability leak is a *consequence* of a genuine
# stale-SQN comparison (replay_rejected verified, resync_recovers verified, base
# obs-equiv falsified/linkable). README section 5 lists the commands and verdicts.

_LINKABILITY_TEMPLATE = """theory {name}
begin

functions: mac/2, senc/2

// UE and home network share a long-term key (unbounded UEs).
rule Register_LTK:
  [ Fr(~k) ] --> [ !LTK($UE, ~k) ]

// UE receives a (possibly replayed) challenge and its MAC check fails. The
// observable it emits is the only thing that differs between the two worlds.
rule UE_Fail:
  [ !LTK($UE, ~k), In(<'chal', r>) ]
  --[ Fail($UE) ]->
  [ Out({failure_obs}) ]

// Non-vacuity: a failing run exists.
lemma executable: exists-trace
  "Ex ue #i. Fail(ue) @ i"

end
"""

# base: type-revealing failure cause — the two diff-worlds output distinct public
# constants, so the Dolev-Yao attacker distinguishes them (linkable).
_LINK_FAIL_BASE = "diff('sync_failure', 'mac_failure')"
# fixed: uniform opaque failure — identical term in both worlds (unlinkable).
_LINK_FAIL_FIXED = "diff(senc('failure', ~k), senc('failure', ~k))"


def compile_linkability_theory(*, with_mitigation: bool) -> str:
    """Compile the CCS'18-class failure-message linkability obs-equiv diffTheory.

    Must be proved with ``tamarin-prover --diff`` (see :func:`run_linkability`).
    """
    if with_mitigation:
        return _LINKABILITY_TEMPLATE.format(
            name="FiveG_AKA_Linkability_Fixed", failure_obs=_LINK_FAIL_FIXED)
    return _LINKABILITY_TEMPLATE.format(
        name="FiveG_AKA_Linkability_Base", failure_obs=_LINK_FAIL_BASE)


# ---------------------------------------------------------------------------
# Proving: run tamarin-prover and parse per-lemma verdicts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LemmaResult:
    lemma: str
    verdict: str  # "verified" | "falsified" | "analysis incomplete"
    steps: int | None


def tamarin_available() -> bool:
    return shutil.which(TAMARIN_BINARY) is not None


def require_tamarin(what: str) -> None:
    """Abort before writing anything if the unbounded backend is unavailable.

    Scripts that measure the unbounded leg must call this *before* they open
    their output file. Degrading to a Z3-only result and still writing would
    silently replace a recorded measurement with a different, weaker one --
    which reads downstream as the paper failing to reproduce rather than as a
    missing dependency.
    """
    if tamarin_available():
        return
    sys.stderr.write(
        f"error: {TAMARIN_BINARY} is not on PATH, so {what} cannot be measured.\n"
        f"       Refusing to run: a Z3-only result would overwrite the recorded\n"
        f"       output with a weaker measurement. Nothing was written.\n"
        f"       Install Tamarin 1.12.0 (https://tamarin-prover.com), or run\n"
        f"       `make reproduce-bounded` for the four Z3-only measurements.\n"
    )
    raise SystemExit(1)


_SUMMARY_RE = re.compile(
    r"^\s*(?P<lemma>\w+)\s+\((?:all-traces|exists-trace)\):\s+"
    r"(?P<verdict>verified|falsified)[^(]*(?:\((?P<steps>\d+)\s+steps\))?",
    re.MULTILINE,
)


def run_tamarin(spthy_text: str, workdir: Path, name: str, timeout: int = 300) -> list[LemmaResult]:
    """Write the theory and prove all lemmas; returns per-lemma verdicts.

    Raises RuntimeError if the binary is missing. The pytest suite skips on
    that; scripts must instead call require_tamarin() up front, so they abort
    before overwriting a recorded output (see that function).
    """
    if not tamarin_available():
        raise RuntimeError(f"{TAMARIN_BINARY} not found on PATH")
    workdir.mkdir(parents=True, exist_ok=True)
    theory_path = workdir / f"{name}.spthy"
    theory_path.write_text(spthy_text)
    proc = subprocess.run(
        [TAMARIN_BINARY, "--prove", str(theory_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    results = [
        LemmaResult(
            lemma=m.group("lemma"),
            verdict=m.group("verdict"),
            steps=int(m.group("steps")) if m.group("steps") else None,
        )
        for m in _SUMMARY_RE.finditer(proc.stdout)
    ]
    if not results:
        raise RuntimeError(
            f"tamarin-prover produced no lemma summary for {name} "
            f"(exit {proc.returncode}): {proc.stderr[-500:]}"
        )
    return results


# --diff output is not line-start anchored: obs-equiv appears as
# "DiffLemma:  Observational_equivalence : falsified - found trace (9 steps)"
# and the sanity lemma as "RHS :  executable (exists-trace): verified (3 steps)".
_OBSEQUIV_RE = re.compile(
    r"Observational_equivalence\s*:\s*(?P<verdict>verified|falsified|analysis incomplete)"
    r"(?:[^(]*\((?P<steps>\d+)\s+steps\))?",
)
_DIFF_LEMMA_RE = re.compile(
    r"(?P<lemma>\w+)\s+\((?:all-traces|exists-trace)\):\s+"
    r"(?P<verdict>verified|falsified)(?:\s*\((?P<steps>\d+)\s+steps\))?",
)


def run_diff_tamarin(spthy_text: str, workdir: Path, name: str, timeout: int = 300) -> list[LemmaResult]:
    """Prove an obs-equiv diffTheory (``--diff``); returns the lemma verdicts.

    Parses both the ordinary lemma summary (``executable``) and the
    ``Observational_equivalence`` line that only ``--diff`` emits.
    """
    if not tamarin_available():
        raise RuntimeError(f"{TAMARIN_BINARY} not found on PATH")
    workdir.mkdir(parents=True, exist_ok=True)
    theory_path = workdir / f"{name}.spthy"
    theory_path.write_text(spthy_text)
    proc = subprocess.run(
        [TAMARIN_BINARY, "--diff", "--prove", str(theory_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    # only parse the "summary of summaries" tail so intermediate progress lines
    # can't be mistaken for verdicts
    tail = proc.stdout.split("summary of summaries", 1)[-1]
    results = [
        LemmaResult(lemma=m.group("lemma"), verdict=m.group("verdict"),
                    steps=int(m.group("steps")) if m.group("steps") else None)
        for m in _DIFF_LEMMA_RE.finditer(tail)
    ]
    results += [
        LemmaResult(lemma="Observational_equivalence", verdict=m.group("verdict"),
                    steps=int(m.group("steps")) if m.group("steps") else None)
        for m in _OBSEQUIV_RE.finditer(tail)
    ]
    if not any(r.lemma == "Observational_equivalence" for r in results):
        raise RuntimeError(
            f"tamarin-prover --diff produced no Observational_equivalence summary "
            f"for {name} (exit {proc.returncode}): {proc.stderr[-500:]}"
        )
    return results


def run_linkability(workdir: Path, timeout: int = 300) -> dict:
    """Alpha unbounded discover -> fix -> certify record for failure-message linkability.

    Linkability holds iff the base diffTheory is observationally *in*equivalent
    (attacker distinguishes the failure branches) and the uniform-failure fix
    restores observational equivalence.
    """
    base = {r.lemma: r for r in run_diff_tamarin(
        compile_linkability_theory(with_mitigation=False), workdir, "linkability_base", timeout)}
    fixed = {r.lemma: r for r in run_diff_tamarin(
        compile_linkability_theory(with_mitigation=True), workdir, "linkability_fixed", timeout)}

    linkable_base = base["Observational_equivalence"].verdict == "falsified"
    base_executable = base.get("executable", LemmaResult("executable", "missing", None)).verdict == "verified"
    fix_unlinkable = fixed["Observational_equivalence"].verdict == "verified"
    fix_nonvacuous = fixed.get("executable", LemmaResult("executable", "missing", None)).verdict == "verified"

    return {
        "record": "failure-message linkability, observational equivalence",
        "property": "failure-message linkability (Borgaonkar et al. CCS'18 class)",
        "setting": "UNBOUNDED sessions/UEs, observational equivalence (tamarin-prover --diff)",
        "base": {lm: {"verdict": r.verdict, "steps": r.steps} for lm, r in base.items()},
        "fixed": {lm: {"verdict": r.verdict, "steps": r.steps} for lm, r in fixed.items()},
        "linkability_shown_base": linkable_base and base_executable,
        "fix_unlinkable": fix_unlinkable and fix_nonvacuous,
        "validated": linkable_base and base_executable and fix_unlinkable and fix_nonvacuous,
        "scope_caveat": (
            "minimal obs-equiv witness: isolates the type-revealing-vs-uniform failure "
            "observable, NOT the full SQN-desync protocol (no sequence-number window / "
            "re-sync). Demonstrates the diff-mode pipeline and the uniform-failure "
            "mitigation; the faithful real-counter SQN desync/re-sync model is shipped as "
            "the sqn_replay.spthy / sqn_resync.spthy theories (README section 5)."
        ),
    }


def run_snbinding(workdir: Path, timeout: int = 300) -> dict:
    """The unbounded discover -> fix -> certify record for the SN-binding property."""
    base = {r.lemma: r for r in run_tamarin(
        compile_theory(with_mitigation=False), workdir, "snbinding_handwritten_base", timeout)}
    fixed = {r.lemma: r for r in run_tamarin(
        compile_theory(with_mitigation=True), workdir, "snbinding_handwritten_fixed", timeout)}

    attack_found = base["sn_binding"].verdict == "falsified"
    base_executable = base["executable"].verdict == "verified"
    fix_proven = fixed["sn_binding"].verdict == "verified"
    fix_nonvacuous = fixed["executable"].verdict == "verified"

    return {
        "record": "serving-network binding, hand-written theory",
        "property": "serving-network / KSEAF binding (Cremers & Dehnel-Wild NDSS'19 class)",
        "setting": "UNBOUNDED sessions/UEs/SNs, Dolev-Yao attacker (tamarin-prover)",
        "rule_mapping": RULE_MAPPING,
        "base": {l: {"verdict": r.verdict, "steps": r.steps} for l, r in base.items()},
        "fixed": {l: {"verdict": r.verdict, "steps": r.steps} for l, r in fixed.items()},
        "attack_rediscovered_unbounded": attack_found and base_executable,
        "fix_proven_unbounded": fix_proven and fix_nonvacuous,
        "validated": attack_found and base_executable and fix_proven and fix_nonvacuous,
        "scope_caveat": (
            "alpha: one property (SN binding) over an abstracted AKA core "
            "(mac/kdf, no SQN or full key schedule); CCS'18 linkability needs "
            "observational equivalence and is not compiled."
        ),
    }
