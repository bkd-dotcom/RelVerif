"""Relational slice -> Tamarin compiler.

Emission tests always run (pure string generation). Prover tests run only when
tamarin-prover is installed; they pin the headline result — the base theory
is falsifiable (Tamarin rediscovers the NDSS'19 SN-confusion attack unbounded)
and the R9-fixed theory proves, non-vacuously.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verification.tamarin_relational_compiler import (  # noqa: E402
    compile_linkability_theory,
    compile_theory,
    run_linkability,
    run_snbinding,
    tamarin_available,
)


# --- emission (offline) -------------------------------------------------------

def test_base_theory_omits_snn_binding():
    theory = compile_theory(with_mitigation=False)
    assert "theory FiveG_AKA_SNBinding_Base" in theory
    assert "mac(~r, ~k)" in theory           # MAC does not cover the SN name
    assert "kdf(~k, ~r)" in theory           # key not bound to the SN
    assert "lemma sn_binding" in theory
    assert "lemma executable" in theory


def test_fixed_theory_binds_snn_in_mac_and_kdf():
    theory = compile_theory(with_mitigation=True)
    assert "theory FiveG_AKA_SNBinding_Fixed" in theory
    assert "mac(<~r, $SN>, ~k)" in theory    # R9: MAC covers the SN name
    assert "kdf(~k, <~r, $SN>)" in theory    # R9: SNN in the key derivation
    assert "kdf(~k, <r, sn>)" in theory      # UE derives the same bound key


def test_variants_share_property_lemmas():
    base, fixed = compile_theory(with_mitigation=False), compile_theory(with_mitigation=True)
    for t in (base, fixed):
        assert t.count("lemma") == 2         # executable + sn_binding, no drift


# --- prover-backed (skipped without tamarin) -----------------------------------

needs_tamarin = pytest.mark.skipif(
    not tamarin_available(), reason="tamarin-prover not installed"
)


@needs_tamarin
def test_unbounded_attack_and_fix(tmp_path):
    record = run_snbinding(tmp_path)
    # base: honest run exists AND the attack is rediscovered (unbounded)
    assert record["base"]["executable"]["verdict"] == "verified"
    assert record["base"]["sn_binding"]["verdict"] == "falsified"
    # fixed: the property is proven for all traces, non-vacuously
    assert record["fixed"]["executable"]["verdict"] == "verified"
    assert record["fixed"]["sn_binding"]["verdict"] == "verified"
    assert record["validated"] is True


# --- linkability: obs-equiv emission (offline) ----------------------------

def test_linkability_base_reveals_failure_type():
    theory = compile_linkability_theory(with_mitigation=False)
    assert "theory FiveG_AKA_Linkability_Base" in theory
    assert "diff('sync_failure', 'mac_failure')" in theory  # distinguishable
    assert "lemma executable" in theory


def test_linkability_fixed_uses_uniform_opaque_failure():
    theory = compile_linkability_theory(with_mitigation=True)
    assert "theory FiveG_AKA_Linkability_Fixed" in theory
    assert "diff(senc('failure', ~k), senc('failure', ~k))" in theory  # identical worlds


@needs_tamarin
def test_linkability_shown_and_fixed_unbounded(tmp_path):
    record = run_linkability(tmp_path)
    # base: a failing run exists AND the two worlds are distinguishable (linkable)
    assert record["base"]["executable"]["verdict"] == "verified"
    assert record["base"]["Observational_equivalence"]["verdict"] == "falsified"
    # fixed: uniform failure restores observational equivalence, non-vacuously
    assert record["fixed"]["executable"]["verdict"] == "verified"
    assert record["fixed"]["Observational_equivalence"]["verdict"] == "verified"
    assert record["validated"] is True
