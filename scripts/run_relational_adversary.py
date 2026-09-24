# ruff: noqa: INP001
"""Run the relational Dolev-Yao adversary demonstrations and write the artifact.

    PYTHONPATH=src .venv/bin/python scripts/run_relational_adversary.py

Writes data/gold/relational_adversary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verification.relational_adversary import run_adversary_demos  # noqa: E402

if __name__ == "__main__":
    out = run_adversary_demos()
    out["scope"] = (
        "Bounded relational Dolev-Yao adversary: composable capabilities (intercept / "
        "decompose / compose / key-secrecy / replay) over an explicit knows() relation. "
        "Derivability uses the sound entailment reading (knows(X) iff negation UNSAT). "
        "Demonstrates: (a) decompose leaks a secret ONLY when the key is known; (b) a "
        "cross-session replay is accepted without freshness and blocked with it, while a "
        "legitimate fresh message is still accepted. Bounded, not an unbounded proof."
    )
    dest = ROOT / "data" / "gold" / "relational_adversary.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {dest}")
