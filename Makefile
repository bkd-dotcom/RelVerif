# RelVerif -- reproduce every measurement in the ICTAI 2026 paper.
#
# Interpreter. CPython 3.11 is a hard floor, set by datetime.UTC (3.11) and, below
# that, zip(strict=True) and match statements (3.10). `make test` additionally needs
# 3.12, because the scipy pin in requirements-test.txt declares it.
#
# `python3` is NOT a safe default: on macOS /usr/bin/python3 is still 3.9, where the
# code dies with a confusing TypeError from inside zip() rather than a clear message.
# So search by name and skip anything below the floor. `python3` is tried first so
# that an activated virtualenv wins; a versioned name would bypass it. Every measured
# value in data/gold/ was re-verified identical on 3.11.15, 3.12.13, 3.13.13 and 3.14.5,
# and the shipped numbers were produced on 3.13.13.
# Override with e.g. `make reproduce PY=/usr/local/bin/python3.12`.
PY ?= $(shell for c in python3 python3.13 python3.14 python3.12 python3.11 python; do \
	  command -v $$c >/dev/null 2>&1 || continue; \
	  $$c -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null || continue; \
	  echo $$c; break; \
	done)

.PHONY: help reproduce reproduce-bounded test clean checkpy checkdeps checktamarin

help:
	@echo "make reproduce           every measurement the paper cites  (needs Z3 + Tamarin, ~80 s)"
	@echo "make reproduce-bounded   the four Z3-only measurements      (no Tamarin needed, ~35 s)"
	@echo "make test                unit tests for the relational subsystem (needs requirements-test.txt)"
	@echo "make clean               remove Python caches"
	@echo
	@echo "interpreter: $(if $(PY),$(PY),NONE FOUND -- see the comment at the top of this Makefile)"

checkpy:
	@test -n "$(PY)" || { \
	  printf '\n  No CPython >= 3.11 found on PATH (3.11 is a hard floor: datetime.UTC,\n'; \
	  printf '  plus zip(strict=True) and match statements below it).\n'; \
	  printf '  Install one, or point PY at an existing interpreter:\n'; \
	  printf '      make reproduce PY=/path/to/python3.12\n\n'; \
	  exit 1; }
	@$(PY) -c 'import sys; v = sys.version_info; \
	  sys.exit(None if v >= (3, 11) else \
	  "\n  PY=%s is Python %d.%d.%d, but this artifact needs >= 3.11 (datetime.UTC).\n" \
	  "  Point PY at a newer interpreter: make reproduce PY=/path/to/python3.13\n" \
	  % (sys.executable, v[0], v[1], v[2]))'
	@$(PY) -c 'import sys; print("  using %s (Python %s)" % (sys.executable, ".".join(map(str, sys.version_info[:3]))))'

checkdeps: checkpy
	@$(PY) -c 'import z3' 2>/dev/null || { \
	  printf '\n  z3 is not importable by $(PY).\n'; \
	  printf '  Create a virtualenv and install the one dependency:\n'; \
	  printf '      $(PY) -m venv .venv && . .venv/bin/activate\n'; \
	  printf '      pip install -r requirements.txt\n'; \
	  printf '  then re-run make. (If you already have a venv, activate it first --\n'; \
	  printf '  an unactivated venv is not on PATH and will not be picked up.)\n\n'; \
	  exit 1; }
	@$(PY) -c 'import z3; print("  using z3", z3.get_version_string())'

# A convenience so a missing dependency surfaces in a second rather than mid-run.
# It is not the only defence: the two scripts that drive the unbounded leg call
# require_tamarin() before they open their output file, so they exit non-zero and
# write nothing if the binary is absent. Neither path can leave a recorded
# measurement overwritten with a weaker Z3-only one.
checktamarin:
	@command -v tamarin-prover >/dev/null 2>&1 || { \
	  printf '\n  tamarin-prover is not on PATH.\n'; \
	  printf '  The unbounded leg (Table III bottom, Table IV lower, RQ4) needs it.\n'; \
	  printf '  Install tamarin-prover 1.12.0 (https://tamarin-prover.com), or run\n'; \
	  printf '  `make reproduce-bounded` for the four measurements that only need Z3.\n\n'; \
	  exit 1; }
	@tamarin-prover --version 2>/dev/null | head -1 | sed 's/^/  using /'

# Every measurement in the paper, regenerated with local solvers only -- no API keys,
# no network. Each line names the file it overwrites under data/gold/, so the mapping
# from a claim in the paper to the file that backs it is readable here.
#
# Table II (extraction accuracy) is deliberately NOT here: it calls an LLM provider, so
# it cannot be part of a one-command offline reproduce. Its recorded outputs are shipped
# as data/gold/relational_extraction_report_{gptoss,llama4}.json -- see README § 'What is not regenerated offline'.
reproduce: checkdeps checktamarin
	$(PY) scripts/run_relational_adversary.py    # -> relational_adversary.json     Table I  (bounded base/fixed verdicts, non-vacuity)
	$(PY) scripts/run_relational_closed_loop.py  # -> relational_closed_loop.json   Table I  (end-to-end loop, per extractor family)
	$(PY) scripts/run_repair_candidate_sweep.py  # -> repair_candidate_sweep.json   Table III top (first check to reject; collateral damage)
	$(PY) scripts/run_semantic_preservation.py   # -> semantic_preservation.json    Table III bottom + RQ4 (gate-by-error-class; leg agreement)
	$(PY) scripts/run_relational_scalability.py  # -> relational_scalability.json   Table IV upper (runtimes; bound and rule-count arms)
	$(PY) scripts/run_expert_baseline.py         # -> expert_baseline.json          Table IV lower (generated vs hand-written theory)

# The subset that needs no Tamarin install: Table I's bounded columns, Table III top
# and Table IV upper. The two unbounded-leg scripts are omitted rather than run in a
# degraded mode, so this target cannot change what the other two recorded.
reproduce-bounded: checkdeps
	$(PY) scripts/run_relational_adversary.py    # -> relational_adversary.json     Table I  (bounded base/fixed verdicts, non-vacuity)
	$(PY) scripts/run_relational_closed_loop.py  # -> relational_closed_loop.json   Table I  (end-to-end loop, per extractor family)
	$(PY) scripts/run_repair_candidate_sweep.py  # -> repair_candidate_sweep.json   Table III top (first check to reject; collateral damage)
	$(PY) scripts/run_relational_scalability.py  # -> relational_scalability.json   Table IV upper (runtimes; bound and rule-count arms)

test: checkdeps
	@$(PY) -c 'import pytest, numpy, scipy' 2>/dev/null || { \
	  printf '\n  The test dependencies are not installed. They are separate from the\n'; \
	  printf '  reproduce dependencies because numpy/scipy are reached by one test only:\n'; \
	  printf '      $(PY) -m pip install -r requirements-test.txt\n'; \
	  printf '  scipy 1.18 needs Python >= 3.12, so use a 3.12+ interpreter for this target.\n\n'; \
	  exit 1; }
	$(PY) -m pytest -q

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
