# RelVerif: research artifact

Code and data for the ICTAI 2026 paper:

> **A Neuro-Symbolic Verification Loop for 5G-AKA: From LLM-Extracted Requirements to
> Solver-Checked Repairs**
>
> Binay Dalai, Mahfuza Farooque
> School of Electrical Engineering and Computer Science, Pennsylvania State University

This artifact regenerates **every measurement in the paper that does not require an LLM
provider**, offline and with no network access, in about 80 seconds. That is everything
except Table II's extraction accuracy, whose recorded reports and raw per-sentence model
responses are both included so it can be re-scored without a provider (Sec. 4).

It contains nothing else: no paper sources, no PDFs, no figures, no prose write-ups, and
no material from any other project.

```
make reproduce            # every measurement the paper cites   (Z3 + Tamarin, ~80 s)
make reproduce-bounded    # the four that need only Z3          (no Tamarin,   ~35 s)
make test                 # 58 unit tests for the relational subsystem
```

---

## 1. Requirements

| | version | why |
|---|---|---|
| CPython | **>= 3.11** for `reproduce`; **>= 3.12** for `test` | `datetime.UTC` sets the 3.11 floor (`zip(strict=True)` and `match` sit below it at 3.10); the `scipy` pin declares `>= 3.12` |
| [`z3-solver`](https://pypi.org/project/z3-solver/) | **4.12.4.0** | the bounded relational leg: **the only runtime dependency** |
| [`tamarin-prover`](https://tamarin-prover.com) | **1.12.0** | the unbounded leg, invoked as a subprocess |
| `pytest`, `numpy`, `scipy` | 8.4.1, 2.4.4, 1.18.0 | `make test` only; see `requirements-test.txt` |

```sh
python3.11 -m venv .venv                          # any 3.11+ interpreter
./.venv/bin/pip install -r requirements.txt       # z3-solver, and nothing else
make reproduce PY=./.venv/bin/python
```

`requirements.txt` is one line on purpose. The six measurement scripts import only `z3`
from outside the standard library; `numpy` and `scipy` are reached by exactly one test
(`tests/test_relational_eval.py` -> `src/evaluation/significance.py`), so they live in
`requirements-test.txt` and are not needed to reproduce a single number in the paper.
That split is what lets `reproduce` run on 3.11 while `test` needs 3.12.

Every measured value in `data/gold/` was re-verified **identical on 3.11.15, 3.12.13,
3.13.13 and 3.14.5**; the shipped numbers were produced on 3.13.13.

Two notes that will otherwise cost you time:

* **`/usr/bin/python3` on macOS is 3.9.6** and dies with a confusing `TypeError` from
  inside `zip()`. `make` finds a suitable interpreter itself, and refuses up front with
  the version it found if none qualifies. Override with `PY=/path/to/python3.12`.
* **Without `tamarin-prover` on `PATH`, the two unbounded-leg scripts refuse to run.**
  They call `require_tamarin()` *before* opening their output file, print what is
  missing, and exit non-zero having written nothing, so a missing prover can never
  leave a recorded measurement overwritten by a weaker Z3-only one. Use
  `make reproduce-bounded` for the four measurements that need only Z3.

## 2. Layout

```
src/pipeline/        typed relational AST, schemas, cache-backed extractor
src/verification/    bounded Z3 encoding, the closed loop, both Tamarin compilers
src/evaluation/      extraction scorer (micro/macro-F1, bootstrap CI)
scripts/             the six entry points that produce the paper's numbers
tests/               58 known-answer tests for the above
data/gold/           inputs (gold rules, LLM caches, Tamarin theories) and the
                     recorded outputs, so you can diff your rerun
```

## 3. Which file backs which claim

`make reproduce` runs the six scripts below in order. Each overwrites one JSON file
under `data/gold/`; the copies shipped here are the ones the paper was written from, so
a rerun can be diffed against them.

| Script | Writes | Backs | Needs Tamarin |
|---|---|---|---|
| `run_relational_adversary.py` | `relational_adversary.json` | **Table I**: bounded base/fixed verdicts and non-vacuity | |
| `run_relational_closed_loop.py` | `relational_closed_loop.json` | **Table I**: the end-to-end loop, per extractor family | |
| `run_repair_candidate_sweep.py` | `repair_candidate_sweep.json` | **Table III (top)**: first check to reject; collateral damage | |
| `run_semantic_preservation.py` | `semantic_preservation.json` | **Table III (bottom)**, **RQ4**: gate by error class; agreement between the two legs | yes |
| `run_relational_scalability.py` | `relational_scalability.json` | **Table IV (upper)**: runtimes, bound arm and rule-count arm | |
| `run_expert_baseline.py` | `expert_baseline.json` | **Table IV (lower)**: generated vs hand-written theory | yes |

`make reproduce-bounded` runs the four unmarked rows. It *omits* the other two rather
than running them in a degraded mode, so it cannot change what they recorded.

One further recorded output, `percorpus_eval.json`, ships without a script because its
input is withheld. Section 4 says what it backs and why.

Spot-checks, all verified to regenerate exactly in a clean virtualenv:

* `repair_candidate_sweep.json`: 1809 candidate evaluations (`n_candidate_evaluations`,
  and 1809 `per_candidate` rows); first rejecter, summed over the nine source x anchor
  cells' `counts`: 387 well-typedness / 1155 witness closure / 225 non-vacuity; 42
  accepted, of which 9 regress another secured anchor (`n_accepted_with_regression`).
* `semantic_preservation.json`: of 24 candidates (`gates_by_mutation_kind`): permuted
  arguments caught by the Z3 sort gate 5/5; negated conclusions by the Tamarin fragment
  gate 5/5; **premise/conclusion converse caught by neither gate, 4/5** (this is the
  paper's central point). `agreement_counts` gives 4 agreements against 4 + 4
  disagreements, i.e. the two legs agree on 4 of 12 comparable candidates
  (`agreement_rate_on_comparable` 0.3333).
* `expert_baseline.json`: same outcome for both authorships (attack found *and* fix
  validated); x1.11 code lines, x1.00 proof steps, prove-time parity (Sec. 6);
  expert-declared fraction **0.619**, under the key
  `automation_scope.expert_declared_fraction_of_generated_theory`.
* `relational_scalability.json`: verdicts re-validated to scale 32 with no flips
  (`verdict_stability.model.max_scale_all_validated` 32, `flips` empty); growth of 2.6 x 10^4
  in quantifier instantiations at flat wall-clock (`cost_summary.model.metric_factor`
  26450.6, `seconds_factor` 0.93).

## 4. What is not regenerated offline

**Table II (extraction accuracy) calls an LLM provider**, so it cannot be part of a
one-command offline reproduce. Its recorded outputs are shipped as
`data/gold/relational_extraction_report_{gptoss,llama4}.json`:

| | micro-F1 | exact match | directionality swap | polarity errors |
|---|---|---|---|---|
| `gpt-oss:120b` | 0.9854 | 0.84 | **0.10** | 0 |
| `llama-4-maverick-17b` | 0.9246 | 0.48 | **0.34** | 4 |

Both are over the same *n* = 50 curated gold corpus (`data/gold/relational_gold.jsonl`),
and each report's `per_rule` array has 50 entries whose booleans add up to exactly the
rates above (42 and 24 exact matches; 5 and 17 swaps).

The raw model responses behind them are in `data/gold/relational_extractor_cache/`:
**50 per model, one per corpus sentence**, keyed by `sha1(model_id|source_text)`. Both
caches are complete and carry nothing spare, so the downstream loop replays offline
without a provider: `run_relational_closed_loop.py` reads that cache instead of
re-querying. (One `llama-4` *response* parsed to no rule at all, recorded as
`n_predicted_empty: 1`, which is a model output, not a missing cache entry.)

To regenerate the reports themselves you need a provider, e.g.

```sh
python3 scripts/run_relational_extraction_eval.py \
    --provider ollama --model gpt-oss:120b \
    --out data/gold/relational_extraction_report_gptoss.json
```

That script is **not** included here, because it is the one component that cannot run
offline; the cache and the scorer it feeds (`src/evaluation/relational_eval.py`, covered
by `tests/test_relational_eval.py`) are.

### The per-corpus decomposition behind limitation (v)

The paper's limitations item (v) reports that the semantic-fidelity gate is
domain-sensitive: at the operating point the paper uses (embedding backend, threshold
0.70) its directionality-inversion catch rate ranges from 25% on one corpus to 91% on
another. Those figures come from `data/gold/percorpus_eval.json`, shipped here so they
can be checked. Each row below reads
`per_corpus.<key>.methods["embedding@0.70"]`:

| corpus key | `dir_caught` / `dir_total` | `dir_catch` |
|---|---|---|
| `aka` | 6 / 7 | 0.8571 |
| `nist80063` | 3 / 12 | 0.2500 |
| `nist80063b` | 10 / 11 | 0.9091 |
| `owaspasvs` | 7 / 12 | 0.5833 |
| aggregate, under `overall` | 26 / 42 | 0.6190 |

Two caveats matter more than the numbers.

* **This one file is a record, not a rerun.** The labeled requirement set it was
  computed from is named in its own `source_gold_csv` field, and is not redistributed
  here because it is third-party requirement text. Table II above needs a provider but
  can be rebuilt; this file cannot be rebuilt at all from what ships, which makes it the
  weakest-provenance number in the paper. It is shipped so the figure can at least be
  read back against the claim it supports.
* **The `loco` block is not a held-out experiment.** Its four `held_out=*` entries are
  byte-identical to the matching `per_corpus` entries. For a frozen threshold gate that
  is self-consistent, since there is nothing to train and so nothing to hold out, but
  the key name invites the opposite reading. Do not cite it as leave-one-corpus-out.

## 5. Table I's unbounded column: run the theories directly

All of Table I's Tamarin verdicts come from hand-written and generated theories checked
in under `data/gold/tamarin_theories/`. No script in `make reproduce` regenerates them,
so run them yourself:

```sh
cd data/gold/tamarin_theories
tamarin-prover --prove sqn_replay.spthy                    # replay rejection
tamarin-prover --prove sqn_resync.spthy                    # re-sync recovery
tamarin-prover --prove snbinding_handwritten_base.spthy    # SN binding, hand-written
tamarin-prover --prove snbinding_handwritten_fixed.spthy
tamarin-prover --prove snbinding_generated_base.spthy      # SN binding, AST-compiled
tamarin-prover --prove snbinding_generated_fixed.spthy
tamarin-prover --prove --diff sqn_link_base.spthy          # NOTE: --diff required
tamarin-prover --prove --diff sqn_link_fixed.spthy
```

`sqn_link_*.spthy` state observational equivalence with a `diff` operator. **Without
`--diff` they fail to parse** (`unexpected ")" ... diff operator found, but flag diff
not set`), which looks like a corrupt file rather than a missing flag.

The two `snbinding_generated_*.spthy` theories are shipped as outputs, but they are not
taken on trust: `src/verification/ast_tamarin_compiler.py` emits them from the relational
AST, and its `compile_snbinding_from_ast(with_mitigation=...)` entry point reproduces
both files **byte for byte**.

Verdicts observed on tamarin-prover 1.12.0:

| Theory | Lemma | Verdict |
|---|---|---|
| `sqn_replay` | `replay_rejected` (all-traces) | verified, 95 steps |
| `sqn_resync` | `resync_recovers` (exists-trace) | verified, 16 steps |
| `snbinding_handwritten_base` | `sn_binding` (all-traces) | falsified, trace at 5 steps |
| `snbinding_handwritten_fixed` | `sn_binding` (all-traces) | verified, 5 steps |
| `snbinding_generated_base` | `sn_binding` (all-traces) | falsified, trace at 6 steps |
| `snbinding_generated_fixed` | `sn_binding` (all-traces) | verified, 2 steps |
| `sqn_link_base` | `Observational_equivalence` | falsified, trace at 16 steps |
| `sqn_link_fixed` | `Observational_equivalence` | **does not terminate** |

The last row is the paper's "fixed search incomplete": the repaired linkability theory's
equivalence search did not finish in 150 s here, and the paper claims no verdict for it.
Every theory also carries an `executable` exists-trace lemma, which is the honest-model
check: a theory that proves its safety lemma only because nothing can run would pass
vacuously, so that lemma must be `verified` for the row above it to mean anything.

## 6. Nondeterminism: what will and will not match exactly

**Combinatorial results are deterministic.** Every count, verdict, and rate listed above
reproduced exactly on repeated clean-virtualenv runs and on all four interpreters in Sec. 1;
`relational_scalability.json` asserts `deterministic_across_repeats` and cross-checks its
verdicts against the closed loop. Each JSON also records a `git_sha` and `generated_at`,
which will of course differ in your rerun, so diff the measured fields, not whole files.

**Wall-clock results are not, and the paper does not quote them as if they were.**
Timings are medians of three runs, and the two ratios near the measurement floor are
reported qualitatively on purpose:

* *Prove time, generated vs expert.* Both theories prove in ~0.62 s, so the ratio
  straddles 1.0; observed **0.93, 0.97, 0.99, 0.99, 1.00, 1.00, 1.03, 1.04** over eight
  runs on one machine. The paper reports **parity (~1.0, noise floor)** rather than any
  one of them.
* *Rule arm, 50 -> 800 requirements.* Observed **x4.97, x4.98, x4.99, x5.09, x5.14,
  x5.15** over six runs. The paper reports **~x5 (sub-linear)**. The claim is
  sub-linearity against a x16 increase in rule count, not the third significant
  figure.

If your absolute timings differ from these by a constant factor, that is your machine,
and none of the paper's claims depend on it.

## 7. Reading the recorded outputs: which `false` values are results

The JSON files are diagnostics as well as results, so `false` does not uniformly mean
"something went wrong". Four of the five booleans you will meet are either good news or
bookkeeping:

| Field | `false` count | What `false` means |
|---|---|---|
| `per_rule[].directionality_swapped` | 45/50, 33/50 | **good**: the rule was extracted the right way round |
| `per_rule[].predicted_empty` | 50/50, 49/50 | **good**: the model returned a parseable rule |
| `per_rule[].exact_match` | 8/50, 26/50 | a miss; these counts *are* the 0.84 / 0.48 rates in Sec. 4 |
| `runs[].base_consistent` | 7/9 runs | **bookkeeping**: see below |
| `runs[].end_to_end_validated` | 2/9 runs | **a result**: see below |

`base_consistent` is `not conflicting_rules`: before the search, each extracted rule is
checked against the attack scenario, and any rule inconsistent with it is dropped.
`false` records that this happened; `conflicting_rules` names the rules and
`n_base_rules_compiled` counts what was kept. It is not an extractor fault (the
`gold-control` arm also drops `REL20` on `supi_concealment`); it is a property of the
scenario, and reporting it is what keeps the kept-rule count honest.

`end_to_end_validated: false` **is** a finding, and the paper's point. It occurs twice in
nine runs, both times with `fixed: false`: the extracted mitigation did not make the
violation UNSAT:

| Anchor | `gpt-oss:120b` | `llama-4-maverick` | `gold-control` |
|---|---|---|---|
| `serving_network_binding` | **fails** | closes | closes |
| `supi_concealment` | closes | closes | closes |
| `key_confirmation` | closes | **fails** | closes |

Both extractors discover every vulnerability (`discovered_vulnerability: true`, 9/9) and
every secured world is non-vacuous (`secure_world_exists: true`, 9/9); what differs is
whether the *repair* they proposed survives re-verification. `gold-control` closes all
three, which is what makes the two failures attributable to extraction rather than to the
encoding. `repair_candidate_sweep.json` reproduces the same two failures independently,
in its `cross_check_vs_run_closed_loop` and `regression_baselines` entries.

## 8. License

Apache License 2.0; see `LICENSE`.
