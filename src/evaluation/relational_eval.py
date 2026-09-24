"""Score relational extraction against the hand-authored gold set.

The propositional eval (``extraction_eval.py``) scores an ACCEPT/REJECT decision.
Relational extraction is a *structured* task, so we measure it structurally:

  * **Predicate-level P/R/F1** — treat each rule as its set of predicate signatures
    ``(normalized_name, arg_sorts, negated)``; TP = recovered, FP = spurious,
    FN = missed. Reported both **strict** (name + sorts + polarity must match) and
    **name-only** (relation recovered, sorts/polarity ignored) so the error analysis
    can separate "found the wrong relation" from "right relation, wrong typing".
  * **Rule-level exact match** — premises AND conclusions recovered exactly (the
    hardest bar; this is the number a skeptic will ask for).
  * **Directionality-swap rate** — predicates all recovered but premise/conclusion
    partition inverted (the relational analogue of the propositional directionality
    metric).
  * **Polarity errors** — right relation + sorts but negation flipped.

Per-rule F1 feeds ``significance.bootstrap_ci`` for a CI on macro-F1 — n=25 is small
and we report it honestly with the interval, not as a point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from evaluation.significance import bootstrap_ci

if TYPE_CHECKING:
    from pipeline.schemas_relational import RelationalRule


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def _names(sigs: set) -> set:
    """Collapse signatures to (name, sorts) — drop polarity — for name+sorts matching."""
    return {(name, sorts) for name, sorts, _ in sigs}


@dataclass
class RuleScore:
    rule_id: str
    tp: int
    fp: int
    fn: int
    exact_match: bool
    directionality_swapped: bool
    polarity_errors: int
    name_only_tp: int
    name_only_fp: int
    name_only_fn: int
    predicted_empty: bool

    @property
    def precision(self) -> float:
        return _prf(self.tp, self.fp, self.fn)[0]

    @property
    def recall(self) -> float:
        return _prf(self.tp, self.fp, self.fn)[1]

    @property
    def f1(self) -> float:
        return _prf(self.tp, self.fp, self.fn)[2]

    @property
    def name_only_f1(self) -> float:
        return _prf(self.name_only_tp, self.name_only_fp, self.name_only_fn)[2]


def score_rule(gold: RelationalRule, pred: RelationalRule | None) -> RuleScore:
    """Score one predicted rule against its gold reference."""
    g_all = gold.all_sigs()
    if pred is None:
        return RuleScore(gold.rule_id, 0, 0, len(g_all), exact_match=False,
                         directionality_swapped=False, polarity_errors=0,
                         name_only_tp=0, name_only_fp=0,
                         name_only_fn=len({(n, s) for n, s, _ in g_all}), predicted_empty=True)
    p_all = pred.all_sigs()
    tp = len(g_all & p_all)
    fp = len(p_all - g_all)
    fn = len(g_all - p_all)

    exact = (gold.premise_sigs() == pred.premise_sigs()
             and gold.conclusion_sigs() == pred.conclusion_sigs())
    swapped = (not exact
               and gold.premise_sigs() == pred.conclusion_sigs()
               and gold.conclusion_sigs() == pred.premise_sigs()
               and len(gold.premise_sigs()) + len(gold.conclusion_sigs()) > 0)

    # name+sorts (ignore polarity) matching
    gn, pn = _names(g_all), _names(p_all)
    n_tp = len(gn & pn)
    n_fp = len(pn - gn)
    n_fn = len(gn - pn)
    # polarity errors: (name, sorts) matched but negation differs
    g_pol = {(n, s): neg for n, s, neg in g_all}
    p_pol = {(n, s): neg for n, s, neg in p_all}
    polarity = sum(1 for k in (gn & pn) if g_pol.get(k) != p_pol.get(k))

    return RuleScore(gold.rule_id, tp, fp, fn, exact, swapped, polarity,
                     n_tp, n_fp, n_fn, predicted_empty=not p_all)


@dataclass
class CorpusScore:
    per_rule: list[RuleScore] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.per_rule)

    # --- micro (pooled counts) ---
    def _micro(self, *, name_only: bool = False) -> tuple[float, float, float]:
        if name_only:
            tp = sum(r.name_only_tp for r in self.per_rule)
            fp = sum(r.name_only_fp for r in self.per_rule)
            fn = sum(r.name_only_fn for r in self.per_rule)
        else:
            tp = sum(r.tp for r in self.per_rule)
            fp = sum(r.fp for r in self.per_rule)
            fn = sum(r.fn for r in self.per_rule)
        return _prf(tp, fp, fn)

    @property
    def micro_precision(self) -> float:
        return self._micro()[0]

    @property
    def micro_recall(self) -> float:
        return self._micro()[1]

    @property
    def micro_f1(self) -> float:
        return self._micro()[2]

    @property
    def name_only_micro_f1(self) -> float:
        return self._micro(name_only=True)[2]

    @property
    def macro_f1(self) -> float:
        return sum(r.f1 for r in self.per_rule) / self.n if self.n else 0.0

    @property
    def exact_match_rate(self) -> float:
        return sum(1 for r in self.per_rule if r.exact_match) / self.n if self.n else 0.0

    @property
    def directionality_swap_rate(self) -> float:
        return sum(1 for r in self.per_rule if r.directionality_swapped) / self.n if self.n else 0.0

    @property
    def total_polarity_errors(self) -> int:
        return sum(r.polarity_errors for r in self.per_rule)

    @property
    def n_predicted_empty(self) -> int:
        return sum(1 for r in self.per_rule if r.predicted_empty)

    def macro_f1_ci(self, *, seed: int = 0) -> tuple[float, float, float]:
        ci = bootstrap_ci([r.f1 for r in self.per_rule], seed=seed)
        return ci.point, ci.lo, ci.hi

    def to_dict(self) -> dict:
        point, lo, hi = self.macro_f1_ci()
        return {
            "n": self.n,
            "micro_precision": round(self.micro_precision, 4),
            "micro_recall": round(self.micro_recall, 4),
            "micro_f1": round(self.micro_f1, 4),
            "macro_f1": round(self.macro_f1, 4),
            "macro_f1_bootstrap_ci": [round(point, 4), round(lo, 4), round(hi, 4)],
            "name_only_micro_f1": round(self.name_only_micro_f1, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "directionality_swap_rate": round(self.directionality_swap_rate, 4),
            "total_polarity_errors": self.total_polarity_errors,
            "n_predicted_empty": self.n_predicted_empty,
            "per_rule": [
                {
                    "rule_id": r.rule_id,
                    "precision": round(r.precision, 3),
                    "recall": round(r.recall, 3),
                    "f1": round(r.f1, 3),
                    "exact_match": r.exact_match,
                    "directionality_swapped": r.directionality_swapped,
                    "polarity_errors": r.polarity_errors,
                    "predicted_empty": r.predicted_empty,
                }
                for r in self.per_rule
            ],
        }


def score_corpus(gold: list[RelationalRule], preds: dict[str, RelationalRule | None]) -> CorpusScore:
    """Score a batch: ``preds`` maps rule_id -> predicted rule (or None if extraction failed)."""
    return CorpusScore(per_rule=[score_rule(g, preds.get(g.rule_id)) for g in gold])
