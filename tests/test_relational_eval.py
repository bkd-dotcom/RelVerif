"""The relational extraction scorer (known-answer tests)."""

from __future__ import annotations

from evaluation.relational_eval import score_corpus, score_rule
from pipeline.schemas_relational import Predicate, RelationalRule


def _rule(prem, conc, rid="R"):
    return RelationalRule(rule_id=rid, source_text="", premises=tuple(prem), conclusions=tuple(conc))


ACC = Predicate("accepts", ("s", "u"), ("Session", "UE"))
MAC = Predicate("mac_verified", ("s", "u"), ("Session", "UE"))
SQN = Predicate("sqn_in_range", ("s", "u"), ("Session", "UE"))
NOT_ACC = Predicate("accepts", ("s", "u"), ("Session", "UE"), negated=True)


def test_perfect_match_is_f1_one_and_exact():
    gold = _rule([ACC], [MAC])
    sc = score_rule(gold, _rule([ACC], [MAC]))
    assert sc.f1 == 1.0
    assert sc.exact_match is True
    assert sc.directionality_swapped is False


def test_missed_predicate_lowers_recall():
    gold = _rule([ACC], [MAC, SQN])
    sc = score_rule(gold, _rule([ACC], [MAC]))  # missed SQN
    assert sc.fn == 1
    assert sc.recall < 1.0
    assert sc.exact_match is False


def test_spurious_predicate_lowers_precision():
    gold = _rule([ACC], [MAC])
    sc = score_rule(gold, _rule([ACC], [MAC, SQN]))  # extra SQN
    assert sc.fp == 1
    assert sc.precision < 1.0


def test_directionality_swap_detected():
    gold = _rule([ACC], [MAC])
    sc = score_rule(gold, _rule([MAC], [ACC]))  # premise/conclusion swapped
    assert sc.directionality_swapped is True
    assert sc.exact_match is False


def test_polarity_error_detected():
    gold = _rule([MAC], [ACC])
    sc = score_rule(gold, _rule([MAC], [NOT_ACC]))  # right relation+sorts, flipped negation
    assert sc.polarity_errors == 1
    assert sc.exact_match is False           # strict signature differs
    assert sc.name_only_f1 == 1.0            # name+sorts still match


def test_none_prediction_is_all_false_negatives():
    gold = _rule([ACC], [MAC])
    sc = score_rule(gold, None)
    assert sc.predicted_empty is True
    assert sc.fn == 2
    assert sc.f1 == 0.0


def test_corpus_aggregates_and_ci():
    gold = [_rule([ACC], [MAC], "R1"), _rule([ACC], [SQN], "R2")]
    preds = {"R1": _rule([ACC], [MAC], "R1"), "R2": None}
    cs = score_corpus(gold, preds)
    assert cs.n == 2
    assert 0.0 < cs.micro_f1 < 1.0
    assert cs.exact_match_rate == 0.5
    assert cs.n_predicted_empty == 1
    point, lo, hi = cs.macro_f1_ci()
    assert lo <= point <= hi
