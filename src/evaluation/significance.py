"""Significance testing for the gold-set evaluation.

Turns the fidelity-ablation point estimates into publication-grade claims:
inter-annotator agreement (Cohen's kappa), paired classifier comparison
(exact McNemar), score comparisons (Wilcoxon signed-rank for paired data,
Mann-Whitney U for independent groups), effect size (Cliff's delta), and
bootstrap confidence intervals (mean and F1).

Design:
- ``scipy`` (pinned ``scipy==1.18.0``) supplies the well-tested exact/rank tests;
  ``numpy`` drives the bootstrap. Cohen's kappa and Cliff's delta are hand-rolled
  (tiny, and kappa is shared with ``scripts/compute_held_out_kappa.py``).
- Everything is deterministic: the bootstrap takes an explicit ``seed`` so a run
  is reproducible from the manifest.

All comparisons are pure functions over plain lists/arrays so the module has no
dependency on the pipeline or the gate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

def _sig(x: float, figs: int = 3) -> float:
    """Round to N significant figures, preserving tiny values (p=1.2e-15 stays)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) or x == 0:
        return x
    return float(f"{x:.{figs}g}")


# --------------------------------------------------------------------------- #
# Cohen's kappa (general categorical; binary ACCEPT/REJECT is a special case)  #
# --------------------------------------------------------------------------- #


def cohen_kappa(r1: Sequence[Any], r2: Sequence[Any]) -> tuple[float, float, float, int]:
    """Cohen's kappa for two raters over categorical labels.

    Returns ``(kappa, p_o, p_e, n)`` where ``p_o`` is observed agreement and
    ``p_e`` expected-by-chance agreement. Works for any hashable labels (bool,
    "ACCEPT"/"REJECT", ...). Handles the degenerate constant-marginal case
    (``p_e >= 1``) the way ``compute_held_out_kappa`` does: kappa is 1.0 on
    perfect agreement else 0.0. Raw agreement ``p_o`` is returned too because
    kappa degenerates when a rater's marginal is constant.
    """
    if len(r1) != len(r2):
        msg = "raters must label the same number of items"
        raise ValueError(msg)
    n = len(r1)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    r1 = list(r1)
    r2 = list(r2)
    agree = sum(1 for a, b in zip(r1, r2, strict=True) if a == b)
    p_o = agree / n
    cats = set(r1) | set(r2)
    p_e = sum((r1.count(c) / n) * (r2.count(c) / n) for c in cats)
    if p_e >= 1.0:
        # constant marginal -> kappa undefined; report 1.0 iff perfectly agreed.
        return (1.0 if p_o == 1.0 else 0.0), p_o, p_e, n
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa, p_o, p_e, n


def interpret_kappa(k: float) -> str:
    """Landis-Koch agreement band."""
    if k is None or (isinstance(k, float) and math.isnan(k)):
        return "n/a"
    if k < 0:
        return "poor"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


# --------------------------------------------------------------------------- #
# Exact McNemar — paired comparison of two binary classifiers vs ground truth  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class McNemarResult:
    b: int  # A correct, B wrong  (discordant, favouring A)
    c: int  # A wrong, B correct  (discordant, favouring B)
    n_discordant: int
    p_value: float
    odds_ratio: float  # b / c (inf if c == 0 and b > 0; nan if both 0)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["p_value"] = _sig(self.p_value)  # 3 sig-figs; keeps tiny p (e.g. 1.2e-15) from rounding to 0
        d["odds_ratio"] = self.odds_ratio if math.isinf(self.odds_ratio) else round(self.odds_ratio, 4)
        return d


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> McNemarResult:
    """Exact McNemar's test on two classifiers' per-item correctness.

    Pass ``correct_a = [pred_a[i] == gold[i] ...]`` and likewise ``correct_b``.
    Only the discordant pairs (one correct, the other wrong) carry signal; the
    exact test is a two-sided binomial on ``min(b, c)`` successes in ``b + c``
    trials at p=0.5. This is the right test for "does backend A accept/reject
    the gold pairs more accurately than the other backend" on the SAME items.
    """
    if len(correct_a) != len(correct_b):
        msg = "correctness vectors must be the same length"
        raise ValueError(msg)
    b = sum(1 for a, bb in zip(correct_a, correct_b, strict=True) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b, strict=True) if not a and bb)
    n_disc = b + c
    p_value = (
        1.0 if n_disc == 0 else float(stats.binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue)
    )
    if c > 0:
        odds_ratio = b / c
    elif b > 0:
        odds_ratio = float("inf")
    else:
        odds_ratio = float("nan")
    return McNemarResult(b=b, c=c, n_discordant=n_disc, p_value=p_value, odds_ratio=odds_ratio)


# --------------------------------------------------------------------------- #
# Rank tests over scores                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RankTestResult:
    test: str
    statistic: float
    p_value: float
    n: int

    def as_dict(self) -> dict:
        return {
            "test": self.test,
            "statistic": round(self.statistic, 4) if not math.isnan(self.statistic) else float("nan"),
            "p_value": _sig(self.p_value),
            "n": self.n,
        }


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> RankTestResult:
    """Wilcoxon signed-rank test for PAIRED continuous scores (same items).

    Use to compare two backends' scores item-by-item (e.g. lexical vs judge on
    each gold pair). Zero differences are handled by scipy's default. Degrades
    gracefully (nan statistic, p=1.0) when every paired difference is zero.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape:
        msg = "paired samples must be the same length"
        raise ValueError(msg)
    nz = int(np.count_nonzero(xa - ya))
    if nz == 0:
        return RankTestResult("wilcoxon_signed_rank", float("nan"), 1.0, 0)
    res = stats.wilcoxon(xa, ya)
    return RankTestResult("wilcoxon_signed_rank", float(res.statistic), float(res.pvalue), nz)


def mann_whitney_u(x: Sequence[float], y: Sequence[float]) -> RankTestResult:
    """Mann-Whitney U for TWO INDEPENDENT groups of scores.

    Use for score separability: fidelity scores of gold-ACCEPT pairs vs
    gold-REJECT pairs (different items, not paired). Pair with Cliff's delta
    for the effect size.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size == 0 or ya.size == 0:
        return RankTestResult("mann_whitney_u", float("nan"), 1.0, xa.size + ya.size)
    res = stats.mannwhitneyu(xa, ya, alternative="two-sided")
    return RankTestResult("mann_whitney_u", float(res.statistic), float(res.pvalue), xa.size + ya.size)


# --------------------------------------------------------------------------- #
# Cliff's delta — non-parametric effect size for two independent groups        #
# --------------------------------------------------------------------------- #


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> tuple[float, str]:
    """Cliff's delta and its magnitude band (Romano et al. thresholds).

    delta = (#(xi > yj) - #(xi < yj)) / (|x| * |y|), in [-1, 1]. O(|x|*|y|),
    fine for gold-set sizes. Positive delta => x tends to exceed y.
    """
    xa = list(x)
    ya = list(y)
    n = len(xa) * len(ya)
    if n == 0:
        return float("nan"), "n/a"
    gt = lt = 0
    for xi in xa:
        for yj in ya:
            if xi > yj:
                gt += 1
            elif xi < yj:
                lt += 1
    delta = (gt - lt) / n
    a = abs(delta)
    if a < 0.147:
        mag = "negligible"
    elif a < 0.33:
        mag = "small"
    elif a < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return delta, mag


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lo: float
    hi: float
    alpha: float
    n_resamples: int

    def as_dict(self) -> dict:
        return {
            "point": round(self.point, 4),
            "ci_lo": round(self.lo, 4),
            "ci_hi": round(self.hi, 4),
            "alpha": self.alpha,
            "n_resamples": self.n_resamples,
        }


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for a scalar statistic of one sample.

    ``stat`` must accept an ``axis`` kwarg for the vectorised path (np.mean /
    np.median do); a plain callable is also accepted via a slower loop.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), alpha, n_resamples)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    resamples = arr[idx]
    try:
        boot = np.asarray(stat(resamples, axis=1), dtype=float)  # type: ignore[call-arg]
    except TypeError:
        boot = np.array([float(stat(row)) for row in resamples])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(float(stat(arr)), float(lo), float(hi), alpha, n_resamples)


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def bootstrap_f1_ci(
    pred_accept: Sequence[bool],
    gold_accept: Sequence[bool],
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for F1 (ACCEPT = positive), resampling paired items."""
    pred = np.asarray(pred_accept, dtype=bool)
    gold = np.asarray(gold_accept, dtype=bool)
    if pred.shape != gold.shape:
        msg = "pred and gold must be the same length"
        raise ValueError(msg)
    m = pred.size
    if m == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), alpha, n_resamples)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_resamples, m))
    p = pred[idx]
    g = gold[idx]
    tp = np.sum(p & g, axis=1)
    fp = np.sum(p & ~g, axis=1)
    fn = np.sum(~p & g, axis=1)
    denom = 2 * tp + fp + fn
    boot = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point_tp = int(np.sum(pred & gold))
    point_fp = int(np.sum(pred & ~gold))
    point_fn = int(np.sum(~pred & gold))
    return BootstrapCI(_f1_from_counts(point_tp, point_fp, point_fn), float(lo), float(hi), alpha, n_resamples)
