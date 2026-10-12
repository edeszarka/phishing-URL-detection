"""Headline OOF metrics with group-bootstrap CIs and precision at 1:1000.

Run as `uv run python -m src.oof_metrics` after src.train; reads
data/oof_predictions.csv and writes results/oof_headline.json.
"""

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.features import SEED
from src.train import ABLATION_GROUP, operating_point

ROOT = Path(__file__).resolve().parent.parent
PREVALENCE = 0.001
TARGET_FPRS = (0.01, 0.001)
N_BOOTSTRAP = 2000
PRIMARY_COLUMN = "prob_raw_url"
PROBE_COLUMN = "prob_without_scheme_and_www"


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float = PREVALENCE) -> float:
    """Precision implied by TPR/FPR when a `prevalence` share of URLs is phishing."""
    flagged = tpr * prevalence + fpr * (1 - prevalence)
    return tpr * prevalence / flagged if flagged > 0 else float("nan")


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * log_p
        + (n - i) * log_q
        for i in range(k + 1)
    ]
    peak = max(log_terms)
    return min(1.0, math.exp(peak) * sum(math.exp(t - peak) for t in log_terms))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided binomial interval for k successes in n trials (bisection)."""

    def solve(target: float, cdf_k: int) -> float:
        low, high = 0.0, 1.0
        for _ in range(100):
            mid = (low + high) / 2
            if binomial_cdf(cdf_k, n, mid) > target:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    lower = 0.0 if k == 0 else solve(1 - alpha / 2, k - 1)
    upper = 1.0 if k == n else solve(alpha / 2, k)
    return lower, upper


def rates_at_threshold(
    labels: np.ndarray, probs: np.ndarray, threshold: float, weights: np.ndarray
) -> tuple[float, float]:
    """Weighted (TPR, FPR) when rows with prob >= threshold are flagged."""
    flagged, positive = probs >= threshold, labels == 1
    tpr = (weights * (flagged & positive)).sum() / (weights * positive).sum()
    fpr = (weights * (flagged & ~positive)).sum() / (weights * ~positive).sum()
    return float(tpr), float(fpr)


def group_bootstrap_weights(groups: np.ndarray, n: int, seed: int) -> Iterator[np.ndarray]:
    """Row weights for `n` resamples of whole groups with replacement."""
    codes, uniques = pd.factorize(groups)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        counts = np.bincount(rng.integers(0, uniques.size, uniques.size), minlength=uniques.size)
        yield counts[codes]


def exact_bounds(tp: int, positives: int, fp: int, negatives: int) -> dict:
    """Row-level Clopper-Pearson CIs (ignore grouping) and the precision range they imply."""
    tpr_low, tpr_high = clopper_pearson(tp, positives)
    fpr_low, fpr_high = clopper_pearson(fp, negatives)
    return {
        "tpr": [round(tpr_low, 5), round(tpr_high, 5)],
        "fpr": [round(fpr_low, 6), round(fpr_high, 6)],
        "precision_at_1_in_1000_range": [
            round(precision_at_prevalence(tpr_low, fpr_high), 4),
            round(precision_at_prevalence(tpr_high, fpr_low), 4),
        ],
    }


def false_positive_groups(
    labels: np.ndarray, probs: np.ndarray, groups: np.ndarray, threshold: float, top_n: int = 5
) -> dict:
    """How concentrated the false positives at `threshold` are by group."""
    clean = pd.DataFrame({"group": groups, "flagged": probs >= threshold})[labels == 0]
    per_group = clean.groupby("group")["flagged"].agg(false_positives="sum", clean_rows="size")
    hit = per_group[per_group["false_positives"] > 0]
    hit = hit.sort_values("false_positives", ascending=False)
    return {
        "groups_with_false_positives": len(hit),
        "top_groups": [
            {
                "group": g,
                "false_positives": int(r["false_positives"]),
                "clean_rows": int(r["clean_rows"]),
            }
            for g, r in hit.head(top_n).iterrows()
        ],
    }


def threshold_carry_over(
    labels: np.ndarray, probs: np.ndarray, folds: np.ndarray, threshold: float, target: float
) -> list[dict]:
    """Per fold model: TPR/FPR at the pooled threshold, and that fold's own threshold."""
    rows = []
    for fold in np.unique(folds):
        y, p = labels[folds == fold], probs[folds == fold]
        tpr, fpr = rates_at_threshold(y, p, threshold, np.ones(y.size))
        rows.append(
            {
                "fold": int(fold),
                "clean_rows": int((y == 0).sum()),
                "tpr_at_pooled_threshold": round(tpr, 4),
                "fpr_at_pooled_threshold": round(fpr, 5),
                "own_threshold_for_target": round(operating_point(y, p, target)[0], 5),
            }
        )
    return rows


def interval(values: list[float]) -> list[float]:
    """Percentile 95% interval, ignoring NaN."""
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return [round(float(low), 5), round(float(high), 5)]


def headline(labels: np.ndarray, probs: np.ndarray, groups: np.ndarray, folds: np.ndarray) -> dict:
    """Point estimates and group-bootstrap intervals for each target FPR."""
    ones = np.ones(labels.size)
    thresholds = {t: operating_point(labels, probs, t)[0] for t in TARGET_FPRS}
    boot = {
        t: {"tpr": [], "fpr": [], "precision": [], "recall_reoptimised": []} for t in TARGET_FPRS
    }
    aucs = []
    for w in group_bootstrap_weights(groups, N_BOOTSTRAP, SEED):
        aucs.append(roc_auc_score(labels, probs, sample_weight=w))
        fpr_curve, tpr_curve, _ = roc_curve(labels, probs, sample_weight=w)
        for target, threshold in thresholds.items():
            tpr, fpr = rates_at_threshold(labels, probs, threshold, w)
            boot[target]["tpr"].append(tpr)
            boot[target]["fpr"].append(fpr)
            boot[target]["precision"].append(precision_at_prevalence(tpr, fpr))
            boot[target]["recall_reoptimised"].append(tpr_curve[fpr_curve <= target].max())

    result = {
        "rows": int(labels.size),
        "phishing_rows": int(labels.sum()),
        "clean_rows": int((labels == 0).sum()),
        "groups": int(pd.Series(groups).nunique()),
        "auc": round(float(roc_auc_score(labels, probs)), 4),
        "auc_ci95": interval(aucs),
    }
    for target, threshold in thresholds.items():
        tpr, fpr = rates_at_threshold(labels, probs, threshold, ones)
        flagged = probs >= threshold
        tp, fp = int((flagged & (labels == 1)).sum()), int((flagged & (labels == 0)).sum())
        b = boot[target]
        result[f"target_fpr_{target}"] = {
            "threshold": round(threshold, 5),
            "true_positives": tp,
            "false_positives": fp,
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 5),
            "precision_at_1_in_1000": round(precision_at_prevalence(tpr, fpr), 4),
            "fixed_threshold_ci95": {
                "tpr": interval(b["tpr"]),
                "fpr": interval(b["fpr"]),
                "precision_at_1_in_1000": interval(b["precision"]),
            },
            "recall_at_target_fpr_reoptimised_ci95": interval(b["recall_reoptimised"]),
            "exact_binomial_ci95": exact_bounds(
                tp, int(labels.sum()), fp, int((labels == 0).sum())
            ),
            "false_positive_concentration": false_positive_groups(labels, probs, groups, threshold),
            "threshold_carry_over_by_fold": threshold_carry_over(
                labels, probs, folds, threshold, target
            ),
        }
    return result


def frame_arrays(oof: pd.DataFrame, prob_column: str = PRIMARY_COLUMN) -> tuple[np.ndarray, ...]:
    return tuple(oof[c].to_numpy() for c in ("label", prob_column, "group", "fold"))


def main() -> None:
    oof = pd.read_csv(ROOT / "data" / "oof_predictions.csv")
    without = oof[oof["group"] != ABLATION_GROUP]
    report = {
        "source": "data/oof_predictions.csv (src.train, raw URL, platform-level folds, ADR 0002)",
        "prevalence_assumed": PREVALENCE,
        "bootstrap": (
            f"{N_BOOTSTRAP} resamples of platform-level groups, seed {SEED}, percentile 95% CI;"
            " OOF models held fixed, so CIs cover evaluation-sample variation, not retraining"
        ),
        "threshold_rule": (
            "highest TPR with OOF FPR <= target; operating point FPR <= 1%,"
            " FPR <= 0.1% reported as a scenario (ADR 0003)"
        ),
        "exact_ci_note": "Clopper-Pearson on row counts; ignores grouping, so narrower than real",
        "all_groups": headline(*frame_arrays(oof)),
        "scored_without_duilawyeryork": headline(*frame_arrays(without)),
        "probe_without_scheme_and_www": headline(*frame_arrays(oof, PROBE_COLUMN)),
    }
    out = ROOT / "results" / "oof_headline.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
