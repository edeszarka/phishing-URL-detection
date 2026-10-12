"""Headline OOF metrics with group-bootstrap CIs and precision at 1:1000.

Run as `uv run python -m src.oof_metrics` after src.train; reads
data/oof_predictions.csv and writes results/oof_headline.json.
"""

import json
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
PROB_COLUMN = "prob_raw_url"


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float = PREVALENCE) -> float:
    """Precision implied by TPR/FPR when a `prevalence` share of URLs is phishing."""
    flagged = tpr * prevalence + fpr * (1 - prevalence)
    return tpr * prevalence / flagged if flagged > 0 else float("nan")


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


def interval(values: list[float]) -> list[float]:
    """Percentile 95% interval, ignoring NaN."""
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return [round(float(low), 5), round(float(high), 5)]


def headline(labels: np.ndarray, probs: np.ndarray, groups: np.ndarray) -> dict:
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
        b = boot[target]
        result[f"target_fpr_{target}"] = {
            "threshold": round(threshold, 5),
            "true_positives": int((flagged & (labels == 1)).sum()),
            "false_positives": int((flagged & (labels == 0)).sum()),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 5),
            "precision_at_1_in_1000": round(precision_at_prevalence(tpr, fpr), 4),
            "fixed_threshold_ci95": {
                "tpr": interval(b["tpr"]),
                "fpr": interval(b["fpr"]),
                "precision_at_1_in_1000": interval(b["precision"]),
            },
            "recall_at_target_fpr_reoptimised_ci95": interval(b["recall_reoptimised"]),
        }
    return result


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
        "threshold_rule": "highest TPR with OOF FPR <= target (candidate points, not yet adopted)",
        "all_groups": headline(
            oof["label"].to_numpy(), oof[PROB_COLUMN].to_numpy(), oof["group"].to_numpy()
        ),
        "scored_without_duilawyeryork": headline(
            without["label"].to_numpy(),
            without[PROB_COLUMN].to_numpy(),
            without["group"].to_numpy(),
        ),
    }
    out = ROOT / "results" / "oof_headline.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
