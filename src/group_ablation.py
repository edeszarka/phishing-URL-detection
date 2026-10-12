"""Paired experiment: does limiting large groups in training change recall at FPR <= 1%?

Variants differ only in which training-fold rows each fold model is fit on. Folds,
scored rows and the threshold rule are shared, so differences are paired:
- baseline: all training-fold rows (the primary model, ADR 0003).
- without_duilawyeryork: ABLATION_GROUP removed from training; its rows are still scored.
- cap_K: at most K rows per platform-level group in training, first in file order.

Groups never span folds, so a global per-group mask restricts only training rows.
Recall at FPR <= 1% is re-optimised per variant (and per bootstrap resample).
Run as `uv run python -m src.group_ablation`; writes results/group_ablation.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.clean import load_clean_train
from src.features import SEED, build_pipeline
from src.groups import cv_group_key
from src.oof_metrics import N_BOOTSTRAP, group_bootstrap_weights, interval
from src.train import ABLATION_GROUP, N_FOLDS, REPORTING_FPR, assign_folds, operating_point

ROOT = Path(__file__).resolve().parent.parent
FOLD_SEEDS = (0, 1, 2, 3, 4)
# Per-group cap for the cap variant; None skips it (value set by decision, not tuned).
CAP: int | None = None


def training_masks(groups: pd.Series, cap: int | None) -> dict[str, np.ndarray]:
    """Rows each variant may train on (row order = file order after cleaning)."""
    masks = {
        "baseline": np.ones(len(groups), dtype=bool),
        "without_duilawyeryork": (groups != ABLATION_GROUP).to_numpy(),
    }
    if cap is not None:
        masks[f"cap_{cap}"] = (groups.groupby(groups).cumcount() < cap).to_numpy()
    return masks


def oof_with_mask(
    urls: pd.Series, labels: pd.Series, folds: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """OOF probabilities for every row; fold models train only on masked rows."""
    probs = np.empty(len(urls))
    for fold in range(N_FOLDS):
        train, val = (folds != fold) & mask, folds == fold
        model = build_pipeline().fit(urls[train], labels[train])
        probs[val] = model.predict_proba(urls[val])[:, 1]
    return probs


def recall_at_fpr(labels: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> float:
    """Highest (weighted) TPR with FPR <= REPORTING_FPR, threshold re-optimised."""
    fpr, tpr, _ = roc_curve(labels, probs, sample_weight=weights)
    return float(tpr[fpr <= REPORTING_FPR].max())


def paired_comparison(labels: np.ndarray, probs: dict, groups: np.ndarray) -> dict:
    """Point recall per variant and group-bootstrap CI of each variant minus baseline."""
    ones = np.ones(labels.size)
    point = {v: recall_at_fpr(labels, p, ones) for v, p in probs.items()}
    diffs: dict[str, list[float]] = {v: [] for v in probs if v != "baseline"}
    for w in group_bootstrap_weights(groups, N_BOOTSTRAP, SEED):
        base = recall_at_fpr(labels, probs["baseline"], w)
        for v in diffs:
            diffs[v].append(recall_at_fpr(labels, probs[v], w) - base)
    return {
        "rows": int(labels.size),
        "auc": {v: round(float(roc_auc_score(labels, p)), 4) for v, p in probs.items()},
        "recall_at_fpr_0.01": {v: round(r, 4) for v, r in point.items()},
        "difference_vs_baseline": {
            v: {
                "point": round(point[v] - point["baseline"], 4),
                "group_bootstrap_ci95": interval(d),
                "share_of_resamples_above_0": round(float(np.mean(np.array(d) > 0)), 4),
            }
            for v, d in diffs.items()
        },
    }


def baseline_operating_point(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Baseline threshold for FPR <= 1% on all rows, with its counts, for one fold seed."""
    threshold, tpr, fpr = operating_point(labels, probs, REPORTING_FPR)
    flagged = probs >= threshold
    return {
        "threshold": round(threshold, 5),
        "true_positives": int((flagged & (labels == 1)).sum()),
        "false_positives": int((flagged & (labels == 0)).sum()),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 5),
    }


def main() -> None:
    df = load_clean_train()
    groups = df["url"].map(cv_group_key)
    urls, labels = df["url"].reset_index(drop=True), df["label"].reset_index(drop=True)
    y, g = labels.to_numpy(), groups.to_numpy()
    masks = training_masks(groups.reset_index(drop=True), CAP)
    scored = {"all_rows": np.ones(y.size, dtype=bool), "without_duilawyeryork": g != ABLATION_GROUP}

    by_seed = {}
    for seed in FOLD_SEEDS:
        folds = assign_folds(labels, groups, seed)
        probs = {v: oof_with_mask(urls, labels, folds, m) for v, m in masks.items()}
        by_seed[f"fold_seed_{seed}"] = {
            pop: paired_comparison(y[keep], {v: p[keep] for v, p in probs.items()}, g[keep])
            for pop, keep in scored.items()
        }
        by_seed[f"fold_seed_{seed}"]["baseline_operating_point_fpr_0.01"] = (
            baseline_operating_point(y, probs["baseline"])
        )
        print(f"fold seed {seed} done")

    report = {
        "question": "does limiting large groups in training change recall at FPR <= 1%?",
        "design": (
            "paired: same folds, scored rows and re-optimised threshold rule per variant;"
            f" {N_BOOTSTRAP} platform-level group resamples (seed {SEED}) per fold seed"
        ),
        "fold_seeds": list(FOLD_SEEDS),
        "training_rows_removed": {v: int((~m).sum()) for v, m in masks.items()},
        "training_configs_evaluated": {
            "before_this_experiment": 1,
            "this_experiment_new": len(masks) - 1,
            "note": "baseline is unchanged; no variant replaces it (ADR 0003)",
        },
        "results": by_seed,
    }
    out = ROOT / "results" / "group_ablation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
