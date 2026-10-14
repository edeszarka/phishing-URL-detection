"""Final evaluation of the frozen model and probe (ADR 0003). The ONLY file that reads
the labels of data/test.csv and data/external_test.csv.

  uv run python -m src.evaluate --dry-run   # OOF predictions only; must reproduce
                                            # results/oof_headline.json; reads no test file
  uv run python -m src.evaluate             # one real run -> results/evaluation.json
  uv run python -m src.evaluate --rerun-reason "..."   # logged rerun

Per set, at each fixed threshold: raw counts, TPR, FPR, precision at 1:1000. Headline
interval is the platform-group bootstrap on precision directly; Clopper-Pearson ranges
(alpha 0.025 per rate for >= 95% joint coverage; 0.05 per rate for comparison with
ADR 0003) are cross-checks. They ignore grouping and pair interval ends, so they are
conservative in one way and too narrow in another. Repeated per overlap bin and with
duilawyeryork.com rows removed from scoring.
"""

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from predict import score_urls
from src.features import SEED
from src.groups import platform_group_key
from src.oof_metrics import (
    N_BOOTSTRAP,
    PREVALENCE,
    exact_bounds,
    false_positive_groups,
    group_bootstrap_weights,
    interval,
    precision_at_prevalence,
    rates_at_threshold,
)
from src.overlap import EVAL_SETS, strongest_levels
from src.train import ABLATION_GROUP

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "evaluation.json"
RUN_LOG = ROOT / "results" / "evaluation_runs.jsonl"
OVERLAP_BINS = {
    "same_page": ("exact", "normalized_case", "normalized"),
    "same_site": ("host_first_path", "hostname"),
    "same_group": ("tenant_group", "platform_group"),
    "none": ("none",),
}
ROLES = {"0.01": "operating point (ADR 0003)", "0.001": "scenario only, evidence needed"}


def final_model_meta() -> dict:
    return json.loads((ROOT / "results" / "final_model.json").read_text())


def safe_auc(labels: np.ndarray, probs: np.ndarray, weights: np.ndarray | None = None) -> float:
    """ROC-AUC, or NaN when the (weighted) sample has only one class."""
    w = np.ones(labels.size) if weights is None else weights
    if w[labels == 1].sum() == 0 or w[labels == 0].sum() == 0:
        return float("nan")
    return float(roc_auc_score(labels, probs, sample_weight=w))


def evaluate_block(
    labels: np.ndarray, probs: np.ndarray, groups: np.ndarray, thresholds: dict[str, float]
) -> dict:
    """Metrics at fixed thresholds with group-bootstrap and exact intervals."""
    if labels.size == 0:
        return {"rows": 0}
    boot: dict = {t: {"tpr": [], "fpr": [], "precision": []} for t in thresholds}
    aucs = []
    for w in group_bootstrap_weights(groups, N_BOOTSTRAP, SEED):
        aucs.append(safe_auc(labels, probs, w))
        for target, threshold in thresholds.items():
            tpr, fpr = rates_at_threshold(labels, probs, threshold, w)
            boot[target]["tpr"].append(tpr)
            boot[target]["fpr"].append(fpr)
            boot[target]["precision"].append(precision_at_prevalence(tpr, fpr))
    auc = safe_auc(labels, probs)
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    result = {
        "rows": int(labels.size),
        "phishing_rows": positives,
        "clean_rows": negatives,
        "groups": int(pd.Series(groups).nunique()),
        "auc": None if np.isnan(auc) else round(auc, 4),
        "auc_ci95": None if np.isnan(auc) else interval(aucs),
    }
    for target, threshold in thresholds.items():
        flagged = probs >= threshold
        tp, fp = int((flagged & (labels == 1)).sum()), int((flagged & (labels == 0)).sum())
        tpr, fpr = rates_at_threshold(labels, probs, threshold, np.ones(labels.size))
        b = boot[target]
        fpr_ci = interval(b["fpr"]) if negatives else None
        result[f"target_fpr_{target}"] = {
            "role": ROLES[target],
            "threshold": round(threshold, 5),
            "true_positives": tp,
            "false_positives": fp,
            "tpr": round(tpr, 4) if positives else None,
            "fpr": round(fpr, 5) if negatives else None,
            "precision_at_1_in_1000": round(precision_at_prevalence(tpr, fpr), 4)
            if positives and negatives
            else None,
            "group_bootstrap_ci95": {
                "tpr": interval(b["tpr"]) if positives else None,
                "fpr": fpr_ci,
                "precision_at_1_in_1000": interval(b["precision"])
                if positives and negatives
                else None,
            },
            # Zero-width FPR interval (e.g. no false positives): rely on the exact bound.
            "bootstrap_degenerate": fpr_ci is None or fpr_ci[0] == fpr_ci[1],
            "exact_ci_joint95_alpha0.025": exact_bounds(tp, positives, fp, negatives, 0.025)
            if positives and negatives
            else None,
            "exact_binomial_ci95": exact_bounds(tp, positives, fp, negatives)
            if positives and negatives
            else None,
            "false_positive_concentration": false_positive_groups(labels, probs, groups, threshold),
        }
    return finite_or_none(result)


def finite_or_none(value: object) -> object:
    """Replace NaN/inf (e.g. precision when nothing is flagged) with None for valid JSON."""
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_or_none(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def thresholds_for(name: str) -> dict[str, float]:
    return final_model_meta()["models"][name]["thresholds"]


def headline_fields(block: dict) -> dict:
    """The fields of an evaluate_block that oof_metrics.headline also writes."""
    keep = ("rows", "phishing_rows", "clean_rows", "groups", "auc", "auc_ci95")
    out = {k: block[k] for k in keep}
    for key in (k for k in block if k.startswith("target_fpr_")):
        b = block[key]
        out[key] = {
            "threshold": b["threshold"],
            "true_positives": b["true_positives"],
            "false_positives": b["false_positives"],
            "tpr": b["tpr"],
            "fpr": b["fpr"],
            "precision_at_1_in_1000": b["precision_at_1_in_1000"],
            "fixed_threshold_ci95": b["group_bootstrap_ci95"],
            "exact_binomial_ci95": b["exact_binomial_ci95"],
            "false_positive_concentration": b["false_positive_concentration"],
        }
    return out


def expected_fields(reference: dict) -> dict:
    """oof_headline.json block with fields evaluate_block does not compute removed."""
    out = {k: v for k, v in reference.items() if not k.startswith("target_fpr_")}
    for key in (k for k in reference if k.startswith("target_fpr_")):
        drop = ("recall_at_target_fpr_reoptimised_ci95", "threshold_carry_over_by_fold")
        out[key] = {k: v for k, v in reference[key].items() if k not in drop}
    return out


def dry_run() -> bool:
    """Recompute the OOF headline with evaluate_block; True if every field matches."""
    oof = pd.read_csv(ROOT / "data" / "oof_predictions.csv")
    reference = json.loads((ROOT / "results" / "oof_headline.json").read_text())
    without = oof[oof["group"] != ABLATION_GROUP]
    cases = {
        "all_groups": (oof, "prob_raw_url", "model"),
        "scored_without_duilawyeryork": (without, "prob_raw_url", "model"),
        "probe_without_scheme_and_www": (oof, "prob_without_scheme_and_www", "probe"),
    }
    ok = True
    for name, (frame, column, model) in cases.items():
        block = evaluate_block(
            frame["label"].to_numpy(),
            frame[column].to_numpy(),
            frame["group"].to_numpy(),
            thresholds_for(model),
        )
        match = headline_fields(block) == expected_fields(reference[name])
        ok &= match
        cp = block["target_fpr_0.01"]["exact_ci_joint95_alpha0.025"]
        print(
            f"{name}: {'MATCH' if match else 'MISMATCH'}; joint-95% exact range at 1% FPR:"
            f" {cp['precision_at_1_in_1000_range']}"
        )
        if not match:
            print(json.dumps(headline_fields(block), indent=1)[:3000])
    return ok


def load_labelled(name: str) -> pd.DataFrame:
    """URL and label of an evaluation set. Labels are read here and nowhere else."""
    frame = pd.read_csv(ROOT / "data" / EVAL_SETS[name], dtype={"url": str}, keep_default_na=False)
    if not frame["label"].isin([0, 1]).all():
        raise SystemExit(f"{name}: labels outside {{0, 1}}")
    return frame


def evaluate_set(name: str, train_urls: pd.Series, probe: object) -> dict:
    frame = load_labelled(name)
    urls, labels = frame["url"], frame["label"].to_numpy()
    groups = urls.map(platform_group_key).to_numpy()
    model_probs = np.array(score_urls(urls.tolist()))
    probe_probs = np.array(score_urls(urls.tolist(), probe))
    level = strongest_levels(train_urls, urls)
    model_thr, probe_thr = thresholds_for("model"), thresholds_for("probe")
    by_bin = {}
    for bin_name, levels in OVERLAP_BINS.items():
        mask = level.isin(levels).to_numpy()
        by_bin[bin_name] = evaluate_block(labels[mask], model_probs[mask], groups[mask], model_thr)
    keep = groups != ABLATION_GROUP
    return {
        "model": evaluate_block(labels, model_probs, groups, model_thr),
        "model_by_overlap_bin": by_bin,
        "model_scored_without_duilawyeryork": {
            "rows_removed": int((~keep).sum()),
            **evaluate_block(labels[keep], model_probs[keep], groups[keep], model_thr),
        },
        "probe_without_scheme_and_www": evaluate_block(labels, probe_probs, groups, probe_thr),
    }


def git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def real_run(rerun_reason: str | None) -> None:
    if OUT.exists() and not rerun_reason:
        raise SystemExit(f"{OUT.name} exists; a rerun needs --rerun-reason (ADR 0003, decision 8)")
    from src.clean import load_clean_train

    run = {
        "utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "reason": rerun_reason or "first run",
    }
    with RUN_LOG.open("a") as log:  # logged before any label is read
        log.write(json.dumps(run) + "\n")
    meta = final_model_meta()
    probe = joblib.load(ROOT / meta["models"]["probe"]["path"])
    train_urls = load_clean_train()["url"]
    report = {
        "run": run,
        "prevalence_assumed": PREVALENCE,
        "bootstrap": f"{N_BOOTSTRAP} resamples of platform-level groups, seed {SEED}",
        "headline_interval": "group_bootstrap_ci95; exact ranges are cross-checks",
        "overlap_bins": {k: list(v) for k, v in OVERLAP_BINS.items()},
        "thresholds": {name: spec["thresholds"] for name, spec in meta["models"].items()},
    }
    for name in EVAL_SETS:
        report[name] = evaluate_set(name, train_urls, probe)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-reason")
    args = parser.parse_args()
    if args.dry_run:
        raise SystemExit(0 if dry_run() else 1)
    real_run(args.rerun_reason)


if __name__ == "__main__":
    main()
