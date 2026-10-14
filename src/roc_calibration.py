"""Low-FPR ROC summary and calibration on OOF predictions (diagnostics only).

Run as `uv run python -m src.roc_calibration` after src.train. Reads
data/oof_predictions.csv (training labels only) and writes
results/roc_calibration_oof.json. The low-FPR functions are reused by src.evaluate.
Nothing here changes the model or the threshold (ADR 0003).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.train import operating_point

ROOT = Path(__file__).resolve().parent.parent
LOW_FPRS = (0.001, 0.005, 0.01, 0.02, 0.05)
PAUC_MAX_FPR = 0.01
N_BINS = 10
COLUMNS = {"model": "prob_raw_url", "probe": "prob_without_scheme_and_www"}


def low_fpr_summary(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Highest TPR at each FPR budget, and standardised partial AUC up to PAUC_MAX_FPR.

    Points are read off this sample's own ROC curve, so they describe ranking quality,
    not the fixed operating threshold.
    """
    return {
        "tpr_at_fpr": {
            str(fpr): round(operating_point(labels, probs, fpr)[1], 4) for fpr in LOW_FPRS
        },
        "partial_auc_max_fpr_0.01": round(
            float(roc_auc_score(labels, probs, max_fpr=PAUC_MAX_FPR)), 4
        ),
    }


def reliability_table(labels: np.ndarray, probs: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """Mean score vs observed phishing share in equal-width score bins (empty bins skipped)."""
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append(
                {
                    "bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}",
                    "rows": int(mask.sum()),
                    "mean_score": round(float(probs[mask].mean()), 4),
                    "phishing_share": round(float(labels[mask].mean()), 4),
                }
            )
    return rows


def calibration(labels: np.ndarray, probs: np.ndarray) -> dict:
    return {
        "brier_score": round(float(brier_score_loss(labels, probs)), 4),
        "brier_score_constant_base_rate": round(
            float(brier_score_loss(labels, np.full(labels.size, labels.mean()))), 4
        ),
        "reliability": reliability_table(labels, probs),
    }


def main() -> None:
    oof = pd.read_csv(ROOT / "data" / "oof_predictions.csv")
    labels = oof["label"].to_numpy()
    report: dict = {
        "source": "data/oof_predictions.csv (training labels only; platform-level folds)",
        "note": "diagnostic only; no threshold or model change follows from it (ADR 0003)",
        "population": f"cleaned training mix, {int(labels.sum())} of {labels.size} phishing",
    }
    for name, column in COLUMNS.items():
        probs = oof[column].to_numpy()
        report[name] = {**low_fpr_summary(labels, probs), "calibration": calibration(labels, probs)}
    out = ROOT / "results" / "roc_calibration_oof.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
