import numpy as np

from src.roc_calibration import calibration, low_fpr_summary, reliability_table


def test_perfect_ranking_has_full_tpr_and_pauc():
    labels = np.array([0] * 50 + [1] * 50)
    probs = np.linspace(0, 1, 100)
    summary = low_fpr_summary(labels, probs)
    assert set(summary["tpr_at_fpr"].values()) == {1.0}
    assert summary["partial_auc_max_fpr_0.01"] == 1.0


def test_reliability_bins_cover_all_rows_and_include_score_one():
    labels = np.array([0, 0, 1, 1, 1])
    probs = np.array([0.05, 0.15, 0.95, 1.0, 0.55])
    rows = reliability_table(labels, probs)
    assert sum(r["rows"] for r in rows) == 5
    assert rows[-1] == {"bin": "0.9-1.0", "rows": 2, "mean_score": 0.975, "phishing_share": 1.0}


def test_brier_of_perfect_scores_is_zero():
    labels = np.array([0, 1, 0, 1])
    assert calibration(labels, labels.astype(float))["brier_score"] == 0.0
