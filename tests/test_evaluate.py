import numpy as np

from src.evaluate import evaluate_block, safe_auc
from src.oof_metrics import exact_bounds

THRESHOLDS = {"0.01": 0.5, "0.001": 0.9}


def frame(n_groups: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array([0, 1] * n_groups)
    probs = np.where(labels == 1, 0.8, 0.2)
    groups = np.repeat(np.arange(n_groups), 2).astype(str)
    return labels, probs, groups


def test_empty_block_has_only_row_count():
    empty = np.array([])
    assert evaluate_block(empty, empty, empty, THRESHOLDS) == {"rows": 0}


def test_single_class_block_has_no_auc_or_precision():
    labels, probs, groups = frame()
    keep = labels == 0
    block = evaluate_block(labels[keep], probs[keep], groups[keep], THRESHOLDS)
    assert block["auc"] is None
    assert block["target_fpr_0.01"]["precision_at_1_in_1000"] is None
    assert block["target_fpr_0.01"]["exact_ci_joint95_alpha0.025"] is None


def test_zero_false_positives_flags_degenerate_bootstrap():
    labels, probs, groups = frame()
    point = evaluate_block(labels, probs, groups, THRESHOLDS)["target_fpr_0.01"]
    assert point["false_positives"] == 0
    assert point["bootstrap_degenerate"] is True
    assert point["exact_ci_joint95_alpha0.025"]["fpr"][1] > 0


def test_nothing_flagged_gives_null_precision_not_nan():
    labels, probs, groups = frame()
    point = evaluate_block(labels, probs, groups, THRESHOLDS)["target_fpr_0.001"]
    assert point["true_positives"] == point["false_positives"] == 0
    assert point["precision_at_1_in_1000"] is None
    assert point["group_bootstrap_ci95"]["precision_at_1_in_1000"] == [None, None]


def test_joint_exact_range_is_wider_than_per_rate_range():
    joint = exact_bounds(100, 400, 4, 400, alpha=0.025)["precision_at_1_in_1000_range"]
    single = exact_bounds(100, 400, 4, 400)["precision_at_1_in_1000_range"]
    assert joint[0] <= single[0] and joint[1] >= single[1]


def test_safe_auc_is_nan_for_one_class():
    assert np.isnan(safe_auc(np.zeros(3), np.array([0.1, 0.2, 0.3])))
