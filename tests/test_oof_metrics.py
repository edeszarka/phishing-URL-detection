import math

import numpy as np

from src.oof_metrics import (
    binomial_cdf,
    clopper_pearson,
    false_positive_groups,
    group_bootstrap_weights,
    precision_at_prevalence,
    rates_at_threshold,
)


def test_precision_at_prevalence():
    assert math.isclose(precision_at_prevalence(0.5, 0.01, 0.001), 0.0005 / (0.0005 + 0.00999))
    assert precision_at_prevalence(0.5, 0.0) == 1.0
    assert math.isnan(precision_at_prevalence(0.0, 0.0))


def test_rates_at_threshold_uses_weights():
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.2, 0.8, 0.1])
    assert rates_at_threshold(labels, probs, 0.5, np.ones(4)) == (0.5, 0.5)
    assert rates_at_threshold(labels, probs, 0.5, np.array([3, 1, 0, 1])) == (0.75, 0.0)


def test_group_bootstrap_keeps_groups_whole_and_is_seeded():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    first = list(group_bootstrap_weights(groups, 5, seed=0))
    assert all(w[0] == w[1] and w[3] == w[4] == w[5] for w in first)
    assert all(w.sum() > 0 for w in first)
    again = list(group_bootstrap_weights(groups, 5, seed=0))
    assert all((a == b).all() for a, b in zip(first, again, strict=True))


def test_binomial_cdf_edges():
    assert binomial_cdf(3, 10, 0.0) == 1.0
    assert binomial_cdf(10, 10, 1.0) == 1.0
    assert math.isclose(binomial_cdf(0, 2, 0.5), 0.25)


def test_clopper_pearson_known_values():
    # Zero successes in 100: upper bound is 1 - 0.025 ** (1 / 100).
    low, high = clopper_pearson(0, 100)
    assert low == 0.0
    assert math.isclose(high, 1 - 0.025 ** (1 / 100), rel_tol=1e-6)
    # All successes mirrors zero successes.
    low, high = clopper_pearson(100, 100)
    assert high == 1.0
    assert math.isclose(low, 0.025 ** (1 / 100), rel_tol=1e-6)
    low, high = clopper_pearson(4, 4257)
    assert low < 4 / 4257 < high


def test_false_positive_groups_counts_clean_rows_only():
    labels = np.array([0, 0, 0, 1])
    probs = np.array([0.9, 0.9, 0.1, 0.9])
    groups = np.array(["a", "a", "b", "a"])
    result = false_positive_groups(labels, probs, groups, 0.5)
    assert result["groups_with_false_positives"] == 1
    assert result["top_groups"] == [{"group": "a", "false_positives": 2, "clean_rows": 2}]
