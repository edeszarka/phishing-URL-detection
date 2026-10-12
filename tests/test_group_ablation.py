import numpy as np
import pandas as pd

from src.group_ablation import recall_at_fpr, training_masks
from src.train import ABLATION_GROUP


def test_training_masks_cap_keeps_first_rows_per_group():
    groups = pd.Series(["a", "b", "a", "a", ABLATION_GROUP, "b"])
    masks = training_masks(groups, cap=2)
    assert masks["baseline"].all()
    assert masks["without_duilawyeryork"].tolist() == [True, True, True, True, False, True]
    assert masks["cap_2"].tolist() == [True, True, True, False, True, True]


def test_training_masks_without_cap_has_two_variants():
    assert set(training_masks(pd.Series(["a"]), cap=None)) == {"baseline", "without_duilawyeryork"}


def test_recall_at_fpr_reoptimises_threshold():
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.4, 0.5, 0.1])
    # FPR <= 1% allows no clean row flagged, so only the 0.9 row counts.
    assert recall_at_fpr(labels, probs, np.ones(4)) == 0.5
