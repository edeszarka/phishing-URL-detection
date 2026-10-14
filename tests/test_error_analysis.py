import numpy as np
import pandas as pd

from src.error_analysis import outcome, url_properties


def test_outcome_uses_inclusive_threshold():
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.5, 0.4, 0.5, 0.4])
    assert outcome(labels, probs, 0.5).tolist() == ["TP", "FN", "FP", "TN"]


def test_url_properties_flags():
    urls = pd.Series(["https://www.a.com/", "http://1.2.3.4/login", "b.org/x?y=1"])
    props = url_properties(urls)
    assert props["https"].tolist() == [True, False, False]
    assert props["www"].tolist() == [True, False, False]
    assert props["bare_host"].tolist() == [True, False, False]
    assert props["ip_host"].tolist() == [False, True, False]
