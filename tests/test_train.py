import pickle

import numpy as np
import pandas as pd

from src.features import (
    MAX_URL_CHARS,
    build_pipeline,
    prepare_url,
    prepare_url_without_scheme_and_www,
)
from src.train import assign_folds, recall_at_fpr


def test_prepare_url_coerces_any_input():
    assert prepare_url(None) == ""
    assert prepare_url(42) == "42"
    assert prepare_url("  http://a.com  ") == "http://a.com"
    assert len(prepare_url("x" * 10_000)) == MAX_URL_CHARS


def test_strip_variant_removes_only_leading_scheme_and_www():
    assert prepare_url_without_scheme_and_www("HTTPS://WWW.Ex.com/www.x") == "Ex.com/www.x"
    assert prepare_url_without_scheme_and_www("ex.com/http://y") == "ex.com/http://y"


def test_pipeline_keeps_case_and_pickles_without_main():
    urls = pd.Series(
        ["http://a.com/Login", "http://b.org/x", "http://c.tk/Verify", "http://d.net/"]
    )
    model = build_pipeline().fit(urls, [1, 0, 1, 0])
    assert "Log" in model.named_steps["tfidf"].vocabulary_
    blob = pickle.dumps(model)
    assert b"__main__" not in blob
    assert pickle.loads(blob).predict_proba(["", None])[:, 1].shape == (2,)


def test_assign_folds_keeps_groups_disjoint():
    groups = pd.Series([f"g{i // 3}" for i in range(60)])
    labels = pd.Series([i // 3 % 2 for i in range(60)])
    folds = assign_folds(labels, groups)
    assert (pd.Series(folds).groupby(groups).nunique() == 1).all()
    assert set(folds) == set(range(5))


def test_recall_at_fpr_respects_limit():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.8, 0.95, 0.99])
    result = recall_at_fpr(labels, probs, max_fpr=0.0)
    assert result["fpr"] == 0.0
    assert result["recall"] == 0.5
