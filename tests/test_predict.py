import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import predict as predict_module
from predict import NOT_SCORED, label_for, predict, score_urls, version_mismatches

ROOT = Path(__file__).resolve().parent.parent
UNUSUAL = [
    "https://www.PayPal-secure.login.tk/webscr?cmd=x",
    "example.com",
    "http://192.168.0.1/login",
    "  http://padded.com/  ",
    "http://exämple.de/päth?q=ü",
    "not a url at all",
    "://",
    "\x00\x1f\x7f",
    "x" * 5000,
    "http://a.com/" + "?" * 10_000,
]
NOT_SCORABLE = [None, 12345, 1.5, b"http://bytes.com", "", "   ", "\n\t", ["http://a.com"]]


def assert_contract(result: dict) -> None:
    assert set(result) == {"label", "probability"}
    assert result["label"] in {"phishing", "clean"}
    assert type(result["probability"]) is float
    assert math.isfinite(result["probability"]) and 0.0 <= result["probability"] <= 1.0


def test_model_loaded_with_frozen_threshold():
    assert predict_module._MODEL is not None
    assert round(predict_module.THRESHOLD, 5) == 0.74956


@pytest.mark.parametrize("url", UNUSUAL)
def test_unusual_strings_are_scored(url):
    result = predict(url)
    assert_contract(result)
    assert result["probability"] > 0.0


@pytest.mark.parametrize("url", NOT_SCORABLE)
def test_invalid_inputs_are_not_scored(url):
    assert predict(url) == {"label": "clean", "probability": NOT_SCORED}


def test_score_urls_matches_predict_on_mixed_batch():
    inputs = UNUSUAL + NOT_SCORABLE
    batch = score_urls(inputs)
    assert batch == [predict(url)["probability"] for url in inputs]


def test_long_input_is_cut_to_cap():
    head = "http://a.com/" + "b" * 3987
    assert len(head) == 4000
    assert score_urls([head + "c" * 5000]) == score_urls([head])


def test_label_at_threshold_is_phishing():
    assert label_for(0.5, 0.5) == "phishing"
    assert label_for(np.nextafter(0.5, 0.0), 0.5) == "clean"


def test_model_failure_returns_not_scored(monkeypatch):
    class Broken:
        def predict_proba(self, urls):
            raise RuntimeError("boom")

    monkeypatch.setattr(predict_module, "_MODEL", Broken())
    assert predict("http://a.com") == {"label": "clean", "probability": NOT_SCORED}


def test_non_finite_score_returns_not_scored(monkeypatch):
    class Nan:
        def predict_proba(self, urls):
            return np.full((len(urls), 2), np.nan)

    monkeypatch.setattr(predict_module, "_MODEL", Nan())
    assert predict("http://a.com") == {"label": "clean", "probability": NOT_SCORED}


def test_missing_model_returns_not_scored(monkeypatch):
    monkeypatch.setattr(predict_module, "_MODEL", None)
    assert predict("http://a.com") == {"label": "clean", "probability": NOT_SCORED}


def test_version_mismatch_is_listed():
    recorded = {"numpy": "2.5.4", "joblib": "1.6.0"}
    assert version_mismatches(recorded, dict(recorded)) == []
    assert version_mismatches(recorded, {"numpy": "2.0.0", "joblib": "1.6.0"}) == [
        "numpy 2.5.4 (saved) vs 2.0.0 (installed)"
    ]


def test_import_does_not_train_or_read_data():
    code = (
        "import sys, predict; "
        "print(sorted(m for m in sys.modules if m in "
        "{'src.train', 'src.clean', 'src.train_final', 'src.groups'}))"
    )
    out = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"
