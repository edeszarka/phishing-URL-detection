"""Single-URL phishing prediction: `from predict import predict`.

Loads the frozen model (ADR 0003) from models/model.joblib next to this file. Nothing
is trained or downloaded on import. Invalid-input policy (ADR 0005): a non-str input or
a string that is empty after strip() is not scored and returns
{"label": "clean", "probability": 0.0}. Here 0.0 means "not scored", not "certainly
clean". Strings longer than 4,000 characters are cut before scoring. Never raises.

`probability` is the logistic-regression score for the cleaned training population
(48.9% phishing), not calibrated to deployment prevalence.
"""

import json
import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:  # the pickled pipeline references src.features
    sys.path.insert(0, str(ROOT))

NOT_SCORED = 0.0
_FINAL_MODEL_JSON = ROOT / "results" / "final_model.json"


def installed_versions() -> dict[str, str]:
    import joblib
    import numpy
    import scipy
    import sklearn

    return {
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def version_mismatches(recorded: dict[str, str], installed: dict[str, str]) -> list[str]:
    """Libraries whose installed version differs from the one the model was saved with."""
    return [
        f"{name} {recorded[name]} (saved) vs {installed.get(name)} (installed)"
        for name in recorded
        if installed.get(name) != recorded[name]
    ]


def _load() -> tuple[object | None, float, list[str]]:
    """(model, operating threshold, version mismatches); (None, inf, []) on any failure."""
    try:
        import joblib

        meta = json.loads(_FINAL_MODEL_JSON.read_text())
        mismatches = version_mismatches(meta["library_versions"], installed_versions())
        spec = meta["models"]["model"]
        threshold = float(spec["thresholds"][meta["operating_target_fpr"]])
        return joblib.load(ROOT / spec["path"]), threshold, mismatches
    except Exception as exc:  # never raise on import; every input is then not scored
        warnings.warn(f"model not loaded, all inputs will be not scored: {exc!r}", stacklevel=2)
        return None, math.inf, []


_MODEL, THRESHOLD, _MISMATCHES = _load()
if _MISMATCHES:  # warned outside _load so a warning filter cannot disable the model
    warnings.warn("model saved with other versions: " + "; ".join(_MISMATCHES), stacklevel=2)


def is_scorable(url: object) -> bool:
    return isinstance(url, str) and url.strip() != ""


def score_urls(urls: list[str]) -> list[float]:
    """Phishing probability per input; NOT_SCORED for invalid inputs or any failure."""
    scores = [NOT_SCORED] * len(urls)
    valid = [i for i, url in enumerate(urls) if is_scorable(url)]
    if _MODEL is None or not valid:
        return scores
    try:
        probs = _MODEL.predict_proba([urls[i] for i in valid])[:, 1]
    except Exception:
        return scores
    for i, p in zip(valid, probs, strict=True):
        p = float(p)
        scores[i] = p if math.isfinite(p) and 0.0 <= p <= 1.0 else NOT_SCORED
    return scores


def label_for(probability: float, threshold: float) -> str:
    return "phishing" if probability >= threshold else "clean"


def predict(url: str) -> dict:
    """{"label": "phishing" | "clean", "probability": float in [0, 1]}."""
    try:
        probability = score_urls([url])[0]
    except Exception:
        probability = NOT_SCORED
    return {"label": label_for(probability, THRESHOLD), "probability": probability}
