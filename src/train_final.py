"""Refit the frozen baseline and the shortcut probe on all cleaned rows (ADR 0003).

Run as `uv run python -m src.train_final` after src.train. Reads the cleaned
training data and data/oof_predictions.csv (for the exact OOF thresholds), writes
models/model.joblib, models/probe.joblib and results/final_model.json.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from src.clean import load_clean_train
from src.features import MAX_URL_CHARS, build_pipeline
from src.train import operating_point

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "model": {"column": "prob_raw_url", "strip": False},
    "probe": {"column": "prob_without_scheme_and_www", "strip": True},
}
# Thresholds fixed in ADR 0003, rounded to 5 decimals; the exact values are recomputed
# from the OOF predictions and must round to these.
ADR_THRESHOLDS = {
    "model": {"0.01": 0.74956, "0.001": 0.84127},
    "probe": {"0.01": 0.69256, "0.001": 0.88479},
}


def oof_thresholds(oof: pd.DataFrame, column: str) -> dict[str, float]:
    """Unrounded OOF threshold per target FPR; rows with prob >= threshold are flagged."""
    labels, probs = oof["label"].to_numpy(), oof[column].to_numpy()
    return {str(fpr): operating_point(labels, probs, fpr)[0] for fpr in (0.01, 0.001)}


def library_versions() -> dict[str, str]:
    return {
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    df = load_clean_train()
    oof = pd.read_csv(ROOT / "data" / "oof_predictions.csv")
    if len(oof) != len(df) or not (oof["label"].to_numpy() == df["label"].to_numpy()).all():
        raise SystemExit("oof_predictions.csv does not match the cleaned data; rerun src.train")
    (ROOT / "models").mkdir(exist_ok=True)
    report: dict = {
        "input": "data/train.csv after cleaning (ADR 0001)",
        "rows": len(df),
        "phishing_rows": int(df["label"].sum()),
        "clean_rows": int((df["label"] == 0).sum()),
        "max_url_chars": MAX_URL_CHARS,
        "operating_target_fpr": "0.01",
        "library_versions": library_versions(),
        "models": {},
    }
    for name, spec in MODELS.items():
        thresholds = oof_thresholds(oof, spec["column"])
        rounded = {k: round(v, 5) for k, v in thresholds.items()}
        if rounded != ADR_THRESHOLDS[name]:
            raise SystemExit(f"{name}: OOF thresholds {rounded} differ from ADR 0003")
        model = build_pipeline(spec["strip"]).fit(df["url"], df["label"])
        path = ROOT / "models" / f"{name}.joblib"
        joblib.dump(model, path)
        report["models"][name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "input": "without_scheme_and_www" if spec["strip"] else "raw_url",
            "thresholds": thresholds,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    out = ROOT / "results" / "final_model.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
