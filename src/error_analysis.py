"""What the frozen model gets wrong on OOF predictions (training labels only).

Run as `uv run python -m src.error_analysis` after src.train. At the operating threshold
from results/final_model.json, compares false negatives with true positives and false
positives with true negatives on simple URL properties. Writes counts and shares to
results/error_analysis_oof.json; the example URLs (supplied data) go to the ignored
data/error_examples_oof.csv. Diagnostic only; nothing here changes the model or threshold.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import load_clean_train
from src.groups import cv_group_key, key_kind

ROOT = Path(__file__).resolve().parent.parent
N_EXAMPLES = 15
TOP_GROUPS = 5
_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)


def outcome(labels: np.ndarray, probs: np.ndarray, threshold: float) -> np.ndarray:
    """'TP', 'FN', 'FP' or 'TN' per row; rows with prob >= threshold are flagged."""
    flagged = probs >= threshold
    return np.select(
        [flagged & (labels == 1), ~flagged & (labels == 1), flagged & (labels == 0)],
        ["TP", "FN", "FP"],
        default="TN",
    )


def url_properties(urls: pd.Series) -> pd.DataFrame:
    """Simple per-URL flags used to describe errors."""
    text = urls.str.strip()
    rest = text.str.replace(_SCHEME, "", regex=True)
    return pd.DataFrame(
        {
            "https": text.str.lower().str.startswith("https://"),
            "www": rest.str.lower().str.startswith("www."),
            "bare_host": ~rest.str.rstrip("/").str.contains(r"[/?#]"),
            "ip_host": urls.map(key_kind) == "ip",
            "length": text.str.len(),
        }
    )


def describe(props: pd.DataFrame, groups: pd.Series) -> dict:
    """Shares, length quantiles and top platform groups for one outcome slice."""
    return {
        "rows": len(props),
        "share_https": round(float(props["https"].mean()), 4),
        "share_www": round(float(props["www"].mean()), 4),
        "share_bare_host": round(float(props["bare_host"].mean()), 4),
        "share_ip_host": round(float(props["ip_host"].mean()), 4),
        "length_q25_q50_q75": [float(q) for q in props["length"].quantile([0.25, 0.5, 0.75])],
        "top_groups": [
            {"group": g, "rows": int(n)} for g, n in groups.value_counts().head(TOP_GROUPS).items()
        ],
    }


def main() -> None:
    df = load_clean_train().reset_index(drop=True)
    oof = pd.read_csv(ROOT / "data" / "oof_predictions.csv")
    groups = df["url"].map(cv_group_key)
    aligned = (oof["label"].to_numpy() == df["label"].to_numpy()).all() and (
        oof["group"].to_numpy() == groups.to_numpy()
    ).all()
    if len(oof) != len(df) or not aligned:
        raise SystemExit("oof_predictions.csv does not match the cleaned data; rerun src.train")
    meta = json.loads((ROOT / "results" / "final_model.json").read_text())
    threshold = meta["models"]["model"]["thresholds"][meta["operating_target_fpr"]]
    probs = oof["prob_raw_url"].to_numpy()
    kind = outcome(df["label"].to_numpy(), probs, threshold)
    props = url_properties(df["url"])
    report = {
        "source": "data/oof_predictions.csv + cleaned train URLs (training labels only)",
        "threshold": round(threshold, 5),
        "note": "diagnostic only; no threshold or model change follows from it (ADR 0003)",
        "counts": {k: int((kind == k).sum()) for k in ("TP", "FN", "FP", "TN")},
        "slices": {
            k: describe(props[kind == k], groups[kind == k]) for k in ("TP", "FN", "FP", "TN")
        },
    }
    out = ROOT / "results" / "error_analysis_oof.json"
    out.write_text(json.dumps(report, indent=2))
    examples = pd.concat(
        [
            pd.DataFrame({"kind": "FP", "url": df["url"], "group": groups, "prob": probs})[
                kind == "FP"
            ].nlargest(N_EXAMPLES, "prob"),
            pd.DataFrame({"kind": "FN", "url": df["url"], "group": groups, "prob": probs})[
                kind == "FN"
            ].nsmallest(N_EXAMPLES, "prob"),
        ]
    )
    examples.to_csv(ROOT / "data" / "error_examples_oof.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
