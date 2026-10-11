"""Data-quality counts for data/train.csv (dry run: nothing is dropped yet).

Normalisation follows the rule used to build external_test.csv: lower-case,
strip scheme, leading "www." and trailing slashes.
"""

import json
import re
from pathlib import Path

import pandas as pd

from src.groups import group_key

ROOT = Path(__file__).resolve().parent.parent
_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://")
_NONPRINTABLE = re.compile(r"[\x00-\x1f\x7f]")


def normalize_url(url: str) -> str:
    text = url.strip().lower()
    text = _SCHEME.sub("", text)
    text = text.removeprefix("www.")
    return text.rstrip("/")


def load_raw(path: Path) -> pd.DataFrame:
    """Read every cell as a string so malformed values stay visible."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def duplicate_counts(df: pd.DataFrame, key: pd.Series) -> dict:
    """Redundant copies and label conflicts when rows are compared by `key`."""
    frame = pd.DataFrame({"key": key, "label": df["label"]})
    labels_per_key = frame.groupby("key")["label"].nunique()
    conflicting = labels_per_key[labels_per_key > 1].index
    in_conflict = frame["key"].isin(conflicting)
    consistent = frame[~in_conflict]
    return {
        "redundant_copies_same_label": int(consistent.duplicated().sum()),
        "conflicting_keys": int(conflicting.size),
        "rows_in_conflicting_keys": int(in_conflict.sum()),
    }


def by_label(df: pd.DataFrame, mask: pd.Series) -> dict:
    return {f"label_{k}": int(v) for k, v in df.loc[mask, "label"].value_counts().items()}


def quality_counts(df: pd.DataFrame) -> dict:
    url = df["url"]
    kinds = url.map(lambda u: group_key(u)[1])
    has_scheme = url.str.strip().str.lower().str.match(_SCHEME)
    return {
        "rows": len(df),
        "label_values": {k: int(v) for k, v in df["label"].value_counts().items()},
        "malformed": {
            "blank_url": int((url.str.strip() == "").sum()),
            "label_not_0_or_1": int((~df["label"].isin(["0", "1"])).sum()),
            "surrounding_whitespace": int((url != url.str.strip()).sum()),
            "internal_whitespace": int(url.str.strip().str.contains(r"\s").sum()),
            "nonprintable_chars": int(url.str.contains(_NONPRINTABLE).sum()),
            "non_ascii": int((~url.map(str.isascii)).sum()),
            "no_registered_domain_or_ip": int((kinds == "fallback").sum()),
        },
        "scheme_present_by_label": by_label(df, has_scheme),
        "scheme_absent_by_label": by_label(df, ~has_scheme),
        "exact_url": duplicate_counts(df, url),
        "normalized_url": duplicate_counts(df, url.map(normalize_url)),
    }


def main() -> None:
    train = load_raw(ROOT / "data" / "train.csv")
    counts = {"input": "data/train.csv (raw)", **quality_counts(train)}
    out = ROOT / "results" / "cleaning_counts.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
