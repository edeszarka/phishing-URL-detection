"""Data-quality counts and cleaning rules for data/train.csv.

Normalisation follows the rule used to build external_test.csv: lower-case,
strip scheme, leading "www." and trailing slashes. The normalised key is only
used for matching; cleaned rows keep their original URL string.

Rules, in order (ADR 0001):
1. Drop every row whose normalised key carries both labels.
2. Among the remaining rows, keep the first row per normalised key (file order).
Malformed-looking rows are kept: all parse and all labels are 0/1.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import SEED
from src.groups import group_summary, key_kind, tenant_group_key

ROOT = Path(__file__).resolve().parent.parent
_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)
_HOST_END = re.compile(r"[/?#]")
_NONPRINTABLE = re.compile(r"[\x00-\x1f\x7f]")


def normalize_url(url: str) -> str:
    text = url.strip().lower()
    text = _SCHEME.sub("", text)
    text = text.removeprefix("www.")
    return text.rstrip("/")


def normalize_url_keep_path_case(url: str) -> str:
    """Like normalize_url, but only the host is lower-cased."""
    text = _SCHEME.sub("", url.strip())
    end = _HOST_END.search(text)
    split = end.start() if end else len(text)
    host = text[:split].lower().removeprefix("www.")
    return (host + text[split:]).rstrip("/")


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
    kinds = url.map(key_kind)
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
            "no_domain_or_ip": int((kinds == "fallback").sum()),
        },
        "scheme_present_by_label": by_label(df, has_scheme),
        "scheme_absent_by_label": by_label(df, ~has_scheme),
        "exact_url": duplicate_counts(df, url),
        "normalized_url": duplicate_counts(df, url.map(normalize_url)),
    }


def label_counts(labels: pd.Series) -> dict:
    return {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}


def is_redundant_copy(key: pd.Series, keep: str) -> pd.Series:
    """True for every row except the representative of its key.

    keep: 'first' or 'last' in file order, or 'random' (seeded).
    """
    if keep in ("first", "last"):
        return key.duplicated(keep=keep)
    order = np.random.default_rng(SEED).permutation(len(key))
    return key.iloc[order].duplicated(keep="first").reindex(key.index)


def rule_masks(df: pd.DataFrame, keep: str = "first") -> tuple[pd.Series, pd.Series, pd.Series]:
    """Normalised key, rule-1 mask (conflict) and rule-2 mask (non-representative copy)."""
    key = df["url"].map(normalize_url)
    in_conflict = df.groupby(key)["label"].transform("nunique") > 1
    is_copy = is_redundant_copy(key.where(~in_conflict), keep) & ~in_conflict
    return key, in_conflict, is_copy


def rule2_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Each row dropped by rule 2 next to the kept row of the same key."""
    key, in_conflict, is_copy = rule_masks(df)
    representative = df[~in_conflict & ~is_copy].assign(key=key).set_index("key")["url"]
    dropped = df[is_copy]
    return pd.DataFrame(
        {
            "dropped_url": dropped["url"].to_numpy(),
            "kept_url": key[is_copy].map(representative).to_numpy(),
            "label": dropped["label"].astype(int).to_numpy(),
        }
    )


def clean_train(df: pd.DataFrame, keep: str = "first") -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning rules; return cleaned rows (int labels) and a log.

    keep="first" is the adopted rule (ADR 0001); other values are sensitivity checks.
    """
    key, in_conflict, is_copy = rule_masks(df, keep)
    kept = df[~in_conflict]
    is_copy = is_copy[~in_conflict]
    cleaned = kept[~is_copy].reset_index(drop=True).astype({"label": int})

    case_kept_copies = kept["url"].map(normalize_url_keep_path_case).duplicated().sum()
    log = {
        "rows_in": len(df),
        "rule1_conflicting_keys": int(key[in_conflict].nunique()),
        "rule1_rows_dropped": int(in_conflict.sum()),
        "rule1_dropped_by_label": label_counts(df.loc[in_conflict, "label"]),
        "rule2_rows_dropped": int(is_copy.sum()),
        "rule2_dropped_by_label": label_counts(kept.loc[is_copy, "label"]),
        "rows_out": len(cleaned),
        "label_counts_out": label_counts(cleaned["label"]),
        "option_c_keep_path_case": {
            "rule2_rows_dropped": int(case_kept_copies),
            "rows_merged_only_by_path_case": int(is_copy.sum() - case_kept_copies),
        },
    }
    return cleaned, log


def load_clean_train() -> pd.DataFrame:
    """Cleaned training rows, regenerated from the raw file on every call."""
    return clean_train(load_raw(ROOT / "data" / "train.csv"))[0]


def main() -> None:
    raw = load_raw(ROOT / "data" / "train.csv")
    cleaned, log = clean_train(raw)
    groups = group_summary(cleaned, tenant_group_key)
    report = {
        "input": "data/train.csv",
        "raw_quality": quality_counts(raw),
        "cleaning": log,
        "groups_after_cleaning": groups,
        "max_group_share": groups["top_groups"][0]["share_of_rows"],
        "max_group_share_exceeds_10pct": groups["top_groups"][0]["share_of_rows"] > 0.10,
    }
    out = ROOT / "results" / "cleaning.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "raw_quality"}, indent=2))


if __name__ == "__main__":
    main()
