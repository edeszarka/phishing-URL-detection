"""Control experiments for cleaning and grouping decisions.

Run as `uv run python -m src.controls`; writes results/controls.json.
Same baseline pipeline (raw URL input) as src.train throughout.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.clean import clean_train, load_raw
from src.groups import group_key, registry_domain
from src.inspect_data import bare_host_trailing_slash, profile_frame
from src.train import assign_folds, oof_probabilities, summarize

ROOT = Path(__file__).resolve().parent.parent


def registered_groups(df: pd.DataFrame) -> pd.Series:
    return df["url"].map(lambda u: group_key(u)[0])


def oof(df: pd.DataFrame, folds: np.ndarray) -> np.ndarray:
    return oof_probabilities(
        df["url"].reset_index(drop=True), df["label"].reset_index(drop=True), folds, strip=False
    )


def rule2_sensitivity(raw: pd.DataFrame) -> dict:
    """Rule-2 representative first/last/random; folds fixed per group across variants."""
    variants = {keep: clean_train(raw, keep)[0] for keep in ("first", "last", "random")}
    base = variants["first"]
    base_groups = registered_groups(base)
    fold_of = dict(zip(base_groups, assign_folds(base["label"], base_groups), strict=True))
    result = {}
    for keep, df in variants.items():
        folds = registered_groups(df).map(fold_of)
        assert folds.notna().all(), "variants must share groups"
        labels = df["label"].to_numpy()
        result[keep] = {
            "rows": len(df),
            "urls_not_in_first_variant": int((~df["url"].isin(base["url"])).sum()),
            "oof_raw_url": summarize(labels, oof(df, folds.to_numpy()), folds.to_numpy()),
            "bare_host_trailing_slash": bare_host_trailing_slash(profile_frame(df)),
        }
    return result


def top_group_shares(groups: pd.Series, top_n: int = 5) -> list[dict]:
    sizes = groups.value_counts()
    return [
        {"group": g, "rows": int(n), "share_of_rows": round(n / len(groups), 4)}
        for g, n in sizes.head(top_n).items()
    ]


def subset_summary(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Pooled AUC (if both classes present) and mean probability per class."""
    both = 0 < labels.sum() < labels.size
    return {
        "rows": int(labels.size),
        "phishing_rows": int(labels.sum()),
        "pooled_auc": round(float(roc_auc_score(labels, probs)), 4) if both else None,
        "mean_prob_phishing": round(float(probs[labels == 1].mean()), 4) if labels.any() else None,
        "mean_prob_clean": round(float(probs[labels == 0].mean()), 4) if both else None,
    }


def group_key_control(df: pd.DataFrame) -> dict:
    """Registered domain (adopted) vs registry domain (private suffixes merged)."""
    keys = {
        "registered_domain": registered_groups(df),
        "registry_domain": df["url"].map(registry_domain),
    }
    differs = (keys["registered_domain"] != keys["registry_domain"]).to_numpy()
    labels = df["label"].to_numpy()
    result = {"rows_where_keys_differ": int(differs.sum())}
    for name, groups in keys.items():
        folds = assign_folds(df["label"], groups)
        probs = oof(df, folds)
        top = top_group_shares(groups)
        result[name] = {
            "groups": int(groups.nunique()),
            "top_groups": top,
            "max_group_share_exceeds_10pct": top[0]["share_of_rows"] > 0.10,
            "oof_raw_url": summarize(labels, probs, folds),
            "rows_where_keys_differ": subset_summary(labels[differs], probs[differs]),
        }
    return result


def main() -> None:
    raw = load_raw(ROOT / "data" / "train.csv")
    report = {
        "rule2_representative": rule2_sensitivity(raw),
        "group_key": group_key_control(clean_train(raw)[0]),
    }
    out = ROOT / "results" / "controls.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
