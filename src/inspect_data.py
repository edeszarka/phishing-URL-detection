"""Descriptive statistics on data/train.csv; every number lands in results/."""

import json
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from src.clean import load_raw
from src.groups import group_key

ROOT = Path(__file__).resolve().parent.parent


def host_and_first_segment(url: str) -> tuple[str, str]:
    """Lower-cased host and the first path segment (case kept)."""
    parts = urlsplit(url.strip())
    segment = parts.path.lstrip("/").split("/", 1)[0]
    return (parts.hostname or ""), segment


def label_table(df: pd.DataFrame, by: list[str], top_n: int) -> list[dict]:
    """Rows and phishing rows per value of `by`, largest first."""
    table = (
        df.groupby(by)["label"]
        .agg(rows="size", phishing_rows=lambda s: int((s == "1").sum()))
        .sort_values("rows", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return table.to_dict(orient="records")


def domain_breakdown(df: pd.DataFrame, domain: str, top_n: int = 15) -> dict:
    """Split one registered-domain group into hosts and first path segments."""
    rows = df[df["url"].map(lambda u: group_key(u)[0]) == domain].copy()
    parts = rows["url"].map(host_and_first_segment)
    rows["host"] = parts.str[0]
    rows["first_segment"] = parts.str[1]
    return {
        "domain": domain,
        "rows": len(rows),
        "phishing_rows": int((rows["label"] == "1").sum()),
        "by_host": label_table(rows, ["host"], top_n),
        "by_host_and_first_segment": label_table(rows, ["host", "first_segment"], top_n),
    }


def main() -> None:
    train = load_raw(ROOT / "data" / "train.csv")
    result = {"input": "data/train.csv (raw)", **domain_breakdown(train, "google.com")}
    out = ROOT / "results" / "eda_google_raw.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=int))
    print(json.dumps(result, indent=2, default=int))


if __name__ == "__main__":
    main()
