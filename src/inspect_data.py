"""Descriptive statistics on cleaned training data; every number lands in results/."""

import json
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from src.clean import load_clean_train, load_raw, rule2_pairs
from src.features import prepare_url_without_scheme_and_www
from src.groups import _EXTRACT, group_key

ROOT = Path(__file__).resolve().parent.parent
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def host_and_first_segment(url: str) -> tuple[str, str]:
    """Lower-cased host and the first path segment (case kept)."""
    parts = urlsplit(url.strip())
    segment = parts.path.lstrip("/").split("/", 1)[0]
    return (parts.hostname or ""), segment


def path_state(path: str) -> str:
    """'none' (no path), 'slash' (just '/') or 'nonempty'."""
    if path == "":
        return "none"
    return "slash" if path == "/" else "nonempty"


def top_level_domain(url: str) -> str:
    """Last label of the public suffix; 'ip' for IP hosts, 'none' if absent."""
    ext = _EXTRACT(url.strip().lower())
    if ext.ipv4 or ext.ipv6:
        return "ip"
    return ext.suffix.rsplit(".", 1)[-1] if ext.suffix else "none"


def url_profile(url: str) -> dict:
    """Stateless per-URL attributes used for the per-class comparison."""
    parts = urlsplit(url.strip())
    host = parts.hostname or ""
    return {
        "scheme": parts.scheme.lower() or "none",
        "has_www": host.startswith("www."),
        "path_state": path_state(parts.path),
        "has_query": bool(parts.query),
        "length": len(url),
        "tld": top_level_domain(url),
    }


def class_shares(df: pd.DataFrame, column: str) -> dict:
    """Rows and within-class share of each value of `column`, per label."""
    counts = pd.crosstab(df[column], df["label"])
    shares = counts / counts.sum()
    return {
        str(value): {
            f"label_{label}": {
                "rows": int(counts.loc[value, label]),
                "share_of_class": round(float(shares.loc[value, label]), 4),
            }
            for label in counts.columns
        }
        for value in counts.index
    }


def length_quantiles(df: pd.DataFrame) -> dict:
    return {
        f"label_{label}": {
            f"q{int(q * 100)}": float(v) for q, v in group["length"].quantile(QUANTILES).items()
        }
        for label, group in df.groupby("label")
    }


def top_tlds(df: pd.DataFrame, top_n: int) -> list[dict]:
    """Most frequent TLDs overall, with per-class rows and phishing share."""
    counts = pd.crosstab(df["tld"], df["label"]).reindex(columns=[0, 1], fill_value=0)
    counts["rows"] = counts[0] + counts[1]
    top = counts.sort_values("rows", ascending=False).head(top_n)
    return [
        {
            "tld": tld,
            "rows": int(row["rows"]),
            "clean_rows": int(row[0]),
            "phishing_rows": int(row[1]),
            "phishing_share": round(row[1] / row["rows"], 4),
        }
        for tld, row in top.iterrows()
    ]


def per_class_profile(df: pd.DataFrame, top_n_tlds: int) -> dict:
    return {
        "rows_by_label": {f"label_{k}": int(v) for k, v in df["label"].value_counts().items()},
        "scheme": class_shares(df, "scheme"),
        "has_www": class_shares(df, "has_www"),
        "path_state": class_shares(df, "path_state"),
        "has_query": class_shares(df, "has_query"),
        "length_quantiles": length_quantiles(df),
        "distinct_tlds_by_label": {
            f"label_{k}": int(v) for k, v in df.groupby("label")["tld"].nunique().items()
        },
        "top_tlds": top_tlds(df, top_n_tlds),
    }


def label_table(df: pd.DataFrame, by: list[str], top_n: int) -> list[dict]:
    """Rows and phishing rows per value of `by`, largest first."""
    table = (
        df.groupby(by)["label"]
        .agg(rows="size", phishing_rows=lambda s: int((s == 1).sum()))
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
        "phishing_rows": int((rows["label"] == 1).sum()),
        "by_host": label_table(rows, ["host"], top_n),
        "by_host_and_first_segment": label_table(rows, ["host", "first_segment"], top_n),
    }


def is_bare_host(url: str) -> bool:
    """Scheme and host only: path empty or '/', no query, no fragment."""
    parts = urlsplit(url.strip())
    return parts.path in ("", "/") and not parts.query and not parts.fragment


def registry_domain(url: str) -> str:
    """Domain under the ICANN suffix, ignoring private suffixes (x.netlify.app -> netlify.app)."""
    ext = _EXTRACT(url.strip().lower())
    return ext.ipv4 or ext.ipv6 or ext.top_domain_under_registry_suffix or "none"


def group_support(df: pd.DataFrame) -> dict:
    """How many distinct groups stand behind the rows of `df`, per class."""
    sizes = df["group"].value_counts()
    return {
        "rows": len(df),
        "phishing_rows": int((df["label"] == 1).sum()),
        "distinct_groups": int(df["group"].nunique()),
        "distinct_phishing_groups": int(df.loc[df["label"] == 1, "group"].nunique()),
        "distinct_clean_groups": int(df.loc[df["label"] == 0, "group"].nunique()),
        "distinct_registry_domains": int(df["url"].map(registry_domain).nunique()),
        "top_registry_domains": {
            k: int(v) for k, v in df["url"].map(registry_domain).value_counts().head(3).items()
        },
        "largest_group": {"group": sizes.index[0], "rows": int(sizes.iloc[0])} if len(df) else None,
    }


def tld_support(df: pd.DataFrame, tlds: list[str]) -> dict:
    empty = df["path_state"] != "nonempty"
    return {
        tld: {
            "all_rows": group_support(df[df["tld"] == tld]),
            "empty_path_rows": group_support(df[(df["tld"] == tld) & empty]),
        }
        for tld in tlds
    }


def bare_host_trailing_slash(df: pd.DataFrame) -> dict:
    """Trailing slash vs none among bare-host URLs, per class, with group support."""
    bare = df[df["url"].map(is_bare_host)]
    return {
        "rows_by_label": {f"label_{k}": int(v) for k, v in bare["label"].value_counts().items()},
        "path_state": class_shares(bare, "path_state"),
        "groups_by_path_state": {
            state: group_support(rows) for state, rows in bare.groupby("path_state")
        },
    }


def duplicate_difference(dropped: str, kept: str) -> str:
    """Which surface parts differ between two URLs with the same normalised key."""
    a, b = urlsplit(dropped.strip()), urlsplit(kept.strip())
    diffs = []
    if a.scheme.lower() != b.scheme.lower():
        diffs.append("scheme")
    if (a.hostname or "").startswith("www.") != (b.hostname or "").startswith("www."):
        diffs.append("www")
    if dropped.strip().endswith("/") != kept.strip().endswith("/"):
        diffs.append("trailing_slash")
    rest_a = prepare_url_without_scheme_and_www(dropped).rstrip("/")
    rest_b = prepare_url_without_scheme_and_www(kept).rstrip("/")
    if rest_a != rest_b:
        diffs.append("case")
    return "+".join(diffs) or "identical"


def dropped_duplicate_profile(pairs: pd.DataFrame, top_n: int = 5) -> dict:
    """Difference types and group concentration of rule-2 drops, per class."""
    pairs = pairs.assign(
        difference=[
            duplicate_difference(d, k)
            for d, k in zip(pairs["dropped_url"], pairs["kept_url"], strict=True)
        ],
        group=pairs["dropped_url"].map(lambda u: group_key(u)[0]),
    )
    result = {}
    for label, rows in pairs.groupby("label"):
        sizes = rows["group"].value_counts()
        result[f"label_{label}"] = {
            "rows": len(rows),
            "difference_types": {k: int(v) for k, v in rows["difference"].value_counts().items()},
            "distinct_groups": int(sizes.size),
            "top_groups": [{"group": g, "rows": int(n)} for g, n in sizes.head(top_n).items()],
        }
    return result


def main() -> None:
    train = load_clean_train()
    profiled = pd.concat([train, pd.DataFrame(train["url"].map(url_profile).tolist())], axis=1)
    profiled["group"] = profiled["url"].map(lambda u: group_key(u)[0])
    empty_path = profiled[profiled["path_state"] != "nonempty"]
    result = {
        "input": "data/train.csv after cleaning (ADR 0001)",
        "all_rows": per_class_profile(profiled, top_n_tlds=15),
        "empty_path_rows": per_class_profile(empty_path, top_n_tlds=10),
        "google_com": domain_breakdown(train, "google.com"),
        "tld_group_support": tld_support(profiled, ["app", "ip", "fr", "edu"]),
        "bare_host_trailing_slash": bare_host_trailing_slash(profiled),
        "rule2_dropped_duplicates": dropped_duplicate_profile(
            rule2_pairs(load_raw(ROOT / "data" / "train.csv"))
        ),
    }
    out = ROOT / "results" / "eda.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=int))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
