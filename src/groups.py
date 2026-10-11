"""Group key for grouped splitting: registered domain (eTLD+1).

PSL private suffixes count as public suffixes, so alice.github.io and
bob.github.io are separate groups. IP hosts are their own group. URLs with no
recognisable registered domain fall back to the whole lower-cased string.
Uses tldextract's bundled suffix-list snapshot; never touches the network.
"""

import json
from pathlib import Path

import pandas as pd
import tldextract

ROOT = Path(__file__).resolve().parent.parent
_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)


def group_key(url: str) -> tuple[str, str]:
    """Return (key, kind) where kind is 'domain', 'ip' or 'fallback'."""
    text = str(url).strip().lower()
    ext = _EXTRACT(text)
    if ext.ipv4 or ext.ipv6:
        return ext.ipv4 or ext.ipv6, "ip"
    if ext.top_domain_under_public_suffix:
        return ext.top_domain_under_public_suffix, "domain"
    return text, "fallback"


def group_summary(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Group-size statistics for a frame with url and label columns."""
    keys = df["url"].map(group_key)
    groups = keys.str[0]
    kinds = keys.str[1].value_counts()
    sizes = groups.value_counts()
    top = [
        {
            "group": g,
            "rows": int(n),
            "share_of_rows": round(n / len(df), 4),
            "phishing_rows": int(df.loc[groups == g, "label"].sum()),
        }
        for g, n in sizes.head(top_n).items()
    ]
    return {
        "rows": len(df),
        "groups": int(sizes.size),
        "singleton_groups": int((sizes == 1).sum()),
        "key_kind_rows": {k: int(v) for k, v in kinds.items()},
        "top_groups": top,
    }


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    summary = {"input": "data/train.csv (raw, before cleaning)", **group_summary(train)}
    out = ROOT / "results" / "groups_raw.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
