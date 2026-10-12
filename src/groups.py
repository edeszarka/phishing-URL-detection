"""Group keys for grouped splitting.

- tenant-level: eTLD+1 with PSL private suffixes treated as public, so
  alice.github.io and bob.github.io are separate groups.
- platform-level: eTLD+1 under the ICANN suffix only, so every tenant of a
  hosting platform (x.netlify.app, y.netlify.app) shares one group.

IP hosts are their own group under both keys; URLs with no recognisable domain
fall back to the whole lower-cased string. Uses tldextract's bundled
suffix-list snapshot; never touches the network.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import tldextract

ROOT = Path(__file__).resolve().parent.parent
_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)


def tenant_group_key(url: str) -> str:
    """Tenant-level group: private suffixes count, so each platform tenant is separate."""
    text = str(url).strip().lower()
    ext = _EXTRACT(text)
    return ext.ipv4 or ext.ipv6 or ext.top_domain_under_public_suffix or text


def platform_group_key(url: str) -> str:
    """Platform-level group: private suffixes ignored (x.netlify.app -> netlify.app)."""
    text = str(url).strip().lower()
    ext = _EXTRACT(text)
    return ext.ipv4 or ext.ipv6 or ext.top_domain_under_registry_suffix or text


def key_kind(url: str) -> str:
    """'ip', 'domain', or 'fallback' when no domain is recognisable."""
    ext = _EXTRACT(str(url).strip().lower())
    if ext.ipv4 or ext.ipv6:
        return "ip"
    return "domain" if ext.top_domain_under_registry_suffix else "fallback"


def group_summary(df: pd.DataFrame, key: Callable[[str], str], top_n: int = 5) -> dict:
    """Group-size statistics for a frame with url and label columns."""
    groups = df["url"].map(key)
    kinds = df["url"].map(key_kind).value_counts()
    sizes = groups.value_counts()
    top = [
        {
            "group": g,
            "rows": int(n),
            "share_of_rows": round(n / len(df), 4),
            "phishing_rows": int(df.loc[groups == g, "label"].astype(int).sum()),
        }
        for g, n in sizes.head(top_n).items()
    ]
    return {
        "rows": len(df),
        "groups": int(sizes.size),
        "singleton_groups": int((sizes == 1).sum()),
        "key_kind_rows": {k: int(v) for k, v in kinds.items()},
        "top_groups": top,
        "max_group_share_exceeds_10pct": top[0]["share_of_rows"] > 0.10,
    }


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    summary = {
        "input": "data/train.csv (raw, before cleaning)",
        "tenant_level": group_summary(train, tenant_group_key),
        "platform_level": group_summary(train, platform_group_key),
    }
    out = ROOT / "results" / "groups_raw.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
