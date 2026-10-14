"""Overlap between training URLs and the evaluation sets (ADR 0003, decision 7).

Run as `uv run python -m src.overlap`. Reads the cleaned training data and ONLY the
url column of data/test.csv and data/external_test.csv; no evaluation label is read.
Writes results/overlap.json and data/overlap_levels.csv (one strongest level per
evaluation row, derived from supplied data, so kept out of git).

Levels, strictest first; each implies every looser one, so a URL's strongest level
is the first that matches:
  exact             identical URL string
  normalized_case   normalised, path case kept (src.clean.normalize_url_keep_path_case)
  normalized        external-set rule: lower-case, no scheme, no leading www., no
                    trailing slash (src.clean.normalize_url)
  host_first_path   hostname plus first path segment (near-duplicate pages)
  hostname          lower-cased host, no leading www., userinfo or port
  tenant_group      eTLD+1 with private suffixes (src.groups.tenant_group_key)
  platform_group    eTLD+1, ICANN suffixes only (src.groups.platform_group_key, ADR 0002)
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from predict import is_scorable
from src.clean import load_clean_train, load_raw, normalize_url, normalize_url_keep_path_case
from src.features import MAX_URL_CHARS
from src.groups import platform_group_key, tenant_group_key

ROOT = Path(__file__).resolve().parent.parent
EVAL_SETS = {"test": "test.csv", "external": "external_test.csv"}
_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)
_HOST_END = re.compile(r"[/?#]")


def split_host(url: str) -> tuple[str, str]:
    """(host, rest): host lower-cased without www., userinfo or port; rest keeps case."""
    text = _SCHEME.sub("", url.strip())
    end = _HOST_END.search(text)
    split = end.start() if end else len(text)
    host = text[:split].rpartition("@")[2].split(":")[0].lower().removeprefix("www.")
    return host, text[split:]


def hostname(url: str) -> str:
    return split_host(url)[0]


def host_first_path(url: str) -> str:
    host, rest = split_host(url)
    first = rest.lstrip("/").split("/")[0]
    return f"{host}/{_HOST_END.split(first)[0].lower()}"


LEVELS: dict[str, Callable[[str], str]] = {
    "exact": lambda url: url,
    "normalized_case": normalize_url_keep_path_case,
    "normalized": normalize_url,
    "host_first_path": host_first_path,
    "hostname": hostname,
    "tenant_group": tenant_group_key,
    "platform_group": platform_group_key,
}


def strongest_levels(train_urls: pd.Series, urls: pd.Series) -> pd.Series:
    """Strictest level at which each URL matches some training URL, else 'none'."""
    level = pd.Series("none", index=urls.index)
    for name in reversed(LEVELS):  # loosest first, so stricter levels overwrite
        key = LEVELS[name]
        level[urls.map(key).isin(set(train_urls.map(key)))] = name
    return level


def load_eval_urls(name: str) -> pd.Series:
    """URL column only; labels are never loaded here."""
    path = ROOT / "data" / EVAL_SETS[name]
    return pd.read_csv(path, usecols=["url"], dtype=str, keep_default_na=False)["url"]


def summarize(level: pd.Series, urls: pd.Series) -> dict:
    counts = level.value_counts()
    at_least = {}
    for i, name in enumerate(LEVELS):
        at_least[name] = int(level.isin(list(LEVELS)[: i + 1]).sum())
    return {
        "rows": len(urls),
        "strongest_level": {n: int(counts.get(n, 0)) for n in [*LEVELS, "none"]},
        "matches_at_level_or_stricter": at_least,
        "not_scored": int((~urls.map(is_scorable)).sum()),
        "truncated": int((urls.str.strip().str.len() > MAX_URL_CHARS).sum()),
    }


PAGE_KEYS: dict[str, Callable[[str], str]] = {
    "exact": LEVELS["exact"],
    "normalized_case": LEVELS["normalized_case"],
    "normalized": LEVELS["normalized"],
}


def within_file_duplicates(urls: pd.Series) -> dict[str, int]:
    """Rows that repeat an earlier row of the same file, per page-level key."""
    return {name: int(urls.map(key).duplicated().sum()) for name, key in PAGE_KEYS.items()}


def rows_matching(reference: pd.Series, urls: pd.Series) -> dict[str, int]:
    """Rows of `urls` whose page-level key appears anywhere in `reference`."""
    return {
        name: int(urls.map(key).isin(set(reference.map(key))).sum())
        for name, key in PAGE_KEYS.items()
    }


def duplication(raw_train: pd.Series, clean_train: pd.Series, sets: dict) -> dict:
    """Page-level duplication within and between the three supplied files (URLs only)."""
    removed_keys = set(raw_train.map(normalize_url)) - set(clean_train.map(normalize_url))
    return {
        "within_file": {
            "train_raw": within_file_duplicates(raw_train),
            **{name: within_file_duplicates(urls) for name, urls in sets.items()},
        },
        "train_raw_vs": {name: rows_matching(raw_train, urls) for name, urls in sets.items()},
        "matching_keys_removed_by_cleaning": {
            name: int(urls.map(normalize_url).isin(removed_keys).sum())
            for name, urls in sets.items()
        },
        "normalized_keys_in_all_three": len(
            set(raw_train.map(normalize_url))
            & set(sets["test"].map(normalize_url))
            & set(sets["external"].map(normalize_url))
        ),
    }


def main() -> None:
    train_urls = load_clean_train()["url"]
    sets = {name: load_eval_urls(name) for name in EVAL_SETS}
    report: dict = {
        "reference": "data/train.csv after cleaning (ADR 0001), the rows the model saw",
        "levels_strictest_first": list(LEVELS),
        "labels_read": False,
    }
    frames = []
    for name, urls in sets.items():
        level = strongest_levels(train_urls, urls)
        report[name] = summarize(level, urls)
        frames.append(pd.DataFrame({"set": name, "row": urls.index, "level": level}))
    report["test_vs_external"] = summarize(
        strongest_levels(sets["test"], sets["external"]), sets["external"]
    )
    raw_train = load_raw(ROOT / "data" / "train.csv")["url"]
    report["duplication"] = duplication(raw_train, train_urls, sets)
    out = ROOT / "results" / "overlap.json"
    out.write_text(json.dumps(report, indent=2))
    pd.concat(frames).to_csv(ROOT / "data" / "overlap_levels.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
