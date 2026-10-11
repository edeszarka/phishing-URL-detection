import pandas as pd

from src.clean import duplicate_counts, normalize_url


def test_normalize_url_follows_external_rule():
    assert normalize_url(" HTTPS://WWW.Example.com/Path/ ") == "example.com/path"
    assert normalize_url("example.com") == "example.com"


def test_duplicate_counts_separates_copies_from_conflicts():
    df = pd.DataFrame(
        {"url": ["a", "a", "a", "b", "b", "c"], "label": ["1", "1", "1", "0", "1", "0"]}
    )
    assert duplicate_counts(df, df["url"]) == {
        "redundant_copies_same_label": 2,
        "conflicting_keys": 1,
        "rows_in_conflicting_keys": 2,
    }
