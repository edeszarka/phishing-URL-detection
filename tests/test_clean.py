import pandas as pd

from src.clean import (
    clean_train,
    duplicate_counts,
    normalize_url,
    normalize_url_keep_path_case,
)


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


def test_normalize_keep_path_case_lowers_host_only():
    assert normalize_url_keep_path_case("HTTPS://WWW.Ex.com/DropboxF/") == "ex.com/DropboxF"
    assert normalize_url_keep_path_case("http://Ex.com?Q=1") == "ex.com?Q=1"


def test_clean_drops_conflicts_before_dedupe_and_keeps_first_original():
    df = pd.DataFrame(
        {
            "url": [
                "http://A.com/X",  # kept: first of its key, original case
                "https://a.com/x/",  # copy of row 0 -> dropped by rule 2
                "http://c.com",  # conflict -> dropped by rule 1
                "https://c.com/",  # conflict -> dropped by rule 1
                "http://c.com",  # same-label copy inside a conflict: still rule 1
                "http://b.com",
            ],
            "label": ["1", "1", "0", "1", "0", "0"],
        }
    )
    cleaned, log = clean_train(df)
    assert cleaned["url"].tolist() == ["http://A.com/X", "http://b.com"]
    assert cleaned["label"].tolist() == [1, 0]
    assert (log["rule1_rows_dropped"], log["rule2_rows_dropped"]) == (3, 1)
    assert log["option_c_keep_path_case"]["rows_merged_only_by_path_case"] == 1
