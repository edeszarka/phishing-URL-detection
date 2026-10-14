import pandas as pd

from src.overlap import LEVELS, duplication, host_first_path, hostname, strongest_levels


def test_hostname_drops_scheme_www_userinfo_and_port():
    assert hostname("https://user:pw@WWW.Example.com:8080/Path?q=1") == "example.com"
    assert hostname("example.com") == "example.com"
    assert hostname("paypal.com@evil.tk/login") == "evil.tk"


def test_host_first_path_lowercases_first_segment_only():
    assert host_first_path("http://a.com/Login/step2?x=1") == "a.com/login"
    assert host_first_path("a.com") == "a.com/"
    assert host_first_path("a.com/?q=1") == "a.com/"


def test_strongest_level_is_strictest_match():
    train = pd.Series(["http://a.com/Login/x", "http://sub.b.co.uk/p", "c.blogspot.com"])
    urls = pd.Series(
        [
            "http://a.com/Login/x",  # identical string
            "https://www.a.com/Login/x/",  # same after normalisation, path case kept
            "http://a.com/login/x",  # same only after lower-casing the path
            "http://a.com/Login/y",  # same host and first segment
            "http://a.com/other",  # same host
            "http://other.b.co.uk/",  # same tenant group
            "d.blogspot.com",  # same platform group only
            "http://new.org/",  # no overlap
        ]
    )
    expected = [*LEVELS, "none"]
    assert strongest_levels(train, urls).tolist() == expected


def test_duplication_counts_within_and_between_files():
    raw = pd.Series(["http://a.com/x", "https://www.a.com/x/", "http://c.com/", "http://c.com"])
    clean = pd.Series(["http://a.com/x"])  # c.com removed by cleaning
    sets = {
        "test": pd.Series(["http://a.com/x", "a.com/x", "http://c.com/"]),
        "external": pd.Series(["http://z.org/"]),
    }
    d = duplication(raw, clean, sets)
    assert d["within_file"]["train_raw"] == {"exact": 0, "normalized_case": 2, "normalized": 2}
    assert d["within_file"]["test"]["normalized"] == 1
    assert d["train_raw_vs"]["test"] == {"exact": 2, "normalized_case": 3, "normalized": 3}
    assert d["matching_keys_removed_by_cleaning"] == {"test": 1, "external": 0}
    assert d["normalized_keys_in_all_three"] == 0
