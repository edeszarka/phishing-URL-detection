import pandas as pd

from src.groups import platform_group_key
from src.inspect_data import (
    domain_breakdown,
    duplicate_difference,
    host_and_first_segment,
    is_bare_host,
    label_runs_test,
    path_state,
    top_level_domain,
    url_profile,
)


def test_host_and_first_segment():
    assert host_and_first_segment("https://Docs.Google.com/Forms/d/x") == (
        "docs.google.com",
        "Forms",
    )
    assert host_and_first_segment("not a url") == ("", "not a url")


def test_path_state_distinguishes_missing_slash_and_path():
    assert [path_state(p) for p in ["", "/", "/a"]] == ["none", "slash", "nonempty"]


def test_top_level_domain():
    assert top_level_domain("http://x.example.co.uk/a") == "uk"
    assert top_level_domain("http://alice.github.io") == "io"
    assert top_level_domain("http://10.0.0.1/") == "ip"
    assert top_level_domain("not a url") == "none"


def test_url_profile():
    assert url_profile("HTTPS://www.Example.com/?q=1") == {
        "scheme": "https",
        "has_www": True,
        "path_state": "slash",
        "has_query": True,
        "length": 28,
        "tld": "com",
    }
    assert url_profile("")["path_state"] == "none"


def test_domain_breakdown_counts_hosts():
    df = pd.DataFrame(
        {
            "url": [
                "https://docs.google.com/forms/a",
                "https://docs.google.com/forms/b",
                "https://www.google.com/",
                "https://evil.tk/",
            ],
            "label": [1, 1, 0, 1],
        }
    )
    result = domain_breakdown(df, "google.com")
    assert (result["rows"], result["phishing_rows"]) == (3, 2)
    assert result["by_host"][0] == {"host": "docs.google.com", "rows": 2, "phishing_rows": 2}


def test_is_bare_host():
    assert is_bare_host("http://a.com") and is_bare_host("https://www.a.com/")
    assert not is_bare_host("http://a.com/?q=1")
    assert not is_bare_host("http://a.com/x")


def test_duplicate_difference_names_each_part():
    assert duplicate_difference("http://a.com/x/", "http://a.com/x") == "trailing_slash"
    assert duplicate_difference("https://www.a.com/X", "http://a.com/x/") == (
        "scheme+www+trailing_slash+case"
    )
    assert duplicate_difference("http://a.com", "http://a.com") == "identical"


def test_platform_group_key_ignores_private_suffix():
    assert platform_group_key("https://musing-villani.netlify.app/") == "netlify.app"
    assert platform_group_key("http://10.0.0.1/x") == "10.0.0.1"


def test_label_runs_test_flags_sorted_order():
    import numpy as np

    sorted_labels = np.array([0] * 50 + [1] * 50)
    alternating = np.array([0, 1] * 50)
    assert label_runs_test(sorted_labels)["z"] < -5
    assert label_runs_test(alternating)["z"] > 5
