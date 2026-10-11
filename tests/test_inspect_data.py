import pandas as pd

from src.inspect_data import (
    domain_breakdown,
    host_and_first_segment,
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
