import pandas as pd

from src.inspect_data import domain_breakdown, host_and_first_segment


def test_host_and_first_segment():
    assert host_and_first_segment("https://Docs.Google.com/Forms/d/x") == (
        "docs.google.com",
        "Forms",
    )
    assert host_and_first_segment("not a url") == ("", "not a url")


def test_domain_breakdown_counts_hosts():
    df = pd.DataFrame(
        {
            "url": [
                "https://docs.google.com/forms/a",
                "https://docs.google.com/forms/b",
                "https://www.google.com/",
                "https://evil.tk/",
            ],
            "label": ["1", "1", "0", "1"],
        }
    )
    result = domain_breakdown(df, "google.com")
    assert (result["rows"], result["phishing_rows"]) == (3, 2)
    assert result["by_host"][0] == {"host": "docs.google.com", "rows": 2, "phishing_rows": 2}
