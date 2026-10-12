from src.groups import group_key, registry_domain


def test_subdomains_share_registered_domain():
    assert group_key("https://login.evil.co.uk/a")[0] == group_key("pay.evil.co.uk")[0]


def test_private_suffix_splits_tenants():
    assert group_key("http://alice.github.io")[0] != group_key("http://bob.github.io")[0]


def test_ip_host_is_own_group():
    assert group_key("http://192.168.0.1/login") == ("192.168.0.1", "ip")


def test_unparseable_falls_back_without_raising():
    assert group_key("")[1] == "fallback"
    assert group_key("not a url")[1] == "fallback"


def test_scheme_less_matches_schemed():
    assert group_key("www.example.co.uk/path") == group_key("https://www.example.co.uk/x")
    assert group_key("www.example.co.uk/path") == ("example.co.uk", "domain")


def test_scheme_less_deceptive_subdomain_uses_real_domain():
    assert group_key("PayPal.com.evil.tk/login") == ("evil.tk", "domain")


def test_case_and_whitespace_do_not_change_group():
    assert group_key("  HTTP://Login.Evil.TK/A ") == group_key("http://login.evil.tk/a")


def test_userinfo_and_port_are_ignored():
    assert group_key("http://user@evil.tk:443/x") == ("evil.tk", "domain")


def test_scheme_less_ip_and_ipv6():
    assert group_key("10.0.0.1/admin") == ("10.0.0.1", "ip")
    assert group_key("http://[::1]:8080/x") == ("::1", "ip")


def test_garbage_inputs_fall_back_without_raising():
    for garbage in ["   ", "http://", "@@@", "\x00\n", "javascript:alert(1)", "a" * 5000]:
        key, kind = group_key(garbage)
        assert kind == "fallback"
        assert key == garbage.strip().lower()


def test_registry_domain_falls_back_like_group_key():
    assert registry_domain("Not A URL") == "not a url"
    assert registry_domain("https://a.netlify.app") == registry_domain("https://b.netlify.app")
