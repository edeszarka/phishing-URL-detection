from src.groups import key_kind, platform_group_key, tenant_group_key

BOTH_KEYS = (tenant_group_key, platform_group_key)


def test_subdomains_share_a_group_under_both_keys():
    for key in BOTH_KEYS:
        assert key("https://login.evil.co.uk/a") == key("pay.evil.co.uk") == "evil.co.uk"


def test_private_suffix_tenants_split_only_at_tenant_level():
    assert tenant_group_key("http://alice.github.io") != tenant_group_key("http://bob.github.io")
    assert platform_group_key("http://alice.github.io") == "github.io"
    assert platform_group_key("https://a.netlify.app") == platform_group_key(
        "https://b.netlify.app"
    )


def test_ip_host_is_own_group():
    for key in BOTH_KEYS:
        assert key("http://192.168.0.1/login") == "192.168.0.1"
        assert key("10.0.0.1/admin") == "10.0.0.1"
        assert key("http://[::1]:8080/x") == "::1"
    assert key_kind("http://192.168.0.1/login") == "ip"


def test_scheme_less_matches_schemed():
    for key in BOTH_KEYS:
        assert key("www.example.co.uk/path") == key("https://www.example.co.uk/x")


def test_scheme_less_deceptive_subdomain_uses_real_domain():
    for key in BOTH_KEYS:
        assert key("PayPal.com.evil.tk/login") == "evil.tk"


def test_case_whitespace_userinfo_and_port_are_ignored():
    for key in BOTH_KEYS:
        assert key("  HTTP://Login.Evil.TK/A ") == key("http://user@evil.tk:443/x") == "evil.tk"


def test_garbage_inputs_fall_back_without_raising():
    for garbage in [
        "",
        "   ",
        "not a url",
        "http://",
        "@@@",
        "\x00\n",
        "javascript:alert(1)",
        "a" * 5000,
    ]:
        assert key_kind(garbage) == "fallback"
        for key in BOTH_KEYS:
            assert key(garbage) == garbage.strip().lower()
