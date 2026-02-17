"""Tests for API request source slug inference."""

from types import SimpleNamespace

import pytest

from taxomind.services.api.request_source import _host_to_slug, resolve_source_slug


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("api.example.com", "api-example"),
        ("foo.bar.example.com", "foo-bar-example"),
        ("www.test.app.request.com", "test-app-request"),
        ("example.com", "example"),
        ("api.example.com:8080", "api-example"),
        ("api.example.com, proxy.local", "api-example"),
        ("127.0.0.1", "127-0-0-1"),
        ("", "unknown"),
    ],
)
def test_host_to_slug(host: str, expected: str) -> None:
    assert _host_to_slug(host) == expected


def test_resolve_source_slug_prefers_payload_source_slug() -> None:
    request = SimpleNamespace(
        headers={"host": "subdomain.domani1.com"},
        url=SimpleNamespace(hostname="subdomain.domani1.com"),
    )
    assert resolve_source_slug(request, "ClassifAI Tenant 01") == "classifai-tenant-01"


def test_resolve_source_slug_infers_from_host_when_missing() -> None:
    request = SimpleNamespace(
        headers={"host": "subdomain.domani1.com"},
        url=SimpleNamespace(hostname="subdomain.domani1.com"),
    )
    assert resolve_source_slug(request, None) == "subdomain-domani1"
