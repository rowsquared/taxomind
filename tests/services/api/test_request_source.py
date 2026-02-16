"""Tests for API request source slug inference."""

import pytest

from taxomind.services.api.request_source import _host_to_slug


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("api.example.com", "api-example"),
        ("foo.bar.example.com", "foo-bar-example"),
        ("example.com", "example"),
        ("api.example.com:8080", "api-example"),
        ("api.example.com, proxy.local", "api-example"),
        ("127.0.0.1", "127-0-0-1"),
        ("", "unknown"),
    ],
)
def test_host_to_slug(host: str, expected: str) -> None:
    assert _host_to_slug(host) == expected
