"""Utilities for inferring a stable source slug for incoming API requests."""

from __future__ import annotations

import ipaddress
import re
from typing import Optional

from fastapi import Request


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def _host_to_slug(host: str) -> str:
    if not host:
        return "unknown"

    candidate = host.strip().lower()
    if not candidate:
        return "unknown"

    # Some proxies send comma-separated hosts.
    candidate = candidate.split(",")[0].strip()
    # Strip a simple :port suffix when present.
    if ":" in candidate and candidate.count(":") == 1:
        candidate = candidate.split(":", 1)[0]

    if not candidate:
        return "unknown"

    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        parts = [part for part in candidate.split(".") if part]
        if parts and parts[0] == "www":
            parts = parts[1:]
        if len(parts) >= 2:
            # Keep subdomains + registrable label, drop only the last TLD label.
            candidate = "-".join(parts[:-1])
        elif parts:
            candidate = parts[0]
    else:
        candidate = candidate.replace(".", "-").replace(":", "-")

    slug = _slugify(candidate)
    return slug or "unknown"


def resolve_source_slug(request: Request, provided_slug: Optional[str]) -> str:
    """Return explicit source slug when provided, otherwise infer it from host."""
    if provided_slug:
        explicit = _slugify(provided_slug)
        if explicit:
            return explicit

    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("x-original-host")
        or request.headers.get("host")
        or request.url.hostname
        or ""
    )
    return _host_to_slug(host)


def scoped_taxonomy_key(source_slug: str, taxonomy_key: str) -> str:
    """Build scoped taxonomy key as '<sourceSlug>_<taxonomyKey>'."""
    base_key = str(taxonomy_key or "").strip()
    if not base_key:
        return _slugify(source_slug) or "unknown"

    scope = _slugify(source_slug) or "unknown"
    prefix = f"{scope}_"
    if base_key.startswith(prefix):
        return base_key
    return f"{scope}_{base_key}"
