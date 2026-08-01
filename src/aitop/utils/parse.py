"""Lenient parsers for the untrusted JSON/CLI output engines hand us.

Every helper here returns `None` rather than raising: a runtime that changes
its response shape between releases must degrade a single field, not the poll.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_FRACTION = re.compile(r"\.(\d+)")


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 string. Tolerates Go's 9-digit nanosecond precision."""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    # datetime.fromisoformat accepts at most 6 fractional digits; Ollama emits 9.
    text = _FRACTION.sub(lambda m: "." + m.group(1)[:6].ljust(6, "0"), text, count=1)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def to_int(value: Any) -> int | None:
    """Coerce to int, stripping unit suffixes like '4096 MiB' or '81 %'."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+", value.replace(",", ""))
        if match:
            return int(match.group())
    return None


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group())
    return None


def first(mapping: Any, *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None key from a dict-ish object."""
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return default


def split_host_port(value: str, default_port: int) -> tuple[str, int]:
    """Parse `host`, `host:port`, `http://host:port`, `[::1]:port`."""
    text = value.strip().removeprefix("http://").removeprefix("https://").rstrip("/")
    text = text.split("/", 1)[0]
    if text.startswith("["):
        host, _, rest = text.partition("]")
        port = rest.lstrip(":")
        return host.lstrip("["), int(port) if port.isdigit() else default_port
    head, sep, tail = text.rpartition(":")
    if sep and tail.isdigit():
        return (head or "127.0.0.1"), int(tail)
    return (text or "127.0.0.1"), default_port
