"""Formatting helpers shared by every renderer."""

from __future__ import annotations

from datetime import UTC, datetime

_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def bytes_human(value: int | float | None, precision: int = 1) -> str:
    """1_073_741_824 -> '1.0 GB'. None -> '—'."""
    if value is None:
        return "—"
    size = float(value)
    for unit in _UNITS:
        if abs(size) < 1024.0 or unit == _UNITS[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.{precision}f} {unit}"
        size /= 1024.0
    return f"{size:.{precision}f} PB"  # pragma: no cover


def percent(value: float | None, precision: int = 0) -> str:
    return "—" if value is None else f"{value:.{precision}f}%"


def watts(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} W"


def celsius(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}°C"


def duration_human(seconds: float | None) -> str:
    """90061 -> '1d 1h 1m'."""
    if seconds is None:
        return "—"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def relative_time(when: datetime | None) -> str:
    """'in 4m' / '3h ago' / '—'."""
    if when is None:
        return "—"
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delta = (when - datetime.now(UTC)).total_seconds()
    future = delta > 0
    delta = abs(delta)
    if delta < 60:
        text = f"{int(delta)}s"
    elif delta < 3600:
        text = f"{int(delta // 60)}m"
    elif delta < 86400:
        text = f"{int(delta // 3600)}h"
    else:
        text = f"{int(delta // 86400)}d"
    return f"in {text}" if future else f"{text} ago"


def ratio_bar(fraction: float | None, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """A plain-text meter. Colour is applied by the caller's markup."""
    if fraction is None:
        return empty * width
    clamped = max(0.0, min(1.0, fraction))
    filled = round(clamped * width)
    return fill * filled + empty * (width - filled)


def heat_color(fraction: float | None) -> str:
    """Rich colour name for a 0..1 utilisation value."""
    if fraction is None:
        return "grey42"
    if fraction < 0.60:
        return "green"
    if fraction < 0.85:
        return "yellow"
    return "red"


def sparkline(values: list[float], width: int = 24, *, loft: float = 100.0) -> str:
    """Compact Unicode sparkline. Values are scaled against `loft` (or the max)."""
    if width <= 0:
        return ""
    if not values:
        return " " * width
    window = values[-width:]
    if len(window) < width:
        window = [0.0] * (width - len(window)) + window
    peak = max(loft, max(window) if window else 0.0, 1.0)
    chars = " ▁▂▃▄▅▆▇█"
    out: list[str] = []
    for value in window:
        idx = int(max(0.0, min(1.0, value / peak)) * (len(chars) - 1))
        out.append(chars[idx])
    return "".join(out)


def truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"
