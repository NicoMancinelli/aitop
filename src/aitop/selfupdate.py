"""Version checking and self-update.

aitop can be installed four ways, and each updates differently:

    uv tool install   -> uv tool install --force git+URL@tag
    pipx install      -> pipx install --force git+URL@tag
    pip install       -> pip install --upgrade git+URL@tag
    git clone + -e .  -> not ours to touch; tell the user to `git pull`

`detect_install_method()` works out which one is in play by looking at where
the package was imported from, so `aitop update` does the right thing without
the user having to remember how they installed it.

The startup check is deliberately unobtrusive: it is cached for a day, runs
concurrently with telemetry collection, times out fast, and never fails the
command. Set `updates.check: false` or `AITOP_NO_UPDATE_CHECK=1` to disable it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx

from aitop import __version__

log = logging.getLogger(__name__)

REPO_SLUG = "NicoMancinelli/aitop"
REPO_URL = f"https://github.com/{REPO_SLUG}"
GIT_SPEC = f"git+{REPO_URL}"
RELEASES_API = f"https://api.github.com/repos/{REPO_SLUG}/releases/latest"

CHECK_TIMEOUT = 3.0
CACHE_TTL_SECONDS = 24 * 3600


class InstallMethod(StrEnum):
    UV_TOOL = "uv tool"
    PIPX = "pipx"
    PIP = "pip"
    GIT_CHECKOUT = "git checkout"
    UNKNOWN = "unknown"

    @property
    def self_updatable(self) -> bool:
        """A dev checkout is the user's working tree — we never rewrite it."""
        return self in (InstallMethod.UV_TOOL, InstallMethod.PIPX, InstallMethod.PIP)


@dataclass(frozen=True, slots=True)
class UpdateStatus:
    current: str
    latest: str | None = None
    url: str = REPO_URL
    checked: bool = False
    """False when the check was skipped, disabled, or failed."""

    error: str | None = None

    @property
    def available(self) -> bool:
        if not self.latest:
            return False
        return parse_version(self.latest) > parse_version(self.current)

    def describe(self) -> str:
        if not self.checked:
            return self.error or "update check skipped"
        if self.available:
            return f"aitop {self.latest} is available (you have {self.current})"
        return f"aitop {self.current} is up to date"


# --------------------------------------------------------------------------- #
# Version comparison
# --------------------------------------------------------------------------- #

_VERSION_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?(.*)")


def parse_version(value: str) -> tuple[int, int, int, int, str]:
    """Compare release versions without pulling in `packaging`.

    Returns a sortable tuple where a pre-release sorts *below* the matching
    final release: 1.2.0rc1 < 1.2.0. Unparseable input sorts lowest.
    """
    text = (value or "").strip().lstrip("vV")
    match = _VERSION_RE.match(text)
    if not match:
        return (0, 0, 0, 0, "")
    major, minor, patch, rest = match.groups()
    suffix = (rest or "").strip(".-_")
    # 0 => pre-release, 1 => final. Keeps 1.0.0rc1 below 1.0.0.
    stage = 0 if suffix else 1
    return (int(major), int(minor or 0), int(patch or 0), stage, suffix)


# --------------------------------------------------------------------------- #
# Install method detection
# --------------------------------------------------------------------------- #


def package_root() -> Path:
    import aitop

    return Path(aitop.__file__).resolve().parent


def detect_install_method() -> InstallMethod:
    """Infer how this copy of aitop got onto the machine."""
    root = str(package_root())

    if repo_checkout() is not None:
        return InstallMethod.GIT_CHECKOUT
    if (
        f"{os.sep}uv{os.sep}tools{os.sep}" in root
        or f"{os.sep}uv{os.sep}tools{os.sep}" in sys.prefix
    ):
        return InstallMethod.UV_TOOL
    if (
        f"{os.sep}pipx{os.sep}venvs{os.sep}" in root
        or f"{os.sep}pipx{os.sep}venvs{os.sep}" in sys.prefix
    ):
        return InstallMethod.PIPX
    if "site-packages" in root or "dist-packages" in root:
        return InstallMethod.PIP
    return InstallMethod.UNKNOWN


def repo_checkout() -> Path | None:
    """The git working tree this package lives in, if it is a dev checkout.

    An installed copy lives under site-packages and has no `.git` above it;
    an editable install points straight at `<repo>/src/aitop`.
    """
    root = package_root()
    for candidate in (root, *root.parents):
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").is_file():
            return candidate
        if candidate == candidate.parent:
            break
    return None


# --------------------------------------------------------------------------- #
# Release lookup, with an on-disk cache
# --------------------------------------------------------------------------- #


def cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "aitop" / "update-check.json"


def _read_cache(ttl: float) -> str | None:
    try:
        payload = json.loads(cache_path().read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if time.time() - float(payload.get("checked_at", 0)) > ttl:
        return None
    latest = payload.get("latest")
    return latest if isinstance(latest, str) else None


def _write_cache(latest: str) -> None:
    try:
        path = cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checked_at": time.time(), "latest": latest}))
    except OSError as exc:  # a read-only cache dir must not break the command
        log.debug("could not write update cache: %s", exc)


def checks_disabled() -> bool:
    return os.environ.get("AITOP_NO_UPDATE_CHECK", "").strip().lower() in ("1", "true", "yes")


async def fetch_latest_version(timeout: float = CHECK_TIMEOUT) -> tuple[str | None, str | None]:
    """(version, error). Returns (None, reason) rather than raising."""
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(
                RELEASES_API,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": f"aitop/{__version__}",
                },
            )
        if response.status_code == 404:
            return None, "no releases published yet"
        response.raise_for_status()
        tag = response.json().get("tag_name")
    except httpx.HTTPError as exc:
        return None, f"update check failed: {type(exc).__name__}"
    except (ValueError, AttributeError):
        return None, "update check failed: unexpected response"

    if not isinstance(tag, str) or not tag:
        return None, "update check failed: release has no tag"
    return tag.lstrip("vV"), None


async def check_for_update(
    *,
    enabled: bool = True,
    force: bool = False,
    ttl: float = CACHE_TTL_SECONDS,
    timeout: float = CHECK_TIMEOUT,
) -> UpdateStatus:
    """Look up the newest release, preferring a fresh cache entry.

    `force` skips the cache (used by `aitop update`); the startup path relies
    on the cache so the common case costs nothing.
    """
    if not enabled or (checks_disabled() and not force):
        return UpdateStatus(current=__version__, error="update checks disabled")

    if not force:
        cached = _read_cache(ttl)
        if cached is not None:
            return UpdateStatus(current=__version__, latest=cached, checked=True)

    latest, error = await fetch_latest_version(timeout=timeout)
    if latest is None:
        return UpdateStatus(current=__version__, error=error)

    _write_cache(latest)
    return UpdateStatus(current=__version__, latest=latest, checked=True)


# --------------------------------------------------------------------------- #
# Applying an update
# --------------------------------------------------------------------------- #


def update_command(method: InstallMethod, ref: str | None) -> list[str] | None:
    """The exact argv that upgrades this installation, or None if we can't."""
    spec = f"{GIT_SPEC}@{ref}" if ref else GIT_SPEC
    match method:
        case InstallMethod.UV_TOOL:
            return ["uv", "tool", "install", "--force", spec]
        case InstallMethod.PIPX:
            return ["pipx", "install", "--force", spec]
        case InstallMethod.PIP:
            return [sys.executable, "-m", "pip", "install", "--upgrade", spec]
        case _:
            return None


async def apply_update(
    *, ref: str | None = None, dry_run: bool = False
) -> tuple[bool, str, list[str] | None]:
    """Upgrade this install in place. Returns (ok, message, argv_that_ran)."""
    from aitop.utils.proc import run

    method = detect_install_method()

    if method is InstallMethod.GIT_CHECKOUT:
        checkout = repo_checkout()
        return (
            False,
            f"this is a development checkout at {checkout} — run `git pull` there instead",
            None,
        )

    argv = update_command(method, ref)
    if argv is None:
        return (
            False,
            f"cannot self-update a {method.value} install; reinstall from {REPO_URL}",
            None,
        )

    if dry_run:
        return True, "would run: " + " ".join(argv), argv

    result = await run(*argv, timeout=300.0)
    if result.ok:
        return True, f"updated via {method.value} — restart aitop to use the new version", argv
    return False, result.reason or "update command failed", argv
