"""Subprocess helpers that never raise.

Every hardware probe shells out to a vendor tool that may be missing, may need
root, or may hang. `run()` turns all of those into a `CommandResult` with
`ok == False` so collectors can degrade instead of crashing the dashboard.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 4.0


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    missing: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.missing

    @property
    def reason(self) -> str:
        if self.missing:
            return f"{self.argv[0]}: not installed"
        if self.timed_out:
            return f"{self.argv[0]}: timed out"
        if self.returncode != 0:
            detail = (self.stderr or self.stdout).strip().splitlines()
            return f"{self.argv[0]}: exit {self.returncode}" + (f" ({detail[0]})" if detail else "")
        return ""


def which(binary: str) -> str | None:
    """Locate a binary, also checking the sbin dirs GUI sessions tend to drop."""
    found = shutil.which(binary)
    if found:
        return found
    for extra in ("/usr/sbin", "/sbin", "/usr/local/bin", "/opt/homebrew/bin"):
        candidate = os.path.join(extra, binary)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


async def run(
    *argv: str,
    timeout: float = DEFAULT_TIMEOUT,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Run a command, capturing output. Never raises."""
    if not argv:
        raise ValueError("run() needs at least one argument")

    resolved = which(argv[0])
    if resolved is None:
        return CommandResult(argv, 127, "", "", missing=True)

    full = (resolved, *argv[1:])
    try:
        proc = await asyncio.create_subprocess_exec(
            *full,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env={**os.environ, **(env or {})},
        )
    except OSError as exc:  # pragma: no cover - exec-time failures are rare
        log.debug("exec failed for %s: %s", argv[0], exc)
        return CommandResult(argv, 126, "", str(exc))

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        _kill(proc)
        return CommandResult(argv, -1, "", "", timed_out=True)

    return CommandResult(
        argv,
        proc.returncode or 0,
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


def _kill(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


async def can_sudo_nopasswd(binary: str) -> bool:
    """True if `sudo -n <binary>` would run without prompting for a password.

    Used to decide whether `powermetrics` (root-only on macOS) is reachable.
    We never prompt: a dashboard that blocks on a password prompt is a hang.
    """
    if os.geteuid() == 0:
        return True
    if which("sudo") is None or which(binary) is None:
        return False
    result = await run("sudo", "-n", "-l", binary, timeout=2.0)
    return result.ok
