"""Supervisor-aware start/stop/restart for local AI runtimes.

Lifecycle actions go through the process supervisor when we can detect one
(`systemd`, `launchd`, `docker`), and fall back to starting a known binary or
signalling the PID directly. Remote endpoints always refuse — we never reach
across the network to kill someone else's daemon.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

from aitop.utils.proc import run, which

log = logging.getLogger(__name__)

# Well-known unit / label names tried in order for each engine kind.
_SYSTEMD_UNITS: dict[str, tuple[str, ...]] = {
    "ollama": ("ollama.service", "ollama"),
    "lmstudio": ("lmstudio.service", "lm-studio.service"),
    "vllm": ("vllm.service",),
    "llama-server": ("llama-server.service", "llamacpp.service"),
    "mlx": ("mlx-openai-server.service", "mlx_lm.service"),
}

_LAUNCHD_LABELS: dict[str, tuple[str, ...]] = {
    "ollama": ("com.ollama.ollama", "homebrew.mxcl.ollama"),
    "lmstudio": ("com.lmstudio.LMStudio",),
}

_START_BINARIES: dict[str, tuple[str, ...]] = {
    "ollama": ("ollama", "serve"),
    "lmstudio": ("lms", "server", "start"),
    "vllm": ("vllm", "serve"),
    "llama-server": ("llama-server",),
    "mlx": ("mlx_lm.server",),
}


@dataclass(frozen=True, slots=True)
class LifecycleResult:
    ok: bool
    message: str
    action: str
    managed_by: str | None = None


async def start_engine(
    kind: str,
    *,
    managed_by: str | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> LifecycleResult:
    """Bring a local engine up via supervisor or its CLI."""
    managed = managed_by or await _guess_supervisor(kind)

    if managed == "systemd":
        result = await _systemd("start", kind)
        if result.ok:
            return LifecycleResult(True, f"started {kind} via systemd", "start", "systemd")
    if managed == "launchd":
        result = await _launchd_kickstart(kind)
        if result.ok:
            return LifecycleResult(True, f"started {kind} via launchd", "start", "launchd")
    if managed == "docker":
        return LifecycleResult(
            False, f"{kind}: docker start needs an explicit container name", "start", "docker"
        )

    argv = _START_BINARIES.get(kind)
    if not argv or which(argv[0]) is None:
        return LifecycleResult(
            False,
            f"{kind}: no supervisor and `{argv[0] if argv else kind}` not on PATH",
            "start",
            managed,
        )

    env: dict[str, str] = {}
    if kind == "ollama" and (host != "127.0.0.1" or port):
        env["OLLAMA_HOST"] = f"{host}:{port}" if port else host

    # Detach: start the daemon and return; we do not wait for it to exit.
    resolved = which(argv[0])
    assert resolved is not None
    try:
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            resolved,
            *argv[1:],
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
            env={**os.environ, **env},
            start_new_session=True,
        )
    except OSError as exc:
        return LifecycleResult(False, f"failed to exec {argv[0]}: {exc}", "start", managed)

    return LifecycleResult(
        True,
        f"launched `{' '.join(argv)}` (pid {proc.pid})",
        "start",
        managed or "manual",
    )


async def stop_engine(
    kind: str,
    *,
    pid: int | None = None,
    managed_by: str | None = None,
) -> LifecycleResult:
    """Stop a local engine via supervisor, docker, or SIGTERM."""
    managed = managed_by or await _guess_supervisor(kind)

    if managed == "systemd":
        result = await _systemd("stop", kind)
        if result.ok:
            return LifecycleResult(True, f"stopped {kind} via systemd", "stop", "systemd")
        if result.message:
            log.debug("systemd stop failed: %s", result.message)

    if managed == "launchd":
        result = await _launchd_bootout(kind)
        if result.ok:
            return LifecycleResult(True, f"stopped {kind} via launchd", "stop", "launchd")

    if managed == "docker" and pid is not None:
        # Best-effort: map PID → container is unreliable; refuse rather than guess.
        return LifecycleResult(
            False,
            f"{kind}: docker-managed — stop the container explicitly",
            "stop",
            "docker",
        )

    if kind == "lmstudio" and which("lms") is not None:
        result = await run("lms", "server", "stop", timeout=15.0)
        if result.ok:
            return LifecycleResult(True, "stopped LM Studio server via `lms`", "stop", managed)

    if pid is None:
        return LifecycleResult(
            False, f"{kind}: no PID and no working supervisor stop", "stop", managed
        )

    try:
        import signal

        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return LifecycleResult(True, f"{kind}: process {pid} already gone", "stop", managed)
    except PermissionError:
        return LifecycleResult(
            False, f"{kind}: permission denied signalling pid {pid}", "stop", managed
        )

    return LifecycleResult(True, f"sent SIGTERM to {kind} (pid {pid})", "stop", managed or "manual")


async def restart_engine(
    kind: str,
    *,
    pid: int | None = None,
    managed_by: str | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> LifecycleResult:
    managed = managed_by or await _guess_supervisor(kind)

    if managed == "systemd":
        result = await _systemd("restart", kind)
        if result.ok:
            return LifecycleResult(True, f"restarted {kind} via systemd", "restart", "systemd")

    if managed == "launchd":
        result = await _launchd_kickstart(kind, kill=True)
        if result.ok:
            return LifecycleResult(True, f"restarted {kind} via launchd", "restart", "launchd")

    stopped = await stop_engine(kind, pid=pid, managed_by=managed)
    if not stopped.ok and "already gone" not in stopped.message:
        # Still attempt start — the process may have been down already.
        log.debug("stop before restart: %s", stopped.message)

    import asyncio

    await asyncio.sleep(0.4)
    started = await start_engine(kind, managed_by=managed, host=host, port=port)
    if started.ok:
        return LifecycleResult(True, f"restarted {kind}", "restart", started.managed_by)
    return LifecycleResult(False, started.message, "restart", managed)


async def _guess_supervisor(kind: str) -> str | None:
    if sys.platform == "darwin":
        return "launchd"
    if sys.platform.startswith("linux"):
        # Prefer systemd when the unit exists.
        for unit in _SYSTEMD_UNITS.get(kind, ()):
            probe = await run("systemctl", "cat", unit, timeout=2.0)
            if probe.ok:
                return "systemd"
            user = await run("systemctl", "--user", "cat", unit, timeout=2.0)
            if user.ok:
                return "systemd"
        return "manual"
    return None


async def _systemd(action: str, kind: str) -> LifecycleResult:
    units = _SYSTEMD_UNITS.get(kind, ())
    if not units:
        return LifecycleResult(False, f"no systemd unit mapping for {kind}", action, "systemd")

    last_reason = ""
    for unit in units:
        for argv in (("systemctl", action, unit), ("systemctl", "--user", action, unit)):
            result = await run(*argv, timeout=20.0)
            if result.ok:
                return LifecycleResult(True, f"{action} {unit}", action, "systemd")
            last_reason = result.reason
    return LifecycleResult(False, last_reason or f"systemd {action} failed", action, "systemd")


async def _launchd_kickstart(kind: str, *, kill: bool = False) -> LifecycleResult:
    labels = _LAUNCHD_LABELS.get(kind, ())
    if not labels:
        return LifecycleResult(False, f"no launchd label mapping for {kind}", "start", "launchd")

    uid = os.getuid()
    domain = f"gui/{uid}"
    last_reason = ""
    for label in labels:
        target = f"{domain}/{label}"
        argv = (
            ("launchctl", "kickstart", "-k", target) if kill else ("launchctl", "kickstart", target)
        )
        result = await run(*argv, timeout=15.0)
        if result.ok:
            return LifecycleResult(
                True, f"kickstart {label}", "restart" if kill else "start", "launchd"
            )
        # Some installs need bootstrap first — try kickstart without -k as fallback.
        last_reason = result.reason
    return LifecycleResult(False, last_reason or "launchctl kickstart failed", "start", "launchd")


async def _launchd_bootout(kind: str) -> LifecycleResult:
    labels = _LAUNCHD_LABELS.get(kind, ())
    uid = os.getuid()
    domain = f"gui/{uid}"
    last_reason = ""
    for label in labels:
        result = await run("launchctl", "bootout", f"{domain}/{label}", timeout=15.0)
        if result.ok:
            return LifecycleResult(True, f"bootout {label}", "stop", "launchd")
        # kill via kickstart -k then bootout may fail if not bootstrapped by us
        kill = await run("launchctl", "kill", "SIGTERM", f"{domain}/{label}", timeout=10.0)
        if kill.ok:
            return LifecycleResult(True, f"signalled {label}", "stop", "launchd")
        last_reason = result.reason or kill.reason
    return LifecycleResult(False, last_reason or "launchctl stop failed", "stop", "launchd")
