"""Background service management for aitop serve (launchd & systemd)."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SERVICE_NAME = "com.aitop.serve"
SYSTEMD_UNIT = "aitop.service"


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    platform: str
    installed: bool
    active: bool
    pid: int | None
    detail: str
    unit_path: Path | None


# --------------------------------------------------------------------------- #
# Executable & Path Resolution
# --------------------------------------------------------------------------- #


def resolve_aitop_exec() -> list[str]:
    """Find the best command invocation for `aitop`."""
    which_aitop = shutil.which("aitop")
    if which_aitop:
        return [which_aitop]

    venv_bin = Path(sys.prefix) / "bin" / "aitop"
    if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
        return [str(venv_bin.resolve())]

    local_bin = Path.home() / ".local" / "bin" / "aitop"
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return [str(local_bin.resolve())]

    which_uv = shutil.which("uv")
    if which_uv:
        return [which_uv, "tool", "run", "aitop"]

    return [sys.executable, "-m", "aitop"]


def get_launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_NAME}.plist"


def get_systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT


def get_log_paths() -> tuple[Path, Path]:
    log_dir = (
        Path.home() / "Library" / "Logs"
        if sys.platform == "darwin"
        else Path.home() / ".local" / "state" / "aitop"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "aitop.log", log_dir / "aitop-error.log"


# --------------------------------------------------------------------------- #
# Templates / Generators
# --------------------------------------------------------------------------- #


def generate_launchd_plist(
    *,
    host: str = "0.0.0.0",
    port: int = 9090,
    exec_args: list[str] | None = None,
    stdout_log: Path | None = None,
    stderr_log: Path | None = None,
) -> str:
    """Generate XML plist for macOS launchd user agent."""
    if exec_args is None:
        exec_args = resolve_aitop_exec()

    cmd = [*exec_args, "serve", "--host", host, "--port", str(port)]
    out_log, err_log = (
        get_log_paths() if stdout_log is None or stderr_log is None else (stdout_log, stderr_log)
    )

    env_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
    home = str(Path.home())
    extra_dirs = [f"{home}/.local/bin", f"{home}/.cargo/bin", "/opt/homebrew/bin"]
    full_path = ":".join(d for d in extra_dirs if Path(d).exists()) + f":{env_path}"

    payload: dict[str, Any] = {
        "Label": SERVICE_NAME,
        "ProgramArguments": cmd,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(out_log.resolve()),
        "StandardErrorPath": str(err_log.resolve()),
        "WorkingDirectory": home,
        "EnvironmentVariables": {
            "PATH": full_path,
            "PYTHONUNBUFFERED": "1",
        },
    }
    return plistlib.dumps(payload, sort_keys=True).decode("utf-8")


def generate_systemd_unit(
    *,
    host: str = "0.0.0.0",
    port: int = 9090,
    exec_args: list[str] | None = None,
) -> str:
    """Generate systemd user service unit string."""
    if exec_args is None:
        exec_args = resolve_aitop_exec()

    exec_start = " ".join([*exec_args, "serve", "--host", host, "--port", str(port)])

    return f"""[Unit]
Description=aitop serve fleet gateway
Documentation=https://github.com/NicoMancinelli/aitop
After=network.target

[Service]
Type=exec
ExecStart={exec_start}
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""


# --------------------------------------------------------------------------- #
# macOS Management (launchd)
# --------------------------------------------------------------------------- #


def _run_cmd(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def install_macos(host: str = "0.0.0.0", port: int = 9090) -> tuple[bool, str]:
    plist_path = get_launchd_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    get_log_paths()

    _run_cmd(["launchctl", "unload", "-w", str(plist_path)])

    content = generate_launchd_plist(host=host, port=port)
    plist_path.write_text(content, encoding="utf-8")
    plist_path.chmod(0o644)

    res = _run_cmd(["launchctl", "load", "-w", str(plist_path)])
    if res.returncode != 0:
        return False, f"Failed to load launchd agent: {res.stderr.strip() or res.stdout.strip()}"

    return True, f"Installed and loaded launchd user agent at {plist_path}"


def status_macos() -> ServiceStatus:
    plist_path = get_launchd_plist_path()
    installed = plist_path.is_file()

    if not installed:
        return ServiceStatus(
            platform="darwin",
            installed=False,
            active=False,
            pid=None,
            detail=f"Plist not found at {plist_path}",
            unit_path=None,
        )

    res = _run_cmd(["launchctl", "list", SERVICE_NAME])
    if res.returncode != 0:
        return ServiceStatus(
            platform="darwin",
            installed=True,
            active=False,
            pid=None,
            detail="Plist exists but agent is not loaded in launchd",
            unit_path=plist_path,
        )

    pid = None
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith('"PID"') or line.startswith("PID ="):
            parts = line.replace("=", " ").replace(";", " ").replace('"', " ").split()
            for p in parts:
                if p.isdigit():
                    pid = int(p)
                    break

    is_active = pid is not None
    detail = f"Running (PID {pid})" if is_active else "Loaded in launchd (idle or restarting)"
    return ServiceStatus(
        platform="darwin",
        installed=True,
        active=is_active,
        pid=pid,
        detail=detail,
        unit_path=plist_path,
    )


def uninstall_macos() -> tuple[bool, str]:
    plist_path = get_launchd_plist_path()
    if not plist_path.is_file():
        return True, "Service not installed (no plist found)."

    _run_cmd(["launchctl", "unload", "-w", str(plist_path)])

    try:
        plist_path.unlink()
    except OSError as exc:
        return False, f"Failed to delete {plist_path}: {exc}"

    return True, f"Unloaded and removed {plist_path}"


# --------------------------------------------------------------------------- #
# Linux Management (systemd)
# --------------------------------------------------------------------------- #


def install_linux(host: str = "0.0.0.0", port: int = 9090) -> tuple[bool, str]:
    unit_path = get_systemd_unit_path()
    unit_path.parent.mkdir(parents=True, exist_ok=True)

    content = generate_systemd_unit(host=host, port=port)
    unit_path.write_text(content, encoding="utf-8")
    unit_path.chmod(0o644)

    reload_res = _run_cmd(["systemctl", "--user", "daemon-reload"])
    if reload_res.returncode != 0:
        return False, f"daemon-reload failed: {reload_res.stderr.strip()}"

    enable_res = _run_cmd(["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT])
    if enable_res.returncode != 0:
        return False, f"Failed to enable/start service: {enable_res.stderr.strip()}"

    return True, f"Installed and started systemd user service at {unit_path}"


def status_linux() -> ServiceStatus:
    unit_path = get_systemd_unit_path()
    installed = unit_path.is_file()

    if not installed:
        return ServiceStatus(
            platform="linux",
            installed=False,
            active=False,
            pid=None,
            detail=f"Unit file not found at {unit_path}",
            unit_path=None,
        )

    active_res = _run_cmd(["systemctl", "--user", "is-active", SYSTEMD_UNIT])
    active_state = active_res.stdout.strip()
    is_active = active_state == "active"

    pid = None
    show_res = _run_cmd(["systemctl", "--user", "show", SYSTEMD_UNIT, "--property=MainPID"])
    if show_res.returncode == 0:
        for line in show_res.stdout.splitlines():
            if line.startswith("MainPID="):
                val = line.split("=", 1)[1].strip()
                if val.isdigit() and int(val) > 0:
                    pid = int(val)

    detail = f"Active ({active_state})" + (f", PID {pid}" if pid else "")
    return ServiceStatus(
        platform="linux",
        installed=True,
        active=is_active,
        pid=pid,
        detail=detail,
        unit_path=unit_path,
    )


def uninstall_linux() -> tuple[bool, str]:
    unit_path = get_systemd_unit_path()
    if not unit_path.is_file():
        return True, "Service not installed (no unit file found)."

    _run_cmd(["systemctl", "--user", "stop", SYSTEMD_UNIT])
    _run_cmd(["systemctl", "--user", "disable", SYSTEMD_UNIT])

    try:
        unit_path.unlink()
    except OSError as exc:
        return False, f"Failed to delete {unit_path}: {exc}"

    _run_cmd(["systemctl", "--user", "daemon-reload"])
    _run_cmd(["systemctl", "--user", "reset-failed", SYSTEMD_UNIT])

    return True, f"Stopped, disabled, and removed {unit_path}"


# --------------------------------------------------------------------------- #
# Public Dispatch API
# --------------------------------------------------------------------------- #


def install_service(host: str = "0.0.0.0", port: int = 9090) -> tuple[bool, str]:
    """Install and start the background service on the current platform."""
    if sys.platform == "darwin":
        return install_macos(host=host, port=port)
    elif sys.platform.startswith("linux"):
        return install_linux(host=host, port=port)
    return False, f"Background service management not supported on platform: {sys.platform}"


def status_service() -> ServiceStatus:
    """Query background service status on the current platform."""
    if sys.platform == "darwin":
        return status_macos()
    elif sys.platform.startswith("linux"):
        return status_linux()
    return ServiceStatus(
        platform=sys.platform,
        installed=False,
        active=False,
        pid=None,
        detail=f"Unsupported platform {sys.platform}",
        unit_path=None,
    )


def uninstall_service() -> tuple[bool, str]:
    """Stop and uninstall the background service on the current platform."""
    if sys.platform == "darwin":
        return uninstall_macos()
    elif sys.platform.startswith("linux"):
        return uninstall_linux()
    return False, f"Background service management not supported on platform: {sys.platform}"
