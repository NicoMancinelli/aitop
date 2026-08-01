"""Cross-platform CPU / memory / host facts via psutil.

Everything here works identically on macOS and Linux; platform-specific
enrichment (P/E core split, thermals, power) is layered on top by the probes
in `apple.py`, `nvidia.py` and `amd.py`.
"""

from __future__ import annotations

import platform
import sys
import time

import psutil

from aitop.models import CPUSnapshot, HostSnapshot, MemorySnapshot


def platform_id() -> str:
    """'darwin-arm64', 'linux-x86_64', ..."""
    return f"{sys.platform}-{platform.machine()}"


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def collect_host() -> HostSnapshot:
    uname = platform.uname()
    return HostSnapshot(
        hostname=uname.node,
        os_name=_os_name(),
        os_version=platform.mac_ver()[0] or uname.release,
        kernel=f"{uname.system} {uname.release}",
        platform_id=platform_id(),
        uptime_seconds=max(0.0, time.time() - psutil.boot_time()),
        python_version=platform.python_version(),
    )


def _os_name() -> str:
    if sys.platform == "darwin":
        return "macOS"
    if sys.platform.startswith("linux"):
        try:
            with open("/etc/os-release", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return "Linux"
    return platform.system()


def collect_cpu(per_core: bool = True) -> CPUSnapshot:
    """Non-blocking CPU sample.

    `interval=None` compares against the previous call, so the first sample in
    a process reads 0% — the collector primes it once at startup.
    """
    per_core_values = psutil.cpu_percent(interval=None, percpu=True) if per_core else []
    overall = (
        sum(per_core_values) / len(per_core_values)
        if per_core_values
        else psutil.cpu_percent(interval=None)
    )
    freq = _cpu_freq()
    return CPUSnapshot(
        model=cpu_model(),
        arch=platform.machine(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        load_percent=round(overall, 1),
        per_core_percent=[round(v, 1) for v in per_core_values],
        frequency_mhz=freq,
        temperature_c=cpu_temperature(),
    )


def _cpu_freq() -> float | None:
    """psutil.cpu_freq() is unimplemented on Apple Silicon and some VMs."""
    try:
        freq = psutil.cpu_freq()
    except (NotImplementedError, AttributeError, OSError):
        return None
    return freq.current if freq and freq.current else None


def cpu_model() -> str:
    if sys.platform == "darwin":
        return _sysctl_str("machdep.cpu.brand_string") or platform.processor() or "Apple Silicon"
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith(("model name", "Model")):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def cpu_temperature() -> float | None:
    """Package temperature on Linux. macOS needs powermetrics — see apple.py."""
    try:
        sensors = psutil.sensors_temperatures()
    except (AttributeError, NotImplementedError, OSError):
        return None
    for key in ("coretemp", "k10temp", "cpu_thermal", "zenpower", "acpitz"):
        readings = sensors.get(key)
        if readings:
            package = next((r for r in readings if "package" in (r.label or "").lower()), None)
            return (package or readings[0]).current
    return None


def collect_memory() -> MemorySnapshot:
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return MemorySnapshot(
        total_bytes=virtual.total,
        used_bytes=virtual.total - virtual.available,
        available_bytes=virtual.available,
        swap_total_bytes=swap.total,
        swap_used_bytes=swap.used,
        unified=is_apple_silicon(),
    )


# --------------------------------------------------------------------------- #
# sysctl helpers (macOS)
# --------------------------------------------------------------------------- #


def _sysctl_str(key: str) -> str | None:
    if sys.platform != "darwin":
        return None
    import subprocess  # local import: only macOS pays for it

    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


def sysctl_int(key: str) -> int | None:
    value = _sysctl_str(key)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def apple_core_split() -> tuple[int | None, int | None]:
    """(performance cores, efficiency cores) on Apple Silicon."""
    if not is_apple_silicon():
        return None, None
    return sysctl_int("hw.perflevel0.logicalcpu"), sysctl_int("hw.perflevel1.logicalcpu")
