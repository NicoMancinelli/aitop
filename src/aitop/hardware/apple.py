"""Apple Silicon probe.

Three sources, in increasing order of privilege:

* `ioreg -rc AGXAccelerator`  — GPU utilisation and in-use GPU memory. No root.
* `system_profiler SPDisplaysDataType -json` — GPU name, core count, Metal
  family. No root, but slow (~1s), so it is cached for the process lifetime.
* `powermetrics` — package/GPU/ANE power and die temperature. **Root only.**
  We probe it with `sudo -n` and, if that would prompt, skip it and record a
  degraded note rather than hanging the dashboard on a password prompt.

Unified memory means there is no separate VRAM pool. What we report as
`vram_total_bytes` is the GPU *wired limit* — the ceiling Metal will let a
process allocate — because that, not total RAM, is what a model has to fit in.
"""

from __future__ import annotations

import json
import re
from typing import Any

from aitop.hardware.base import HardwareProbe, ProbeResult
from aitop.hardware.system import apple_core_split, is_apple_silicon, sysctl_int
from aitop.models import GPUSnapshot, Vendor
from aitop.utils.parse import to_float, to_int
from aitop.utils.proc import can_sudo_nopasswd, run

GB = 1024**3

_IOREG_UTIL = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
_IOREG_IN_USE = re.compile(r'"In use system memory"\s*=\s*(\d+)')
_IOREG_ALLOC = re.compile(r'"Alloc system memory"\s*=\s*(\d+)')

_PM_GPU_POWER = re.compile(r"^GPU Power:\s*([\d.]+)\s*mW", re.MULTILINE)
_PM_CPU_POWER = re.compile(r"^CPU Power:\s*([\d.]+)\s*mW", re.MULTILINE)
_PM_ANE_POWER = re.compile(r"^ANE Power:\s*([\d.]+)\s*mW", re.MULTILINE)
_PM_COMBINED = re.compile(r"^Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW", re.MULTILINE)
_PM_GPU_RESIDENCY = re.compile(r"^GPU HW active residency:\s*([\d.]+)%", re.MULTILINE)
_PM_DIE_TEMP = re.compile(r"die temperature:\s*([\d.]+)\s*C", re.IGNORECASE)


class AppleSiliconProbe(HardwareProbe):
    name = "apple"

    def __init__(self, allow_powermetrics: bool = True) -> None:
        self._static: dict[str, Any] | None = None
        self._allow_powermetrics = allow_powermetrics
        self._powermetrics_ok: bool | None = None

    async def available(self) -> bool:
        return is_apple_silicon()

    async def probe(self) -> ProbeResult:
        result = ProbeResult()
        static = await self._static_info(result)

        util, in_use = await self._ioreg(result)
        wired_limit, estimated = gpu_wired_limit()
        if estimated:
            result.note("GPU memory ceiling is an estimate (iogpu.wired_limit_mb unset)")

        power = await self._powermetrics(result)

        result.gpus.append(
            GPUSnapshot(
                index=0,
                name=static.get("name", "Apple GPU"),
                vendor=Vendor.APPLE,
                driver_version=static.get("driver"),
                api_version=static.get("metal"),
                core_count=static.get("cores"),
                utilization_percent=power.get("gpu_residency", util),
                vram_total_bytes=wired_limit,
                vram_used_bytes=in_use,
                temperature_c=power.get("die_temp"),
                power_watts=power.get("gpu_watts"),
                unified_memory=True,
            )
        )

        perf, eff = apple_core_split()
        result.cpu_overrides.update(
            {
                "performance_cores": perf,
                "efficiency_cores": eff,
                "power_watts": power.get("cpu_watts"),
                "temperature_c": power.get("die_temp"),
            }
        )
        result.total_power_watts = power.get("combined_watts")
        return result

    # -- ioreg --------------------------------------------------------------- #

    async def _ioreg(self, result: ProbeResult) -> tuple[float | None, int | None]:
        """GPU utilisation and in-use GPU memory, without root."""
        out = await run("ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator", timeout=3.0)
        if not out.ok:
            result.note(f"GPU utilisation unavailable ({out.reason or 'ioreg failed'})")
            return None, None

        util_match = _IOREG_UTIL.search(out.stdout)
        mem_match = _IOREG_IN_USE.search(out.stdout) or _IOREG_ALLOC.search(out.stdout)
        return (
            float(util_match.group(1)) if util_match else None,
            int(mem_match.group(1)) if mem_match else None,
        )

    # -- system_profiler ----------------------------------------------------- #

    async def _static_info(self, result: ProbeResult) -> dict[str, Any]:
        """GPU name / core count / Metal family. Slow, so cached per process."""
        if self._static is not None:
            return self._static

        out = await run("system_profiler", "-json", "SPDisplaysDataType", timeout=10.0)
        info: dict[str, Any] = {}
        if out.ok:
            try:
                displays = json.loads(out.stdout).get("SPDisplaysDataType") or []
            except (json.JSONDecodeError, AttributeError):
                displays = []
            if displays:
                entry = displays[0]
                info["name"] = entry.get("sppci_model") or entry.get("_name") or "Apple GPU"
                info["cores"] = to_int(entry.get("sppci_cores"))
                # The key was renamed in macOS 15; accept both spellings.
                info["metal"] = _metal_family(
                    entry.get("spdisplays_mtlgpufamilysupport")
                    or entry.get("spdisplays_metalfamily")
                )
                info["driver"] = entry.get("spdisplays_gmux-version")
        else:
            result.note(f"GPU details unavailable ({out.reason})")

        self._static = info
        return info

    # -- powermetrics (root) -------------------------------------------------- #

    async def _powermetrics(self, result: ProbeResult) -> dict[str, float]:
        if not self._allow_powermetrics:
            return {}

        if self._powermetrics_ok is None:
            self._powermetrics_ok = await can_sudo_nopasswd("powermetrics")
            if not self._powermetrics_ok:
                result.note("power/thermal data needs root — run `sudo aitop` for watts and °C")
        if not self._powermetrics_ok:
            return {}

        out = await run(
            "sudo",
            "-n",
            "powermetrics",
            "--samplers",
            "cpu_power,gpu_power,thermal",
            "-n",
            "1",
            "-i",
            "200",
            timeout=8.0,
        )
        if not out.ok:
            result.note(f"powermetrics failed ({out.reason})")
            self._powermetrics_ok = False
            return {}
        return _parse_powermetrics(out.stdout)


def _parse_powermetrics(text: str) -> dict[str, float]:
    """Pull the handful of numbers we care about out of the plain-text report."""
    values: dict[str, float] = {}
    for key, pattern, scale in (
        ("gpu_watts", _PM_GPU_POWER, 0.001),
        ("cpu_watts", _PM_CPU_POWER, 0.001),
        ("ane_watts", _PM_ANE_POWER, 0.001),
        ("combined_watts", _PM_COMBINED, 0.001),
        ("gpu_residency", _PM_GPU_RESIDENCY, 1.0),
        ("die_temp", _PM_DIE_TEMP, 1.0),
    ):
        match = pattern.search(text)
        if match:
            parsed = to_float(match.group(1))
            if parsed is not None:
                values[key] = parsed * scale

    if "combined_watts" not in values:
        parts = [values[k] for k in ("cpu_watts", "gpu_watts", "ane_watts") if k in values]
        if parts:
            values["combined_watts"] = sum(parts)
    return values


def _metal_family(raw: Any) -> str | None:
    """'spdisplays_metal3' -> 'Metal 3'."""
    if not isinstance(raw, str) or not raw:
        return None
    match = re.search(r"metal\s*(\d+)", raw, re.IGNORECASE)
    return f"Metal {match.group(1)}" if match else raw.replace("spdisplays_", "")


def gpu_wired_limit() -> tuple[int | None, bool]:
    """GPU-allocatable ceiling in bytes, and whether it had to be estimated.

    `iogpu.wired_limit_mb` is authoritative when a user has raised it (a common
    tweak for large models). It reads 0 by default, in which case we fall back
    to Metal's `recommendedMaxWorkingSetSize` ratios: ~2/3 of unified memory up
    to 36 GB of RAM, ~3/4 above it.
    """
    explicit = sysctl_int("iogpu.wired_limit_mb")
    if explicit:
        return explicit * 1024 * 1024, False

    total = sysctl_int("hw.memsize")
    if not total:
        return None, True
    ratio = 2 / 3 if total <= 36 * GB else 0.75
    return int(total * ratio), True
