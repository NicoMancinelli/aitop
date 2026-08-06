"""Intel GPU probe via `xpu-smi` (preferred) or `intel_gpu_top`.

`xpu-smi` covers Arc and Data Center GPUs when XPU-SMI / XPU Manager is
installed. Consumer iGPUs more often expose `intel_gpu_top` from
igt-gpu-tools. Either path degrades cleanly when the tool is missing or the
JSON shape drifts — same contract as the NVIDIA / AMD probes.
"""

from __future__ import annotations

import json
import re
from typing import Any

from aitop.hardware.base import HardwareProbe, ProbeResult
from aitop.models import GPUSnapshot, Vendor
from aitop.utils.parse import first, to_float, to_int
from aitop.utils.proc import run, which

MIB = 1024 * 1024

_NAME_KEYS = (
    "device_name",
    "Device Name",
    "deviceName",
    "name",
    "model",
)
_DRIVER_KEYS = ("driver_version", "Driver Version", "driverVersion")
_VRAM_TOTAL_KEYS = (
    "memory_physical_size_byte",
    "memory_physical_size",
    "Memory Physical Size",
    "memoryPhysicalSize",
)
_VRAM_USED_KEYS = (
    "gpu_memory_used_bytes",
    "gpu_memory_used",
    "GPU Memory Used (MiB)",
    "GPU Memory Used",
    "memory_used",
)
_UTIL_KEYS = (
    "gpu_utilization",
    "GPU Utilization (%)",
    "Average % utilization of all GPU Engines",
    "gpu_engine_util",
    "gpu_util",
)
_TEMP_KEYS = (
    "gpu_core_temperature",
    "GPU Core Temperature (C)",
    "GPU Core Temperature (Celsius Degree)",
    "temperature",
    "temp",
)
_POWER_KEYS = (
    "gpu_power",
    "GPU Power (W)",
    "power",
    "power_watts",
)
_EU_KEYS = ("number_of_eus", "Number of EUs", "num_eus", "eu_count")
_API_KEYS = ("level_zero_version", "oneapi_version", "api_version")


class IntelProbe(HardwareProbe):
    name = "intel"

    async def available(self) -> bool:
        return which("xpu-smi") is not None or which("intel_gpu_top") is not None

    async def probe(self) -> ProbeResult:
        if which("xpu-smi") is not None:
            result = await self._probe_xpu_smi()
            if result.gpus:
                return result
            # Fall through to intel_gpu_top when xpu-smi listed nothing useful.
            if which("intel_gpu_top") is None:
                return result
            fallback = await self._probe_intel_gpu_top()
            if fallback.gpus:
                # Keep any notes from the primary attempt.
                for note in result.degraded:
                    fallback.note(note)
                return fallback
            for note in fallback.degraded:
                result.note(note)
            return result

        return await self._probe_intel_gpu_top()

    # -- xpu-smi ------------------------------------------------------------ #

    async def _probe_xpu_smi(self) -> ProbeResult:
        result = ProbeResult()
        out = await run("xpu-smi", "discovery", "-j", timeout=6.0)
        if not out.ok:
            result.note(f"Intel xpu-smi unavailable ({out.reason})")
            return result

        payload = _loads_json(out.stdout)
        if payload is None:
            result.note("xpu-smi discovery returned non-JSON output")
            return result

        devices = _device_list(payload)
        if not devices:
            result.note("xpu-smi reported no GPUs")
            return result

        total_power = 0.0
        for index, device in enumerate(devices):
            device_id = first(device, "device_id", "deviceId", default=index)
            detail = await self._xpu_discovery_detail(device_id)
            stats = await self._xpu_stats(device_id)
            gpu = _snapshot_from_xpu(index, device, detail, stats)
            result.gpus.append(gpu)
            if gpu.power_watts:
                total_power += gpu.power_watts

        if total_power:
            result.total_power_watts = round(total_power, 1)
        return result

    async def _xpu_discovery_detail(self, device_id: Any) -> dict[str, Any]:
        out = await run("xpu-smi", "discovery", "-d", str(device_id), "-j", timeout=6.0)
        if out.ok:
            payload = _loads_json(out.stdout)
            if isinstance(payload, dict):
                return _flatten_device_payload(payload)
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                return _flatten_device_payload(payload[0])
        # Text table fallback — still gets Memory Physical Size / Driver Version.
        out = await run("xpu-smi", "discovery", "-d", str(device_id), timeout=6.0)
        if out.ok:
            return _parse_xpu_table(out.stdout)
        return {}

    async def _xpu_stats(self, device_id: Any) -> dict[str, Any]:
        out = await run("xpu-smi", "stats", "-d", str(device_id), "-j", timeout=6.0)
        if out.ok:
            payload = _loads_json(out.stdout)
            if isinstance(payload, dict):
                return _flatten_device_payload(payload)
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                return _flatten_device_payload(payload[0])
        out = await run("xpu-smi", "stats", "-d", str(device_id), timeout=6.0)
        if out.ok:
            return _parse_xpu_table(out.stdout)
        return {}

    # -- intel_gpu_top ------------------------------------------------------ #

    async def _probe_intel_gpu_top(self) -> ProbeResult:
        result = ProbeResult()
        out = await run(
            "intel_gpu_top",
            "-J",
            "-s",
            "100",
            "-n",
            "1",
            timeout=5.0,
        )
        if not out.ok:
            result.note(f"Intel intel_gpu_top unavailable ({out.reason})")
            return result

        samples = _parse_intel_gpu_top_json(out.stdout)
        if not samples:
            result.note("intel_gpu_top returned no usable samples")
            return result

        total_power = 0.0
        for index, sample in enumerate(samples):
            gpu = _snapshot_from_igt(index, sample)
            result.gpus.append(gpu)
            if gpu.power_watts:
                total_power += gpu.power_watts

        if total_power:
            result.total_power_watts = round(total_power, 1)
        return result


def _snapshot_from_xpu(
    index: int,
    device: dict[str, Any],
    detail: dict[str, Any],
    stats: dict[str, Any],
) -> GPUSnapshot:
    merged = {**device, **detail, **stats}
    name = _lookup_str(merged, _NAME_KEYS) or "Intel GPU"
    vram_total = _vram_bytes(merged, _VRAM_TOTAL_KEYS)
    vram_used = _vram_bytes(merged, _VRAM_USED_KEYS)
    if vram_used is None:
        util_mem = _lookup_float(
            merged,
            ("GPU Memory Util (%)", "gpu_memory_utilization", "memory_utilization"),
            fuzzy="memory util",
        )
        if util_mem is not None and vram_total:
            vram_used = int(vram_total * util_mem / 100.0)

    api = _lookup_str(merged, _API_KEYS)
    return GPUSnapshot(
        index=to_int(first(device, "device_id", "deviceId", default=index)) or index,
        name=name,
        vendor=Vendor.INTEL,
        driver_version=_lookup_str(merged, _DRIVER_KEYS),
        api_version=api or "Level Zero",
        core_count=to_int(_lookup_float(merged, _EU_KEYS, fuzzy="eus")),
        utilization_percent=_lookup_float(merged, _UTIL_KEYS, fuzzy="util"),
        vram_total_bytes=vram_total,
        vram_used_bytes=vram_used,
        temperature_c=_lookup_float(merged, _TEMP_KEYS, fuzzy="temperature"),
        power_watts=_lookup_float(merged, _POWER_KEYS, fuzzy="power"),
    )


def _snapshot_from_igt(index: int, sample: dict[str, Any]) -> GPUSnapshot:
    engines = sample.get("engines")
    util = _igt_engine_busy(engines)
    power = None
    power_block = sample.get("power")
    if isinstance(power_block, dict):
        power = to_float(first(power_block, "GPU", "gpu", "Package", "package"))
    freq = sample.get("frequency")
    # Frequency is informational; GPUSnapshot has no dedicated field.
    _ = freq
    name = "Intel GPU"
    client_card = sample.get("drm-card") or sample.get("card")
    if isinstance(client_card, str) and client_card.strip():
        name = client_card.strip()
    return GPUSnapshot(
        index=index,
        name=name,
        vendor=Vendor.INTEL,
        api_version="i915/Xe",
        utilization_percent=util,
        power_watts=power,
        # iGPU VRAM is typically system RAM; leave unset rather than guess.
        unified_memory=True,
    )


def _igt_engine_busy(engines: Any) -> float | None:
    if not isinstance(engines, dict) or not engines:
        return None
    busies: list[float] = []
    for meta in engines.values():
        if not isinstance(meta, dict):
            continue
        busy = to_float(first(meta, "busy", "Busy"))
        if busy is not None:
            busies.append(busy)
    if not busies:
        return None
    # Peak engine busy is the most btop-like single number for a busy GPU.
    return max(busies)


def _device_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        entries = first(payload, "device_list", "deviceList", "devices", default=None)
        if isinstance(entries, list):
            return [e for e in entries if isinstance(e, dict)]
        # Single-device detail payload.
        if any(k in payload for k in ("device_id", "device_name", "Device Name")):
            return [payload]
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    return []


def _flatten_device_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull nested device/tile stats into one flat dict for key lookups."""
    flat: dict[str, Any] = dict(payload)
    for key in ("device_level", "device", "stats", "data", "tile_level"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            flat.update(nested)
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    flat.update(item)
    # discovery -d -j sometimes wraps the card under device_list[0].
    for device in _device_list(payload):
        flat.update(device)
    return flat


def _parse_xpu_table(text: str) -> dict[str, Any]:
    """Parse the ascii-table form of `xpu-smi discovery/stats -d N`."""
    out: dict[str, Any] = {}
    # Stick to single-line rows — `[^|]` would otherwise span newlines and
    # swallow the real metric rows that follow a truncated discovery cell.
    row = re.compile(r"\|\s*([^|\n]+?)\s*\|\s*([^|\n]*?)\s*\|")
    for match in row.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if not key or not value:
            continue
        if key.lower() in {"device id", "device information"}:
            continue
        if key not in out:
            out[key] = value
    # Also accept "Key: Value" lines from some builds / wrapped cells.
    for line in text.splitlines():
        if ":" not in line or line.strip().startswith("+"):
            continue
        # Prefer the segment after the last pipe before the colon.
        working = line
        if "|" in working:
            # "|           | Driver Version: 16929133" → "Driver Version: 16929133"
            working = working.rsplit("|", 1)[-1]
        key, _, value = working.partition(":")
        key, value = key.strip(" |"), value.strip(" |")
        if key and value and key not in out:
            out[key] = value
    return out


def _vram_bytes(card: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key not in card:
            continue
        raw = card[key]
        if isinstance(raw, str) and re.search(r"miB|mib|MB", raw, re.IGNORECASE):
            mib = to_float(raw)
            return int(mib * MIB) if mib is not None else None
        if isinstance(raw, str) and re.search(r"giB|gib|GB", raw, re.IGNORECASE):
            gib = to_float(raw)
            return int(gib * MIB * 1024) if gib is not None else None
        # Numeric: treat large values as bytes, small as MiB (xpu-smi tables).
        number = to_float(raw)
        if number is None:
            continue
        if number >= 8 * MIB:  # already bytes (≥ 8 MiB)
            return int(number)
        return int(number * MIB)
    # Fuzzy: any key mentioning physical size / memory used.
    for key, raw in card.items():
        lower = key.lower()
        if "physical size" in lower or lower.endswith("memory used") or "memory used" in lower:
            return _vram_bytes({key: raw}, (key,))
    return None


def _lookup_str(card: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = card.get(key)
        if isinstance(value, str) and value.strip() and value.strip().upper() != "N/A":
            return value.strip()
        if isinstance(value, (int, float)) and key.lower().endswith("version"):
            return str(value)
    return None


def _lookup_float(
    card: dict[str, Any], keys: tuple[str, ...], fuzzy: str | None = None
) -> float | None:
    for key in keys:
        if key in card:
            value = to_float(card[key])
            if value is not None:
                return value
    if fuzzy:
        needle = fuzzy.lower()
        for key, raw in card.items():
            if needle in key.lower():
                value = to_float(raw)
                if value is not None:
                    return value
    return None


def _loads_json(text: str) -> Any | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # intel_gpu_top can emit concatenated objects / trailing junk.
        start = text.find("{")
        start_arr = text.find("[")
        if start_arr != -1 and (start == -1 or start_arr < start):
            start = start_arr
        if start == -1:
            return None
        chunk = text[start:]
        # Close a truncated JSON array from a killed intel_gpu_top.
        if chunk.startswith("[") and not chunk.rstrip().endswith("]"):
            chunk = chunk.rstrip().rstrip(",") + "]"
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return None


def _parse_intel_gpu_top_json(text: str) -> list[dict[str, Any]]:
    payload = _loads_json(text)
    if isinstance(payload, list):
        return [s for s in payload if isinstance(s, dict) and "engines" in s]
    if isinstance(payload, dict) and "engines" in payload:
        return [payload]
    return []
