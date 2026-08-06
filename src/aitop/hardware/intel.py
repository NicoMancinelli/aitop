"""Intel GPU probe via `xpu-smi` (XPU Manager CLI).

Uses discovery for identity + VRAM size, then a one-shot `dump` for live
utilization / power / temperature / memory used. The older `xpumcli` binary
name is accepted as a fallback.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Any

from aitop.hardware.base import HardwareProbe, ProbeResult
from aitop.models import GPUSnapshot, Vendor
from aitop.utils.parse import first, to_float, to_int
from aitop.utils.proc import run, which

MIB = 1024 * 1024

# discovery --dump columns: Device ID, Device Name, Memory Physical Size (MiB)
_DISCOVERY_DUMP = "1,2,16"
# dump metrics: util %, power W, temp C, memory used MiB
_LIVE_METRICS = "0,1,3,18"


class IntelProbe(HardwareProbe):
    name = "intel"

    def __init__(self) -> None:
        self._bin: str | None = None

    async def available(self) -> bool:
        for name in ("xpu-smi", "xpumcli"):
            if which(name) is not None:
                self._bin = name
                return True
        return False

    @property
    def binary(self) -> str:
        return self._bin or "xpu-smi"

    async def probe(self) -> ProbeResult:
        result = ProbeResult()
        devices = await self._discover()
        if devices is None:
            result.note(f"Intel telemetry unavailable ({self.binary} discovery failed)")
            return result
        if not devices:
            result.note(f"{self.binary} reported no GPUs")
            return result

        live = await self._live_metrics([d["index"] for d in devices])
        total_power = 0.0
        for device in devices:
            metrics = live.get(device["index"], {})
            used_mib = to_float(metrics.get("mem_used_mib"))
            total_mib = to_float(device.get("vram_total_mib"))
            # If discovery omitted size but dump gave util% + used, back out total.
            mem_util = to_float(metrics.get("mem_util"))
            if total_mib is None and used_mib is not None and mem_util and mem_util > 0:
                total_mib = used_mib / (mem_util / 100.0)

            gpu = GPUSnapshot(
                index=device["index"],
                name=str(device.get("name") or "Intel GPU"),
                vendor=Vendor.INTEL,
                driver_version=device.get("driver"),
                api_version="Level Zero / XPU",
                utilization_percent=to_float(metrics.get("util")),
                vram_total_bytes=int(total_mib * MIB) if total_mib else None,
                vram_used_bytes=int(used_mib * MIB) if used_mib is not None else None,
                temperature_c=to_float(metrics.get("temp_c")),
                power_watts=to_float(metrics.get("power_w")),
            )
            result.gpus.append(gpu)
            if gpu.power_watts:
                total_power += gpu.power_watts

        if total_power:
            result.total_power_watts = round(total_power, 1)
        return result

    async def _discover(self) -> list[dict[str, Any]] | None:
        """Return device dicts, or None when the tool itself failed."""
        # Prefer JSON discovery — richer and stable across XPUM versions.
        out = await run(self.binary, "discovery", "-j", timeout=5.0)
        if out.ok:
            devices = _parse_discovery_json(out.stdout)
            if devices is not None:
                # Enrich with per-device memory size when the list payload is thin.
                for device in devices:
                    if device.get("vram_total_mib") is not None:
                        continue
                    detail = await run(
                        self.binary,
                        "discovery",
                        "-d",
                        str(device["index"]),
                        "-j",
                        timeout=5.0,
                    )
                    if detail.ok:
                        size = _memory_mib_from_detail(detail.stdout)
                        if size is not None:
                            device["vram_total_mib"] = size
                return devices

        # CSV dump fallback (used by some packaging / older builds).
        out = await run(
            self.binary,
            "discovery",
            "--dump",
            _DISCOVERY_DUMP,
            timeout=5.0,
        )
        if not out.ok:
            return None
        return _parse_discovery_csv(out.stdout)

    async def _live_metrics(self, indexes: list[int]) -> dict[int, dict[str, float | None]]:
        if not indexes:
            return {}
        device_arg = ",".join(str(i) for i in indexes)
        out = await run(
            self.binary,
            "dump",
            "-d",
            device_arg,
            "-m",
            _LIVE_METRICS,
            "-n",
            "1",
            timeout=6.0,
        )
        if not out.ok:
            return {}
        return _parse_dump_csv(out.stdout)


def _parse_discovery_json(text: str) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    entries: list[Any]
    if isinstance(payload, dict):
        entries = payload.get("device_list") or payload.get("deviceList") or []
    elif isinstance(payload, list):
        entries = payload
    else:
        return None
    if not isinstance(entries, list):
        return None

    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        dtype = str(first(entry, "device_type", "deviceType", default="GPU") or "GPU")
        if dtype.upper() not in {"GPU", "DEVICE", ""}:
            continue
        index = to_int(first(entry, "device_id", "deviceId", "id"))
        if index is None:
            continue
        name = first(entry, "device_name", "deviceName", "name") or "Intel GPU"
        driver = first(entry, "driver_version", "driverVersion", "kernel_version")
        mem = to_float(
            first(
                entry,
                "memory_physical_size_mb",
                "memoryPhysicalSize",
                "memory_physical_size",
            )
        )
        out.append(
            {
                "index": index,
                "name": str(name),
                "driver": str(driver) if driver else None,
                "vram_total_mib": mem,
            }
        )
    return out


def _memory_mib_from_detail(text: str) -> float | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    # Detail payloads vary: sometimes nested under device_id / device.
    candidates: list[Any] = [payload]
    for key in ("device", "device_list", "deviceList"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
        elif isinstance(value, list):
            candidates.extend(v for v in value if isinstance(v, dict))
    for entry in candidates:
        for key, raw in entry.items():
            if "memory" in str(key).lower() and "physical" in str(key).lower():
                value = to_float(raw)
                if value is not None:
                    return value
            if str(key).lower() in {
                "memory_physical_size_mb",
                "memoryphysicalsize",
                "memory_physical_size",
            }:
                value = to_float(raw)
                if value is not None:
                    return value
    return None


def _parse_discovery_csv(text: str) -> list[dict[str, Any]]:
    rows = _csv_rows(text)
    out: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < 2:
            continue
        # Skip header-ish first cell.
        index = to_int(row[0])
        if index is None:
            continue
        name = row[1].strip() or "Intel GPU"
        mem = to_float(row[2]) if len(row) > 2 else None
        out.append({"index": index, "name": name, "driver": None, "vram_total_mib": mem})
    return out


def _parse_dump_csv(text: str) -> dict[int, dict[str, float | None]]:
    """Parse `xpu-smi dump -m 0,1,3,18 -n 1` CSV into per-device metrics."""
    rows = _csv_rows(text)
    out: dict[int, dict[str, float | None]] = {}
    for row in rows:
        # Typical: Timestamp, DeviceId, util, power, temp, mem_used
        if len(row) < 3:
            continue
        # Device id is usually column 1 after timestamp; tolerate missing stamp.
        index = to_int(row[1]) if len(row) >= 6 else to_int(row[0])
        values = row[2:] if len(row) >= 6 else row[1:]
        if index is None or len(values) < 1:
            continue
        out[index] = {
            "util": to_float(values[0]) if len(values) > 0 else None,
            "power_w": to_float(values[1]) if len(values) > 1 else None,
            "temp_c": to_float(values[2]) if len(values) > 2 else None,
            "mem_used_mib": to_float(values[3]) if len(values) > 3 else None,
            "mem_util": None,
        }
    return out


def _csv_rows(text: str) -> list[list[str]]:
    reader = csv.reader(StringIO(text))
    rows: list[list[str]] = []
    for row in reader:
        cleaned = [cell.strip() for cell in row]
        if not cleaned or all(not c for c in cleaned):
            continue
        # Drop trailing empty-only rows sometimes emitted by xpu-smi.
        if len(cleaned) == 1 and cleaned[0].upper() in {"N/A", ""}:
            continue
        rows.append(cleaned)
    return rows
