"""AMD ROCm probe via `rocm-smi --json`.

ROCm's JSON keys are verbose and have drifted across releases (`GPU use (%)`,
`GPU Utilization (%)`, ...), so every field is looked up through a list of
known spellings and a fuzzy fallback rather than a fixed key.
"""

from __future__ import annotations

import json
from typing import Any

from aitop.hardware.base import HardwareProbe, ProbeResult
from aitop.models import GPUSnapshot, Vendor
from aitop.utils.parse import to_float, to_int
from aitop.utils.proc import run, which

_UTIL_KEYS = ("GPU use (%)", "GPU Utilization (%)", "GPU use", "gfx_activity")
_VRAM_TOTAL_KEYS = ("VRAM Total Memory (B)", "vram_total", "Total Memory (B)")
_VRAM_USED_KEYS = ("VRAM Total Used Memory (B)", "vram_used", "Used Memory (B)")
_TEMP_KEYS = (
    "Temperature (Sensor junction) (C)",
    "Temperature (Sensor edge) (C)",
    "Temperature (Sensor memory) (C)",
)
_POWER_KEYS = (
    "Average Graphics Package Power (W)",
    "Current Socket Graphics Package Power (W)",
    "Average Socket Power (W)",
)
_NAME_KEYS = ("Card series", "Card Series", "Card model", "Device Name", "Market Name")
_DRIVER_KEYS = ("Driver version", "Driver Version")


class AMDProbe(HardwareProbe):
    name = "amd"

    async def available(self) -> bool:
        return which("rocm-smi") is not None

    async def probe(self) -> ProbeResult:
        result = ProbeResult()
        out = await run(
            "rocm-smi",
            "--showproductname",
            "--showuse",
            "--showmemuse",
            "--showmeminfo",
            "vram",
            "--showtemp",
            "--showpower",
            "--showdriverversion",
            "--json",
            timeout=6.0,
        )
        if not out.ok:
            result.note(f"ROCm telemetry unavailable ({out.reason})")
            return result

        try:
            payload = json.loads(out.stdout)
        except json.JSONDecodeError:
            result.note("rocm-smi returned non-JSON output")
            return result
        if not isinstance(payload, dict):
            result.note("rocm-smi returned an unexpected JSON shape")
            return result

        total_power = 0.0
        for key, card in sorted(payload.items()):
            if not key.startswith("card") or not isinstance(card, dict):
                continue
            gpu = _to_snapshot(key, card)
            result.gpus.append(gpu)
            if gpu.power_watts:
                total_power += gpu.power_watts

        if not result.gpus:
            result.note("rocm-smi reported no cards")
        elif total_power:
            result.total_power_watts = round(total_power, 1)
        return result


def _to_snapshot(card_key: str, card: dict[str, Any]) -> GPUSnapshot:
    return GPUSnapshot(
        index=to_int(card_key.removeprefix("card")) or 0,
        name=_lookup_str(card, _NAME_KEYS) or "AMD GPU",
        vendor=Vendor.AMD,
        driver_version=_lookup_str(card, _DRIVER_KEYS),
        api_version="ROCm",
        utilization_percent=_lookup_float(card, _UTIL_KEYS, fuzzy="use"),
        vram_total_bytes=to_int(_lookup_float(card, _VRAM_TOTAL_KEYS)),
        vram_used_bytes=to_int(_lookup_float(card, _VRAM_USED_KEYS)),
        temperature_c=_lookup_float(card, _TEMP_KEYS, fuzzy="temperature"),
        power_watts=_lookup_float(card, _POWER_KEYS, fuzzy="power"),
    )


def _lookup_str(card: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = card.get(key)
        if isinstance(value, str) and value.strip() and value.strip().upper() != "N/A":
            return value.strip()
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
        for key, raw in card.items():
            if fuzzy in key.lower():
                value = to_float(raw)
                if value is not None:
                    return value
    return None
