"""Hardware probes: parsing, and the guarantee that failures degrade."""

from __future__ import annotations

import json

import pytest

from aitop.hardware.amd import AMDProbe, _to_snapshot
from aitop.hardware.apple import _metal_family, _parse_powermetrics
from aitop.hardware.base import HardwareProbe, ProbeResult
from aitop.hardware.collector import HardwareCollector
from aitop.hardware.nvidia import NvidiaProbe
from aitop.hardware.system import collect_cpu, collect_host, collect_memory
from aitop.models import Vendor
from aitop.utils.proc import CommandResult

POWERMETRICS_SAMPLE = """\
*** Sampled system activity ***

**** Processor usage ****
CPU Power: 1543 mW
GPU Power: 312 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 1855 mW

**** GPU usage ****
GPU HW active residency:  23.41% (444 MHz: 12% 612 MHz: 11%)

**** Thermals ****
CPU die temperature: 48.23 C
GPU die temperature: 45.10 C
"""

NVIDIA_CSV = (
    "0, NVIDIA GeForce RTX 4090, 550.54.14, 37, 24564, 12800, 61, 145.32, 450.00\n"
    "1, NVIDIA GeForce RTX 3090, 550.54.14, [N/A], 24576, 1024, 44, [N/A], 350.00\n"
)


# --------------------------------------------------------------------------- #
# Cross-platform basics
# --------------------------------------------------------------------------- #


def test_host_snapshot_is_populated():
    host = collect_host()
    assert host.hostname
    assert host.platform_id.count("-") >= 1
    assert host.uptime_seconds is not None and host.uptime_seconds > 0


def test_memory_snapshot_is_coherent():
    memory = collect_memory()
    assert memory.total_bytes > 0
    assert 0 <= memory.used_percent <= 100
    assert memory.used_bytes + memory.available_bytes == memory.total_bytes


def test_cpu_snapshot_is_coherent():
    cpu = collect_cpu(per_core=True)
    assert cpu.logical_cores and cpu.logical_cores > 0
    assert 0 <= cpu.load_percent <= 100
    assert len(cpu.per_core_percent) == cpu.logical_cores


# --------------------------------------------------------------------------- #
# Apple parsing
# --------------------------------------------------------------------------- #


def test_parse_powermetrics_converts_milliwatts():
    values = _parse_powermetrics(POWERMETRICS_SAMPLE)
    assert values["cpu_watts"] == pytest.approx(1.543)
    assert values["gpu_watts"] == pytest.approx(0.312)
    assert values["combined_watts"] == pytest.approx(1.855)
    assert values["gpu_residency"] == pytest.approx(23.41)
    assert values["die_temp"] == pytest.approx(48.23)


def test_parse_powermetrics_sums_when_combined_line_absent():
    text = "CPU Power: 1000 mW\nGPU Power: 500 mW\n"
    values = _parse_powermetrics(text)
    assert values["combined_watts"] == pytest.approx(1.5)


def test_parse_powermetrics_on_empty_input():
    assert _parse_powermetrics("") == {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("spdisplays_metal3", "Metal 3"),
        ("spdisplays_metal4", "Metal 4"),
        ("Metal 2", "Metal 2"),
        (None, None),
        ("", None),
    ],
)
def test_metal_family(raw, expected):
    assert _metal_family(raw) == expected


# --------------------------------------------------------------------------- #
# NVIDIA parsing
# --------------------------------------------------------------------------- #


async def test_nvidia_probe_parses_csv(monkeypatch):
    async def fake_run(*argv, **kwargs):
        if "--version" in argv:
            return CommandResult(argv, 0, "NVIDIA-SMI version: 550\nCUDA Version : 12.4\n", "")
        return CommandResult(argv, 0, NVIDIA_CSV, "")

    monkeypatch.setattr("aitop.hardware.nvidia.run", fake_run)

    result = await NvidiaProbe().probe()
    assert len(result.gpus) == 2
    first, second = result.gpus

    assert first.name == "NVIDIA GeForce RTX 4090"
    assert first.vendor is Vendor.NVIDIA
    assert first.api_version == "CUDA 12.4"
    assert first.utilization_percent == pytest.approx(37)
    assert first.vram_total_bytes == 24564 * 1024 * 1024
    assert first.power_watts == pytest.approx(145.32)
    assert first.vram_used_percent == pytest.approx(12800 / 24564 * 100)

    # [N/A] fields become None rather than raising.
    assert second.utilization_percent is None
    assert second.power_watts is None
    assert result.total_power_watts == pytest.approx(145.3)  # summed and rounded to 0.1 W


async def test_nvidia_probe_degrades_when_tool_missing(monkeypatch):
    async def fake_run(*argv, **kwargs):
        return CommandResult(argv, 127, "", "", missing=True)

    monkeypatch.setattr("aitop.hardware.nvidia.run", fake_run)

    result = await NvidiaProbe().probe()
    assert result.gpus == []
    assert result.degraded and "not installed" in result.degraded[0]


# --------------------------------------------------------------------------- #
# AMD parsing
# --------------------------------------------------------------------------- #


def test_amd_snapshot_from_rocm_json():
    card = {
        "Card series": "Radeon RX 7900 XTX",
        "GPU use (%)": "42",
        "VRAM Total Memory (B)": "25753026560",
        "VRAM Total Used Memory (B)": "8589934592",
        "Temperature (Sensor junction) (C)": "63.0",
        "Average Graphics Package Power (W)": "212.0",
        "Driver version": "6.7.0",
    }
    gpu = _to_snapshot("card1", card)
    assert gpu.index == 1
    assert gpu.vendor is Vendor.AMD
    assert gpu.name == "Radeon RX 7900 XTX"
    assert gpu.utilization_percent == pytest.approx(42)
    assert gpu.vram_total_bytes == 25753026560
    assert gpu.temperature_c == pytest.approx(63.0)
    assert gpu.power_watts == pytest.approx(212.0)


def test_amd_snapshot_tolerates_renamed_keys():
    gpu = _to_snapshot(
        "card0", {"GPU Utilization (%)": "77", "Temperature (Sensor edge) (C)": "55"}
    )
    assert gpu.utilization_percent == pytest.approx(77)
    assert gpu.temperature_c == pytest.approx(55)
    assert gpu.name == "AMD GPU"


async def test_amd_probe_handles_non_json(monkeypatch):
    async def fake_run(*argv, **kwargs):
        return CommandResult(argv, 0, "ERROR: could not open device", "")

    monkeypatch.setattr("aitop.hardware.amd.run", fake_run)
    result = await AMDProbe().probe()
    assert result.gpus == []
    assert "non-JSON" in result.degraded[0]


# --------------------------------------------------------------------------- #
# Intel / xpu-smi parsing
# --------------------------------------------------------------------------- #


async def test_intel_probe_parses_discovery_and_dump(monkeypatch):
    from aitop.hardware.intel import IntelProbe

    async def fake_run(*argv, **kwargs):
        if "discovery" in argv and "-j" in argv and "-d" not in argv:
            return CommandResult(
                argv,
                0,
                json.dumps(
                    {
                        "device_list": [
                            {
                                "device_id": 0,
                                "device_name": "Intel(R) Arc(TM) B580 Graphics",
                                "device_type": "GPU",
                                "vendor_name": "Intel(R) Corporation",
                            }
                        ]
                    }
                ),
                "",
            )
        if "discovery" in argv and "-d" in argv:
            return CommandResult(
                argv,
                0,
                json.dumps({"device_id": 0, "memory_physical_size_mb": 12216.0}),
                "",
            )
        if "dump" in argv:
            return CommandResult(
                argv,
                0,
                "2026-01-01 00:00:00.000, 0, 41.5, 88.0, 52.0, 4096.0\n",
                "",
            )
        return CommandResult(argv, 1, "", "unexpected")

    monkeypatch.setattr("aitop.hardware.intel.run", fake_run)
    monkeypatch.setattr("aitop.hardware.intel.which", lambda name: "/usr/bin/xpu-smi")

    probe = IntelProbe()
    assert await probe.available()
    result = await probe.probe()
    assert len(result.gpus) == 1
    gpu = result.gpus[0]
    assert gpu.vendor is Vendor.INTEL
    assert "Arc" in gpu.name
    assert gpu.utilization_percent == pytest.approx(41.5)
    assert gpu.power_watts == pytest.approx(88.0)
    assert gpu.temperature_c == pytest.approx(52.0)
    assert gpu.vram_used_bytes == 4096 * 1024 * 1024
    assert gpu.vram_total_bytes == int(12216 * 1024 * 1024)
    assert result.total_power_watts == pytest.approx(88.0)


async def test_intel_probe_csv_discovery_fallback(monkeypatch):
    from aitop.hardware.intel import IntelProbe

    async def fake_run(*argv, **kwargs):
        if "discovery" in argv and "-j" in argv:
            return CommandResult(argv, 1, "", "no json")
        if "discovery" in argv and "--dump" in argv:
            return CommandResult(argv, 0, "0, Intel Graphics, 8192.00\n", "")
        if "dump" in argv:
            return CommandResult(argv, 0, "0, 12.0, 40.0, 45.0, 1024.0\n", "")
        return CommandResult(argv, 1, "", "nope")

    monkeypatch.setattr("aitop.hardware.intel.run", fake_run)
    probe = IntelProbe()
    probe._bin = "xpu-smi"
    result = await probe.probe()
    assert len(result.gpus) == 1
    assert result.gpus[0].name == "Intel Graphics"
    assert result.gpus[0].vram_total_bytes == 8192 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Collector resilience
# --------------------------------------------------------------------------- #


class ExplodingProbe(HardwareProbe):
    name = "exploding"

    async def available(self) -> bool:
        return True

    async def probe(self) -> ProbeResult:
        raise RuntimeError("vendor tool went sideways")


async def test_safe_probe_converts_exceptions_into_notes():
    result = await ExplodingProbe().safe_probe()
    assert result.gpus == []
    assert "exploding" in result.degraded[0]
    assert "vendor tool went sideways" in result.degraded[0]


async def test_collector_produces_a_snapshot_even_with_a_broken_probe(monkeypatch):
    collector = HardwareCollector(per_core=True)
    monkeypatch.setattr(collector, "probes", _returns([ExplodingProbe()]))

    snapshot = await collector.collect()
    assert snapshot.host.hostname
    assert snapshot.memory.total_bytes > 0
    assert any("exploding" in note for note in snapshot.degraded)


async def test_collector_runs_on_this_machine():
    """Smoke test: whatever this box is, a snapshot comes out."""
    snapshot = await HardwareCollector().collect()
    assert snapshot.cpu.logical_cores
    assert snapshot.memory.total_bytes > 0
    assert isinstance(snapshot.degraded, list)


def _returns(value):
    """An async no-arg callable yielding `value` — a stand-in for a bound method."""

    async def _inner():
        return value

    return _inner
