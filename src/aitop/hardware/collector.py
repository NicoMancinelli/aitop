"""Hardware collector — assembles one `HardwareSnapshot` per pass.

Probes are selected once at construction (`available()` is cheap but not free)
and then run concurrently on every pass. A probe that fails contributes a note
to `degraded` and nothing else; the snapshot is always produced.
"""

from __future__ import annotations

import asyncio
import logging

import psutil

from aitop.hardware.amd import AMDProbe
from aitop.hardware.apple import AppleSiliconProbe
from aitop.hardware.base import HardwareProbe, ProbeResult
from aitop.hardware.intel import IntelProbe
from aitop.hardware.nvidia import NvidiaProbe
from aitop.hardware.system import collect_cpu, collect_host, collect_memory
from aitop.models import HardwareSnapshot

log = logging.getLogger(__name__)

ALL_PROBES: tuple[type[HardwareProbe], ...] = (
    AppleSiliconProbe,
    NvidiaProbe,
    AMDProbe,
    IntelProbe,
)


class HardwareCollector:
    def __init__(self, per_core: bool = True, allow_privileged: bool = True) -> None:
        self.per_core = per_core
        self.allow_privileged = allow_privileged
        self._probes: list[HardwareProbe] | None = None
        self._primed = False

    async def probes(self) -> list[HardwareProbe]:
        """Instantiate and filter probes once, on first use."""
        if self._probes is not None:
            return self._probes

        candidates: list[HardwareProbe] = []
        for cls in ALL_PROBES:
            probe = (
                AppleSiliconProbe(allow_powermetrics=self.allow_privileged)
                if cls is AppleSiliconProbe
                else cls()
            )
            candidates.append(probe)

        checks = await asyncio.gather(*(p.available() for p in candidates), return_exceptions=True)
        self._probes = [
            probe
            for probe, ok in zip(candidates, checks, strict=True)
            if ok is True  # exceptions and False alike disable the probe
        ]
        log.debug("active hardware probes: %s", [p.name for p in self._probes])
        return self._probes

    def prime(self) -> None:
        """Seed psutil's CPU counters so the first real sample isn't 0%."""
        psutil.cpu_percent(interval=None, percpu=True)
        psutil.cpu_percent(interval=None)
        self._primed = True

    async def collect(self) -> HardwareSnapshot:
        if not self._primed:
            # A single blocking 100 ms sample beats reporting a bogus 0%.
            await asyncio.to_thread(psutil.cpu_percent, 0.1, True)
            self._primed = True

        probes = await self.probes()
        host, cpu, memory, results = await asyncio.gather(
            asyncio.to_thread(collect_host),
            asyncio.to_thread(collect_cpu, self.per_core),
            asyncio.to_thread(collect_memory),
            asyncio.gather(*(p.safe_probe() for p in probes)),
        )

        merged = ProbeResult()
        for result in results:
            merged.merge(result)

        if not merged.gpus:
            merged.note(
                "no GPU telemetry source found "
                "(nvidia-smi / rocm-smi / xpu-smi / intel_gpu_top / Apple GPU)"
            )

        return HardwareSnapshot(
            host=host,
            cpu=merged.apply_cpu(cpu),
            memory=memory,
            gpus=sorted(merged.gpus, key=lambda g: g.index),
            total_power_watts=merged.total_power_watts,
            degraded=merged.degraded,
        )
