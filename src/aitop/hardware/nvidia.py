"""NVIDIA probe via `nvidia-smi`.

One CSV query covers every field we need for all GPUs in the box. Fields that
a given card doesn't report come back as `[N/A]`, which `to_float` turns into
`None` rather than an exception.
"""

from __future__ import annotations

from typing import ClassVar

from aitop.hardware.base import HardwareProbe, ProbeResult
from aitop.models import GPUSnapshot, Vendor
from aitop.utils.parse import to_float, to_int
from aitop.utils.proc import run, which

MIB = 1024 * 1024


class NvidiaProbe(HardwareProbe):
    name = "nvidia"

    QUERY: ClassVar[tuple[str, ...]] = (
        "index",
        "name",
        "driver_version",
        "utilization.gpu",
        "memory.total",
        "memory.used",
        "temperature.gpu",
        "power.draw",
        "power.limit",
    )

    def __init__(self) -> None:
        self._cuda_version: str | None = None
        self._cuda_checked = False

    async def available(self) -> bool:
        return which("nvidia-smi") is not None

    async def probe(self) -> ProbeResult:
        result = ProbeResult()
        out = await run(
            "nvidia-smi",
            f"--query-gpu={','.join(self.QUERY)}",
            "--format=csv,noheader,nounits",
            timeout=5.0,
        )
        if not out.ok:
            result.note(f"NVIDIA telemetry unavailable ({out.reason})")
            return result

        cuda = await self._cuda()
        total_power = 0.0
        for line in out.stdout.splitlines():
            fields = [f.strip() for f in line.split(",")]
            if len(fields) < len(self.QUERY):
                continue
            gpu = self._to_snapshot(fields, cuda)
            result.gpus.append(gpu)
            if gpu.power_watts:
                total_power += gpu.power_watts

        if not result.gpus:
            result.note("nvidia-smi returned no GPUs")
        elif total_power:
            result.total_power_watts = round(total_power, 1)
        return result

    def _to_snapshot(self, fields: list[str], cuda: str | None) -> GPUSnapshot:
        total_mib = to_float(fields[4])
        used_mib = to_float(fields[5])
        return GPUSnapshot(
            index=to_int(fields[0]) or 0,
            name=fields[1] or "NVIDIA GPU",
            vendor=Vendor.NVIDIA,
            driver_version=fields[2] or None,
            api_version=f"CUDA {cuda}" if cuda else None,
            utilization_percent=to_float(fields[3]),
            vram_total_bytes=int(total_mib * MIB) if total_mib else None,
            vram_used_bytes=int(used_mib * MIB) if used_mib is not None else None,
            temperature_c=to_float(fields[6]),
            power_watts=to_float(fields[7]),
            power_limit_watts=to_float(fields[8]),
        )

    async def _cuda(self) -> str | None:
        """`nvidia-smi --version` reports the CUDA runtime on driver 535+."""
        if self._cuda_checked:
            return self._cuda_version
        self._cuda_checked = True
        out = await run("nvidia-smi", "--version", timeout=4.0)
        if out.ok:
            for line in out.stdout.splitlines():
                if line.lower().startswith("cuda version"):
                    self._cuda_version = line.split(":", 1)[-1].strip() or None
                    break
        return self._cuda_version
