"""Hardware probe contract.

Probes wrap platform-specific tools (`powermetrics`, `ioreg`, `nvidia-smi`,
`rocm-smi`, `xpu-smi`, `intel_gpu_top`). Two rules hold everywhere:

* `available()` is cheap and side-effect free — it decides whether the probe
  is even applicable on this box.
* `probe()` never raises and never blocks on a password prompt. When a tool is
  missing or needs privileges we don't have, the probe returns partial data
  and appends a note to `degraded`.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field

from aitop.models import CPUSnapshot, GPUSnapshot

log = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """What a probe managed to collect this pass."""

    gpus: list[GPUSnapshot] = field(default_factory=list)
    cpu_overrides: dict[str, object] = field(default_factory=dict)
    """Fields to merge into the psutil-derived `CPUSnapshot`."""

    total_power_watts: float | None = None
    degraded: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and message not in self.degraded:
            self.degraded.append(message)

    def merge(self, other: ProbeResult) -> None:
        self.gpus.extend(other.gpus)
        self.cpu_overrides.update(other.cpu_overrides)
        if other.total_power_watts is not None:
            self.total_power_watts = (self.total_power_watts or 0.0) + other.total_power_watts
        for note in other.degraded:
            self.note(note)

    def apply_cpu(self, cpu: CPUSnapshot) -> CPUSnapshot:
        clean = {k: v for k, v in self.cpu_overrides.items() if v is not None}
        return cpu.model_copy(update=clean) if clean else cpu


class HardwareProbe(abc.ABC):
    """A platform- or vendor-specific source of hardware telemetry."""

    name: str = "probe"

    @abc.abstractmethod
    async def available(self) -> bool:
        """Is this probe applicable and usable on this machine right now?"""

    @abc.abstractmethod
    async def probe(self) -> ProbeResult:
        """Collect one pass. Must not raise."""

    async def safe_probe(self) -> ProbeResult:
        try:
            return await self.probe()
        except Exception as exc:
            log.debug("probe %s failed: %s", self.name, exc, exc_info=True)
            result = ProbeResult()
            result.note(f"{self.name}: {type(exc).__name__}: {exc}")
            return result

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__}>"
