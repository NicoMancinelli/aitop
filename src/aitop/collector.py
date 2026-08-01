"""The backend's single public entry point.

`SnapshotCollector.collect()` fans out to hardware probes, engine adapters and
Tailscale concurrently and returns one `SystemSnapshot`. `stream()` does the
same on an interval and publishes to the event bus.

Renderers consume snapshots and nothing else — no renderer imports
`aitop.hardware` or `aitop.engines`. That is what makes swapping the Textual UI
for a web UI (or a Prometheus exporter, or a WebSocket fleet feed) a matter of
adding a subscriber.
"""

from __future__ import annotations

import asyncio
import logging
import time

from aitop.bus import EventBus, Topic
from aitop.config import Config
from aitop.engines.registry import EngineRegistry
from aitop.hardware.collector import HardwareCollector
from aitop.models import SystemSnapshot
from aitop.net.tailscale import collect_tailscale

log = logging.getLogger(__name__)


class SnapshotCollector:
    def __init__(
        self,
        config: Config | None = None,
        bus: EventBus | None = None,
        *,
        node: str = "local",
        allow_privileged: bool = True,
    ) -> None:
        self.config = config or Config()
        self.bus = bus or EventBus()
        self.node = node
        self.hardware = HardwareCollector(
            per_core=self.config.ui.show_per_core,
            allow_privileged=allow_privileged,
        )
        self.engines = EngineRegistry(self.config)

    async def collect(self) -> SystemSnapshot:
        """One full pass. Sub-collectors never raise, so neither does this."""
        started = time.perf_counter()
        hardware, engines, tailscale = await asyncio.gather(
            self.hardware.collect(),
            self.engines.poll_all(),
            collect_tailscale(),
        )
        return SystemSnapshot(
            node=self.node,
            hardware=hardware,
            engines=sorted(engines, key=lambda e: (not e.online, e.kind.value)),
            tailscale=tailscale,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def stream(self, interval: float | None = None) -> None:
        """Collect forever, publishing each snapshot to the bus.

        Cancel the task to stop. Errors are logged and retried — a transient
        probe failure must not tear down the dashboard.
        """
        period = interval or self.config.polling.hardware_interval
        while True:
            cycle_start = time.perf_counter()
            try:
                snapshot = await self.collect()
                self.bus.publish(Topic.SNAPSHOT, snapshot, source=self.node)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("collection cycle failed: %s", exc, exc_info=True)
                self.bus.publish(Topic.ERROR, str(exc), source=self.node)
            elapsed = time.perf_counter() - cycle_start
            await asyncio.sleep(max(0.1, period - elapsed))

    async def aclose(self) -> None:
        await self.engines.aclose()

    async def __aenter__(self) -> SnapshotCollector:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
