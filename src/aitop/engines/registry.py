"""Engine discovery and parallel polling.

Discovery is two-pronged:

* **Endpoint probing** — every well-known port from `config.DEFAULT_PORTS`
  plus anything the user configured, probed concurrently.
* **Process scanning** — psutil finds running daemons by name so we can
  attach a PID and a supervisor (`launchd` / `systemd` / `docker` / manual),
  and catch daemons listening on a non-default port.

Only adapters registered here exist as far as the rest of aitop is concerned.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import ClassVar

import httpx
import psutil

from aitop.config import Config, EndpointConfig
from aitop.engines.base import BaseEngine
from aitop.engines.lmstudio import LMStudioEngine
from aitop.engines.ollama import OllamaEngine
from aitop.engines.openai_compat import LlamaServerEngine, MLXEngine, VLLMEngine
from aitop.models import EngineKind, EngineSnapshot
from aitop.version import __version__

log = logging.getLogger(__name__)

ADAPTERS: dict[EngineKind, type[BaseEngine]] = {
    EngineKind.OLLAMA: OllamaEngine,
    EngineKind.LMSTUDIO: LMStudioEngine,
    EngineKind.VLLM: VLLMEngine,
    EngineKind.LLAMA_SERVER: LlamaServerEngine,
    EngineKind.MLX: MLXEngine,
}


class EngineRegistry:
    """Owns the adapter instances and the shared HTTP client."""

    supported: ClassVar[tuple[EngineKind, ...]] = tuple(ADAPTERS)

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(4.0, connect=1.0),
            headers={"User-Agent": f"aitop/{__version__}"},
            trust_env=False,
            limits=httpx.Limits(max_connections=16),
        )
        self._engines: list[BaseEngine] = []
        self._processes: dict[EngineKind, ProcessInfo] = {}

    # -- construction ------------------------------------------------------- #

    def build(self, endpoints: list[EndpointConfig] | None = None) -> list[BaseEngine]:
        """Instantiate an adapter per configured endpoint we know how to speak."""
        targets = endpoints if endpoints is not None else self.config.all_endpoints()
        self._engines = [
            ADAPTERS[ep.kind](ep, client=self._client) for ep in targets if ep.kind in ADAPTERS
        ]
        return self._engines

    @property
    def engines(self) -> list[BaseEngine]:
        return self._engines or self.build()

    # -- polling ------------------------------------------------------------ #

    async def discover(self) -> list[EngineSnapshot]:
        """Poll every endpoint concurrently; return only the reachable ones,
        enriched with process metadata."""
        snapshots = await self.poll_all()
        return [s for s in snapshots if s.state.value != "offline"]

    async def poll_all(self, *, include_offline: bool = True) -> list[EngineSnapshot]:
        engines = self.engines
        if self.config.discovery.scan_processes:
            self._processes = await asyncio.to_thread(scan_processes)

        results = await asyncio.gather(
            *(engine.poll() for engine in engines), return_exceptions=True
        )

        out: list[EngineSnapshot] = []
        for engine, result in zip(engines, results, strict=True):
            if isinstance(result, BaseException):  # poll() should never raise
                log.debug("engine %s raised: %s", engine.name, result)
                snapshot = engine.offline(error=str(result))
            else:
                snapshot = result
            snapshot = self._attach_process(snapshot)
            if include_offline or snapshot.state.value != "offline":
                out.append(snapshot)
        return _dedupe(out)

    def _attach_process(self, snapshot: EngineSnapshot) -> EngineSnapshot:
        info = self._processes.get(snapshot.kind)
        if info is None or snapshot.remote:
            return snapshot
        return snapshot.model_copy(
            update={
                "pid": info.pid,
                "process_name": info.name,
                "managed_by": info.managed_by,
            }
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> EngineRegistry:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


# --------------------------------------------------------------------------- #
# Process scanning
# --------------------------------------------------------------------------- #


class ProcessInfo:
    __slots__ = ("pid", "name", "managed_by", "ports")

    def __init__(self, pid: int, name: str, managed_by: str | None, ports: set[int]) -> None:
        self.pid = pid
        self.name = name
        self.managed_by = managed_by
        self.ports = ports

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ProcessInfo {self.name} pid={self.pid} ports={sorted(self.ports)}>"


def scan_processes() -> dict[EngineKind, ProcessInfo]:
    """Match running processes against each adapter's `process_names`.

    Blocking (psutil); call via `asyncio.to_thread`. Access errors are
    expected for processes owned by other users and are simply skipped.
    """
    matchers = {kind: adapter.process_names for kind, adapter in ADAPTERS.items()}
    found: dict[EngineKind, ProcessInfo] = {}

    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            name = proc.info.get("name") or ""
            if not name:
                continue
            lowered = name.lower()
            for kind, needles in matchers.items():
                if kind in found:
                    continue
                if not any(needle.lower() in lowered for needle in needles):
                    continue
                found[kind] = ProcessInfo(
                    pid=proc.info["pid"],
                    name=name,
                    managed_by=detect_supervisor(proc.info["pid"]),
                    ports=listening_ports(proc),
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return found


def listening_ports(proc: psutil.Process) -> set[int]:
    """TCP ports the process listens on. Empty set when the OS denies us."""
    try:
        connections = proc.net_connections(kind="inet")
    except (psutil.AccessDenied, psutil.NoSuchProcess, NotImplementedError, OSError):
        return set()
    return {
        c.laddr.port
        for c in connections
        if c.status == psutil.CONN_LISTEN and getattr(c, "laddr", None)
    }


def detect_supervisor(pid: int) -> str | None:
    """Best-effort guess at how a daemon must be stopped and restarted."""
    if sys.platform == "darwin":
        return "launchd"
    if sys.platform.startswith("linux"):
        try:
            cgroup = f"/proc/{pid}/cgroup"
            with open(cgroup, encoding="utf-8") as handle:
                content = handle.read()
        except OSError:
            return None
        if "docker" in content or "containerd" in content:
            return "docker"
        if ".service" in content:
            return "systemd"
        return "manual"
    return None


def _dedupe(snapshots: list[EngineSnapshot]) -> list[EngineSnapshot]:
    """Collapse duplicate (kind, binding) pairs, preferring the online one."""
    best: dict[tuple[str, str], EngineSnapshot] = {}
    for snap in snapshots:
        key = (snap.kind.value, str(snap.binding))
        current = best.get(key)
        if current is None or (not current.online and snap.online):
            best[key] = snap
    return list(best.values())


def running_under_root() -> bool:
    return os.geteuid() == 0
