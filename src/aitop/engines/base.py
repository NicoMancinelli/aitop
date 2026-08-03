"""`BaseEngine` — the contract every AI runtime adapter implements.

An engine adapter is responsible for exactly three things:

1. **Detect** — is this runtime present and reachable at a given endpoint?
2. **Poll**   — return an `EngineSnapshot` describing models, residency, stats.
3. **Control** — start/stop/restart/unload/rebind/pull. The hooks are declared
   here so the UI can grey out unsupported actions via `capabilities`.

Adapters never render, never print, and never raise out of `poll()`. A dead
daemon is a snapshot with `state=OFFLINE`, not an exception.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, ClassVar

import httpx

from aitop.config import EndpointConfig
from aitop.models import (
    Binding,
    BindScope,
    EngineKind,
    EngineSnapshot,
    EngineState,
    LoadedModel,
    ModelInfo,
)
from aitop.version import __version__

log = logging.getLogger(__name__)


class EngineCapability:
    """Bit-flags for what an adapter can actually do, so the UI can adapt."""

    TELEMETRY = "telemetry"
    LIST_MODELS = "list_models"
    LIST_LOADED = "list_loaded"
    UNLOAD = "unload"
    LIFECYCLE = "lifecycle"
    REBIND = "rebind"
    PULL = "pull"


class BaseEngine(abc.ABC):
    """Abstract adapter for one runtime at one endpoint."""

    kind: ClassVar[EngineKind] = EngineKind.UNKNOWN
    display_name: ClassVar[str] = "unknown"
    capabilities: ClassVar[frozenset[str]] = frozenset({EngineCapability.TELEMETRY})
    process_names: ClassVar[tuple[str, ...]] = ()
    """Substrings matched against running process names during discovery."""

    def __init__(self, endpoint: EndpointConfig, client: httpx.AsyncClient | None = None) -> None:
        self.endpoint = endpoint
        self._client = client
        self._owns_client = client is None

    # -- identity ---------------------------------------------------------- #

    @property
    def name(self) -> str:
        return self.endpoint.name or self.display_name

    @property
    def host(self) -> str:
        return self.endpoint.host

    @property
    def port(self) -> int:
        return self.endpoint.resolved_port()

    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    @property
    def binding(self) -> Binding:
        return Binding(host=self.host, port=self.port, scope=classify_scope(self.host))

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    # -- HTTP plumbing ----------------------------------------------------- #

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"User-Agent": f"aitop/{__version__}"}
            if self.endpoint.api_key:
                headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.endpoint.timeout),
                headers=headers,
                trust_env=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _get_json(self, path: str) -> Any | None:
        """GET a JSON endpoint. Returns None on any failure — never raises."""
        url = f"{self.base_url}{path}"
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # transport, HTTP, decode, test-double gaps
            log.debug("%s GET %s failed: %s", self.name, url, exc)
            return None

    # -- lifecycle of a poll ----------------------------------------------- #

    async def detect(self) -> bool:
        """Cheap reachability check. Subclasses override with a real ping."""
        return await self._get_json("/") is not None

    @abc.abstractmethod
    async def _collect(self) -> EngineSnapshot:
        """Do the actual probing. Called only by `poll()`."""

    async def poll(self) -> EngineSnapshot:
        """Public entry point: always returns a snapshot, never raises."""
        started = time.perf_counter()
        try:
            snapshot = await self._collect()
        except Exception as exc:  # adapter bug or exotic transport error
            log.debug("%s poll failed: %s", self.name, exc, exc_info=True)
            return self.offline(error=f"{type(exc).__name__}: {exc}")
        latency = (time.perf_counter() - started) * 1000
        return snapshot.model_copy(update={"latency_ms": round(latency, 2)})

    # -- snapshot builders -------------------------------------------------- #

    def snapshot(
        self,
        *,
        state: EngineState,
        version: str | None = None,
        models: list[ModelInfo] | None = None,
        loaded: list[LoadedModel] | None = None,
        error: str | None = None,
        **extra: Any,
    ) -> EngineSnapshot:
        return EngineSnapshot(
            kind=self.kind,
            name=self.name,
            state=state,
            binding=self.binding,
            version=version,
            remote=self.endpoint.remote,
            models=models or [],
            loaded=loaded or [],
            error=error,
            **extra,
        )

    def offline(self, error: str | None = None) -> EngineSnapshot:
        return self.snapshot(state=EngineState.OFFLINE, error=error)

    # -- control surface (Phase 2 — declared now so the UI can bind to it) -- #

    async def start(self) -> tuple[bool, str]:
        return False, f"{self.display_name}: start not implemented"

    async def stop(self) -> tuple[bool, str]:
        return False, f"{self.display_name}: stop not implemented"

    async def restart(self) -> tuple[bool, str]:
        return False, f"{self.display_name}: restart not implemented"

    async def unload(self, model_id: str | None = None) -> tuple[bool, str]:
        """Evict resident weights and flush KV cache. `None` means all models."""
        return False, f"{self.display_name}: unload not implemented"

    async def rebind(self, host: str) -> tuple[bool, str]:
        return False, f"{self.display_name}: rebind not implemented"

    async def pull(self, model: str, *, on_progress=None) -> tuple[bool, str]:
        return False, f"{self.display_name}: pull not implemented"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.host}:{self.port}>"


def classify_scope(host: str) -> BindScope:
    """Map a bind address to the network exposure it implies."""
    if host in ("127.0.0.1", "::1", "localhost"):
        return BindScope.LOOPBACK
    if host in ("0.0.0.0", "::", "*"):
        return BindScope.LAN
    if host.startswith("100.") and _is_cgnat(host):
        return BindScope.TAILSCALE
    if host.startswith(("10.", "192.168.", "172.")):
        return BindScope.LAN
    return BindScope.OTHER


def _is_cgnat(host: str) -> bool:
    """Tailscale hands out 100.64.0.0/10 — distinguish it from 100.0.0.0/8."""
    parts = host.split(".")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        return False
    return 64 <= int(parts[1]) <= 127
