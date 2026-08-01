"""Serializable telemetry contract shared by every aitop consumer.

Everything the backend produces is a pydantic model in this module. The TUI,
the (future) web UI, the Prometheus exporter and the remote-fleet WebSocket
stream all speak this schema and nothing else — no renderer ever touches a
subprocess or an HTTP client directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Frozen(BaseModel):
    """Immutable base — snapshots are values, never mutated in place."""

    model_config = ConfigDict(frozen=True, extra="ignore")


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class EngineKind(StrEnum):
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    MLX = "mlx"
    VLLM = "vllm"
    LLAMA_SERVER = "llama-server"
    UNKNOWN = "unknown"


class EngineState(StrEnum):
    ONLINE = "online"
    """Reachable and answering telemetry probes."""

    OFFLINE = "offline"
    """Probed, nothing listening."""

    DEGRADED = "degraded"
    """Reachable but a telemetry endpoint failed or returned garbage."""

    UNKNOWN = "unknown"
    """Not probed yet."""


class BindScope(StrEnum):
    LOOPBACK = "loopback"
    LAN = "lan"
    TAILSCALE = "tailscale"
    OTHER = "other"


class Vendor(StrEnum):
    APPLE = "apple"
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #


class CPUSnapshot(Frozen):
    model: str = "unknown"
    arch: str = "unknown"
    physical_cores: int | None = None
    logical_cores: int | None = None
    performance_cores: int | None = None
    efficiency_cores: int | None = None
    load_percent: float = 0.0
    per_core_percent: list[float] = Field(default_factory=list)
    frequency_mhz: float | None = None
    temperature_c: float | None = None
    power_watts: float | None = None


class MemorySnapshot(Frozen):
    total_bytes: int = 0
    used_bytes: int = 0
    available_bytes: int = 0
    swap_total_bytes: int = 0
    swap_used_bytes: int = 0
    unified: bool = False
    """True on Apple Silicon, where this pool is also the VRAM pool."""

    @property
    def used_percent(self) -> float:
        return (self.used_bytes / self.total_bytes * 100) if self.total_bytes else 0.0


class GPUSnapshot(Frozen):
    index: int = 0
    name: str = "unknown"
    vendor: Vendor = Vendor.UNKNOWN
    driver_version: str | None = None
    api_version: str | None = None
    """Metal version on macOS, CUDA/ROCm version on Linux."""

    core_count: int | None = None
    utilization_percent: float | None = None
    vram_total_bytes: int | None = None
    vram_used_bytes: int | None = None
    temperature_c: float | None = None
    power_watts: float | None = None
    power_limit_watts: float | None = None
    unified_memory: bool = False

    @property
    def vram_used_percent(self) -> float | None:
        if not self.vram_total_bytes or self.vram_used_bytes is None:
            return None
        return self.vram_used_bytes / self.vram_total_bytes * 100


class HostSnapshot(Frozen):
    hostname: str = "localhost"
    os_name: str = "unknown"
    os_version: str = ""
    kernel: str = ""
    platform_id: str = "unknown"
    """One of: darwin-arm64, linux-x86_64, ..."""

    uptime_seconds: float | None = None
    python_version: str = ""


class HardwareSnapshot(Frozen):
    """One full pass of the hardware collectors."""

    collected_at: datetime = Field(default_factory=_utcnow)
    host: HostSnapshot = Field(default_factory=HostSnapshot)
    cpu: CPUSnapshot = Field(default_factory=CPUSnapshot)
    memory: MemorySnapshot = Field(default_factory=MemorySnapshot)
    gpus: list[GPUSnapshot] = Field(default_factory=list)
    total_power_watts: float | None = None
    degraded: list[str] = Field(default_factory=list)
    """Human-readable notes about probes that were unavailable (never fatal)."""


# --------------------------------------------------------------------------- #
# Networking
# --------------------------------------------------------------------------- #


class TailscaleStatus(Frozen):
    available: bool = False
    running: bool = False
    hostname: str | None = None
    ipv4: str | None = None
    ipv6: str | None = None
    tailnet: str | None = None
    peer_count: int = 0


class Binding(Frozen):
    host: str
    port: int
    scope: BindScope = BindScope.OTHER

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::", "*") else self.host
        return f"http://{host}:{self.port}"

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"


# --------------------------------------------------------------------------- #
# Engines & models
# --------------------------------------------------------------------------- #


class ModelInfo(Frozen):
    """A model known to an engine — on disk, whether or not it is resident."""

    id: str
    name: str
    engine: EngineKind = EngineKind.UNKNOWN
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    format: str | None = None
    """gguf, mlx, safetensors, ..."""

    size_bytes: int | None = None
    max_context: int | None = None
    modified_at: datetime | None = None


class LoadedModel(Frozen):
    """A model currently resident in memory/VRAM."""

    id: str
    name: str
    engine: EngineKind = EngineKind.UNKNOWN
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    size_bytes: int | None = None
    vram_bytes: int | None = None
    """Portion of `size_bytes` offloaded to the GPU."""

    context_length: int | None = None
    context_used: int | None = None
    expires_at: datetime | None = None

    @property
    def gpu_fraction(self) -> float | None:
        if not self.size_bytes or self.vram_bytes is None:
            return None
        return self.vram_bytes / self.size_bytes

    @property
    def context_fill(self) -> float | None:
        if not self.context_length or self.context_used is None:
            return None
        return self.context_used / self.context_length


class InferenceStats(Frozen):
    """Live inference counters. Populated in Phase 4; zeroed until then."""

    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None
    queue_depth: int = 0
    active_requests: int = 0
    total_requests: int = 0


class EngineSnapshot(Frozen):
    """One full pass of a single engine's telemetry probes."""

    kind: EngineKind
    name: str
    state: EngineState = EngineState.UNKNOWN
    binding: Binding | None = None
    version: str | None = None
    pid: int | None = None
    process_name: str | None = None
    managed_by: str | None = None
    """launchd, systemd, docker, manual — how lifecycle actions must be issued."""

    remote: bool = False
    latency_ms: float | None = None
    models: list[ModelInfo] = Field(default_factory=list)
    loaded: list[LoadedModel] = Field(default_factory=list)
    stats: InferenceStats = Field(default_factory=InferenceStats)
    error: str | None = None
    collected_at: datetime = Field(default_factory=_utcnow)

    @property
    def online(self) -> bool:
        return self.state is EngineState.ONLINE

    @property
    def resident_bytes(self) -> int:
        return sum(m.size_bytes or 0 for m in self.loaded)


# --------------------------------------------------------------------------- #
# Top-level snapshot
# --------------------------------------------------------------------------- #


class SystemSnapshot(Frozen):
    """The single object every consumer renders. Fully JSON-serializable."""

    collected_at: datetime = Field(default_factory=_utcnow)
    node: str = "local"
    hardware: HardwareSnapshot = Field(default_factory=HardwareSnapshot)
    engines: list[EngineSnapshot] = Field(default_factory=list)
    tailscale: TailscaleStatus = Field(default_factory=TailscaleStatus)
    duration_ms: float | None = None

    @property
    def online_engines(self) -> list[EngineSnapshot]:
        return [e for e in self.engines if e.online]

    @property
    def all_loaded(self) -> list[LoadedModel]:
        return [m for e in self.engines for m in e.loaded]
