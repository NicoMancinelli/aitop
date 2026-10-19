"""Serializable telemetry contract shared by every aitop consumer.

Everything the backend produces is a pydantic model in this module. The TUI,
Prometheus exporter (`aitop metrics` / `GET /metrics`), and remote
fleet stream all speak this schema and nothing else — no renderer ever touches
a subprocess or an HTTP client directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

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
    SGLANG = "sglang"
    APHRODITE = "aphrodite"
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
    """Live inference counters. Populated from engine /metrics when available."""

    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None
    queue_depth: int = 0
    active_requests: int = 0
    total_requests: int = 0
    latency_summary: LatencySummary | None = None


class DownloadProgress(Frozen):
    """One progress tick from a model pull / HF download."""

    model: str
    engine: EngineKind = EngineKind.UNKNOWN
    status: str = "pending"
    completed_bytes: int | None = None
    total_bytes: int | None = None
    error: str | None = None
    done: bool = False

    @property
    def fraction(self) -> float | None:
        if not self.total_bytes or self.completed_bytes is None:
            return None
        return self.completed_bytes / self.total_bytes


class HubModel(Frozen):
    """A model discovered via the Hugging Face hub search."""

    id: str
    author: str | None = None
    downloads: int | None = None
    likes: int | None = None
    tags: list[str] = Field(default_factory=list)
    pipeline_tag: str | None = None
    last_modified: datetime | None = None
    url: str | None = None


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
# Benchmarking & Telemetry
# --------------------------------------------------------------------------- #


class BenchmarkResult(Frozen):
    """Result of an inference latency / throughput benchmark run."""

    model: str
    engine: EngineKind = EngineKind.UNKNOWN
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_s: float = 0.0
    ttft_ms: float | None = None
    tokens_per_second: float = 0.0
    error: str | None = None
    success: bool = True


class ModelFitResult(Frozen):
    """Estimation of model fit, VRAM requirements, and context headroom."""

    fits: bool = True
    """Whether the model weights and context fit entirely in available VRAM."""

    gpu_offload_fraction: float = 1.0
    """Estimated fraction of model/context that can be offloaded to GPU (0.0 to 1.0)."""

    exceeds_memory: bool = False
    """True if total memory required exceeds available VRAM."""

    max_context_length: int = 0
    """Maximum context length (tokens) that can fit in remaining VRAM headroom."""

    model_size_bytes: int = 0
    """Model weight size in bytes."""

    kv_cache_bytes: int = 0
    """KV cache memory required for requested context length in bytes."""

    total_required_bytes: int = 0
    """Total memory required (model weights + KV cache) in bytes."""

    available_vram_bytes: int = 0
    """Available VRAM capacity in bytes."""

    headroom_bytes: int = 0
    """Remaining VRAM headroom in bytes after model and context (negative if exceeds)."""

    @property
    def fits_in_vram(self) -> bool:
        """Alias for fits (whether model weights and context fit in VRAM)."""
        return self.fits

    @property
    def max_context_tokens(self) -> int:
        """Alias for max_context_length."""
        return self.max_context_length


class LatencySummary(Frozen):
    """Summary of inference request latency percentiles and throughput."""

    sample_count: int = 0
    ttft_p50: float | None = None
    ttft_p90: float | None = None
    ttft_p95: float | None = None
    ttft_p99: float | None = None
    ttft_min: float | None = None
    ttft_max: float | None = None
    ttft_mean: float | None = None

    tok_per_sec_p50: float | None = None
    tok_per_sec_p90: float | None = None
    tok_per_sec_p95: float | None = None
    tok_per_sec_p99: float | None = None
    tok_per_sec_min: float | None = None
    tok_per_sec_max: float | None = None
    tok_per_sec_mean: float | None = None

    duration_p50: float | None = None
    duration_p90: float | None = None
    duration_p95: float | None = None
    duration_p99: float | None = None
    duration_min: float | None = None
    duration_max: float | None = None
    duration_mean: float | None = None

    @property
    def tokens_per_sec_p50(self) -> float | None:
        return self.tok_per_sec_p50

    @property
    def tokens_per_sec_p90(self) -> float | None:
        return self.tok_per_sec_p90

    @property
    def tokens_per_sec_p95(self) -> float | None:
        return self.tok_per_sec_p95

    @property
    def tokens_per_sec_p99(self) -> float | None:
        return self.tok_per_sec_p99

    @property
    def tps_p50(self) -> float | None:
        return self.tok_per_sec_p50

    @property
    def tps_p90(self) -> float | None:
        return self.tok_per_sec_p90

    @property
    def tps_p95(self) -> float | None:
        return self.tok_per_sec_p95

    @property
    def tps_p99(self) -> float | None:
        return self.tok_per_sec_p99


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


# --------------------------------------------------------------------------- #
# Alerts & Cluster
# --------------------------------------------------------------------------- #


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertEvent(Frozen):
    """Telemetry or lifecycle event that crossed an alert threshold."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    severity: AlertSeverity = AlertSeverity.INFO
    title: str
    message: str
    source: str = "local"
    timestamp: datetime = Field(default_factory=_utcnow)
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None


class ClusterSnapshot(Frozen):
    """Aggregated fleet-wide telemetry across multiple aitop nodes."""

    collected_at: datetime = Field(default_factory=_utcnow)
    nodes: list[SystemSnapshot] = Field(default_factory=list)

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def online_nodes(self) -> int:
        return sum(1 for n in self.nodes if n.online_engines or n.hardware.host.hostname)

    @property
    def total_gpus(self) -> int:
        return sum(len(n.hardware.gpus) for n in self.nodes)

    @property
    def total_vram_bytes(self) -> int:
        return sum(gpu.vram_total_bytes or 0 for n in self.nodes for gpu in n.hardware.gpus)

    @property
    def used_vram_bytes(self) -> int:
        return sum(gpu.vram_used_bytes or 0 for n in self.nodes for gpu in n.hardware.gpus)

    @property
    def online_engines(self) -> int:
        return sum(len(n.online_engines) for n in self.nodes)

    @property
    def total_loaded_models(self) -> int:
        return sum(len(n.all_loaded) for n in self.nodes)

    @property
    def cluster_tokens_per_second(self) -> float:
        return sum(e.stats.tokens_per_second or 0.0 for n in self.nodes for e in n.online_engines)
