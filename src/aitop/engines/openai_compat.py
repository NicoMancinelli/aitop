"""Shared adapter for OpenAI-compatible local servers.

Covers vLLM, llama.cpp's `llama-server`, and MLX OpenAI-style frontends.
All speak some subset of:

  GET /v1/models          — model catalogue (always)
  GET /health             — liveness (llama-server, some MLX)
  GET /v1/models/{id}     — detail (rare)
  GET /metrics            — Prometheus counters (vLLM)

Residency detail is sparse on these runtimes: if the server lists a model it
is typically loaded. Inference stats are best-effort from `/metrics`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from aitop.engines.base import BaseEngine, EngineCapability
from aitop.engines.lifecycle import restart_engine, start_engine, stop_engine
from aitop.engines.stats import parse_prometheus_stats
from aitop.models import (
    EngineKind,
    EngineSnapshot,
    EngineState,
    InferenceStats,
    LoadedModel,
    ModelInfo,
)
from aitop.utils.parse import first, to_int

# Re-export for callers that imported the parser from this module.
__all__ = (
    "LlamaServerEngine",
    "MLXEngine",
    "OpenAICompatEngine",
    "VLLMEngine",
    "parse_prometheus_stats",
)


class OpenAICompatEngine(BaseEngine):
    """Base for runtimes that expose an OpenAI-compatible HTTP surface."""

    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            EngineCapability.TELEMETRY,
            EngineCapability.LIST_MODELS,
            EngineCapability.LIST_LOADED,
            EngineCapability.LIFECYCLE,
        }
    )
    health_paths: ClassVar[tuple[str, ...]] = ("/health", "/v1/models")
    metrics_path: ClassVar[str | None] = "/metrics"
    assume_listed_are_loaded: ClassVar[bool] = True
    """When True, every model returned by /v1/models is treated as resident."""

    async def detect(self) -> bool:
        for path in self.health_paths:
            if await self._get_json(path) is not None or await self._get_ok(path):
                return True
        return False

    async def _get_ok(self, path: str) -> bool:
        url = f"{self.base_url}{path}"
        try:
            response = await self.client.get(url)
            return response.status_code < 500
        except Exception:
            return False

    async def _collect(self) -> EngineSnapshot:
        models_payload = await self._get_json("/v1/models")
        if models_payload is None:
            # Some servers only answer /health with plain text.
            if await self._get_ok("/health"):
                return self.snapshot(
                    state=EngineState.DEGRADED,
                    error="/v1/models unavailable; server is up",
                )
            return self.offline()

        models = [self._parse_model(e) for e in _data_array(models_payload)]
        loaded = [self._as_loaded(m) for m in models] if self.assume_listed_are_loaded else []
        stats = await self._collect_stats()
        version = await self._server_version()

        return self.snapshot(
            state=EngineState.ONLINE,
            version=version,
            models=sorted(models, key=lambda m: m.name),
            loaded=sorted(loaded, key=lambda m: m.name),
            stats=stats,
        )

    def _parse_model(self, entry: dict[str, Any]) -> ModelInfo:
        model_id = str(first(entry, "id", "model", default="unknown"))
        return ModelInfo(
            id=model_id,
            name=model_id,
            engine=self.kind,
            family=first(entry, "family", "owned_by"),
            max_context=to_int(first(entry, "max_model_len", "context_length", "n_ctx")),
            size_bytes=to_int(first(entry, "size", "size_bytes")),
        )

    def _as_loaded(self, model: ModelInfo) -> LoadedModel:
        return LoadedModel(
            id=model.id,
            name=model.name,
            engine=self.kind,
            family=model.family,
            size_bytes=model.size_bytes,
            context_length=model.max_context,
        )

    async def _server_version(self) -> str | None:
        for path in ("/version", "/props", "/v1/models"):
            payload = await self._get_json(path)
            if not isinstance(payload, dict):
                continue
            version = first(payload, "version", "default_generation_settings")
            if isinstance(version, str):
                return version
            if isinstance(version, dict):
                # llama-server /props embeds nested settings — not a version.
                continue
        return None

    async def _collect_stats(self) -> InferenceStats:
        if not self.metrics_path:
            return InferenceStats()
        url = f"{self.base_url}{self.metrics_path}"
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            text = response.text
        except Exception:
            return InferenceStats()
        return parse_prometheus_stats(text)

    # -- lifecycle ---------------------------------------------------------- #

    async def start(self) -> tuple[bool, str]:
        if self.endpoint.remote:
            return False, "cannot start a remote engine"
        result = await start_engine(
            self.kind.value,
            host=self.host,
            port=self.port,
            container=self.endpoint.container,
        )
        return result.ok, result.message

    async def stop(self) -> tuple[bool, str]:
        if self.endpoint.remote:
            return False, "cannot stop a remote engine"
        snap = await self.poll()
        result = await stop_engine(
            self.kind.value,
            pid=snap.pid,
            managed_by=snap.managed_by,
            container=self.endpoint.container,
        )
        return result.ok, result.message

    async def restart(self) -> tuple[bool, str]:
        if self.endpoint.remote:
            return False, "cannot restart a remote engine"
        snap = await self.poll()
        result = await restart_engine(
            self.kind.value,
            pid=snap.pid,
            managed_by=snap.managed_by,
            host=self.host,
            port=self.port,
            container=self.endpoint.container,
        )
        return result.ok, result.message


class VLLMEngine(OpenAICompatEngine):
    kind: ClassVar[EngineKind] = EngineKind.VLLM
    display_name: ClassVar[str] = "vLLM"
    process_names: ClassVar[tuple[str, ...]] = ("vllm", "VLLM")
    health_paths: ClassVar[tuple[str, ...]] = ("/health", "/v1/models")
    metrics_path: ClassVar[str | None] = "/metrics"


class LlamaServerEngine(OpenAICompatEngine):
    kind: ClassVar[EngineKind] = EngineKind.LLAMA_SERVER
    display_name: ClassVar[str] = "llama-server"
    process_names: ClassVar[tuple[str, ...]] = ("llama-server", "llama_server")
    health_paths: ClassVar[tuple[str, ...]] = ("/health", "/v1/models")
    # Recent llama.cpp builds expose Prometheus at /metrics.
    metrics_path: ClassVar[str | None] = "/metrics"

    async def _server_version(self) -> str | None:
        payload = await self._get_json("/props")
        if isinstance(payload, dict):
            return first(payload, "version") or "llama-server"
        return "llama-server"


class MLXEngine(OpenAICompatEngine):
    kind: ClassVar[EngineKind] = EngineKind.MLX
    display_name: ClassVar[str] = "MLX"
    process_names: ClassVar[tuple[str, ...]] = ("mlx_lm", "mlx-openai", "mlx_lm.server")
    health_paths: ClassVar[tuple[str, ...]] = ("/v1/models", "/health")
    metrics_path: ClassVar[str | None] = None


def _data_array(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        entries = payload.get("data")
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = None
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]
