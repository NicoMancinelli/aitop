"""Shared adapter for OpenAI-compatible local servers.

Covers vLLM, llama.cpp's `llama-server`, and MLX OpenAI-style frontends.
All speak some subset of:

  GET /v1/models          — model catalogue (always)
  GET /health             — liveness (llama-server, some MLX)
  GET /v1/models/{id}     — detail (rare)
  GET /metrics            — Prometheus counters (vLLM)
  GET /models             — llama-server router catalogue with load status
  POST /models/load|unload — llama-server router model lifecycle

Residency detail is sparse on these runtimes: if the server lists a model it
is typically loaded. Inference stats are best-effort from `/metrics`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

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

_LOADED_STATUS = frozenset({"loaded", "loading", "ready", "online"})


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
    """llama.cpp `llama-server` — single-model or multi-model router."""

    kind: ClassVar[EngineKind] = EngineKind.LLAMA_SERVER
    display_name: ClassVar[str] = "llama-server"
    process_names: ClassVar[tuple[str, ...]] = ("llama-server", "llama_server")
    health_paths: ClassVar[tuple[str, ...]] = ("/health", "/v1/models", "/models")
    metrics_path: ClassVar[str | None] = "/metrics"
    # Classic single-model mode lists only the resident weights; router mode
    # reports status on GET /models and we filter to loaded entries instead.
    assume_listed_are_loaded: ClassVar[bool] = True
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            EngineCapability.TELEMETRY,
            EngineCapability.LIST_MODELS,
            EngineCapability.LIST_LOADED,
            EngineCapability.LOAD,
            EngineCapability.UNLOAD,
            EngineCapability.LIFECYCLE,
        }
    )

    async def _server_version(self) -> str | None:
        payload = await self._get_json("/props")
        if isinstance(payload, dict):
            return first(payload, "version") or "llama-server"
        return "llama-server"

    async def _collect(self) -> EngineSnapshot:
        router = await self._get_json("/models")
        if router is not None:
            entries = _data_array(router)
            if entries:
                models = [self._parse_model(e) for e in entries]
                loaded = [
                    self._as_loaded(self._parse_model(e))
                    for e in entries
                    if _router_status(e) in _LOADED_STATUS
                ]
                # If the router omits status, treat listed models as resident
                # only when every entry lacks a status field (older builds).
                if not loaded and entries and all(_router_status(e) is None for e in entries):
                    loaded = [self._as_loaded(m) for m in models]
                return self.snapshot(
                    state=EngineState.ONLINE,
                    version=await self._server_version(),
                    models=sorted(models, key=lambda m: m.name),
                    loaded=sorted(loaded, key=lambda m: m.name),
                    stats=await self._collect_stats(),
                )
        return await super()._collect()

    async def load(self, model_id: str) -> tuple[bool, str]:
        """`POST /models/load` — llama-server router mode."""
        ok, message = await self._router_action("/models/load", model_id)
        if ok:
            return True, f"loaded {model_id}"
        return False, message

    async def unload(self, model_id: str | None = None) -> tuple[bool, str]:
        """`POST /models/unload` — one model, or every currently loaded model."""
        targets = [model_id] if model_id else await self._loaded_ids()
        if not targets:
            return True, "no resident models"
        failed: list[str] = []
        for target in targets:
            ok, message = await self._router_action("/models/unload", target)
            if not ok:
                failed.append(f"{target} ({message})")
        if failed:
            return False, "failed to unload: " + ", ".join(failed)
        return True, f"unloaded {len(targets)} model(s)"

    async def _loaded_ids(self) -> list[str]:
        snap = await self.poll()
        return [m.id for m in snap.loaded]

    async def _router_action(self, path: str, model_id: str) -> tuple[bool, str]:
        url = f"{self.base_url}{path}"
        try:
            response = await self.client.post(
                url,
                json={"model": model_id},
                timeout=httpx.Timeout(120.0, connect=5.0),
            )
            if response.status_code == 404:
                return False, (
                    "llama-server router API unavailable — start with "
                    "`--models-dir` / router mode for load/unload"
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = (exc.response.text or "")[:200]
            return False, f"HTTP {exc.response.status_code}" + (f": {detail}" if detail else "")
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return True, "ok"


class MLXEngine(OpenAICompatEngine):
    kind: ClassVar[EngineKind] = EngineKind.MLX
    display_name: ClassVar[str] = "MLX"
    process_names: ClassVar[tuple[str, ...]] = ("mlx_lm", "mlx-openai", "mlx_lm.server")
    health_paths: ClassVar[tuple[str, ...]] = ("/v1/models", "/health")
    metrics_path: ClassVar[str | None] = None


def _router_status(entry: dict[str, Any]) -> str | None:
    raw = first(entry, "status", "state", "load_state")
    if raw is None:
        return None
    return str(raw).strip().lower() or None


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
