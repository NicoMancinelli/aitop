"""Shared adapter for OpenAI-compatible local servers.

Covers vLLM, llama.cpp's `llama-server`, and MLX OpenAI-style frontends.
All speak some subset of:

  GET /v1/models          — model catalogue (always)
  GET /health             — liveness (llama-server, some MLX)
  GET /v1/models/{id}     — detail (rare)
  GET /metrics            — Prometheus counters (vLLM)

Load / unload where the upstream APIs allow it:

  llama-server router     POST /models/load · POST /models/unload · GET /models
  vLLM sleep mode         POST /wake_up · POST /sleep  (needs VLLM_SERVER_DEV_MODE)
  vLLM LoRA               POST /v1/load_lora_adapter · /v1/unload_lora_adapter
  anything else           warm via a one-token /v1/completions probe (load only)

Residency detail is sparse on these runtimes: if the server lists a model it
is typically loaded, unless llama-server reports per-model status on /models.
Inference stats are best-effort from `/metrics`.
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

# Status values from llama-server router GET /models that mean "in memory".
_LLAMA_RESIDENT = frozenset({"loaded", "loading", "sleeping"})


class OpenAICompatEngine(BaseEngine):
    """Base for runtimes that expose an OpenAI-compatible HTTP surface."""

    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            EngineCapability.TELEMETRY,
            EngineCapability.LIST_MODELS,
            EngineCapability.LIST_LOADED,
            EngineCapability.LIFECYCLE,
            EngineCapability.LOAD,
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

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: httpx.Timeout | None = None,
        params: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        """POST JSON and return (status, parsed-or-text). Never raises."""
        url = f"{self.base_url}{path}"
        try:
            response = await self.client.post(
                url,
                json=payload if payload is not None else None,
                params=params,
                timeout=timeout or self.client.timeout,
            )
        except Exception as exc:
            return 0, f"{type(exc).__name__}: {exc}"
        try:
            body: Any = response.json()
        except Exception:
            body = (response.text or "").strip()
        return response.status_code, body

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

    # -- load / unload ------------------------------------------------------ #

    async def load(self, model_id: str) -> tuple[bool, str]:
        """Warm weights with a one-token OpenAI completions probe."""
        return await self._warm_load(model_id)

    async def unload(self, model_id: str | None = None) -> tuple[bool, str]:
        return False, (
            f"{self.display_name}: unload not supported "
            "(base model is fixed for the life of the process)"
        )

    async def _warm_load(self, model_id: str) -> tuple[bool, str]:
        """Hit /v1/completions (then chat) so the runtime pages weights in."""
        timeout = httpx.Timeout(120.0, connect=5.0)
        status, body = await self._post_json(
            "/v1/completions",
            {
                "model": model_id,
                "prompt": ".",
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            },
            timeout=timeout,
        )
        if 200 <= status < 300:
            return True, f"loaded {model_id}"

        status, body = await self._post_json(
            "/v1/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "."}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            },
            timeout=timeout,
        )
        if 200 <= status < 300:
            return True, f"loaded {model_id}"
        return False, f"load failed: {_format_http_error(status, body)}"

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
    """vLLM: sleep/wake for the base model, LoRA load/unload when enabled."""

    kind: ClassVar[EngineKind] = EngineKind.VLLM
    display_name: ClassVar[str] = "vLLM"
    process_names: ClassVar[tuple[str, ...]] = ("vllm", "VLLM")
    health_paths: ClassVar[tuple[str, ...]] = ("/health", "/v1/models")
    metrics_path: ClassVar[str | None] = "/metrics"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            EngineCapability.TELEMETRY,
            EngineCapability.LIST_MODELS,
            EngineCapability.LIST_LOADED,
            EngineCapability.LIFECYCLE,
            EngineCapability.LOAD,
            EngineCapability.UNLOAD,
        }
    )

    async def load(self, model_id: str) -> tuple[bool, str]:
        # Prefer waking a sleeping engine (frees/restores the base model).
        status, body = await self._post_json("/wake_up")
        if 200 <= status < 300:
            return True, f"woke vLLM ({model_id})"
        # 404/405 → sleep mode off; other errors fall through to LoRA / warm.

        # LoRA adapters: model_id may be "name=/path" or just a path/name.
        lora_name, lora_path = _split_lora_ref(model_id)
        status, body = await self._post_json(
            "/v1/load_lora_adapter",
            {"lora_name": lora_name, "lora_path": lora_path},
            timeout=httpx.Timeout(120.0, connect=5.0),
        )
        if 200 <= status < 300:
            return True, f"loaded LoRA {lora_name}"
        if status not in (0, 404, 405):
            return False, f"LoRA load failed: {_format_http_error(status, body)}"

        return await self._warm_load(model_id)

    async def unload(self, model_id: str | None = None) -> tuple[bool, str]:
        if model_id:
            status, body = await self._post_json(
                "/v1/unload_lora_adapter",
                {"lora_name": model_id},
            )
            if 200 <= status < 300:
                return True, f"unloaded LoRA {model_id}"
            if status not in (0, 404, 405):
                return False, f"LoRA unload failed: {_format_http_error(status, body)}"

        # Sleep mode releases VRAM for the base model (dev-mode endpoints).
        status, body = await self._post_json("/sleep", params={"level": 1})
        if 200 <= status < 300:
            target = model_id or "base model"
            return True, f"put vLLM to sleep ({target})"
        if status in (0, 404, 405):
            return False, (
                "vLLM unload needs sleep mode "
                "(VLLM_SERVER_DEV_MODE=1 --enable-sleep-mode) "
                "or a LoRA name with runtime LoRA updating enabled"
            )
        return False, f"sleep failed: {_format_http_error(status, body)}"


class LlamaServerEngine(OpenAICompatEngine):
    """llama-server: router /models/load|/models/unload when multi-model."""

    kind: ClassVar[EngineKind] = EngineKind.LLAMA_SERVER
    display_name: ClassVar[str] = "llama-server"
    process_names: ClassVar[tuple[str, ...]] = ("llama-server", "llama_server")
    health_paths: ClassVar[tuple[str, ...]] = ("/health", "/v1/models")
    # Recent llama.cpp builds expose Prometheus at /metrics.
    metrics_path: ClassVar[str | None] = "/metrics"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            EngineCapability.TELEMETRY,
            EngineCapability.LIST_MODELS,
            EngineCapability.LIST_LOADED,
            EngineCapability.LIFECYCLE,
            EngineCapability.LOAD,
            EngineCapability.UNLOAD,
        }
    )

    async def _collect(self) -> EngineSnapshot:
        # Prefer router catalogue — it distinguishes loaded vs cached-on-disk.
        router = await self._get_json("/models")
        if isinstance(router, dict) and _data_array(router):
            models: list[ModelInfo] = []
            loaded: list[LoadedModel] = []
            for entry in _data_array(router):
                model = self._parse_model(entry)
                models.append(model)
                status = _llama_status_value(entry)
                if status in _LLAMA_RESIDENT:
                    loaded.append(self._as_loaded(model))
            stats = await self._collect_stats()
            version = await self._server_version()
            return self.snapshot(
                state=EngineState.ONLINE,
                version=version,
                models=sorted(models, key=lambda m: m.name),
                loaded=sorted(loaded, key=lambda m: m.name),
                stats=stats,
            )
        return await super()._collect()

    async def _server_version(self) -> str | None:
        payload = await self._get_json("/props")
        if isinstance(payload, dict):
            return first(payload, "version") or "llama-server"
        return "llama-server"

    async def load(self, model_id: str) -> tuple[bool, str]:
        status, body = await self._post_json(
            "/models/load",
            {"model": model_id},
            timeout=httpx.Timeout(300.0, connect=5.0),
        )
        if 200 <= status < 300:
            return True, f"loaded {model_id}"
        if status in (0, 404, 405):
            # Single-model server (no router) — warm via completions.
            return await self._warm_load(model_id)
        return False, f"load failed: {_format_http_error(status, body)}"

    async def unload(self, model_id: str | None = None) -> tuple[bool, str]:
        targets = [model_id] if model_id else await self._router_loaded_ids()
        if not targets:
            return True, "no resident models"

        failed: list[str] = []
        unloaded = 0
        saw_router = False
        for target in targets:
            status, body = await self._post_json("/models/unload", {"model": target})
            if 200 <= status < 300:
                saw_router = True
                unloaded += 1
                continue
            if status in (0, 404, 405):
                continue
            saw_router = True
            failed.append(f"{target} ({_format_http_error(status, body)})")

        if unloaded:
            if failed:
                return False, "partial unload: " + ", ".join(failed)
            return True, f"unloaded {unloaded} model(s)"
        if saw_router and failed:
            return False, "failed to unload: " + ", ".join(failed)
        return False, (
            "llama-server unload needs router mode "
            "(start without -m / --model so /models/unload exists)"
        )

    async def _router_loaded_ids(self) -> list[str]:
        payload = await self._get_json("/models")
        if not isinstance(payload, dict):
            # Fall back to whatever /v1/models currently lists as resident.
            snap = await self.poll()
            return [m.id for m in snap.loaded]
        return [
            str(first(entry, "id", "model", default=""))
            for entry in _data_array(payload)
            if _llama_status_value(entry) in _LLAMA_RESIDENT and first(entry, "id", "model")
        ]


class MLXEngine(OpenAICompatEngine):
    """MLX frontends pin one model at process start — load warms, unload N/A."""

    kind: ClassVar[EngineKind] = EngineKind.MLX
    display_name: ClassVar[str] = "MLX"
    process_names: ClassVar[tuple[str, ...]] = ("mlx_lm", "mlx-openai", "mlx_lm.server")
    health_paths: ClassVar[tuple[str, ...]] = ("/v1/models", "/health")
    metrics_path: ClassVar[str | None] = None
    # Inherit LOAD from the base; do not advertise UNLOAD.


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


def _llama_status_value(entry: dict[str, Any]) -> str:
    status = entry.get("status")
    if isinstance(status, dict):
        return str(status.get("value") or "").lower()
    if isinstance(status, str):
        return status.lower()
    return "loaded"  # older /v1/models entries have no status → treat as resident


def _split_lora_ref(model_id: str) -> tuple[str, str]:
    """Parse `name=/path/to/adapter` or fall back to basename + full ref."""
    if "=" in model_id:
        name, _, path = model_id.partition("=")
        name, path = name.strip(), path.strip()
        if name and path:
            return name, path
    # Use the last path segment as the LoRA name when given a filesystem path.
    name = model_id.rstrip("/").rsplit("/", 1)[-1] or model_id
    return name, model_id


def _format_http_error(status: int, body: Any) -> str:
    if status == 0:
        return str(body) if body else "connection failed"
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return f"HTTP {status}: {err['message']}"
        if isinstance(err, str) and err:
            return f"HTTP {status}: {err}"
        if body.get("message"):
            return f"HTTP {status}: {body['message']}"
    text = str(body).strip() if body is not None else ""
    if text and len(text) < 200:
        return f"HTTP {status}: {text}"
    return f"HTTP {status}"
