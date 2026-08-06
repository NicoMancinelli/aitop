"""Ollama adapter.

Telemetry comes from three endpoints, all cheap and unauthenticated:

  GET  /api/version  -> daemon version (also our liveness probe)
  GET  /api/tags     -> every model on disk
  GET  /api/ps       -> resident models, with the GPU-offloaded byte count

`size` vs `size_vram` on /api/ps is the interesting bit: their ratio is the
share of the model actually living on the GPU, which is what the KV/VRAM
panel visualises.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any, ClassVar

import httpx

from aitop.engines.base import BaseEngine, EngineCapability
from aitop.engines.lifecycle import restart_engine, start_engine, stop_engine
from aitop.engines.stats import (
    STATS_PROBE_INTERVAL_S,
    inference_stats_from_ollama,
    parse_prometheus_stats,
)
from aitop.models import (
    DownloadProgress,
    EngineKind,
    EngineSnapshot,
    EngineState,
    InferenceStats,
    LoadedModel,
    ModelInfo,
)
from aitop.utils.parse import first, parse_timestamp, to_int


class OllamaEngine(BaseEngine):
    kind: ClassVar[EngineKind] = EngineKind.OLLAMA
    display_name: ClassVar[str] = "Ollama"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            EngineCapability.TELEMETRY,
            EngineCapability.LIST_MODELS,
            EngineCapability.LIST_LOADED,
            EngineCapability.UNLOAD,
            EngineCapability.LOAD,
            EngineCapability.PULL,
            EngineCapability.DELETE,
            EngineCapability.LIFECYCLE,
            EngineCapability.REBIND,
        }
    )
    process_names: ClassVar[tuple[str, ...]] = ("ollama",)

    def __init__(self, endpoint, client=None) -> None:
        super().__init__(endpoint, client)
        self._last_stats = InferenceStats()
        self._last_stats_mono = 0.0

    async def detect(self) -> bool:
        return await self._get_json("/api/version") is not None

    async def _collect(self) -> EngineSnapshot:
        version_payload = await self._get_json("/api/version")
        if version_payload is None:
            return self.offline()

        version = first(version_payload, "version")
        tags = await self._get_json("/api/tags")
        ps = await self._get_json("/api/ps")
        loaded = self._parse_loaded(ps)

        degraded = tags is None or ps is None
        return self.snapshot(
            state=EngineState.DEGRADED if degraded else EngineState.ONLINE,
            version=version,
            models=self._parse_models(tags),
            loaded=loaded,
            stats=await self._collect_stats(loaded),
            error="one or more telemetry endpoints did not respond" if degraded else None,
        )

    def _remember_stats(self, stats: InferenceStats) -> None:
        if stats.tokens_per_second is None and stats.prompt_tokens_per_second is None:
            return
        self._last_stats = stats
        self._last_stats_mono = time.monotonic()

    async def _collect_stats(self, loaded: list[LoadedModel]) -> InferenceStats:
        """Prefer /metrics (when present), else cache / soft-probe generate."""
        metrics = await self._try_prometheus_stats()
        if metrics.tokens_per_second is not None or metrics.prompt_tokens_per_second is not None:
            self._remember_stats(metrics)
            return metrics

        now = time.monotonic()
        due = (now - self._last_stats_mono) >= STATS_PROBE_INTERVAL_S
        if loaded and due:
            # Advance the clock even on failure so we don't probe every poll tick.
            self._last_stats_mono = now
            probed = await self._probe_generate_stats(loaded[0].id)
            if probed.tokens_per_second is not None:
                self._remember_stats(probed)
                return probed

        return self._last_stats

    async def _try_prometheus_stats(self) -> InferenceStats:
        url = f"{self.base_url}/metrics"
        try:
            response = await self.client.get(url)
            if response.status_code >= 400:
                return InferenceStats()
            return parse_prometheus_stats(response.text)
        except Exception:
            return InferenceStats()

    async def _probe_generate_stats(self, model_id: str) -> InferenceStats:
        """One-token generate to sample tok/s without changing keep-alive."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_id,
                    "prompt": ".",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=httpx.Timeout(20.0, connect=3.0),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return InferenceStats()
        if not isinstance(payload, dict):
            return InferenceStats()
        return inference_stats_from_ollama(payload)

    # -- parsing ------------------------------------------------------------ #

    def _parse_models(self, payload: Any) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for entry in _models_array(payload):
            details = entry.get("details") or {}
            name = first(entry, "name", "model", default="unknown")
            out.append(
                ModelInfo(
                    id=first(entry, "model", "name", default=name),
                    name=name,
                    engine=self.kind,
                    family=first(details, "family"),
                    parameter_size=first(details, "parameter_size"),
                    quantization=first(details, "quantization_level"),
                    format=first(details, "format"),
                    size_bytes=to_int(entry.get("size")),
                    modified_at=parse_timestamp(entry.get("modified_at")),
                )
            )
        return sorted(out, key=lambda m: m.name)

    def _parse_loaded(self, payload: Any) -> list[LoadedModel]:
        out: list[LoadedModel] = []
        for entry in _models_array(payload):
            details = entry.get("details") or {}
            name = first(entry, "name", "model", default="unknown")
            out.append(
                LoadedModel(
                    id=first(entry, "model", "name", default=name),
                    name=name,
                    engine=self.kind,
                    family=first(details, "family"),
                    parameter_size=first(details, "parameter_size"),
                    quantization=first(details, "quantization_level"),
                    size_bytes=to_int(entry.get("size")),
                    vram_bytes=to_int(entry.get("size_vram")),
                    context_length=to_int(first(entry, "context_length", "num_ctx")),
                    expires_at=parse_timestamp(entry.get("expires_at")),
                )
            )
        return sorted(out, key=lambda m: m.name)

    # -- control ------------------------------------------------------------ #

    async def unload(self, model_id: str | None = None) -> tuple[bool, str]:
        """Evict resident weights by re-requesting them with keep_alive=0.

        Ollama has no explicit unload API; a zero keep-alive on an empty
        generate request drops the model and its KV cache immediately.
        """
        targets = [model_id] if model_id else [m.id for m in (await self.poll()).loaded]
        if not targets:
            return True, "no resident models"

        failed: list[str] = []
        for target in targets:
            try:
                response = await self.client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": target, "keep_alive": 0},
                )
                response.raise_for_status()
            except Exception as exc:
                failed.append(f"{target} ({type(exc).__name__})")

        if failed:
            return False, "failed to unload: " + ", ".join(failed)
        return True, f"unloaded {len(targets)} model(s)"

    async def load(self, model_id: str) -> tuple[bool, str]:
        """Warm a model into memory with a keep-alive generate request."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_id,
                    "prompt": ".",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"num_predict": 1},
                },
                timeout=httpx.Timeout(120.0, connect=5.0),
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                self._remember_stats(inference_stats_from_ollama(payload))
        except Exception as exc:
            return False, f"load failed: {type(exc).__name__}: {exc}"
        return True, f"loaded {model_id}"

    async def delete(self, model_id: str) -> tuple[bool, str]:
        """`DELETE /api/delete` removes a model from disk."""
        try:
            response = await self.client.request(
                "DELETE",
                f"{self.base_url}/api/delete",
                json={"model": model_id},
            )
            # Older Ollamas used POST /api/delete.
            if response.status_code == 404:
                response = await self.client.post(
                    f"{self.base_url}/api/delete",
                    json={"name": model_id},
                )
            response.raise_for_status()
        except Exception as exc:
            return False, f"delete failed: {type(exc).__name__}: {exc}"
        return True, f"deleted {model_id}"

    async def start(self) -> tuple[bool, str]:
        if self.endpoint.remote:
            return False, "cannot start a remote engine"
        result = await start_engine(
            "ollama",
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
            "ollama",
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
            "ollama",
            pid=snap.pid,
            managed_by=snap.managed_by,
            host=self.host,
            port=self.port,
            container=self.endpoint.container,
        )
        return result.ok, result.message

    async def rebind(self, host: str) -> tuple[bool, str]:
        """Restart Ollama bound to a new host (e.g. Tailscale IP or 0.0.0.0)."""
        if self.endpoint.remote:
            return False, "cannot rebind a remote engine"
        from aitop.utils.parse import split_host_port

        new_host, new_port = split_host_port(host, self.port)
        snap = await self.poll()
        stopped = await stop_engine(
            "ollama",
            pid=snap.pid,
            managed_by=snap.managed_by,
            container=self.endpoint.container,
        )
        if not stopped.ok and "already gone" not in stopped.message:
            return False, f"rebind stop failed: {stopped.message}"

        import asyncio

        await asyncio.sleep(0.5)
        started = await start_engine(
            "ollama",
            managed_by=snap.managed_by if snap.managed_by != "systemd" else "manual",
            host=new_host,
            port=new_port,
            container=self.endpoint.container,
        )
        if not started.ok:
            return False, (
                f"stopped ollama but failed to restart on {new_host}:{new_port}: "
                f"{started.message}. For systemd/launchd, set OLLAMA_HOST in the "
                "unit/plist and restart the service."
            )
        self.endpoint.host = new_host
        self.endpoint.port = new_port
        return True, f"rebound Ollama to {new_host}:{new_port}"

    async def pull(
        self,
        model: str,
        *,
        on_progress: Callable[[DownloadProgress], None] | None = None,
    ) -> tuple[bool, str]:
        """Stream `POST /api/pull` and optionally report progress."""
        url = f"{self.base_url}/api/pull"
        try:
            async with self.client.stream(
                "POST",
                url,
                json={"name": model, "stream": True},
                timeout=None,
            ) as response:
                response.raise_for_status()
                async for progress in self._iter_pull_progress(response, model):
                    if on_progress:
                        on_progress(progress)
                    if progress.status == "error":
                        return False, progress.error or "pull failed"
                    if progress.done:
                        return True, f"pulled {model}"
        except Exception as exc:
            return False, f"pull failed: {type(exc).__name__}: {exc}"
        return True, f"pulled {model}"

    async def _iter_pull_progress(self, response, model: str) -> AsyncIterator[DownloadProgress]:
        import json

        async for line in response.aiter_lines():
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            error = payload.get("error")
            status = str(payload.get("status") or ("error" if error else "pulling"))
            completed = to_int(payload.get("completed"))
            total = to_int(payload.get("total"))
            done = status.lower() in {"success", "complete", "completed"} or (
                error is None and status == "success"
            )
            if error:
                yield DownloadProgress(
                    model=model,
                    engine=self.kind,
                    status="error",
                    error=str(error),
                    done=True,
                )
                return
            yield DownloadProgress(
                model=model,
                engine=self.kind,
                status=status,
                completed_bytes=completed,
                total_bytes=total,
                done=done,
            )
            if done:
                return


def _models_array(payload: Any) -> list[dict[str, Any]]:
    """Both /api/tags and /api/ps wrap their list in a `models` key."""
    if isinstance(payload, dict):
        entries = payload.get("models")
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = None
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]
