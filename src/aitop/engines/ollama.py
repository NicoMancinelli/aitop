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

from collections.abc import AsyncIterator, Callable
from typing import Any, ClassVar

import httpx

from aitop.engines.base import BaseEngine, EngineCapability
from aitop.engines.lifecycle import restart_engine, start_engine, stop_engine
from aitop.models import (
    DownloadProgress,
    EngineKind,
    EngineSnapshot,
    EngineState,
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
            EngineCapability.LIFECYCLE,
            EngineCapability.REBIND,
        }
    )
    process_names: ClassVar[tuple[str, ...]] = ("ollama",)

    async def detect(self) -> bool:
        return await self._get_json("/api/version") is not None

    async def _collect(self) -> EngineSnapshot:
        version_payload = await self._get_json("/api/version")
        if version_payload is None:
            return self.offline()

        version = first(version_payload, "version")
        tags = await self._get_json("/api/tags")
        ps = await self._get_json("/api/ps")

        degraded = tags is None or ps is None
        return self.snapshot(
            state=EngineState.DEGRADED if degraded else EngineState.ONLINE,
            version=version,
            models=self._parse_models(tags),
            loaded=self._parse_loaded(ps),
            error="one or more telemetry endpoints did not respond" if degraded else None,
        )

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
                json={"model": model_id, "keep_alive": "30m"},
                timeout=httpx.Timeout(120.0, connect=5.0),
            )
            response.raise_for_status()
        except Exception as exc:
            return False, f"load failed: {type(exc).__name__}: {exc}"
        return True, f"loaded {model_id}"

    async def start(self) -> tuple[bool, str]:
        if self.endpoint.remote:
            return False, "cannot start a remote engine"
        result = await start_engine("ollama", host=self.host, port=self.port)
        return result.ok, result.message

    async def stop(self) -> tuple[bool, str]:
        if self.endpoint.remote:
            return False, "cannot stop a remote engine"
        snap = await self.poll()
        result = await stop_engine("ollama", pid=snap.pid, managed_by=snap.managed_by)
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
        )
        return result.ok, result.message

    async def rebind(self, host: str) -> tuple[bool, str]:
        """Restart Ollama bound to a new host (e.g. Tailscale IP or 0.0.0.0)."""
        if self.endpoint.remote:
            return False, "cannot rebind a remote engine"
        from aitop.utils.parse import split_host_port

        new_host, new_port = split_host_port(host, self.port)
        snap = await self.poll()
        stopped = await stop_engine("ollama", pid=snap.pid, managed_by=snap.managed_by)
        if not stopped.ok and "already gone" not in stopped.message:
            return False, f"rebind stop failed: {stopped.message}"

        import asyncio

        await asyncio.sleep(0.5)
        started = await start_engine(
            "ollama",
            managed_by=snap.managed_by if snap.managed_by != "systemd" else "manual",
            host=new_host,
            port=new_port,
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
