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

from typing import Any, ClassVar

from aitop.engines.base import BaseEngine, EngineCapability
from aitop.models import EngineKind, EngineSnapshot, EngineState, LoadedModel, ModelInfo
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
            EngineCapability.PULL,
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
