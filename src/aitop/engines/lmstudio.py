"""LM Studio adapter.

LM Studio exposes two HTTP surfaces:

  GET /api/v0/models  -- native REST (0.3.x+). Rich: arch, quantization,
                         load state, max/loaded context length.
  GET /v1/models      -- OpenAI-compatible. Only ids, but always present.

We prefer the native endpoint and fall back to the OpenAI one, which means a
very old LM Studio still shows up as online with a model list, just without
residency detail. The `lms` CLI is consulted opportunistically for the app
version and for loaded-model detail the HTTP API omits.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from aitop.engines.base import BaseEngine, EngineCapability
from aitop.models import EngineKind, EngineSnapshot, EngineState, LoadedModel, ModelInfo
from aitop.utils.parse import first, to_int
from aitop.utils.proc import run, which

_LOADED_STATES = {"loaded", "loading"}


class LMStudioEngine(BaseEngine):
    kind: ClassVar[EngineKind] = EngineKind.LMSTUDIO
    display_name: ClassVar[str] = "LM Studio"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            EngineCapability.TELEMETRY,
            EngineCapability.LIST_MODELS,
            EngineCapability.LIST_LOADED,
            EngineCapability.UNLOAD,
        }
    )
    process_names: ClassVar[tuple[str, ...]] = ("LM Studio", "lms", "lmstudio")

    async def detect(self) -> bool:
        return (
            await self._get_json("/api/v0/models") is not None
            or await self._get_json("/v1/models") is not None
        )

    async def _collect(self) -> EngineSnapshot:
        native = await self._get_json("/api/v0/models")
        if native is not None:
            models = [self._model_from_native(e) for e in _data_array(native)]
            loaded = [
                self._loaded_from_native(e)
                for e in _data_array(native)
                if str(e.get("state", "")).lower() in _LOADED_STATES
            ]
            state = EngineState.ONLINE
            error = None
        else:
            openai = await self._get_json("/v1/models")
            if openai is None:
                return self.offline()
            models = [self._model_from_openai(e) for e in _data_array(openai)]
            loaded = []
            state = EngineState.DEGRADED
            error = "native /api/v0 unavailable; residency detail unknown"

        if not loaded:
            cli_loaded = await self._loaded_from_cli()
            if cli_loaded:
                loaded = cli_loaded
                if state is EngineState.DEGRADED:
                    error = "native /api/v0 unavailable; residency read from `lms`"

        return self.snapshot(
            state=state,
            version=await self._cli_version(),
            models=sorted(models, key=lambda m: m.name),
            loaded=sorted(loaded, key=lambda m: m.name),
            error=error,
        )

    # -- parsing ------------------------------------------------------------ #

    def _model_from_native(self, entry: dict[str, Any]) -> ModelInfo:
        model_id = str(first(entry, "id", "modelKey", default="unknown"))
        return ModelInfo(
            id=model_id,
            name=model_id,
            engine=self.kind,
            family=first(entry, "arch", "architecture"),
            quantization=first(entry, "quantization"),
            format=first(entry, "compatibility_type", "format"),
            max_context=to_int(first(entry, "max_context_length", "maxContextLength")),
        )

    def _loaded_from_native(self, entry: dict[str, Any]) -> LoadedModel:
        model_id = str(first(entry, "id", "modelKey", default="unknown"))
        return LoadedModel(
            id=model_id,
            name=model_id,
            engine=self.kind,
            family=first(entry, "arch", "architecture"),
            quantization=first(entry, "quantization"),
            context_length=to_int(
                first(entry, "loaded_context_length", "max_context_length", "contextLength")
            ),
        )

    def _model_from_openai(self, entry: dict[str, Any]) -> ModelInfo:
        model_id = str(first(entry, "id", default="unknown"))
        return ModelInfo(id=model_id, name=model_id, engine=self.kind)

    # -- CLI hooks ---------------------------------------------------------- #

    async def _cli_version(self) -> str | None:
        """`lms version` -> 'lms version 0.0.44'. Local endpoints only."""
        if self.endpoint.remote or which("lms") is None:
            return None
        result = await run("lms", "version", timeout=3.0)
        if not result.ok:
            return None
        tokens = result.stdout.split()
        return tokens[-1].strip() if tokens else None

    async def _loaded_from_cli(self) -> list[LoadedModel]:
        """`lms ps --json` reports resident models with byte sizes."""
        if self.endpoint.remote or which("lms") is None:
            return []
        result = await run("lms", "ps", "--json", timeout=5.0)
        if not result.ok:
            return []
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

        entries = payload if isinstance(payload, list) else _data_array(payload)
        out: list[LoadedModel] = []
        for entry in entries:
            model_id = str(first(entry, "identifier", "modelKey", "path", default="unknown"))
            out.append(
                LoadedModel(
                    id=model_id,
                    name=model_id,
                    engine=self.kind,
                    family=first(entry, "architecture", "arch"),
                    quantization=first(entry, "quantization"),
                    size_bytes=to_int(first(entry, "sizeBytes", "size_bytes", "size")),
                    context_length=to_int(first(entry, "contextLength", "context_length")),
                )
            )
        return out

    # -- control ------------------------------------------------------------ #

    async def unload(self, model_id: str | None = None) -> tuple[bool, str]:
        """`lms unload` drops weights and the KV cache. Requires the CLI."""
        if self.endpoint.remote:
            return False, "unload requires a local `lms` CLI"
        if which("lms") is None:
            return False, "`lms` CLI not found on PATH"
        argv = ("lms", "unload", model_id) if model_id else ("lms", "unload", "--all")
        result = await run(*argv, timeout=20.0)
        if result.ok:
            return True, f"unloaded {model_id or 'all models'}"
        return False, result.reason


def _data_array(payload: Any) -> list[dict[str, Any]]:
    """Both LM Studio surfaces wrap results in an OpenAI-style `data` key."""
    if isinstance(payload, dict):
        entries = payload.get("data")
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = None
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]
