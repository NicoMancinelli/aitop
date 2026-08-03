"""Discovery: endpoint fan-out, process attachment, de-duplication."""

from __future__ import annotations

import httpx
import respx

from aitop.config import Config, EndpointConfig
from aitop.engines.registry import ADAPTERS, EngineRegistry, ProcessInfo, _dedupe
from aitop.models import Binding, BindScope, EngineKind, EngineSnapshot, EngineState

OLLAMA = "http://127.0.0.1:11434"
LMSTUDIO = "http://127.0.0.1:1234"


def _config(*endpoints: EndpointConfig) -> Config:
    return Config(discovery={"auto": False, "scan_processes": False}, endpoints=list(endpoints))


@respx.mock
async def test_registry_polls_every_endpoint_concurrently(ollama_tags, ollama_ps):
    respx.get(f"{OLLAMA}/api/version").mock(httpx.Response(200, json={"version": "0.5.7"}))
    respx.get(f"{OLLAMA}/api/tags").mock(httpx.Response(200, json=ollama_tags))
    respx.get(f"{OLLAMA}/api/ps").mock(httpx.Response(200, json=ollama_ps))
    respx.get(f"{LMSTUDIO}/api/v0/models").mock(side_effect=httpx.ConnectError("refused"))
    respx.get(f"{LMSTUDIO}/v1/models").mock(side_effect=httpx.ConnectError("refused"))

    config = _config(
        EndpointConfig(kind=EngineKind.OLLAMA),
        EndpointConfig(kind=EngineKind.LMSTUDIO),
    )
    async with EngineRegistry(config) as registry:
        snapshots = await registry.poll_all()
        online = await registry.discover()

    assert {s.kind for s in snapshots} == {EngineKind.OLLAMA, EngineKind.LMSTUDIO}
    assert [s.kind for s in online] == [EngineKind.OLLAMA]


@respx.mock
async def test_registry_builds_vllm_adapter():
    """vLLM now has a real adapter and must be constructed like any other kind."""
    config = _config(EndpointConfig(kind=EngineKind.VLLM))
    async with EngineRegistry(config) as registry:
        engines = registry.build()
        assert len(engines) == 1
        assert engines[0].kind is EngineKind.VLLM


@respx.mock
async def test_registry_skips_unknown_engine_kinds():
    """Kinds without an adapter must not blow up build()."""
    config = _config(EndpointConfig(kind=EngineKind.UNKNOWN))
    async with EngineRegistry(config) as registry:
        assert registry.build() == []
        assert await registry.poll_all() == []


@respx.mock
async def test_registry_attaches_process_metadata(monkeypatch, ollama_tags, ollama_ps):
    respx.get(f"{OLLAMA}/api/version").mock(httpx.Response(200, json={"version": "0.5.7"}))
    respx.get(f"{OLLAMA}/api/tags").mock(httpx.Response(200, json=ollama_tags))
    respx.get(f"{OLLAMA}/api/ps").mock(httpx.Response(200, json=ollama_ps))

    monkeypatch.setattr(
        "aitop.engines.registry.scan_processes",
        lambda: {EngineKind.OLLAMA: ProcessInfo(4242, "ollama", "launchd", {11434})},
    )

    config = Config(
        discovery={"auto": False, "scan_processes": True},
        endpoints=[EndpointConfig(kind=EngineKind.OLLAMA)],
    )
    async with EngineRegistry(config) as registry:
        snapshot = (await registry.poll_all())[0]

    assert snapshot.pid == 4242
    assert snapshot.managed_by == "launchd"
    assert snapshot.process_name == "ollama"


@respx.mock
async def test_remote_endpoints_do_not_inherit_local_pids(monkeypatch, ollama_tags, ollama_ps):
    respx.get("http://100.100.1.7:11434/api/version").mock(
        httpx.Response(200, json={"version": "0.5.7"})
    )
    respx.get("http://100.100.1.7:11434/api/tags").mock(httpx.Response(200, json=ollama_tags))
    respx.get("http://100.100.1.7:11434/api/ps").mock(httpx.Response(200, json=ollama_ps))

    monkeypatch.setattr(
        "aitop.engines.registry.scan_processes",
        lambda: {EngineKind.OLLAMA: ProcessInfo(4242, "ollama", "launchd", {11434})},
    )

    config = Config(
        discovery={"auto": False, "scan_processes": True},
        endpoints=[EndpointConfig(kind=EngineKind.OLLAMA, host="100.100.1.7", remote=True)],
    )
    async with EngineRegistry(config) as registry:
        snapshot = (await registry.poll_all())[0]

    assert snapshot.pid is None
    assert snapshot.binding is not None
    assert snapshot.binding.scope is BindScope.TAILSCALE


def test_dedupe_prefers_the_online_snapshot():
    binding = Binding(host="127.0.0.1", port=11434, scope=BindScope.LOOPBACK)
    offline = EngineSnapshot(
        kind=EngineKind.OLLAMA, name="Ollama", state=EngineState.OFFLINE, binding=binding
    )
    online = EngineSnapshot(
        kind=EngineKind.OLLAMA, name="Ollama", state=EngineState.ONLINE, binding=binding
    )
    assert [s.state for s in _dedupe([offline, online])] == [EngineState.ONLINE]
    assert len(_dedupe([online, offline])) == 1


def test_every_adapter_declares_its_identity():
    for kind, adapter in ADAPTERS.items():
        assert adapter.kind is kind
        assert adapter.display_name != "unknown"
        assert adapter.process_names, f"{adapter.__name__} needs process_names for discovery"
