"""Engine adapter behaviour, against mocked HTTP surfaces."""

from __future__ import annotations

import httpx
import pytest
import respx

from aitop.engines.base import classify_scope
from aitop.engines.lmstudio import LMStudioEngine
from aitop.engines.ollama import OllamaEngine
from aitop.models import BindScope, EngineKind, EngineState

OLLAMA = "http://127.0.0.1:11434"
LMSTUDIO = "http://127.0.0.1:1234"


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


@respx.mock
async def test_ollama_online(ollama_endpoint, ollama_tags, ollama_ps):
    respx.get(f"{OLLAMA}/api/version").mock(httpx.Response(200, json={"version": "0.5.7"}))
    respx.get(f"{OLLAMA}/api/tags").mock(httpx.Response(200, json=ollama_tags))
    respx.get(f"{OLLAMA}/api/ps").mock(httpx.Response(200, json=ollama_ps))

    engine = OllamaEngine(ollama_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.state is EngineState.ONLINE
    assert snapshot.version == "0.5.7"
    assert snapshot.kind is EngineKind.OLLAMA
    assert snapshot.latency_ms is not None
    assert [m.name for m in snapshot.models] == ["llama3.2:3b", "qwen2.5-coder:7b"]
    assert snapshot.models[0].quantization == "Q4_K_M"
    assert snapshot.models[0].parameter_size == "3.2B"
    # 9-digit nanosecond precision must not defeat the timestamp parser.
    assert snapshot.models[0].modified_at is not None

    loaded = snapshot.loaded[0]
    assert loaded.name == "llama3.2:3b"
    assert loaded.context_length == 8192
    assert loaded.gpu_fraction == 1.0  # fully offloaded
    assert loaded.expires_at is not None


@respx.mock
async def test_ollama_offline_when_version_unreachable(ollama_endpoint):
    respx.get(f"{OLLAMA}/api/version").mock(side_effect=httpx.ConnectError("refused"))

    engine = OllamaEngine(ollama_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.state is EngineState.OFFLINE
    assert snapshot.models == []
    assert snapshot.binding is not None and snapshot.binding.port == 11434


@respx.mock
async def test_ollama_degraded_when_one_endpoint_fails(ollama_endpoint, ollama_tags):
    respx.get(f"{OLLAMA}/api/version").mock(httpx.Response(200, json={"version": "0.5.7"}))
    respx.get(f"{OLLAMA}/api/tags").mock(httpx.Response(200, json=ollama_tags))
    respx.get(f"{OLLAMA}/api/ps").mock(httpx.Response(500))

    engine = OllamaEngine(ollama_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.state is EngineState.DEGRADED
    assert snapshot.error is not None
    assert len(snapshot.models) == 2  # the endpoint that worked still reports


@respx.mock
async def test_ollama_survives_garbage_payloads(ollama_endpoint):
    respx.get(f"{OLLAMA}/api/version").mock(httpx.Response(200, json={"version": "0.5.7"}))
    respx.get(f"{OLLAMA}/api/tags").mock(httpx.Response(200, json={"models": "not-a-list"}))
    respx.get(f"{OLLAMA}/api/ps").mock(httpx.Response(200, json={"models": [None, 42, {}]}))

    engine = OllamaEngine(ollama_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.state is EngineState.ONLINE
    assert snapshot.models == []
    assert len(snapshot.loaded) == 1  # the one dict entry, with everything unknown
    assert snapshot.loaded[0].name == "unknown"


@respx.mock
async def test_ollama_partial_gpu_offload(ollama_endpoint):
    respx.get(f"{OLLAMA}/api/version").mock(httpx.Response(200, json={"version": "0.5.7"}))
    respx.get(f"{OLLAMA}/api/tags").mock(httpx.Response(200, json={"models": []}))
    respx.get(f"{OLLAMA}/api/ps").mock(
        httpx.Response(
            200,
            json={
                "models": [{"name": "big:70b", "size": 40_000_000_000, "size_vram": 10_000_000_000}]
            },
        )
    )

    engine = OllamaEngine(ollama_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.loaded[0].gpu_fraction == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# LM Studio
# --------------------------------------------------------------------------- #


@respx.mock
async def test_lmstudio_native_api(lmstudio_endpoint, lmstudio_native, monkeypatch):
    monkeypatch.setattr("aitop.engines.lmstudio.which", lambda _: None)  # no CLI on PATH
    respx.get(f"{LMSTUDIO}/api/v0/models").mock(httpx.Response(200, json=lmstudio_native))

    engine = LMStudioEngine(lmstudio_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.state is EngineState.ONLINE
    assert len(snapshot.models) == 2
    assert len(snapshot.loaded) == 1
    assert snapshot.loaded[0].id == "qwen2.5-7b-instruct"
    assert snapshot.loaded[0].context_length == 4096  # loaded, not max
    mlx = next(m for m in snapshot.models if m.format == "mlx")
    assert mlx.quantization == "4bit"
    assert mlx.max_context == 131072


@respx.mock
async def test_lmstudio_falls_back_to_openai_api(lmstudio_endpoint, monkeypatch):
    monkeypatch.setattr("aitop.engines.lmstudio.which", lambda _: None)
    respx.get(f"{LMSTUDIO}/api/v0/models").mock(httpx.Response(404))
    respx.get(f"{LMSTUDIO}/v1/models").mock(
        httpx.Response(200, json={"data": [{"id": "some-model", "object": "model"}]})
    )

    engine = LMStudioEngine(lmstudio_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.state is EngineState.DEGRADED
    assert [m.id for m in snapshot.models] == ["some-model"]
    assert snapshot.loaded == []
    assert "residency" in (snapshot.error or "")


@respx.mock
async def test_lmstudio_offline(lmstudio_endpoint, monkeypatch):
    monkeypatch.setattr("aitop.engines.lmstudio.which", lambda _: None)
    respx.get(f"{LMSTUDIO}/api/v0/models").mock(side_effect=httpx.ConnectError("refused"))
    respx.get(f"{LMSTUDIO}/v1/models").mock(side_effect=httpx.ConnectError("refused"))

    engine = LMStudioEngine(lmstudio_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.state is EngineState.OFFLINE


# --------------------------------------------------------------------------- #
# Bind scope classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", BindScope.LOOPBACK),
        ("::1", BindScope.LOOPBACK),
        ("localhost", BindScope.LOOPBACK),
        ("0.0.0.0", BindScope.LAN),
        ("192.168.86.56", BindScope.LAN),
        ("10.0.0.4", BindScope.LAN),
        ("100.100.1.2", BindScope.TAILSCALE),
        ("100.76.70.44", BindScope.TAILSCALE),
        ("100.5.1.1", BindScope.OTHER),  # 100.0.0.0/8 outside the CGNAT range
        ("203.0.113.9", BindScope.OTHER),
    ],
)
def test_classify_scope(host, expected):
    assert classify_scope(host) is expected
