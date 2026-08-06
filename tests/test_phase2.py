"""OpenAI-compatible engines, lifecycle helpers, hub search, and serve."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from aitop.config import Config, EndpointConfig
from aitop.engines.lifecycle import start_engine
from aitop.engines.ollama import OllamaEngine
from aitop.engines.openai_compat import (
    LlamaServerEngine,
    MLXEngine,
    VLLMEngine,
    parse_prometheus_stats,
)
from aitop.engines.registry import ADAPTERS, EngineRegistry
from aitop.hub import search_hub
from aitop.models import EngineKind, EngineState, SystemSnapshot
from aitop.serve import FleetNode, SnapshotServer, fetch_remote_snapshot, merge_fleet

VLLM = "http://127.0.0.1:8000"
LLAMA = "http://127.0.0.1:8080"


@pytest.fixture
def vllm_endpoint() -> EndpointConfig:
    return EndpointConfig(kind=EngineKind.VLLM, host="127.0.0.1", port=8000)


@pytest.fixture
def llama_endpoint() -> EndpointConfig:
    return EndpointConfig(kind=EngineKind.LLAMA_SERVER, host="127.0.0.1", port=8080)


# --------------------------------------------------------------------------- #
# OpenAI-compatible adapters
# --------------------------------------------------------------------------- #


@respx.mock
async def test_vllm_online_with_metrics(vllm_endpoint):
    respx.get(f"{VLLM}/v1/models").mock(
        httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "meta-llama/Llama-3.1-8B-Instruct",
                        "object": "model",
                        "max_model_len": 8192,
                    }
                ]
            },
        )
    )
    respx.get(f"{VLLM}/metrics").mock(
        httpx.Response(
            200,
            text=(
                "# HELP vllm:num_requests_running Running requests\n"
                "vllm:num_requests_running 2.0\n"
                "vllm:num_requests_waiting 1.0\n"
                "vllm:avg_generation_throughput_toks_per_s 42.5\n"
                "vllm:avg_prompt_throughput_toks_per_s 120.0\n"
                "vllm:request_success_total 99.0\n"
            ),
        )
    )

    engine = VLLMEngine(vllm_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.state is EngineState.ONLINE
    assert snapshot.kind is EngineKind.VLLM
    assert len(snapshot.models) == 1
    assert len(snapshot.loaded) == 1  # listed == resident for vLLM
    assert snapshot.stats.tokens_per_second == pytest.approx(42.5)
    assert snapshot.stats.active_requests == 2
    assert snapshot.stats.queue_depth == 1
    assert snapshot.stats.total_requests == 99


@respx.mock
async def test_llama_server_online(llama_endpoint):
    respx.get(f"{LLAMA}/v1/models").mock(
        httpx.Response(200, json={"data": [{"id": "qwen2.5-7b", "object": "model"}]})
    )
    respx.get(f"{LLAMA}/props").mock(httpx.Response(200, json={"version": "b1234"}))

    engine = LlamaServerEngine(llama_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.state is EngineState.ONLINE
    assert snapshot.version == "b1234"
    assert snapshot.loaded[0].id == "qwen2.5-7b"


@respx.mock
async def test_openai_compat_offline(vllm_endpoint):
    respx.get(f"{VLLM}/v1/models").mock(side_effect=httpx.ConnectError("refused"))
    respx.get(f"{VLLM}/health").mock(side_effect=httpx.ConnectError("refused"))

    engine = VLLMEngine(vllm_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()
    assert snapshot.state is EngineState.OFFLINE


def test_parse_prometheus_stats_ignores_comments():
    stats = parse_prometheus_stats("# HELP foo bar\nvllm:num_requests_running 3\n")
    assert stats.active_requests == 3
    assert stats.tokens_per_second is None


def test_all_expected_adapters_registered():
    assert set(ADAPTERS) >= {
        EngineKind.OLLAMA,
        EngineKind.LMSTUDIO,
        EngineKind.VLLM,
        EngineKind.LLAMA_SERVER,
        EngineKind.MLX,
    }
    assert EngineKind.MLX in EngineRegistry.supported


# --------------------------------------------------------------------------- #
# Ollama pull + lifecycle stubs
# --------------------------------------------------------------------------- #


@respx.mock
async def test_ollama_pull_streams_progress(ollama_endpoint):
    def _stream(request):
        body = (
            b'{"status":"pulling manifest"}\n'
            b'{"status":"downloading","completed":50,"total":100}\n'
            b'{"status":"success"}\n'
        )
        return httpx.Response(200, content=body)

    respx.post("http://127.0.0.1:11434/api/pull").mock(side_effect=_stream)

    ticks: list = []
    engine = OllamaEngine(ollama_endpoint)
    ok, message = await engine.pull("llama3.2:3b", on_progress=ticks.append)
    await engine.aclose()

    assert ok
    assert "pulled" in message
    assert ticks
    assert ticks[-1].done
    assert any(t.fraction == 0.5 for t in ticks if t.fraction is not None)


@respx.mock
async def test_ollama_pull_reports_errors(ollama_endpoint):
    respx.post("http://127.0.0.1:11434/api/pull").mock(
        httpx.Response(200, content=b'{"error":"model not found"}\n')
    )
    engine = OllamaEngine(ollama_endpoint)
    ok, message = await engine.pull("nope")
    await engine.aclose()
    assert not ok
    assert "model not found" in message


async def test_remote_engine_refuses_lifecycle():
    ep = EndpointConfig(kind=EngineKind.OLLAMA, host="100.100.1.7", remote=True)
    engine = OllamaEngine(ep)
    ok, message = await engine.start()
    await engine.aclose()
    assert not ok
    assert "remote" in message


async def test_lifecycle_missing_binary(monkeypatch):
    monkeypatch.setattr("aitop.engines.lifecycle.which", lambda _: None)

    async def manual(_kind):
        return "manual"

    monkeypatch.setattr("aitop.engines.lifecycle._guess_supervisor", manual)
    result = await start_engine("ollama", managed_by="manual")
    assert not result.ok
    assert "PATH" in result.message


# --------------------------------------------------------------------------- #
# Hub search
# --------------------------------------------------------------------------- #


@respx.mock
async def test_hub_search_parses_results():
    respx.get("https://huggingface.co/api/models").mock(
        httpx.Response(
            200,
            json=[
                {
                    "id": " Quen/Qwen2.5-7B-Instruct-GGUF".strip(),
                    "downloads": 12000,
                    "likes": 40,
                    "tags": ["gguf", "text-generation"],
                    "pipeline_tag": "text-generation",
                    "lastModified": "2025-01-01T00:00:00.000Z",
                }
            ],
        )
    )
    results = await search_hub("qwen2.5", limit=5)
    assert len(results) == 1
    assert results[0].id.endswith("Qwen2.5-7B-Instruct-GGUF")
    assert results[0].downloads == 12000
    assert results[0].url and results[0].url.startswith("https://huggingface.co/")


@respx.mock
async def test_hub_search_degrades_on_error():
    respx.get("https://huggingface.co/api/models").mock(httpx.Response(500))
    assert await search_hub("anything") == []


# --------------------------------------------------------------------------- #
# Serve / fleet
# --------------------------------------------------------------------------- #


@respx.mock
async def test_fetch_remote_snapshot():
    payload = SystemSnapshot(node="peer").model_dump(mode="json")
    respx.get("http://100.100.1.7:9090/api/snapshot").mock(httpx.Response(200, json=payload))
    snap = await fetch_remote_snapshot(FleetNode(name="peer", url="http://100.100.1.7:9090"))
    assert snap is not None
    assert snap.node == "peer"


@respx.mock
async def test_merge_fleet_skips_down_nodes():
    local = SystemSnapshot(node="local")
    respx.get("http://a:9090/api/snapshot").mock(side_effect=httpx.ConnectError("down"))
    respx.get("http://b:9090/api/snapshot").mock(
        httpx.Response(200, json=SystemSnapshot(node="ignored").model_dump(mode="json"))
    )
    merged = await merge_fleet(
        local,
        [
            FleetNode(name="a", url="http://a:9090"),
            FleetNode(name="b", url="http://b:9090"),
        ],
    )
    assert [s.node for s in merged] == ["local", "b"]


def test_fleet_config_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "fleet:\n"
        "  serve_port: 9191\n"
        "  nodes:\n"
        "    - name: pveclaw\n"
        "      url: http://100.100.1.7:9090\n"
    )
    config = Config.load(path)
    assert config.fleet.serve_port == 9191
    assert config.fleet.nodes[0].name == "pveclaw"


async def test_snapshot_server_health_and_snapshot():
    from aitop.collector import SnapshotCollector

    class FakeCollector(SnapshotCollector):
        def __init__(self):
            self.config = Config()
            self.node = "local"

        async def collect(self) -> SystemSnapshot:
            return SystemSnapshot(node="local")

        async def aclose(self) -> None:
            return None

    server = SnapshotServer(FakeCollector(), host="127.0.0.1", port=0)
    # Bind to an ephemeral port by starting then reading the socket.
    srv = await asyncio.start_server(server._handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    server._server = srv

    async with srv, httpx.AsyncClient() as client:
        health = await client.get(f"http://127.0.0.1:{port}/healthz")
        assert health.status_code == 200
        assert health.json()["ok"] is True

        snap = await client.get(f"http://127.0.0.1:{port}/api/snapshot")
        assert snap.status_code == 200
        body = snap.json()
        assert body["node"] == "local"

        missing = await client.get(f"http://127.0.0.1:{port}/nope")
        assert missing.status_code == 404

    await srv.wait_closed()


# --------------------------------------------------------------------------- #
# MLX smoke
# --------------------------------------------------------------------------- #


@respx.mock
async def test_mlx_engine():
    ep = EndpointConfig(kind=EngineKind.MLX, host="127.0.0.1", port=8080)
    respx.get("http://127.0.0.1:8080/v1/models").mock(
        httpx.Response(200, json={"data": [{"id": "mlx-community/Llama-3.2-1B"}]})
    )
    engine = MLXEngine(ep)
    snapshot = await engine.poll()
    await engine.aclose()
    assert snapshot.state is EngineState.ONLINE
    assert snapshot.models[0].id.startswith("mlx-community")


# --------------------------------------------------------------------------- #
# OpenAI-compat load / unload
# --------------------------------------------------------------------------- #


@respx.mock
async def test_llama_server_router_load_unload(llama_endpoint):
    respx.get(f"{LLAMA}/models").mock(
        httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "qwen2.5-7b",
                        "status": {"value": "loaded"},
                    },
                    {
                        "id": "gemma-3",
                        "status": {"value": "unloaded"},
                    },
                ]
            },
        )
    )
    respx.get(f"{LLAMA}/props").mock(httpx.Response(200, json={"version": "b9999"}))
    respx.get(f"{LLAMA}/metrics").mock(httpx.Response(404))
    load_route = respx.post(f"{LLAMA}/models/load").mock(
        httpx.Response(200, json={"success": True})
    )
    unload_route = respx.post(f"{LLAMA}/models/unload").mock(
        httpx.Response(200, json={"success": True})
    )

    engine = LlamaServerEngine(llama_endpoint)
    assert engine.supports("load") and engine.supports("unload")

    snap = await engine.poll()
    assert [m.id for m in snap.models] == ["gemma-3", "qwen2.5-7b"]
    assert [m.id for m in snap.loaded] == ["qwen2.5-7b"]

    ok, message = await engine.load("gemma-3")
    assert ok and "loaded gemma-3" in message
    assert load_route.called
    assert load_route.calls.last.request.content == b'{"model":"gemma-3"}'

    ok, message = await engine.unload("qwen2.5-7b")
    assert ok and "unloaded" in message
    assert unload_route.called

    await engine.aclose()


@respx.mock
async def test_llama_server_load_falls_back_to_warm(llama_endpoint):
    respx.post(f"{LLAMA}/models/load").mock(httpx.Response(404))
    warm = respx.post(f"{LLAMA}/v1/completions").mock(
        httpx.Response(200, json={"choices": [{"text": "."}]})
    )

    engine = LlamaServerEngine(llama_endpoint)
    ok, message = await engine.load("qwen2.5-7b")
    await engine.aclose()

    assert ok
    assert "loaded qwen2.5-7b" in message
    assert warm.called


@respx.mock
async def test_llama_server_unload_without_router(llama_endpoint):
    respx.post(f"{LLAMA}/models/unload").mock(httpx.Response(404))
    respx.get(f"{LLAMA}/models").mock(httpx.Response(404))
    respx.get(f"{LLAMA}/v1/models").mock(httpx.Response(200, json={"data": [{"id": "solo"}]}))
    respx.get(f"{LLAMA}/props").mock(httpx.Response(200, json={"version": "b1"}))

    engine = LlamaServerEngine(llama_endpoint)
    ok, message = await engine.unload()
    await engine.aclose()

    assert not ok
    assert "router mode" in message


@respx.mock
async def test_vllm_load_wake_and_unload_sleep(vllm_endpoint):
    wake = respx.post(f"{VLLM}/wake_up").mock(httpx.Response(200, content=b"ok"))
    sleep = respx.post(f"{VLLM}/sleep").mock(httpx.Response(200, content=b"ok"))

    engine = VLLMEngine(vllm_endpoint)
    assert engine.supports("load") and engine.supports("unload")

    ok, message = await engine.load("meta-llama/Llama-3.1-8B-Instruct")
    assert ok and "woke" in message
    assert wake.called

    ok, message = await engine.unload()
    assert ok and "sleep" in message
    assert sleep.called
    assert sleep.calls.last.request.url.params.get("level") == "1"

    await engine.aclose()


@respx.mock
async def test_vllm_lora_load_unload(vllm_endpoint):
    respx.post(f"{VLLM}/wake_up").mock(httpx.Response(404))
    load = respx.post(f"{VLLM}/v1/load_lora_adapter").mock(httpx.Response(200, content=b"Success"))
    unload = respx.post(f"{VLLM}/v1/unload_lora_adapter").mock(
        httpx.Response(200, content=b"Success")
    )

    engine = VLLMEngine(vllm_endpoint)
    ok, message = await engine.load("sql=/models/sql-lora")
    assert ok and "LoRA sql" in message
    assert b'"lora_name":"sql"' in load.calls.last.request.content

    ok, message = await engine.unload("sql")
    assert ok and "LoRA sql" in message
    assert unload.called

    await engine.aclose()


@respx.mock
async def test_mlx_warm_load():
    ep = EndpointConfig(kind=EngineKind.MLX, host="127.0.0.1", port=8080)
    respx.post("http://127.0.0.1:8080/v1/completions").mock(httpx.Response(404))
    chat = respx.post("http://127.0.0.1:8080/v1/chat/completions").mock(
        httpx.Response(200, json={"choices": [{"message": {"content": "."}}]})
    )

    engine = MLXEngine(ep)
    assert engine.supports("load")
    assert not engine.supports("unload")

    ok, message = await engine.load("mlx-community/Llama-3.2-1B")
    await engine.aclose()

    assert ok and "loaded" in message
    assert chat.called
