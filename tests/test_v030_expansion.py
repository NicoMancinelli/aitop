"""Comprehensive test suite for v0.3.0 features:
- Inference benchmarking (Ollama native + OpenAI SSE)
- SGLang and Aphrodite adapters & metrics
- Serve POST /api/models/bench endpoint
- CLI bench subcommands
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import respx

from aitop import cli
from aitop.bench import run_benchmark
from aitop.config import Config, EndpointConfig
from aitop.engines.openai_compat import AphroditeEngine, SGLangEngine
from aitop.models import BenchmarkResult, EngineKind, EngineState, SystemSnapshot
from aitop.serve import SnapshotServer

# --------------------------------------------------------------------------- #
# Benchmarking
# --------------------------------------------------------------------------- #


@respx.mock
async def test_run_benchmark_ollama():
    ollama_url = "http://127.0.0.1:11434/api/generate"
    chunks = [
        json.dumps({"response": "The ", "done": False}) + "\n",
        json.dumps({"response": "sky ", "done": False}) + "\n",
        json.dumps({"response": "is ", "done": False}) + "\n",
        json.dumps({"response": "blue.", "done": True, "prompt_eval_count": 8, "eval_count": 4})
        + "\n",
    ]
    respx.post(ollama_url).mock(
        httpx.Response(
            200, content="".join(chunks).encode(), headers={"content-type": "application/x-ndjson"}
        )
    )

    result = await run_benchmark(
        engine="ollama", model_id="llama3.2:3b", prompt="Sky?", max_tokens=10
    )
    assert isinstance(result, BenchmarkResult)
    assert result.engine == EngineKind.OLLAMA
    assert result.model == "llama3.2:3b"
    assert result.completion_tokens == 4
    assert result.tokens_per_second > 0
    assert result.ttft_ms is not None and result.ttft_ms >= 0
    assert result.success is True


@respx.mock
async def test_run_benchmark_openai_chat():
    vllm_url = "http://127.0.0.1:8000/v1/chat/completions"
    sse_events = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
        'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
        'data: {"choices": [{"delta": {}}], "usage": {"completion_tokens": 2, "prompt_tokens": 4}}\n\n',
        "data: [DONE]\n\n",
    ]
    respx.post(vllm_url).mock(
        httpx.Response(
            200, content="".join(sse_events).encode(), headers={"content-type": "text/event-stream"}
        )
    )

    result = await run_benchmark(engine="vllm", model_id="qwen2.5:7b", max_tokens=10)
    assert isinstance(result, BenchmarkResult)
    assert result.engine == EngineKind.VLLM
    assert result.model == "qwen2.5:7b"
    assert result.completion_tokens == 2
    assert result.tokens_per_second > 0
    assert result.success is True


# --------------------------------------------------------------------------- #
# SGLang & Aphrodite Engines
# --------------------------------------------------------------------------- #


@respx.mock
async def test_sglang_online_with_metrics():
    base = "http://127.0.0.1:30000"
    respx.get(f"{base}/health").mock(httpx.Response(200))
    respx.get(f"{base}/v1/models").mock(
        httpx.Response(
            200,
            json={"data": [{"id": "meta-llama/Llama-3-8B", "created": 1700000000}]},
        )
    )
    prometheus_data = (
        "# HELP sglang:gen_throughput Gen throughput\n"
        "# TYPE sglang:gen_throughput gauge\n"
        "sglang:gen_throughput 42.5\n"
        "# HELP sglang:num_running_reqs Running reqs\n"
        "sglang:num_running_reqs 3\n"
    )
    respx.get(f"{base}/metrics").mock(httpx.Response(200, text=prometheus_data))

    endpoint = EndpointConfig(kind=EngineKind.SGLANG, host="127.0.0.1", port=30000)
    engine = SGLangEngine(endpoint)
    snap = await engine.poll()
    await engine.aclose()

    assert snap.state is EngineState.ONLINE
    assert snap.kind is EngineKind.SGLANG
    assert len(snap.models) == 1
    assert snap.models[0].name == "meta-llama/Llama-3-8B"
    assert snap.stats.tokens_per_second == 42.5
    assert snap.stats.active_requests == 3


@respx.mock
async def test_aphrodite_online_with_metrics():
    base = "http://127.0.0.1:2242"
    respx.get(f"{base}/health").mock(httpx.Response(200))
    respx.get(f"{base}/v1/models").mock(
        httpx.Response(
            200,
            json={"data": [{"id": "PygmalionAI/my-model", "created": 1700000000}]},
        )
    )
    prometheus_data = (
        "aphrodite:avg_generation_throughput_toks_per_s 88.0\naphrodite:num_requests_running 5\n"
    )
    respx.get(f"{base}/metrics").mock(httpx.Response(200, text=prometheus_data))

    endpoint = EndpointConfig(kind=EngineKind.APHRODITE, host="127.0.0.1", port=2242)
    engine = AphroditeEngine(endpoint)
    snap = await engine.poll()
    await engine.aclose()

    assert snap.state is EngineState.ONLINE
    assert snap.kind is EngineKind.APHRODITE
    assert len(snap.models) == 1
    assert snap.models[0].name == "PygmalionAI/my-model"
    assert snap.stats.tokens_per_second == 88.0
    assert snap.stats.active_requests == 5


# --------------------------------------------------------------------------- #
# Serve /api/models/bench endpoint
# --------------------------------------------------------------------------- #


class FakeCollector:
    def __init__(self):
        self.config = Config()
        self.node = "local"
        self.engines = type("R", (), {"engines": []})()

    async def collect(self) -> SystemSnapshot:
        return SystemSnapshot(node="local")

    async def collect_fleet(self) -> list[SystemSnapshot]:
        return [await self.collect()]

    async def aclose(self) -> None:
        return None


async def _boot(server: SnapshotServer):
    srv = await asyncio.start_server(server._handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    server._server = srv
    return srv, port


async def test_serve_bench_endpoint():
    server = SnapshotServer(FakeCollector(), host="127.0.0.1", port=0)  # type: ignore[arg-type]
    srv, port = await _boot(server)
    base = f"http://127.0.0.1:{port}"

    mock_bench = BenchmarkResult(
        engine=EngineKind.OLLAMA,
        model="llama3.2:3b",
        prompt_tokens=4,
        completion_tokens=16,
        total_duration_s=0.5,
        ttft_ms=50.0,
        tokens_per_second=32.0,
        success=True,
    )

    with patch("aitop.bench.run_benchmark", AsyncMock(return_value=mock_bench)):
        async with srv, httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base}/api/models/bench",
                json={"engine": "ollama", "model": "llama3.2:3b", "tokens": 16},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["engine"] == "ollama"
            assert data["tokens_per_second"] == 32.0

            # Missing model
            resp_bad = await client.post(
                f"{base}/api/models/bench",
                json={"engine": "ollama"},
            )
            assert resp_bad.status_code == 400


# --------------------------------------------------------------------------- #
# CLI aitop bench
# --------------------------------------------------------------------------- #


def test_cli_bench_json(capsys):
    mock_bench = BenchmarkResult(
        engine=EngineKind.OLLAMA,
        model="test-model",
        prompt_tokens=5,
        completion_tokens=10,
        total_duration_s=0.2,
        ttft_ms=20.0,
        tokens_per_second=50.0,
        success=True,
    )

    with patch("aitop.cli.run_benchmark", AsyncMock(return_value=mock_bench)):
        rc = cli.main(["bench", "ollama", "test-model", "--tokens", "10", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["engine"] == "ollama"
        assert data["tokens_per_second"] == 50.0


def test_cli_bench_table(capsys):
    mock_bench = BenchmarkResult(
        engine=EngineKind.OLLAMA,
        model="test-model",
        prompt_tokens=5,
        completion_tokens=10,
        total_duration_s=0.2,
        ttft_ms=20.0,
        tokens_per_second=50.0,
        success=True,
    )

    with patch("aitop.cli.run_benchmark", AsyncMock(return_value=mock_bench)):
        rc = cli.main(["bench", "ollama", "test-model", "--tokens", "10"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "50.0 tok/s" in out
        assert "Generation Speed" in out
