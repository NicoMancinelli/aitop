"""InferenceStats for Ollama/LM Studio, fleet API, EventSource token, HF ingest."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from aitop.engines.lmstudio import LMStudioEngine
from aitop.engines.ollama import OllamaEngine
from aitop.engines.stats import (
    inference_stats_from_lmstudio,
    inference_stats_from_ollama,
    parse_prometheus_stats,
)
from aitop.hub import ollama_hub_ref
from aitop.models import EngineState, InferenceStats, SystemSnapshot
from aitop.serve import SnapshotServer
from aitop.views.web import render_dashboard

OLLAMA = "http://127.0.0.1:11434"
LMSTUDIO = "http://127.0.0.1:1234"


def test_inference_stats_from_ollama_generate():
    stats = inference_stats_from_ollama(
        {
            "eval_count": 100,
            "eval_duration": 2_000_000_000,  # 2s → 50 tok/s
            "prompt_eval_count": 10,
            "prompt_eval_duration": 500_000_000,  # 0.5s → 20 tok/s
        }
    )
    assert stats.tokens_per_second == pytest.approx(50.0)
    assert stats.prompt_tokens_per_second == pytest.approx(20.0)
    assert stats.ttft_ms == pytest.approx(500.0)


def test_inference_stats_from_lmstudio_payload():
    stats = inference_stats_from_lmstudio(
        {
            "stats": {
                "tokens_per_second": 41.5,
                "time_to_first_token": 0.12,
            },
            "usage": {"total_tokens": 12},
        }
    )
    assert stats.tokens_per_second == pytest.approx(41.5)
    assert stats.ttft_ms == pytest.approx(120.0)


def test_parse_prometheus_llama_cpp_names():
    text = (
        "# HELP llamacpp:predicted_tokens_seconds\n"
        "llamacpp:predicted_tokens_seconds 33.0\n"
        "llamacpp:prompt_tokens_seconds 90.0\n"
        "llamacpp:requests_processing 1\n"
    )
    stats = parse_prometheus_stats(text)
    assert stats.tokens_per_second == pytest.approx(33.0)
    assert stats.prompt_tokens_per_second == pytest.approx(90.0)
    assert stats.active_requests == 1


def test_ollama_hub_ref_mapping():
    assert ollama_hub_ref("bartowski/Foo-GGUF") == "hf.co/bartowski/Foo-GGUF"
    assert ollama_hub_ref("hf.co/org/model") == "hf.co/org/model"
    assert ollama_hub_ref("https://huggingface.co/org/model") == "hf.co/org/model"
    assert ollama_hub_ref("llama3.2:3b") == "llama3.2:3b"


@respx.mock
async def test_ollama_collects_stats_from_metrics(ollama_endpoint, ollama_tags, ollama_ps):
    respx.get(f"{OLLAMA}/api/version").mock(httpx.Response(200, json={"version": "0.5.7"}))
    respx.get(f"{OLLAMA}/api/tags").mock(httpx.Response(200, json=ollama_tags))
    respx.get(f"{OLLAMA}/api/ps").mock(httpx.Response(200, json=ollama_ps))
    respx.get(f"{OLLAMA}/metrics").mock(
        httpx.Response(
            200,
            text="ollama_tokens_per_second 55.0\nollama_prompt_tokens_per_second 120.0\n",
        )
    )

    engine = OllamaEngine(ollama_endpoint)
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.state is EngineState.ONLINE
    assert snapshot.stats.tokens_per_second == pytest.approx(55.0)
    assert snapshot.stats.prompt_tokens_per_second == pytest.approx(120.0)


@respx.mock
async def test_ollama_load_caches_generate_stats(ollama_endpoint):
    respx.post(f"{OLLAMA}/api/generate").mock(
        httpx.Response(
            200,
            json={
                "model": "llama3.2:3b",
                "eval_count": 1,
                "eval_duration": 25_000_000,  # 40 tok/s
                "prompt_eval_count": 1,
                "prompt_eval_duration": 10_000_000,
            },
        )
    )
    engine = OllamaEngine(ollama_endpoint)
    ok, _ = await engine.load("llama3.2:3b")
    await engine.aclose()
    assert ok
    assert engine._last_stats.tokens_per_second == pytest.approx(40.0)


@respx.mock
async def test_lmstudio_soft_probe_stats(lmstudio_endpoint, lmstudio_native, monkeypatch):
    monkeypatch.setattr("aitop.engines.lmstudio.which", lambda _: None)
    monkeypatch.setattr("aitop.engines.lmstudio.STATS_PROBE_INTERVAL_S", 0.0)
    respx.get(f"{LMSTUDIO}/api/v0/models").mock(httpx.Response(200, json=lmstudio_native))
    respx.get(f"{LMSTUDIO}/metrics").mock(httpx.Response(404))
    respx.post(f"{LMSTUDIO}/api/v0/chat/completions").mock(
        httpx.Response(
            200,
            json={
                "stats": {"tokens_per_second": 77.2, "time_to_first_token": 0.05},
                "usage": {"total_tokens": 2},
            },
        )
    )

    engine = LMStudioEngine(lmstudio_endpoint)
    # Force probe due immediately.
    engine._last_stats_mono = 0.0
    snapshot = await engine.poll()
    await engine.aclose()

    assert snapshot.stats.tokens_per_second == pytest.approx(77.2)


def test_dashboard_eventsource_uses_token_query():
    html = render_dashboard(auth_required=True, auth_all=True).decode()
    assert "withToken" in html
    assert "token=" in html
    assert "/api/fleet" in html
    assert "AUTH_ALL" in html


async def test_auth_all_accepts_query_token():
    from aitop.config import Config, FleetConfig

    class FakeCollector:
        def __init__(self):
            self.config = Config(fleet=FleetConfig(serve_token="lock", serve_auth_all=True))
            self.node = "local"
            self.engines = type("R", (), {"engines": []})()

        async def collect(self) -> SystemSnapshot:
            return SystemSnapshot(node="local")

        async def collect_fleet(self) -> list[SystemSnapshot]:
            return [SystemSnapshot(node="local"), SystemSnapshot(node="peer")]

        async def aclose(self) -> None:
            return None

    server = SnapshotServer(
        FakeCollector(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=0,
        token="lock",
        auth_all=True,
    )
    srv = await asyncio.start_server(server._handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    server._server = srv
    base = f"http://127.0.0.1:{port}"

    async with srv, httpx.AsyncClient() as client:
        assert (await client.get(f"{base}/api/snapshot")).status_code == 401
        ok = await client.get(f"{base}/api/snapshot?token=lock")
        assert ok.status_code == 200
        fleet = await client.get(f"{base}/api/fleet?token=lock")
        assert fleet.status_code == 200
        assert len(fleet.json()) == 2
        # /ui stays reachable so the page can inject ?token= into EventSource.
        assert (await client.get(f"{base}/ui")).status_code == 200

    await srv.wait_closed()


def test_empty_inference_stats_helpers():
    assert inference_stats_from_ollama({}) == InferenceStats()
    assert inference_stats_from_lmstudio({}) == InferenceStats()
