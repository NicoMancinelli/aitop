"""Prometheus exporter, sparkline, load(), and config init."""

from __future__ import annotations

import httpx
import pytest
import respx

from aitop.engines.lmstudio import LMStudioEngine
from aitop.engines.ollama import OllamaEngine
from aitop.models import (
    CPUSnapshot,
    EngineKind,
    EngineSnapshot,
    EngineState,
    GPUSnapshot,
    HardwareSnapshot,
    InferenceStats,
    MemorySnapshot,
    SystemSnapshot,
    TailscaleStatus,
    Vendor,
)
from aitop.prometheus import render_prometheus
from aitop.utils.fmt import sparkline

OLLAMA = "http://127.0.0.1:11434"


def _snapshot() -> SystemSnapshot:
    return SystemSnapshot(
        node="nM1",
        duration_ms=42.5,
        hardware=HardwareSnapshot(
            cpu=CPUSnapshot(model="Apple M1", load_percent=11.0, power_watts=4.2),
            memory=MemorySnapshot(
                total_bytes=16 * 1024**3,
                used_bytes=8 * 1024**3,
                available_bytes=8 * 1024**3,
                unified=True,
            ),
            gpus=[
                GPUSnapshot(
                    index=0,
                    name="Apple M1",
                    vendor=Vendor.APPLE,
                    utilization_percent=57.0,
                    vram_total_bytes=10 * 1024**3,
                    vram_used_bytes=6 * 1024**3,
                    power_watts=3.1,
                )
            ],
            total_power_watts=12.0,
        ),
        engines=[
            EngineSnapshot(
                kind=EngineKind.OLLAMA,
                name="Ollama",
                state=EngineState.ONLINE,
                latency_ms=3.2,
                models=[],
                loaded=[],
                stats=InferenceStats(tokens_per_second=40.0, active_requests=1, queue_depth=0),
            ),
            EngineSnapshot(
                kind=EngineKind.VLLM,
                name="vLLM",
                state=EngineState.OFFLINE,
            ),
        ],
        tailscale=TailscaleStatus(available=True, running=True, peer_count=4),
    )


def test_prometheus_renders_core_gauges():
    text = render_prometheus(_snapshot())
    assert "aitop_info{version=" in text
    assert 'aitop_cpu_load_percent{node="nM1"} 11.0' in text
    assert 'aitop_memory_used_percent{node="nM1"}' in text
    assert "aitop_gpu_utilization_percent" in text
    assert 'aitop_engine_up{node="nM1",engine="ollama",name="Ollama"} 1' in text
    assert 'aitop_engine_up{node="nM1",engine="vllm",name="vLLM"} 0' in text
    assert "aitop_engine_tokens_per_second" in text
    assert 'aitop_tailscale_peers{node="nM1"} 4' in text
    assert text.endswith("\n")


def test_prometheus_escapes_label_quotes():
    snap = SystemSnapshot(node='a"b')
    text = render_prometheus(snap)
    assert 'node="a\\"b"' in text


def test_sparkline_scales_and_pads():
    assert len(sparkline([], width=8)) == 8
    assert len(sparkline([0, 50, 100], width=8)) == 8
    assert sparkline([100], width=1) == "█"
    assert sparkline([0], width=1) == " "


@respx.mock
async def test_ollama_load(ollama_endpoint):
    respx.post(f"{OLLAMA}/api/generate").mock(httpx.Response(200, json={}))
    engine = OllamaEngine(ollama_endpoint)
    ok, message = await engine.load("llama3.2:3b")
    await engine.aclose()
    assert ok
    assert "loaded llama3.2:3b" in message
    assert engine.supports("load")


@respx.mock
async def test_ollama_load_failure(ollama_endpoint):
    respx.post(f"{OLLAMA}/api/generate").mock(httpx.Response(500, text="boom"))
    engine = OllamaEngine(ollama_endpoint)
    ok, message = await engine.load("nope")
    await engine.aclose()
    assert not ok
    assert "load failed" in message


async def test_lmstudio_load_needs_cli(lmstudio_endpoint, monkeypatch):
    monkeypatch.setattr("aitop.engines.lmstudio.which", lambda _: None)
    engine = LMStudioEngine(lmstudio_endpoint)
    ok, message = await engine.load("qwen")
    await engine.aclose()
    assert not ok
    assert "lms" in message


def test_metrics_cli(fake_collector_module, capsys):
    from aitop import cli

    assert cli.main(["metrics", "--no-privileged"]) == 0
    out = capsys.readouterr().out
    assert "aitop_info" in out
    assert "aitop_engine_up" in out


def test_config_init(tmp_path, capsys):
    from aitop import cli

    path = tmp_path / "aitop" / "config.yaml"
    assert cli.main(["config", "init", "--path", str(path), "--no-color"]) == 0
    assert path.is_file()
    assert "fleet:" in path.read_text()
    assert cli.main(["config", "init", "--path", str(path), "--no-color"]) == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "already exists" in out
    assert cli.main(["config", "init", "--path", str(path), "--force", "--no-color"]) == 0
    assert "wrote" in capsys.readouterr().out


def test_load_cli(monkeypatch, capsys):
    from aitop import cli

    class Eng:
        name = "Ollama"
        kind = EngineKind.OLLAMA

        def supports(self, cap):
            return cap == "load"

        async def load(self, model):
            return True, f"loaded {model}"

    class Reg:
        async def aclose(self):
            return None

    async def resolve(config, kind, host=None):
        return Eng(), Reg(), None

    monkeypatch.setattr(cli, "_resolve_engine", resolve)
    assert cli.main(["load", "ollama", "llama3.2"]) == 0
    assert "loaded llama3.2" in capsys.readouterr().out


@pytest.fixture
def fake_collector_module(monkeypatch):
    """Stand in for SnapshotCollector used by `aitop metrics`."""
    from aitop import cli

    class Fake:
        def __init__(self, *a, **k):
            pass

        async def collect(self):
            return _snapshot()

        async def aclose(self):
            return None

    monkeypatch.setattr(cli, "SnapshotCollector", Fake)
    return Fake
