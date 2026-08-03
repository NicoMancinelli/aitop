"""CLI coverage for lifecycle, pull, models, serve, tui, fleet subcommands."""

from __future__ import annotations

import json

import pytest

from aitop import cli
from aitop.models import DownloadProgress, EngineKind, EngineSnapshot, EngineState, SystemSnapshot


class FakeCollector:
    instances: list[FakeCollector] = []

    def __init__(self, config=None, bus=None, **kwargs):
        self.config = config
        self.bus = bus
        self.kwargs = kwargs
        self.closed = False
        self.engines = FakeRegistry()
        FakeCollector.instances.append(self)

    async def collect(self) -> SystemSnapshot:
        return SystemSnapshot(
            engines=[
                EngineSnapshot(
                    kind=EngineKind.OLLAMA,
                    name="Ollama",
                    state=EngineState.ONLINE,
                    version="0.5.7",
                )
            ]
        )

    async def collect_fleet(self):
        return [await self.collect()]

    async def aclose(self) -> None:
        self.closed = True

    async def stream(self, interval=None):
        import asyncio

        self.bus.publish("snapshot", await self.collect())  # type: ignore[arg-type]
        await asyncio.Event().wait()


class FakeRegistry:
    def __init__(self):
        self.engines = []

    async def aclose(self):
        return None


class FakeEngine:
    kind = EngineKind.OLLAMA
    name = "Ollama"
    display_name = "Ollama"
    host = "127.0.0.1"
    port = 11434

    def __init__(self):
        self.calls: list[tuple] = []

    def supports(self, cap: str) -> bool:
        return cap in {"unload", "pull", "rebind", "lifecycle"}

    async def start(self):
        self.calls.append(("start",))
        return True, "started"

    async def stop(self):
        self.calls.append(("stop",))
        return True, "stopped"

    async def restart(self):
        self.calls.append(("restart",))
        return True, "restarted"

    async def unload(self, model=None):
        self.calls.append(("unload", model))
        return True, f"unloaded {model or 'all'}"

    async def rebind(self, host):
        self.calls.append(("rebind", host))
        return True, f"rebound to {host}"

    async def pull(self, model, *, on_progress=None):
        self.calls.append(("pull", model))
        if on_progress:
            on_progress(
                DownloadProgress(
                    model=model,
                    engine=EngineKind.OLLAMA,
                    status="success",
                    completed_bytes=10,
                    total_bytes=10,
                    done=True,
                )
            )
        return True, f"pulled {model}"


@pytest.fixture(autouse=True)
def fake_collector(monkeypatch):
    FakeCollector.instances.clear()
    monkeypatch.setattr(cli, "SnapshotCollector", FakeCollector)
    return FakeCollector


@pytest.fixture
def fake_engine(monkeypatch):
    engine = FakeEngine()

    async def resolve(config, kind_or_name, host=None):
        class Reg:
            async def aclose(self):
                return None

        return engine, Reg(), None

    monkeypatch.setattr(cli, "_resolve_engine", resolve)
    return engine


def test_parser_knows_new_subcommands():
    parser = cli.build_parser()
    for cmd in (
        "tui",
        "serve",
        "fleet",
        "start",
        "stop",
        "restart",
        "unload",
        "rebind",
        "pull",
        "models",
    ):
        args = parser.parse_args(
            [cmd]
            + (["ollama"] if cmd in {"start", "stop", "restart", "unload", "rebind"} else [])
            + (["llama"] if cmd == "pull" else [])
            + (["x"] if cmd == "rebind" else [])
            + (["search", "qwen"] if cmd == "models" else [])
        )
        assert args.command == cmd


def test_start_stop_restart(fake_engine, capsys):
    assert cli.main(["start", "ollama"]) == 0
    assert cli.main(["stop", "ollama"]) == 0
    assert cli.main(["restart", "ollama"]) == 0
    assert [c[0] for c in fake_engine.calls] == ["start", "stop", "restart"]
    out = capsys.readouterr().out
    assert "started" in out and "stopped" in out


def test_unload(fake_engine, capsys):
    assert cli.main(["unload", "ollama", "llama3.2"]) == 0
    assert fake_engine.calls[-1] == ("unload", "llama3.2")


def test_pull(fake_engine, capsys):
    assert cli.main(["pull", "llama3.2:3b"]) == 0
    assert fake_engine.calls[-1] == ("pull", "llama3.2:3b")
    assert "pulled" in capsys.readouterr().out


def test_rebind(fake_engine, capsys):
    assert cli.main(["rebind", "ollama", "0.0.0.0"]) == 0
    assert fake_engine.calls[-1] == ("rebind", "0.0.0.0")


def test_fleet_json(capsys):
    assert cli.main(["fleet", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert payload[0]["engines"][0]["kind"] == "ollama"


def test_models_search(monkeypatch, capsys):
    from aitop.models import HubModel

    async def fake_search(query, **kwargs):
        return [
            HubModel(
                id="org/model-GGUF",
                downloads=1000,
                likes=5,
                tags=["gguf"],
                url="https://huggingface.co/org/model-GGUF",
            )
        ]

    monkeypatch.setattr(cli, "search_hub", fake_search)
    assert cli.main(["models", "search", "qwen"]) == 0
    out = capsys.readouterr().out
    assert "org/model-GGUF" in out
    assert "1,000" in out


def test_unknown_engine_lifecycle(monkeypatch, capsys):
    async def resolve(config, kind_or_name, host=None):
        class Reg:
            async def aclose(self):
                return None

        return None, Reg(), "unknown engine 'nope'"

    monkeypatch.setattr(cli, "_resolve_engine", resolve)
    assert cli.main(["start", "nope"]) == 1
    assert "unknown" in capsys.readouterr().out
