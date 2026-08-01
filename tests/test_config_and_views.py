"""Config loading, the event bus, formatting, and the neofetch renderer."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from rich.console import Console

from aitop.bus import EventBus, Topic
from aitop.config import Config, EndpointConfig
from aitop.models import (
    Binding,
    BindScope,
    CPUSnapshot,
    EngineKind,
    EngineSnapshot,
    EngineState,
    GPUSnapshot,
    HardwareSnapshot,
    HostSnapshot,
    LoadedModel,
    MemorySnapshot,
    ModelInfo,
    SystemSnapshot,
    TailscaleStatus,
    Vendor,
)
from aitop.utils.fmt import bytes_human, duration_human, ratio_bar, relative_time
from aitop.utils.parse import parse_timestamp, split_host_port, to_float, to_int
from aitop.views.logo import LOGO, render_logo
from aitop.views.neofetch import render_neofetch

GB = 1024**3


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_defaults_need_no_file(tmp_path):
    config = Config.load(tmp_path / "nope.yaml")
    assert config.source is None
    kinds = {ep.kind for ep in config.all_endpoints()}
    assert EngineKind.OLLAMA in kinds
    assert EngineKind.LMSTUDIO in kinds


def test_config_file_adds_endpoints(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "polling:\n"
        "  hardware_interval: 5.0\n"
        "endpoints:\n"
        "  - kind: ollama\n"
        "    host: 100.100.1.7\n"
        "    name: pveclaw\n"
        "    remote: true\n"
    )
    config = Config.load(path)
    assert config.polling.hardware_interval == 5.0
    remote = [ep for ep in config.all_endpoints() if ep.remote]
    assert len(remote) == 1
    assert remote[0].host == "100.100.1.7"
    assert remote[0].resolved_port() == 11434  # default filled in


def test_disabled_endpoint_removes_the_default(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("endpoints:\n  - kind: lmstudio\n    enabled: false\n")
    config = Config.load(path)
    assert not any(ep.kind is EngineKind.LMSTUDIO for ep in config.all_endpoints())


def test_invalid_config_falls_back_to_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("polling: [this is not a mapping]\n")
    config = Config.load(path)
    assert config.polling.hardware_interval == 2.0


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11500")
    config = Config()
    ollama = next(ep for ep in config.default_endpoints() if ep.kind is EngineKind.OLLAMA)
    assert (ollama.host, ollama.resolved_port()) == ("0.0.0.0", 11500)


def test_env_var_without_port(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://100.100.1.7")
    config = Config()
    ollama = next(ep for ep in config.default_endpoints() if ep.kind is EngineKind.OLLAMA)
    assert (ollama.host, ollama.resolved_port()) == ("100.100.1.7", 11434)


def test_endpoint_label_and_port_defaults():
    ep = EndpointConfig(kind=EngineKind.VLLM)
    assert ep.resolved_port() == 8000
    assert ep.label() == "vllm"


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    [
        "2025-01-14T10:22:31.833753871-08:00",  # Go nanoseconds
        "2025-01-14T10:22:31.833753-08:00",
        "2025-01-14T10:22:31Z",
        "2025-01-14T10:22:31+00:00",
    ],
)
def test_parse_timestamp_accepts_engine_formats(raw):
    assert parse_timestamp(raw) is not None


@pytest.mark.parametrize("raw", [None, "", "not a date", 42, []])
def test_parse_timestamp_rejects_junk(raw):
    assert parse_timestamp(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("4096 MiB", 4096), ("81 %", 81), ("1,234", 1234), ("[N/A]", None), (True, None), (3.9, 3)],
)
def test_to_int(raw, expected):
    assert to_int(raw) == expected


def test_to_float_handles_na():
    assert to_float("[N/A]") is None
    assert to_float("145.32") == pytest.approx(145.32)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("127.0.0.1:11434", ("127.0.0.1", 11434)),
        ("http://host:1234/v1", ("host", 1234)),
        ("myhost", ("myhost", 9999)),
        ("[::1]:8080", ("::1", 8080)),
    ],
)
def test_split_host_port(raw, expected):
    assert split_host_port(raw, 9999) == expected


def test_formatting_helpers():
    assert bytes_human(1024**3) == "1.0 GB"
    assert bytes_human(None) == "—"
    assert bytes_human(512) == "512 B"
    assert duration_human(90061) == "1d 1h 1m"
    assert len(ratio_bar(0.5, width=10)) == 10
    assert ratio_bar(None, width=4) == "░░░░"
    assert ratio_bar(2.0, width=4) == "████"  # clamped
    assert relative_time(None) == "—"
    future = datetime.now(UTC) + timedelta(minutes=5)
    assert relative_time(future).startswith("in ")


# --------------------------------------------------------------------------- #
# Event bus
# --------------------------------------------------------------------------- #


async def test_bus_delivers_to_topic_subscribers():
    bus = EventBus()
    with bus.subscribe(Topic.SNAPSHOT) as sub:
        bus.publish(Topic.LOG, "ignored")
        bus.publish(Topic.SNAPSHOT, "payload")
        event = await asyncio.wait_for(sub.get(), timeout=1.0)
    assert event.payload == "payload"
    assert bus.subscriber_count == 0  # context manager unsubscribed


async def test_bus_subscriber_without_topics_gets_everything():
    bus = EventBus()
    with bus.subscribe() as sub:
        bus.publish(Topic.LOG, "a")
        assert (await asyncio.wait_for(sub.get(), timeout=1.0)).payload == "a"


async def test_slow_subscriber_drops_oldest_instead_of_blocking():
    bus = EventBus(maxsize=2)
    with bus.subscribe(Topic.SNAPSHOT) as sub:
        for i in range(5):
            bus.publish(Topic.SNAPSHOT, i)
        assert sub.dropped == 3
        # The newest events survive.
        assert (await sub.get()).payload == 3
        assert (await sub.get()).payload == 4


# --------------------------------------------------------------------------- #
# Neofetch rendering
# --------------------------------------------------------------------------- #


def _sample_snapshot() -> SystemSnapshot:
    return SystemSnapshot(
        hardware=HardwareSnapshot(
            host=HostSnapshot(
                hostname="nM1",
                os_name="macOS",
                os_version="15.6",
                kernel="Darwin 24.6.0",
                platform_id="darwin-arm64",
                uptime_seconds=270000,
            ),
            cpu=CPUSnapshot(
                model="Apple M1",
                arch="arm64",
                physical_cores=8,
                logical_cores=8,
                performance_cores=4,
                efficiency_cores=4,
                load_percent=11.5,
                per_core_percent=[10.0] * 8,
                power_watts=1.5,
            ),
            memory=MemorySnapshot(
                total_bytes=16 * GB,
                used_bytes=10 * GB,
                available_bytes=6 * GB,
                unified=True,
            ),
            gpus=[
                GPUSnapshot(
                    name="Apple M1",
                    vendor=Vendor.APPLE,
                    api_version="Metal 4",
                    core_count=8,
                    utilization_percent=9.0,
                    vram_total_bytes=int(10.6 * GB),
                    vram_used_bytes=4 * GB,
                    unified_memory=True,
                )
            ],
            total_power_watts=6.2,
            degraded=["power/thermal data needs root"],
        ),
        engines=[
            EngineSnapshot(
                kind=EngineKind.OLLAMA,
                name="Ollama",
                state=EngineState.ONLINE,
                binding=Binding(host="127.0.0.1", port=11434, scope=BindScope.LOOPBACK),
                version="0.5.7",
                pid=4242,
                managed_by="launchd",
                models=[ModelInfo(id="llama3.2:3b", name="llama3.2:3b", engine=EngineKind.OLLAMA)],
                loaded=[
                    LoadedModel(
                        id="llama3.2:3b",
                        name="llama3.2:3b",
                        engine=EngineKind.OLLAMA,
                        parameter_size="3.2B",
                        quantization="Q4_K_M",
                        size_bytes=4 * GB,
                        vram_bytes=4 * GB,
                        context_length=8192,
                    )
                ],
            ),
            EngineSnapshot(
                kind=EngineKind.LMSTUDIO,
                name="LM Studio",
                state=EngineState.OFFLINE,
                binding=Binding(host="127.0.0.1", port=1234, scope=BindScope.LOOPBACK),
            ),
        ],
        tailscale=TailscaleStatus(
            available=True, running=True, hostname="nM1", ipv4="100.100.1.2", peer_count=12
        ),
    )


def _render_to_text(width: int = 140, **kwargs) -> str:
    console = Console(width=width, no_color=True, force_terminal=False)
    with console.capture() as capture:
        console.print(render_neofetch(_sample_snapshot(), **kwargs))
    return capture.get()


def test_neofetch_renders_the_key_facts():
    output = _render_to_text()
    for expected in (
        "nM1",
        "macOS 15.6",
        "Apple M1 (4P + 4E)",
        "unified",
        "Metal 4",
        "Tailscale 100.100.1.2",
        "Ollama",
        "0.5.7",
        "127.0.0.1:11434",
        "loopback",
        "llama3.2:3b",
        "Q4_K_M",
        "8,192",
        "launchd",
        "power/thermal data needs root",
    ):
        assert expected in output, f"missing {expected!r}"


def test_neofetch_hides_offline_engines_by_default():
    assert "LM Studio" not in _render_to_text()
    assert "LM Studio" in _render_to_text(show_offline=True)


def test_neofetch_without_logo_omits_the_art():
    assert LOGO[0].strip() not in _render_to_text(show_logo=False)


def test_neofetch_on_a_bare_snapshot():
    """A snapshot where every probe failed still renders something useful."""
    console = Console(width=100, no_color=True)
    with console.capture() as capture:
        console.print(render_neofetch(SystemSnapshot()))
    output = capture.get()
    assert "No AI runtimes reachable" in output


@pytest.mark.parametrize("width", [60, 72, 80, 92, 112, 140, 200])
def test_neofetch_never_overflows_its_width(width):
    """Whatever the terminal size, no rendered line may exceed it."""
    output = _render_to_text(width=width, show_offline=True)
    longest = max(len(line.rstrip()) for line in output.splitlines())
    assert longest <= width, f"line of {longest} cols at width {width}"


def test_narrow_terminals_drop_detail_columns_instead_of_ellipsizing():
    narrow = _render_to_text(width=80)
    wide = _render_to_text(width=140)

    # Detail columns are present only when there is room for them.
    assert "MANAGED" in wide and "PID" in wide and "EXPIRES" in wide
    assert "MANAGED" not in narrow and "EXPIRES" not in narrow

    # The columns that survive stay legible rather than being cut to "Q4_K…".
    assert "Q4_K_M" in narrow
    assert "llama3.2:3b" in narrow
    assert "…" not in narrow


def test_very_narrow_terminal_drops_the_logo():
    assert LOGO[0].strip() not in _render_to_text(width=60)


def test_logo_lines_are_padded_to_equal_width():
    lines = render_logo(color=False).plain.splitlines()
    assert len({len(line) for line in lines}) == 1


def test_snapshot_round_trips_through_json():
    snapshot = _sample_snapshot()
    restored = SystemSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored.hardware.host.hostname == "nM1"
    assert restored.online_engines[0].loaded[0].gpu_fraction == 1.0
