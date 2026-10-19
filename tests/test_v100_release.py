"""Comprehensive test suite for aitop 1.0.0 features:
- Cluster aggregation & snapshots
- Alerting engine (VRAM, temp, engine transitions, webhooks)
- Background service management (launchd & systemd generator and status)
- Tailscale fleet auto-discovery
- Prometheus cluster metric exposition
- Serve /api/cluster endpoint & CLI commands
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx
import respx

from aitop import cli
from aitop.alerts import AlertManager
from aitop.bus import EventBus, Topic
from aitop.config import AlertConfig, Config
from aitop.discovery import probe_fleet_node
from aitop.models import (
    AlertSeverity,
    ClusterSnapshot,
    EngineKind,
    EngineSnapshot,
    EngineState,
    GPUSnapshot,
    HardwareSnapshot,
    HostSnapshot,
    InferenceStats,
    LoadedModel,
    SystemSnapshot,
    Vendor,
)
from aitop.prometheus import render_cluster_prometheus
from aitop.serve import SnapshotServer
from aitop.service import (
    generate_launchd_plist,
    generate_systemd_unit,
    resolve_aitop_exec,
    status_service,
)

# --------------------------------------------------------------------------- #
# ClusterSnapshot Aggregation
# --------------------------------------------------------------------------- #


def test_cluster_snapshot_aggregation():
    node1 = SystemSnapshot(
        node="node-a",
        hardware=HardwareSnapshot(
            host=HostSnapshot(hostname="host-a"),
            gpus=[
                GPUSnapshot(
                    vendor=Vendor.NVIDIA,
                    name="RTX 4090",
                    index=0,
                    vram_total_bytes=24 * 1024**3,
                    vram_used_bytes=16 * 1024**3,
                )
            ],
        ),
        engines=[
            EngineSnapshot(
                kind=EngineKind.OLLAMA,
                name="Ollama",
                state=EngineState.ONLINE,
                loaded=[
                    LoadedModel(
                        id="llama3", name="llama3", engine=EngineKind.OLLAMA, size_bytes=4 * 1024**3
                    )
                ],
                stats=InferenceStats(tokens_per_second=45.0),
            )
        ],
    )
    node2 = SystemSnapshot(
        node="node-b",
        hardware=HardwareSnapshot(
            host=HostSnapshot(hostname="host-b"),
            gpus=[
                GPUSnapshot(
                    vendor=Vendor.APPLE,
                    name="Apple M1",
                    index=0,
                    vram_total_bytes=16 * 1024**3,
                    vram_used_bytes=8 * 1024**3,
                )
            ],
        ),
        engines=[
            EngineSnapshot(
                kind=EngineKind.MLX,
                name="MLX",
                state=EngineState.ONLINE,
                loaded=[
                    LoadedModel(
                        id="qwen", name="qwen", engine=EngineKind.MLX, size_bytes=7 * 1024**3
                    )
                ],
                stats=InferenceStats(tokens_per_second=30.0),
            )
        ],
    )

    cluster = ClusterSnapshot(nodes=[node1, node2])
    assert cluster.total_nodes == 2
    assert cluster.online_nodes == 2
    assert cluster.total_gpus == 2
    assert cluster.total_vram_bytes == 40 * 1024**3
    assert cluster.used_vram_bytes == 24 * 1024**3
    assert cluster.online_engines == 2
    assert cluster.total_loaded_models == 2
    assert cluster.cluster_tokens_per_second == 75.0


# --------------------------------------------------------------------------- #
# Alerting Engine
# --------------------------------------------------------------------------- #


@respx.mock
async def test_alert_manager_evaluation():
    webhook_url = "https://hooks.example.com/alert"
    respx.post(webhook_url).mock(httpx.Response(200))

    bus = EventBus()
    sub = bus.subscribe(Topic.ALERT)

    cfg = Config(
        alerts=AlertConfig(
            enabled=True,
            vram_threshold_pct=80.0,
            temp_threshold_celsius=80.0,
            engine_offline=True,
            webhook_url=webhook_url,
        )
    )

    manager = AlertManager(cfg, bus=bus)

    # 1. Healthy snapshot -> no alerts
    healthy_snap = SystemSnapshot(
        node="local",
        hardware=HardwareSnapshot(
            gpus=[
                GPUSnapshot(
                    vendor=Vendor.NVIDIA,
                    name="GPU0",
                    index=0,
                    vram_total_bytes=1000,
                    vram_used_bytes=500,  # 50%
                    temperature_c=65.0,
                )
            ]
        ),
        engines=[EngineSnapshot(kind=EngineKind.OLLAMA, name="Ollama", state=EngineState.ONLINE)],
    )
    alerts = await manager.evaluate(healthy_snap)
    assert len(alerts) == 0

    # 2. Critical VRAM and High Temp snapshot
    unhealthy_snap = SystemSnapshot(
        node="local",
        hardware=HardwareSnapshot(
            gpus=[
                GPUSnapshot(
                    vendor=Vendor.NVIDIA,
                    name="GPU0",
                    index=0,
                    vram_total_bytes=1000,
                    vram_used_bytes=960,  # 96% -> Critical
                    temperature_c=86.0,  # > 80C -> Warning
                )
            ]
        ),
        engines=[EngineSnapshot(kind=EngineKind.OLLAMA, name="Ollama", state=EngineState.OFFLINE)],
    )

    alerts = await manager.evaluate(unhealthy_snap)
    assert len(alerts) == 3  # VRAM, temp, engine down
    assert any(a.severity == AlertSeverity.CRITICAL and a.metric == "vram_pct" for a in alerts)
    assert any(a.metric == "temperature_celsius" for a in alerts)
    assert any(a.metric == "engine_state" for a in alerts)

    # EventBus received alerts
    event = await sub.get()
    assert event.topic == Topic.ALERT

    # 3. Cooldown test: immediately re-evaluating should not fire duplicate alerts
    dupe_alerts = await manager.evaluate(unhealthy_snap)
    assert len(dupe_alerts) == 0

    await manager.aclose()


# --------------------------------------------------------------------------- #
# Background Service Management
# --------------------------------------------------------------------------- #


def test_service_generators():
    exec_args = resolve_aitop_exec()
    assert len(exec_args) >= 1

    plist = generate_launchd_plist(host="127.0.0.1", port=9090, exec_args=exec_args)
    assert "com.aitop.serve" in plist
    assert "127.0.0.1" in plist
    assert "9090" in plist

    unit = generate_systemd_unit(host="127.0.0.1", port=9090, exec_args=exec_args)
    assert "[Unit]" in unit
    assert "[Service]" in unit
    assert "serve" in unit
    assert "WantedBy=default.target" in unit


def test_status_service():
    status = status_service()
    assert status.platform in ("darwin", "linux") or status.platform.startswith("linux")
    assert isinstance(status.installed, bool)
    assert isinstance(status.active, bool)


# --------------------------------------------------------------------------- #
# Tailscale Fleet Auto-Discovery
# --------------------------------------------------------------------------- #


@respx.mock
async def test_probe_fleet_node():
    respx.get("http://100.100.1.5:9090/healthz").mock(
        httpx.Response(200, json={"ok": True, "version": "1.0.0"})
    )
    respx.get("http://100.100.1.6:9090/healthz").mock(httpx.Response(404))

    node_ok = await probe_fleet_node("100.100.1.5", 9090, "peer1")
    assert node_ok is not None
    assert node_ok.name == "peer1"
    assert node_ok.url == "http://100.100.1.5:9090"

    node_bad = await probe_fleet_node("100.100.1.6", 9090, "peer2")
    assert node_bad is None


# --------------------------------------------------------------------------- #
# Prometheus Cluster Metrics
# --------------------------------------------------------------------------- #


def test_prometheus_cluster_exposition():
    snap = SystemSnapshot(node="node1")
    cluster = ClusterSnapshot(nodes=[snap])
    prom = render_cluster_prometheus(cluster)
    assert "aitop_cluster_nodes_total 1" in prom
    assert "aitop_cluster_nodes_online 1" in prom
    assert "aitop_cluster_tokens_per_second" in prom
    assert "aitop_info" in prom


# --------------------------------------------------------------------------- #
# Serve /api/cluster & CLI
# --------------------------------------------------------------------------- #


class FakeCollector:
    def __init__(self, config=None, **kwargs):
        self.config = config or Config()
        self.node = "local"
        self.engines = type("R", (), {"engines": []})()

    async def collect(self) -> SystemSnapshot:
        return SystemSnapshot(node="local")

    async def collect_fleet(self) -> list[SystemSnapshot]:
        return [await self.collect()]

    async def collect_cluster(self) -> ClusterSnapshot:
        return ClusterSnapshot(nodes=[await self.collect()])

    async def aclose(self) -> None:
        return None


async def _boot(server: SnapshotServer):
    srv = await asyncio.start_server(server._handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    server._server = srv
    return srv, port


async def test_serve_cluster_endpoint():
    server = SnapshotServer(FakeCollector(), host="127.0.0.1", port=0)  # type: ignore[arg-type]
    srv, port = await _boot(server)
    base = f"http://127.0.0.1:{port}"

    async with srv, httpx.AsyncClient() as client:
        resp = await client.get(f"{base}/api/cluster")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert len(data["nodes"]) == 1


def test_cli_cluster(capsys):
    with patch("aitop.cli.SnapshotCollector", FakeCollector):
        rc = cli.main(["cluster", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "nodes" in data


def test_cli_service_status(capsys):
    rc = cli.main(["service", "status"])
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "Platform:" in out
    assert "Status:" in out
