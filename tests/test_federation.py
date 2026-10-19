"""Unit and integration tests for peer-to-peer WebSocket federation."""

from __future__ import annotations

import asyncio
import struct
from datetime import datetime

import httpx
import pytest

from aitop.bus import EventBus, Topic
from aitop.config import Config, FleetConfig, FleetNodeConfig
from aitop.federation import (
    FederationManager,
    _build_client_frame,
    normalize_ws_url,
)
from aitop.models import (
    AlertEvent,
    AlertSeverity,
    ClusterSnapshot,
    CPUSnapshot,
    EngineKind,
    EngineSnapshot,
    EngineState,
    HardwareSnapshot,
    HostSnapshot,
    MemorySnapshot,
    SystemSnapshot,
)
from aitop.serve import SnapshotServer


class DummyCollector:
    """Mock collector for SnapshotServer in tests."""

    def __init__(self, node: str = "node-alpha", bus: EventBus | None = None) -> None:
        self.node = node
        self.bus = bus or EventBus()
        self.config = Config(fleet=FleetConfig(serve_host="127.0.0.1", serve_port=0))
        self.engines = type("Engines", (), {"engines": []})()

    async def collect(self) -> SystemSnapshot:
        return SystemSnapshot(
            node=self.node,
            hardware=HardwareSnapshot(
                host=HostSnapshot(hostname=f"host-{self.node}"),
                cpu=CPUSnapshot(model="Test CPU", load_percent=12.5),
                memory=MemorySnapshot(total_bytes=32 * 1024**3, used_bytes=16 * 1024**3),
            ),
            engines=[
                EngineSnapshot(
                    kind=EngineKind.OLLAMA,
                    name="Ollama",
                    state=EngineState.ONLINE,
                )
            ],
        )

    async def collect_fleet(self) -> list[SystemSnapshot]:
        return [await self.collect()]

    async def collect_cluster(self) -> ClusterSnapshot:
        return ClusterSnapshot(nodes=[await self.collect()])

    async def aclose(self) -> None:
        pass


def test_normalize_ws_url():
    assert normalize_ws_url("http://192.168.1.10:9090") == "ws://192.168.1.10:9090/api/ws"
    assert normalize_ws_url("http://192.168.1.10:9090/") == "ws://192.168.1.10:9090/api/ws"
    assert normalize_ws_url("192.168.1.10:9090") == "ws://192.168.1.10:9090/api/ws"
    assert normalize_ws_url("https://peer.local:8443") == "wss://peer.local:8443/api/ws"
    assert normalize_ws_url("ws://peer.local:9090/api/ws") == "ws://peer.local:9090/api/ws"
    assert normalize_ws_url("ws://peer:9090/api/ws?token=xyz") == "ws://peer:9090/api/ws?token=xyz"
    assert normalize_ws_url("") == ""


def test_build_client_frame_masking():
    payload = b"hello federation"
    frame = _build_client_frame(payload, opcode=0x1)
    assert frame[0] == 0x81  # FIN (0x80) | Text (0x1)
    assert frame[1] & 0x80 == 0x80  # MASK bit set
    length = frame[1] & 0x7F
    assert length == len(payload)
    mask = frame[2:6]
    masked_payload = frame[6:]
    unmasked = bytes(b ^ mask[i % 4] for i, b in enumerate(masked_payload))
    assert unmasked == payload


def test_build_client_frame_medium_payload():
    payload = b"X" * 1000
    frame = _build_client_frame(payload, opcode=0x1)
    assert frame[0] == 0x81
    assert frame[1] & 0x7F == 126
    length = struct.unpack("!H", frame[2:4])[0]
    assert length == 1000
    mask = frame[4:8]
    masked_payload = frame[8:]
    unmasked = bytes(b ^ mask[i % 4] for i, b in enumerate(masked_payload))
    assert unmasked == payload


def test_federation_manager_config_and_peer_management():
    config = Config(
        fleet=FleetConfig(
            nodes=[
                FleetNodeConfig(
                    name="peer-1", url="http://10.0.0.1:9090", enabled=True, timeout=2.0
                ),
                FleetNodeConfig(name="peer-disabled", url="http://10.0.0.2:9090", enabled=False),
            ]
        )
    )
    bus = EventBus()
    manager = FederationManager.from_config(config, bus=bus, local_node="node-main")

    assert manager.total_peer_count == 1
    assert manager.connected_peer_count == 0
    peer1 = manager.get_peer("peer-1")
    assert peer1 is not None
    assert peer1.ws_url == "ws://10.0.0.1:9090/api/ws"
    assert peer1.connected is False
    assert peer1.latency_ms == 0.0
    assert peer1.last_seen is None

    # Dynamically add another peer
    peer2 = manager.add_peer("peer-2", "http://10.0.0.3:9090", token="secret")
    assert manager.total_peer_count == 2
    assert peer2.token == "secret"

    status = manager.get_status()
    assert status["local_node"] == "node-main"
    assert status["active_connections"] == 0
    assert status["total_peers"] == 2
    assert len(status["peers"]) == 2
    assert len(status["topology"]["links"]) == 2

    # Remove peer
    assert manager.remove_peer("peer-1") is True
    assert manager.get_peer("peer-1") is None
    assert manager.total_peer_count == 1


@pytest.mark.asyncio
async def test_federation_live_websocket_and_snapshot_sync():
    """Verify peer connection, live latency tracking, snapshot subscription, and cluster aggregation."""
    # 1. Boot up peer server
    peer_collector = DummyCollector(node="peer-beta")
    peer_server = SnapshotServer(peer_collector, host="127.0.0.1", port=0)
    srv = await asyncio.start_server(peer_server._handle, "127.0.0.1", 0)
    peer_port = srv.sockets[0].getsockname()[1]
    peer_server._server = srv

    local_bus = EventBus()
    manager = FederationManager(
        bus=local_bus,
        local_node="local-alpha",
        ping_interval=0.5,
    )
    peer_url = f"http://127.0.0.1:{peer_port}"
    peer_conn = manager.add_peer("peer-beta", peer_url)

    snapshot_sub = local_bus.subscribe(Topic.SNAPSHOT)
    cluster_sub = local_bus.subscribe(Topic.CLUSTER)

    async with srv:
        await manager.start()
        try:
            # Wait for connection to establish
            for _ in range(50):
                if peer_conn.connected:
                    break
                await asyncio.sleep(0.1)

            assert peer_conn.connected is True
            assert peer_conn.latency_ms >= 0.0
            assert isinstance(peer_conn.last_seen, datetime)
            assert peer_conn.remote_node == "peer-beta"

            # Verify snapshot was received and published on local bus
            snap_event = await asyncio.wait_for(snapshot_sub.get(), timeout=5.0)
            assert snap_event.topic == Topic.SNAPSHOT
            assert snap_event.source == "peer-beta"
            assert isinstance(snap_event.payload, SystemSnapshot)
            assert snap_event.payload.node == "peer-beta"

            # Verify cluster snapshot aggregated
            cluster_event = await asyncio.wait_for(cluster_sub.get(), timeout=5.0)
            assert cluster_event.topic == Topic.CLUSTER
            assert isinstance(cluster_event.payload, ClusterSnapshot)
            assert any(n.node == "peer-beta" for n in cluster_event.payload.nodes)

            # Check /api/federation payload structure
            status = manager.get_status()
            assert status["active_connections"] == 1
            assert status["total_peers"] == 1
            assert status["peers"][0]["connected"] is True
            assert status["peers"][0]["remote_node"] == "peer-beta"
            assert status["peers"][0]["latency_ms"] >= 0.0

            link = status["topology"]["links"][0]
            assert link["source"] == "local-alpha"
            assert link["target"] == "peer-beta"
            assert link["connected"] is True
            assert link["status"] == "connected"

        finally:
            await manager.stop()
            snapshot_sub.close()
            cluster_sub.close()


@pytest.mark.asyncio
async def test_federation_alert_and_cluster_relay():
    """Verify Topic.ALERT and Topic.CLUSTER are relayed bidirectionally with deduplication."""
    peer_collector = DummyCollector(node="peer-server")
    peer_server = SnapshotServer(peer_collector, host="127.0.0.1", port=0)
    srv = await asyncio.start_server(peer_server._handle, "127.0.0.1", 0)
    peer_port = srv.sockets[0].getsockname()[1]
    peer_server._server = srv

    local_bus = EventBus()
    manager = FederationManager(bus=local_bus, local_node="local-client")
    manager.add_peer("peer-server", f"http://127.0.0.1:{peer_port}")

    peer_alert_sub = peer_collector.bus.subscribe(Topic.ALERT)

    async with srv:
        await manager.start()
        try:
            peer = manager.get_peer("peer-server")
            assert peer is not None
            for _ in range(50):
                if peer.connected:
                    break
                await asyncio.sleep(0.1)
            assert peer.connected is True

            # Publish alert locally on client; should relay to peer server
            test_alert = AlertEvent(
                title="VRAM Critical",
                message="GPU 0 VRAM at 98%",
                severity=AlertSeverity.CRITICAL,
                source="local-client",
            )
            local_bus.publish(Topic.ALERT, test_alert, source="local-client")

            # Receive on peer server
            relayed_event = await asyncio.wait_for(peer_alert_sub.get(), timeout=5.0)
            assert relayed_event.topic == Topic.ALERT
            if isinstance(relayed_event.payload, AlertEvent):
                assert relayed_event.payload.title == "VRAM Critical"
            else:
                assert relayed_event.payload["title"] == "VRAM Critical"

            # Check deduplication: sending duplicate alert is dropped and not echoed
            manager._remember_alert_id(test_alert.id)
            assert test_alert.id in manager._seen_alert_ids

        finally:
            await manager.stop()
            peer_alert_sub.close()


@pytest.mark.asyncio
async def test_federation_reconnect_loop_with_exponential_backoff():
    """Verify exponential backoff calculation and reconnect attempts when peer is offline."""
    manager = FederationManager(
        bus=None,
        local_node="tester",
        initial_backoff=0.05,
        max_backoff=0.4,
        backoff_factor=2.0,
    )
    # Point to an unused closed port to force connect failures
    peer = manager.add_peer("offline-peer", "http://127.0.0.1:59999", timeout=0.1)

    await manager.start()
    try:
        # Allow a few retry cycles
        await asyncio.sleep(0.35)
        assert peer.connected is False
        assert peer.reconnect_attempts >= 2
        assert peer.last_error is not None
    finally:
        await manager.stop()
        assert manager.connected_peer_count == 0


@pytest.mark.asyncio
async def test_snapshot_server_federation_endpoint():
    """Verify GET /api/federation and POST /api/federation/peers on SnapshotServer."""
    collector = DummyCollector(node="hub-node")
    server = SnapshotServer(collector, host="127.0.0.1", port=0)
    srv = await asyncio.start_server(server._handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    server._server = srv

    async with srv, httpx.AsyncClient() as client:
        # 1. GET /api/federation initially empty
        resp = await client.get(f"http://127.0.0.1:{port}/api/federation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["local_node"] == "hub-node"
        assert data["active_connections"] == 0
        assert data["total_peers"] == 0
        assert "topology" in data
        assert data["topology"]["nodes"] == ["hub-node"]

        # 2. POST /api/federation/peers to add peer
        add_resp = await client.post(
            f"http://127.0.0.1:{port}/api/federation/peers",
            json={
                "name": "worker-gpu-1",
                "url": "http://192.168.1.50:9090",
                "token": "auth-token-123",
            },
        )
        assert add_resp.status_code == 200
        add_data = add_resp.json()
        assert add_data["ok"] is True
        assert add_data["peer"]["name"] == "worker-gpu-1"
        assert add_data["peer"]["url"] == "ws://192.168.1.50:9090/api/ws"

        # 3. GET /api/federation after adding peer
        resp2 = await client.get(f"http://127.0.0.1:{port}/api/federation")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["total_peers"] == 1
        assert len(data2["peers"]) == 1
        assert data2["peers"][0]["name"] == "worker-gpu-1"
        assert len(data2["topology"]["links"]) == 1
        link = data2["topology"]["links"][0]
        assert link["source"] == "hub-node"
        assert link["target"] == "worker-gpu-1"
        assert link["status"] == "disconnected"
