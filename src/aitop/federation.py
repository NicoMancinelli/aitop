"""Peer-to-peer WebSocket federation across aitop serve nodes.

Enables distributed cluster monitoring by connecting local aitop instances to
remote aitop serve WebSocket endpoints (/api/ws), synchronizing telemetry
snapshots, and relaying alerts and cluster-wide state across the fleet.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import hashlib
import json
import logging
import os
import ssl
import struct
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from aitop.bus import EventBus, Topic
from aitop.models import AlertEvent, ClusterSnapshot, SystemSnapshot

if TYPE_CHECKING:
    from aitop.config import Config, FleetNodeConfig
    from aitop.serve import FleetNode

log = logging.getLogger(__name__)

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def normalize_ws_url(url: str) -> str:
    """Ensure a URL is formatted as ws:// or wss:// pointing to /api/ws."""
    clean = url.strip()
    if not clean:
        return ""
    if "://" not in clean:
        clean = f"ws://{clean}"
    parsed = urlsplit(clean)
    scheme = parsed.scheme.lower()
    if scheme in ("http", "ws"):
        ws_scheme = "ws"
    elif scheme in ("https", "wss"):
        ws_scheme = "wss"
    else:
        ws_scheme = "ws"

    netloc = parsed.netloc
    path = parsed.path
    if not path or path == "/":
        path = "/api/ws"
    elif not path.endswith("/api/ws") and not path.endswith("/ws"):
        path = f"{path.rstrip('/')}/api/ws"

    query = f"?{parsed.query}" if parsed.query else ""
    return f"{ws_scheme}://{netloc}{path}{query}"


def _build_client_frame(payload: bytes, *, opcode: int = 0x1) -> bytes:
    """Build an RFC 6455 masked WebSocket frame from client to server."""
    fin_and_opcode = 0x80 | (opcode & 0x0F)
    length = len(payload)
    mask = os.urandom(4)
    if length < 126:
        header = bytes([fin_and_opcode, 0x80 | length]) + mask
    elif length < 65536:
        header = bytes([fin_and_opcode, 0x80 | 126]) + struct.pack("!H", length) + mask
    else:
        header = bytes([fin_and_opcode, 0x80 | 127]) + struct.pack("!Q", length) + mask

    masked_payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return header + masked_payload


class WebSocketClient:
    """Lightweight, non-blocking async WebSocket client."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        connect_timeout: float = 5.0,
    ) -> None:
        self.url = normalize_ws_url(url)
        self.token = token
        self.connect_timeout = connect_timeout
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._closed = False
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Establish TCP/TLS connection and complete WebSocket handshake."""
        parsed = urlsplit(self.url)
        host = parsed.hostname or "127.0.0.1"
        is_ssl = parsed.scheme == "wss"
        port = parsed.port or (443 if is_ssl else 9090)
        path = parsed.path or "/api/ws"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        ssl_ctx = ssl.create_default_context() if is_ssl else None
        self.reader, self.writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=ssl_ctx),
            timeout=self.connect_timeout,
        )

        key = base64.b64encode(os.urandom(16)).decode()
        expected_accept = base64.b64encode(
            hashlib.sha1((key + _WS_GUID).encode()).digest()
        ).decode()

        headers = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        if self.token:
            headers.append(f"Authorization: Bearer {self.token}")
        headers.append("\r\n")

        req = "\r\n".join(headers).encode()
        self.writer.write(req)
        await self.writer.drain()

        status_line = await asyncio.wait_for(self.reader.readline(), timeout=self.connect_timeout)
        if not status_line or b"101" not in status_line:
            status_text = status_line.decode(errors="replace").strip() if status_line else "empty"
            raise ConnectionError(f"WebSocket handshake rejected by peer: {status_text}")

        resp_headers: dict[str, str] = {}
        while True:
            line = await asyncio.wait_for(self.reader.readline(), timeout=self.connect_timeout)
            if line in (b"\r\n", b"\n", b""):
                break
            if b":" in line:
                k, _, v = line.partition(b":")
                resp_headers[k.strip().lower().decode("latin-1")] = v.strip().decode("latin-1")

        accept = resp_headers.get("sec-websocket-accept")
        if accept != expected_accept:
            raise ConnectionError(
                f"WebSocket handshake failed: invalid Sec-WebSocket-Accept ({accept})"
            )

    async def send_frame(self, payload: bytes, *, opcode: int = 0x1) -> None:
        """Send a masked WebSocket frame."""
        if self._closed or not self.writer:
            raise ConnectionResetError("WebSocket connection is closed")
        frame = _build_client_frame(payload, opcode=opcode)
        async with self._write_lock:
            self.writer.write(frame)
            await self.writer.drain()

    async def send_text(self, text: str) -> None:
        """Send a UTF-8 text message."""
        await self.send_frame(text.encode("utf-8"), opcode=0x1)

    async def send_ping(self, payload: bytes = b"") -> None:
        """Send a ping frame."""
        await self.send_frame(payload, opcode=0x9)

    async def recv_frame(self, timeout: float | None = None) -> tuple[int, bytes]:
        """Receive the next WebSocket frame with optional timeout."""
        if self._closed or not self.reader:
            raise ConnectionResetError("WebSocket connection is closed")
        coro = self._read_frame()
        if timeout is not None:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    async def _read_frame(self) -> tuple[int, bytes]:
        assert self.reader is not None
        first_two = await self.reader.readexactly(2)
        opcode = first_two[0] & 0x0F
        masked = bool(first_two[1] & 0x80)
        length = first_two[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", await self.reader.readexactly(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", await self.reader.readexactly(8))[0]
        mask = await self.reader.readexactly(4) if masked else b""
        payload = await self.reader.readexactly(length)
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, payload

    async def close(self) -> None:
        """Gracefully close the WebSocket connection."""
        if self._closed:
            return
        self._closed = True
        if self.writer:
            with contextlib.suppress(Exception):
                frame = _build_client_frame(b"", opcode=0x8)
                self.writer.write(frame)
                await asyncio.wait_for(self.writer.drain(), timeout=1.0)
            with contextlib.suppress(Exception):
                self.writer.close()
                await self.writer.wait_closed()
            self.writer = None
            self.reader = None


@dataclass
class PeerConnection:
    """Tracks live state and telemetry metrics for a federated peer node."""

    name: str
    url: str
    ws_url: str
    token: str | None = None
    timeout: float = 5.0
    connected: bool = False
    latency_ms: float = 0.0
    last_seen: datetime | None = None
    reconnect_attempts: int = 0
    last_error: str | None = None
    remote_node: str | None = None
    last_snapshot: SystemSnapshot | None = None
    ws: WebSocketClient | None = None
    task: asyncio.Task[None] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize peer status for API responses."""
        return {
            "name": self.name,
            "url": self.ws_url,
            "connected": self.connected,
            "latency_ms": round(self.latency_ms, 2),
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "reconnect_attempts": self.reconnect_attempts,
            "error": self.last_error,
            "remote_node": self.remote_node,
        }


class FederationManager:
    """Manages peer-to-peer WebSocket connections, reconnects, and bus relaying."""

    def __init__(
        self,
        peers: list[PeerConnection | FleetNode | FleetNodeConfig | str | dict[str, Any]]
        | None = None,
        bus: EventBus | None = None,
        local_node: str = "local",
        *,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_factor: float = 2.0,
        ping_interval: float = 5.0,
    ) -> None:
        self.bus = bus
        self.local_node = local_node
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_factor = backoff_factor
        self.ping_interval = ping_interval

        self._peers: dict[str, PeerConnection] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._stopped = asyncio.Event()
        self._running = False
        self._local_snapshot: SystemSnapshot | None = None

        # Deduplication for alert gossiping (bounded ring buffer)
        self._seen_alert_ids: set[str] = set()
        self._alert_id_order: collections.deque[str] = collections.deque(maxlen=1000)

        if peers:
            for item in peers:
                self.add_peer_any(item)

    @classmethod
    def from_config(
        cls,
        config: Config,
        bus: EventBus | None = None,
        local_node: str = "local",
    ) -> FederationManager:
        """Create a FederationManager from application configuration."""
        manager = cls(bus=bus, local_node=local_node)
        fleet_cfg = getattr(config, "fleet", None)
        if fleet_cfg and hasattr(fleet_cfg, "nodes"):
            for node_cfg in fleet_cfg.nodes:
                if getattr(node_cfg, "enabled", True):
                    manager.add_peer(
                        name=node_cfg.name,
                        url=node_cfg.url,
                        token=getattr(node_cfg, "token", None),
                        timeout=getattr(node_cfg, "timeout", 5.0),
                    )
        return manager

    def add_peer(
        self,
        name: str,
        url: str,
        token: str | None = None,
        timeout: float = 5.0,
    ) -> PeerConnection:
        """Register a peer node and initiate connection if manager is active."""
        ws_url = normalize_ws_url(url)
        peer = PeerConnection(
            name=name,
            url=url,
            ws_url=ws_url,
            token=token,
            timeout=timeout,
        )
        self._peers[name] = peer

        if self._running and not self._stopped.is_set():
            task = asyncio.create_task(self._peer_loop(peer), name=f"federation-peer-{name}")
            peer.task = task
            self._tasks.append(task)

        return peer

    def add_peer_any(self, item: Any) -> PeerConnection:
        """Register a peer from various input formats."""
        if isinstance(item, PeerConnection):
            self._peers[item.name] = item
            return item
        if isinstance(item, str):
            name = urlsplit(item).netloc or item
            return self.add_peer(name, item)
        if isinstance(item, dict):
            name = item.get("name") or "peer"
            url = item.get("url") or ""
            return self.add_peer(
                name,
                url,
                token=item.get("token"),
                timeout=float(item.get("timeout") or 5.0),
            )
        # Handle FleetNode or FleetNodeConfig object
        name = getattr(item, "name", "peer")
        url = getattr(item, "url", "")
        token = getattr(item, "token", None)
        timeout = float(getattr(item, "timeout", 5.0))
        return self.add_peer(name, url, token=token, timeout=timeout)

    def remove_peer(self, name: str) -> bool:
        """Unregister a peer and terminate its active connection task."""
        peer = self._peers.pop(name, None)
        if not peer:
            return False
        if peer.task and not peer.task.done():
            peer.task.cancel()
        if peer.ws:
            asyncio.create_task(peer.ws.close())
        return True

    def get_peer(self, name: str) -> PeerConnection | None:
        return self._peers.get(name)

    @property
    def peers(self) -> list[PeerConnection]:
        return list(self._peers.values())

    @property
    def connected_peer_count(self) -> int:
        return sum(1 for p in self._peers.values() if p.connected)

    @property
    def total_peer_count(self) -> int:
        return len(self._peers)

    async def start(self) -> None:
        """Start all peer connection loops and bus relay worker."""
        if self._running:
            return
        self._running = True
        self._stopped.clear()

        # Start peer connection tasks
        for peer in self._peers.values():
            task = asyncio.create_task(self._peer_loop(peer), name=f"federation-peer-{peer.name}")
            peer.task = task
            self._tasks.append(task)

        # Start event bus listener task
        if self.bus:
            bus_task = asyncio.create_task(self._listen_local_events(), name="federation-bus-relay")
            self._tasks.append(bus_task)

        log.info(
            "FederationManager started with %d peers configured for node '%s'",
            len(self._peers),
            self.local_node,
        )

    async def stop(self) -> None:
        """Gracefully stop all loops and close all peer connections."""
        if not self._running and self._stopped.is_set():
            return
        self._running = False
        self._stopped.set()

        # Cancel all background tasks
        tasks = [t for t in self._tasks if not t.done()]
        for t in tasks:
            t.cancel()

        # Close all active peer WebSockets
        close_futs = [peer.ws.close() for peer in self._peers.values() if peer.ws]
        if close_futs:
            await asyncio.gather(*close_futs, return_exceptions=True)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        for peer in self._peers.values():
            peer.connected = False
            peer.ws = None
            peer.task = None

        self._tasks.clear()
        log.info("FederationManager stopped cleanly")

    async def __aenter__(self) -> FederationManager:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def _remember_alert_id(self, alert_id: str) -> None:
        if alert_id in self._seen_alert_ids:
            return
        if len(self._alert_id_order) >= (self._alert_id_order.maxlen or 1000):
            oldest = self._alert_id_order.popleft()
            self._seen_alert_ids.discard(oldest)
        self._seen_alert_ids.add(alert_id)
        self._alert_id_order.append(alert_id)

    def _aggregate_cluster_snapshot(self) -> ClusterSnapshot | None:
        """Combine local snapshot and all active peer snapshots into ClusterSnapshot."""
        if not self.bus:
            return None
        snapshots: list[SystemSnapshot] = []
        if self._local_snapshot:
            snapshots.append(self._local_snapshot)
        for p in self._peers.values():
            if p.connected and p.last_snapshot:
                snapshots.append(p.last_snapshot)
        if snapshots:
            cluster = ClusterSnapshot(nodes=snapshots)
            self.bus.publish(Topic.CLUSTER, cluster, source=self.local_node)
            return cluster
        return None

    async def broadcast_event(
        self,
        topic: Topic | str,
        payload: Any,
        source: str = "local",
        exclude_peer: str | None = None,
    ) -> int:
        """Relay an event frame to all connected peer WebSockets."""
        topic_str = topic.value if isinstance(topic, Topic) else str(topic)
        msg = json.dumps({"topic": topic_str, "payload": payload, "source": source})
        sent_count = 0
        for name, peer in list(self._peers.items()):
            if name == exclude_peer or not peer.connected or not peer.ws:
                continue
            try:
                await peer.ws.send_text(msg)
                sent_count += 1
            except Exception as exc:
                log.debug("Failed to relay %s to peer %s: %s", topic_str, name, exc)
        return sent_count

    async def _listen_local_events(self) -> None:
        """Subscribe to local EventBus and relay ALERT and CLUSTER events across peers."""
        if not self.bus:
            return
        sub = self.bus.subscribe(Topic.SNAPSHOT, Topic.ALERT, Topic.CLUSTER)
        try:
            async for event in sub:
                if self._stopped.is_set():
                    break
                if event.topic == Topic.SNAPSHOT and event.source in (self.local_node, "local"):
                    if isinstance(event.payload, SystemSnapshot):
                        self._local_snapshot = event.payload
                        self._aggregate_cluster_snapshot()

                elif event.topic == Topic.ALERT:
                    alert_id = getattr(event.payload, "id", None)
                    if isinstance(event.payload, dict):
                        alert_id = event.payload.get("id")
                    if alert_id:
                        if alert_id in self._seen_alert_ids:
                            continue
                        self._remember_alert_id(alert_id)

                    payload_json = (
                        event.payload.model_dump(mode="json")
                        if hasattr(event.payload, "model_dump")
                        else event.payload
                    )
                    await self.broadcast_event(Topic.ALERT, payload_json, source=event.source)

                elif event.topic == Topic.CLUSTER:
                    payload_json = (
                        event.payload.model_dump(mode="json")
                        if hasattr(event.payload, "model_dump")
                        else event.payload
                    )
                    await self.broadcast_event(Topic.CLUSTER, payload_json, source=event.source)
        except asyncio.CancelledError:
            pass
        finally:
            sub.close()

    async def _peer_loop(self, peer: PeerConnection) -> None:
        """Persistent connection loop with exponential backoff on disconnect."""
        backoff = self.initial_backoff
        while not self._stopped.is_set():
            try:
                peer.reconnect_attempts += 1
                peer.last_error = None
                log.debug("Connecting to peer %s at %s...", peer.name, peer.ws_url)

                ws = WebSocketClient(
                    peer.ws_url,
                    token=peer.token,
                    connect_timeout=peer.timeout,
                )
                t0 = time.perf_counter()
                await ws.connect()
                rtt = (time.perf_counter() - t0) * 1000.0

                peer.ws = ws
                peer.connected = True
                peer.latency_ms = max(0.0, round(rtt, 2))
                peer.last_seen = datetime.now(UTC)
                peer.last_error = None
                backoff = self.initial_backoff  # Reset on successful connect

                log.info("Connected to peer %s (latency: %.1fms)", peer.name, peer.latency_ms)

                await self._run_peer_session(peer, ws)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                peer.connected = False
                peer.last_error = str(exc)
                log.debug("Peer %s connection error: %s", peer.name, exc)
            finally:
                peer.connected = False
                if peer.ws:
                    await peer.ws.close()
                    peer.ws = None

            if self._stopped.is_set():
                break

            # Exponential backoff sleep, interruptible by shutdown
            log.debug("Peer %s reconnect backoff: %.1fs", peer.name, backoff)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), timeout=backoff)
            backoff = min(backoff * self.backoff_factor, self.max_backoff)

    async def _run_peer_session(self, peer: PeerConnection, ws: WebSocketClient) -> None:
        """Run concurrently ping heartbeat loop and incoming frame receiver."""

        async def _ping_loop() -> None:
            while not self._stopped.is_set() and peer.connected:
                try:
                    await asyncio.sleep(self.ping_interval)
                    if self._stopped.is_set() or not peer.connected:
                        break
                    ts = struct.pack("!d", time.perf_counter())
                    await ws.send_ping(ts)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    log.debug("Peer %s ping failed: %s", peer.name, exc)
                    break

        async def _read_loop() -> None:
            while not self._stopped.is_set():
                opcode, payload = await ws.recv_frame()
                peer.last_seen = datetime.now(UTC)

                if opcode == 0x8:  # Close frame
                    log.debug("Peer %s initiated close", peer.name)
                    break
                elif opcode == 0x9:  # Ping from server
                    # Respond with pong frame
                    await ws.send_frame(payload, opcode=0xA)
                elif opcode == 0xA:  # Pong response to our ping
                    if len(payload) == 8:
                        sent_time = struct.unpack("!d", payload)[0]
                        peer.latency_ms = max(
                            0.0, round((time.perf_counter() - sent_time) * 1000.0, 2)
                        )
                elif opcode == 0x1:  # Text message
                    text = payload.decode("utf-8", errors="replace")
                    await self._handle_incoming_text(peer, text)

        ping_task = asyncio.create_task(_ping_loop(), name=f"federation-ping-{peer.name}")
        read_task = asyncio.create_task(_read_loop(), name=f"federation-read-{peer.name}")
        try:
            # Wait until either read fails or shutdown occurs
            done, pending = await asyncio.wait(
                [read_task, ping_task], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            # Propagate any exceptions from completed tasks
            for t in done:
                if not t.cancelled() and t.exception():
                    raise t.exception()  # type: ignore[misc]
        finally:
            ping_task.cancel()
            read_task.cancel()

    async def _handle_incoming_text(self, peer: PeerConnection, text: str) -> None:
        """Parse incoming JSON payload from peer and dispatch to bus."""
        try:
            data = json.loads(text)
        except Exception as exc:
            log.debug("Peer %s sent invalid JSON: %s", peer.name, exc)
            return

        if not isinstance(data, dict):
            return

        # Case 1: SystemSnapshot
        if "hardware" in data and "engines" in data:
            try:
                snapshot = SystemSnapshot.model_validate(data)
                if snapshot.node in ("local", ""):
                    snapshot = snapshot.model_copy(update={"node": peer.name})
                peer.last_snapshot = snapshot
                peer.remote_node = snapshot.node
                if self.bus:
                    self.bus.publish(Topic.SNAPSHOT, snapshot, source=peer.name)
                self._aggregate_cluster_snapshot()
            except Exception as exc:
                log.debug("Failed to validate snapshot from peer %s: %s", peer.name, exc)
            return

        # Case 2: Event relay (ALERT or CLUSTER)
        topic_str = data.get("topic")
        if topic_str in (Topic.ALERT.value, "alert"):
            raw_payload = data.get("payload")
            source = data.get("source", peer.name)

            alert_id = None
            if isinstance(raw_payload, dict):
                alert_id = raw_payload.get("id")
            if alert_id:
                if alert_id in self._seen_alert_ids:
                    return
                self._remember_alert_id(alert_id)

            try:
                alert = (
                    AlertEvent.model_validate(raw_payload)
                    if isinstance(raw_payload, dict)
                    else raw_payload
                )
            except Exception:
                alert = raw_payload

            if self.bus:
                self.bus.publish(Topic.ALERT, alert, source=source)
            # Gossip forward to other connected peers
            await self.broadcast_event(
                Topic.ALERT, raw_payload, source=source, exclude_peer=peer.name
            )

        elif topic_str in (Topic.CLUSTER.value, "cluster"):
            raw_payload = data.get("payload")
            source = data.get("source", peer.name)
            try:
                cluster = (
                    ClusterSnapshot.model_validate(raw_payload)
                    if isinstance(raw_payload, dict)
                    else raw_payload
                )
            except Exception:
                cluster = raw_payload

            if self.bus:
                self.bus.publish(Topic.CLUSTER, cluster, source=source)

    def get_status(self) -> dict[str, Any]:
        """Generate full status payload for /api/federation."""
        active_count = self.connected_peer_count
        peers_data = [peer.to_dict() for peer in self._peers.values()]
        node_names = [self.local_node]
        links: list[dict[str, Any]] = []

        for peer in self._peers.values():
            target_name = peer.remote_node or peer.name
            if peer.connected and target_name not in node_names:
                node_names.append(target_name)
            links.append(
                {
                    "source": self.local_node,
                    "target": peer.name,
                    "url": peer.ws_url,
                    "connected": peer.connected,
                    "latency_ms": round(peer.latency_ms, 2),
                    "last_seen": peer.last_seen.isoformat() if peer.last_seen else None,
                    "status": "connected" if peer.connected else "disconnected",
                }
            )

        return {
            "local_node": self.local_node,
            "active_connections": active_count,
            "total_peers": len(self._peers),
            "peers": peers_data,
            "topology": {
                "nodes": node_names,
                "links": links,
            },
        }
