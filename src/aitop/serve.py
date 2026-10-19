"""Local snapshot HTTP server — the fleet gateway seam.

`aitop serve` exposes the same `SystemSnapshot` every other consumer sees:

  GET  /                 -> service index
  GET  /ui               -> live HTML dashboard (with control actions)
  GET  /healthz          -> {"ok": true, "version": "..."}
  GET  /api/snapshot     -> SystemSnapshot JSON
  GET  /api/fleet        -> list of SystemSnapshot JSONs
  GET  /api/cluster      -> aggregated ClusterSnapshot JSON
  GET  /api/federation   -> active peer connections and cluster link topology
  GET  /api/stream       -> text/event-stream of snapshots
  GET  /api/ws           -> WebSocket snapshot and event stream
  GET  /metrics          -> Prometheus text exposition
  POST /api/engines/{kind}/start|stop|restart
  POST /api/models/load|unload|delete
  POST /api/federation/peers

Implemented with the stdlib so the runtime dependency set stays small.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import httpx

from aitop.bus import Topic
from aitop.collector import SnapshotCollector
from aitop.config import Config
from aitop.federation import FederationManager
from aitop.models import AlertEvent, ClusterSnapshot, EngineKind, SystemSnapshot
from aitop.version import __version__

if TYPE_CHECKING:
    from aitop.bus import EventBus

log = logging.getLogger(__name__)

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_ENDPOINTS = [
    "/ui",
    "/healthz",
    "/api/snapshot",
    "/api/fleet",
    "/api/cluster",
    "/api/federation",
    "/api/stream",
    "/api/ws",
    "/metrics",
    "/api/engines/{kind}/start",
    "/api/engines/{kind}/stop",
    "/api/engines/{kind}/restart",
    "/api/models/load",
    "/api/models/unload",
    "/api/models/delete",
    "/api/models/bench",
    "/api/federation/peers",
]
_PUBLIC_GET = frozenset({"/healthz", "/ui"})
_MUTATING_PREFIXES = ("/api/engines/", "/api/models/", "/api/federation/peers")


@dataclass(frozen=True, slots=True)
class FleetNode:
    name: str
    url: str
    timeout: float = 3.0
    token: str | None = None


async def fetch_remote_snapshot(node: FleetNode) -> SystemSnapshot | None:
    """Pull one snapshot from a peer `aitop serve` endpoint."""
    base = node.url.rstrip("/")
    url = f"{base}/api/snapshot" if not base.endswith("/api/snapshot") else base
    headers = {"User-Agent": f"aitop/{__version__}"}
    if node.token:
        headers["Authorization"] = f"Bearer {node.token}"
    try:
        async with httpx.AsyncClient(timeout=node.timeout, trust_env=False) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return SystemSnapshot.model_validate(response.json())
    except Exception as exc:
        log.debug("fleet node %s unreachable: %s", node.name, exc)
        return None


def fleet_nodes_from_config(config: Config) -> list[FleetNode]:
    return [
        FleetNode(name=n.name, url=n.url, timeout=n.timeout, token=n.token)
        for n in config.fleet.nodes
        if n.enabled
    ]


async def merge_fleet(
    local: SystemSnapshot,
    nodes: list[FleetNode],
) -> list[SystemSnapshot]:
    """Fetch remote snapshots concurrently; skip nodes that are down."""
    if not nodes:
        return [local]
    results = await asyncio.gather(*(fetch_remote_snapshot(n) for n in nodes))
    remotes: list[SystemSnapshot] = []
    for node, snap in zip(nodes, results, strict=True):
        if snap is None:
            continue
        remotes.append(snap.model_copy(update={"node": node.name}))
    return [local, *remotes]


class SnapshotServer:
    """Tiny async HTTP/1.1 server for snapshot, SSE, WebSocket, control, /metrics."""

    def __init__(
        self,
        collector: SnapshotCollector,
        *,
        host: str = "127.0.0.1",
        port: int = 9090,
        interval: float | None = None,
        token: str | None = None,
        auth_all: bool = False,
        federation: FederationManager | None = None,
    ) -> None:
        self.collector = collector
        self.host = host
        self.port = port
        self.interval = interval or collector.config.polling.hardware_interval
        self.token = token if token is not None else collector.config.fleet.serve_token
        self.auth_all = auth_all or collector.config.fleet.serve_auth_all
        self._latest: SystemSnapshot | None = None
        self._lock = asyncio.Lock()
        self._server: asyncio.AbstractServer | None = None

        if federation is not None:
            self.federation: FederationManager | None = federation
        else:
            bus = getattr(collector, "bus", None)
            node = getattr(collector, "node", "local")
            cfg = getattr(collector, "config", None)
            if cfg is not None:
                self.federation = FederationManager.from_config(cfg, bus=bus, local_node=node)
            else:
                self.federation = FederationManager(bus=bus, local_node=node)

    async def refresh(self) -> SystemSnapshot:
        snapshot = await self.collector.collect()
        async with self._lock:
            self._latest = snapshot
        return snapshot

    async def latest(self) -> SystemSnapshot:
        async with self._lock:
            if self._latest is not None:
                return self._latest
        return await self.refresh()

    async def stream(self) -> AsyncIterator[SystemSnapshot]:
        while True:
            yield await self.refresh()
            await asyncio.sleep(max(0.25, self.interval))

    async def run(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        sockets = self._server.sockets or []
        bound = sockets[0].getsockname() if sockets else (self.host, self.port)
        log.info("aitop serve listening on http://%s:%s", bound[0], bound[1])
        if self.federation:
            await self.federation.start()
        try:
            async with self._server:
                await self._server.serve_forever()
        finally:
            if self.federation:
                await self.federation.stop()

    async def aclose(self) -> None:
        """Gracefully shut down server, active connections, and federation."""
        if self.federation:
            await self.federation.stop()
        if self._server:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()

    def get_federation_status(self) -> dict[str, Any]:
        """Return active peer connections and cluster link topology."""
        if self.federation:
            return self.federation.get_status()
        local_node = getattr(self.collector, "node", "local")
        return {
            "local_node": local_node,
            "active_connections": 0,
            "total_peers": 0,
            "peers": [],
            "topology": {
                "nodes": [local_node],
                "links": [],
            },
        }

    def _authorized(
        self,
        headers: dict[str, str],
        *,
        mutating: bool,
        query: dict[str, list[str]] | None = None,
    ) -> bool:
        if not self.token:
            return True
        if not mutating and not self.auth_all:
            return True
        provided = headers.get("x-aitop-token")
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        # EventSource cannot set headers — accept ?token= / ?access_token=.
        if not provided and query:
            for key in ("token", "access_token"):
                values = query.get(key) or []
                if values and values[0]:
                    provided = values[0]
                    break
        return bool(provided) and provided == self.token

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        keep_open = False
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            if not request_line:
                writer.close()
                return
            headers: dict[str, str] = {}
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=10.0)
                if line in (b"\r\n", b"\n", b""):
                    break
                text = line.decode("latin-1", "replace")
                if ":" in text:
                    key, _, value = text.partition(":")
                    headers[key.strip().lower()] = value.strip()

            method, raw_path, *_ = request_line.decode("latin-1", "replace").split()
            parsed = urlparse(raw_path)
            path = parsed.path
            query = parse_qs(parsed.query)
            length = int(headers.get("content-length", "0") or "0")
            body = await reader.readexactly(length) if length > 0 else b""

            if method == "OPTIONS":
                await self._write_cors(writer)
                return

            mutating = method in {"POST", "PUT", "PATCH", "DELETE"} or any(
                path.startswith(p) and method != "GET" for p in _MUTATING_PREFIXES
            )
            # Control POSTs are always mutating; optionally gate all GETs.
            needs_auth = mutating or (self.auth_all and path not in _PUBLIC_GET and method == "GET")
            if needs_auth and not self._authorized(headers, mutating=True, query=query):
                await self._write(writer, 401, {"error": "unauthorized"})
                return

            if method == "GET" and path == "/api/ws":
                if self.auth_all and not self._authorized(headers, mutating=True, query=query):
                    await self._write(writer, 401, {"error": "unauthorized"})
                    return
                keep_open = await self._websocket(reader, writer, headers)
                return

            if method == "POST":
                await self._handle_post(writer, path, body)
                return

            if method != "GET":
                await self._write(writer, 405, {"error": "method not allowed"})
                return

            if path in ("/", "/api"):
                await self._write(
                    writer,
                    200,
                    {
                        "service": "aitop",
                        "version": __version__,
                        "ui": "/ui",
                        "auth_required": bool(self.token and self.auth_all),
                        "control_auth_required": bool(self.token),
                        "endpoints": _ENDPOINTS,
                    },
                )
            elif path == "/ui":
                from aitop.views.web import render_dashboard

                await self._write_raw(
                    writer,
                    200,
                    render_dashboard(
                        auth_required=bool(self.token),
                        auth_all=bool(self.token and self.auth_all),
                    ),
                    content_type="text/html; charset=utf-8",
                )
            elif path == "/healthz":
                await self._write(writer, 200, {"ok": True, "version": __version__})
            elif path == "/api/snapshot":
                snapshot = await self.latest()
                await self._write_raw(
                    writer,
                    200,
                    snapshot.model_dump_json().encode(),
                    content_type="application/json",
                )
            elif path == "/api/fleet":
                fleet = await self.collector.collect_fleet()
                payload = json.dumps([s.model_dump(mode="json") for s in fleet]).encode()
                await self._write_raw(
                    writer,
                    200,
                    payload,
                    content_type="application/json",
                )
            elif path == "/api/cluster":
                cluster = await self.collector.collect_cluster()
                await self._write_raw(
                    writer,
                    200,
                    cluster.model_dump_json().encode(),
                    content_type="application/json",
                )
            elif path == "/api/federation":
                fed_status = self.get_federation_status()
                await self._write_raw(
                    writer,
                    200,
                    json.dumps(fed_status).encode(),
                    content_type="application/json",
                )
            elif path == "/api/stream":
                keep_open = True
                await self._write_sse(writer)
            elif path == "/metrics":
                from aitop.prometheus import render_prometheus

                snapshot = await self.latest()
                body_out = render_prometheus(snapshot).encode()
                await self._write_raw(
                    writer,
                    200,
                    body_out,
                    content_type="text/plain; version=0.0.4; charset=utf-8",
                )
            else:
                await self._write(writer, 404, {"error": "not found"})
        except Exception as exc:
            log.debug("serve handler error: %s", exc)
            with contextlib.suppress(Exception):
                await self._write(writer, 500, {"error": "internal"})
        finally:
            if not keep_open:
                with contextlib.suppress(Exception):
                    writer.close()
                    await writer.wait_closed()

    async def _handle_post(
        self,
        writer: asyncio.StreamWriter,
        path: str,
        body: bytes,
    ) -> None:
        payload: dict[str, Any] = {}
        if body:
            try:
                parsed = json.loads(body.decode())
                if isinstance(parsed, dict):
                    payload = parsed
            except (UnicodeDecodeError, json.JSONDecodeError):
                await self._write(writer, 400, {"error": "invalid json"})
                return

        # /api/federation/peers
        if path == "/api/federation/peers":
            name = str(payload.get("name") or "")
            url = str(payload.get("url") or "")
            token = payload.get("token")
            timeout = float(payload.get("timeout") or 5.0)
            if not name or not url:
                await self._write(writer, 400, {"error": "name and url required"})
                return
            if self.federation:
                peer = self.federation.add_peer(name, url, token=token, timeout=timeout)
                await self._write(writer, 200, {"ok": True, "peer": peer.to_dict()})
            else:
                await self._write(writer, 500, {"error": "federation unavailable"})
            return

        # /api/engines/{kind}/{action}
        if path.startswith("/api/engines/"):
            parts = [p for p in path.strip("/").split("/") if p]
            # api, engines, kind, action
            if len(parts) != 4:
                await self._write(writer, 404, {"error": "not found"})
                return
            kind_raw, action = parts[2], parts[3]
            if action not in {"start", "stop", "restart"}:
                await self._write(writer, 404, {"error": "unknown action"})
                return
            ok, message = await self._engine_action(kind_raw, action)
            await self._write(writer, 200 if ok else 400, {"ok": ok, "message": message})
            await self.refresh()
            return

        if path in {"/api/models/load", "/api/models/unload", "/api/models/delete"}:
            action = path.rsplit("/", 1)[-1]
            engine_raw = str(payload.get("engine") or "")
            model = payload.get("model")
            model_id = str(model) if model is not None else None
            if action != "unload" and not model_id:
                await self._write(writer, 400, {"error": "model required"})
                return
            ok, message = await self._model_action(engine_raw, action, model_id)
            await self._write(writer, 200 if ok else 400, {"ok": ok, "message": message})
            await self.refresh()
            return

        if path == "/api/models/bench":
            from aitop.bench import run_benchmark

            engine_raw = str(payload.get("engine") or "")
            model = payload.get("model")
            model_id = str(model) if model is not None else None
            if not model_id:
                await self._write(writer, 400, {"error": "model required"})
                return
            prompt = str(
                payload.get("prompt") or "Why is the sky blue? Answer in one short paragraph."
            )
            max_tokens = int(payload.get("tokens") or payload.get("max_tokens") or 64)
            engine = self._find_engine(engine_raw)
            resolved_engine = engine.kind.value if engine else engine_raw
            try:
                result = await run_benchmark(
                    engine=resolved_engine,
                    model=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    config=self.collector.config,
                )
                await self._write_raw(
                    writer,
                    200,
                    result.model_dump_json().encode(),
                    content_type="application/json",
                )
            except Exception as exc:
                await self._write(writer, 500, {"error": f"benchmark failed: {exc}"})
            return

        await self._write(writer, 404, {"error": "not found"})

    async def _engine_action(self, kind_raw: str, action: str) -> tuple[bool, str]:
        engine = self._find_engine(kind_raw)
        if engine is None:
            return False, f"no adapter for {kind_raw}"
        if not engine.supports("lifecycle"):
            return False, f"{engine.name} has no lifecycle control"
        return await getattr(engine, action)()

    async def _model_action(
        self, kind_raw: str, action: str, model_id: str | None
    ) -> tuple[bool, str]:
        engine = self._find_engine(kind_raw)
        if engine is None:
            return False, f"no adapter for {kind_raw}"
        if not engine.supports(action):
            return False, f"{engine.name} cannot {action}"
        if action == "unload":
            return await engine.unload(model_id)
        if action == "load":
            assert model_id is not None
            return await engine.load(model_id)
        if action == "delete":
            assert model_id is not None
            return await engine.delete(model_id)
        return False, f"unknown action {action}"

    def _find_engine(self, kind_raw: str):
        needle = kind_raw.strip().lower().replace("_", "-")
        engines = self.collector.engines.engines
        for eng in engines:
            if eng.kind.value == needle or eng.name.lower() == needle:
                return eng
        # Accept enum aliases like lmstudio
        with contextlib.suppress(ValueError):
            kind = EngineKind(needle)
            for eng in engines:
                if eng.kind is kind:
                    return eng
        return None

    async def _write_sse(self, writer: asyncio.StreamWriter) -> None:
        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Headers: Authorization, Content-Type, X-Aitop-Token\r\n"
            "\r\n"
        )
        writer.write(header.encode())
        await writer.drain()
        try:
            async for snapshot in self.stream():
                payload = snapshot.model_dump_json()
                writer.write(f"data: {payload}\n\n".encode())
                await writer.drain()
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            return
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _websocket(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        headers: dict[str, str],
    ) -> bool:
        """Upgrade and push snapshot frames and bus events until client disconnects."""
        key = headers.get("sec-websocket-key")
        if not key or headers.get("upgrade", "").lower() != "websocket":
            await self._write(writer, 400, {"error": "websocket upgrade required"})
            return False

        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        )
        writer.write(response.encode())
        await writer.drain()

        bus: EventBus | None = getattr(self.collector, "bus", None)
        sub = bus.subscribe(Topic.ALERT, Topic.CLUSTER) if bus else None
        send_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)

        async def _drain_client() -> None:
            try:
                while True:
                    header = await reader.readexactly(2)
                    opcode = header[0] & 0x0F
                    length = header[1] & 0x7F
                    masked = bool(header[1] & 0x80)
                    if length == 126:
                        length = struct.unpack("!H", await reader.readexactly(2))[0]
                    elif length == 127:
                        length = struct.unpack("!Q", await reader.readexactly(8))[0]
                    mask = await reader.readexactly(4) if masked else b""
                    payload = await reader.readexactly(length)
                    if masked:
                        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

                    if opcode == 0x8:  # Close
                        return
                    elif opcode == 0x9:  # Ping
                        pong = _ws_frame(payload, opcode=0xA)
                        with contextlib.suppress(asyncio.QueueFull):
                            send_queue.put_nowait(pong)
                    elif opcode == 0x1:  # Text frame from client (relay)
                        try:
                            msg = json.loads(payload.decode("utf-8", errors="replace"))
                            if isinstance(msg, dict):
                                topic_str = msg.get("topic")
                                raw_payload = msg.get("payload")
                                source = msg.get("source", "ws-peer")
                                b = getattr(self.collector, "bus", None)
                                if b and topic_str:
                                    if topic_str in (Topic.ALERT.value, "alert"):
                                        alert = (
                                            AlertEvent.model_validate(raw_payload)
                                            if isinstance(raw_payload, dict)
                                            else raw_payload
                                        )
                                        b.publish(Topic.ALERT, alert, source=source)
                                    elif topic_str in (Topic.CLUSTER.value, "cluster"):
                                        cluster = (
                                            ClusterSnapshot.model_validate(raw_payload)
                                            if isinstance(raw_payload, dict)
                                            else raw_payload
                                        )
                                        b.publish(Topic.CLUSTER, cluster, source=source)
                        except Exception as exc:
                            log.debug("error handling client ws frame: %s", exc)
            except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
                return

        async def _produce_snapshots() -> None:
            async for snapshot in self.stream():
                frame = _ws_frame(snapshot.model_dump_json().encode())
                await send_queue.put(frame)

        async def _produce_bus_events() -> None:
            if not sub:
                return
            async for event in sub:
                payload_json = (
                    event.payload.model_dump(mode="json")
                    if hasattr(event.payload, "model_dump")
                    else event.payload
                )
                msg = json.dumps(
                    {
                        "topic": event.topic.value,
                        "payload": payload_json,
                        "source": event.source,
                    }
                )
                frame = _ws_frame(msg.encode())
                await send_queue.put(frame)

        async def _send_loop() -> None:
            while True:
                frame = await send_queue.get()
                writer.write(frame)
                await writer.drain()

        reader_task = asyncio.create_task(_drain_client())
        snap_task = asyncio.create_task(_produce_snapshots())
        bus_task = asyncio.create_task(_produce_bus_events())
        send_task = asyncio.create_task(_send_loop())

        try:
            await reader_task
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            if sub:
                sub.close()
            for t in (reader_task, snap_task, bus_task, send_task):
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()
        return True

    async def _write_cors(self, writer: asyncio.StreamWriter) -> None:
        header = (
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Authorization, Content-Type, X-Aitop-Token\r\n"
            "Content-Length: 0\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        writer.write(header.encode())
        await writer.drain()

    async def _write(self, writer: asyncio.StreamWriter, status: int, body: dict[str, Any]) -> None:
        await self._write_raw(
            writer,
            status,
            json.dumps(body).encode(),
            content_type="application/json",
        )

    async def _write_raw(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes,
        *,
        content_type: str,
    ) -> None:
        reason = {
            200: "OK",
            204: "No Content",
            400: "Bad Request",
            401: "Unauthorized",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error",
        }.get(status, "OK")
        header = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Headers: Authorization, Content-Type, X-Aitop-Token\r\n"
            f"Date: {datetime.now(UTC).strftime('%a, %d %b %Y %H:%M:%S GMT')}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        writer.write(header.encode() + body)
        await writer.drain()


def _ws_frame(payload: bytes, *, opcode: int = 0x1) -> bytes:
    """Build an unmasked server-to-client WebSocket frame."""
    header = bytes([0x80 | (opcode & 0x0F)])
    length = len(payload)
    if length < 126:
        header += bytes([length])
    elif length < 65536:
        header += bytes([126]) + struct.pack("!H", length)
    else:
        header += bytes([127]) + struct.pack("!Q", length)
    return header + payload
