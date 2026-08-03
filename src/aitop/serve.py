"""Local snapshot HTTP server — the fleet gateway seam.

`aitop serve` exposes the same `SystemSnapshot` every other consumer sees:

  GET /                 -> redirect hint / service index
  GET /ui               -> live HTML dashboard
  GET /healthz          -> {"ok": true, "version": "..."}
  GET /api/snapshot     -> SystemSnapshot JSON
  GET /api/stream       -> text/event-stream of snapshots
  GET /api/ws           -> WebSocket snapshot stream
  GET /metrics          -> Prometheus text exposition

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
from typing import Any
from urllib.parse import urlparse

import httpx

from aitop.collector import SnapshotCollector
from aitop.config import Config
from aitop.models import SystemSnapshot
from aitop.version import __version__

log = logging.getLogger(__name__)

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_ENDPOINTS = [
    "/ui",
    "/healthz",
    "/api/snapshot",
    "/api/stream",
    "/api/ws",
    "/metrics",
]


@dataclass(frozen=True, slots=True)
class FleetNode:
    name: str
    url: str
    timeout: float = 3.0


async def fetch_remote_snapshot(node: FleetNode) -> SystemSnapshot | None:
    """Pull one snapshot from a peer `aitop serve` endpoint."""
    base = node.url.rstrip("/")
    url = f"{base}/api/snapshot" if not base.endswith("/api/snapshot") else base
    try:
        async with httpx.AsyncClient(timeout=node.timeout, trust_env=False) as client:
            response = await client.get(url, headers={"User-Agent": f"aitop/{__version__}"})
            response.raise_for_status()
            return SystemSnapshot.model_validate(response.json())
    except Exception as exc:
        log.debug("fleet node %s unreachable: %s", node.name, exc)
        return None


def fleet_nodes_from_config(config: Config) -> list[FleetNode]:
    return [
        FleetNode(name=n.name, url=n.url, timeout=n.timeout)
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
    """Tiny async HTTP/1.1 server for snapshot, SSE, WebSocket, and /metrics."""

    def __init__(
        self,
        collector: SnapshotCollector,
        *,
        host: str = "127.0.0.1",
        port: int = 9090,
        interval: float | None = None,
    ) -> None:
        self.collector = collector
        self.host = host
        self.port = port
        self.interval = interval or collector.config.polling.hardware_interval
        self._latest: SystemSnapshot | None = None
        self._lock = asyncio.Lock()
        self._server: asyncio.AbstractServer | None = None

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
        async with self._server:
            await self._server.serve_forever()

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
            path = urlparse(raw_path).path

            if method == "GET" and path == "/api/ws":
                keep_open = await self._websocket(reader, writer, headers)
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
                        "endpoints": _ENDPOINTS,
                    },
                )
            elif path == "/ui":
                from aitop.views.web import render_dashboard

                await self._write_raw(
                    writer,
                    200,
                    render_dashboard(),
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
            elif path == "/api/stream":
                keep_open = True
                await self._write_sse(writer)
            elif path == "/metrics":
                from aitop.prometheus import render_prometheus

                snapshot = await self.latest()
                body = render_prometheus(snapshot).encode()
                await self._write_raw(
                    writer,
                    200,
                    body,
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

    async def _write_sse(self, writer: asyncio.StreamWriter) -> None:
        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
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
        """Upgrade and push snapshot JSON frames until the client disconnects."""
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

        async def _drain_client() -> None:
            # Consume (and ignore) client frames so the TCP window stays open.
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
                    if opcode == 0x8:  # close
                        return
                    if opcode == 0x9:  # ping -> pong
                        writer.write(_ws_frame(payload, opcode=0xA))
                        await writer.drain()
            except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
                return

        reader_task = asyncio.create_task(_drain_client())
        try:
            async for snapshot in self.stream():
                if reader_task.done():
                    break
                writer.write(_ws_frame(snapshot.model_dump_json().encode()))
                await writer.drain()
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()
        return True

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
            400: "Bad Request",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error",
        }.get(status, "OK")
        header = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Access-Control-Allow-Origin: *\r\n"
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
