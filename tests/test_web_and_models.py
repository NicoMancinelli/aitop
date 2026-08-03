"""Web UI, WebSocket, models list, and ollama delete."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import struct

import httpx
import respx
from rich.console import Console

from aitop.engines.ollama import OllamaEngine
from aitop.models import (
    EngineKind,
    EngineSnapshot,
    EngineState,
    InferenceStats,
    LoadedModel,
    ModelInfo,
    SystemSnapshot,
)
from aitop.serve import SnapshotServer, _ws_frame
from aitop.views.neofetch import render_neofetch
from aitop.views.web import render_dashboard

OLLAMA = "http://127.0.0.1:11434"
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def test_dashboard_html_is_self_contained():
    html = render_dashboard().decode()
    assert "<title>aitop</title>" in html
    assert "/api/snapshot" in html
    assert "/api/stream" in html
    assert "EventSource" in html
    assert "cdn." not in html.lower()


@respx.mock
async def test_ollama_delete(ollama_endpoint):
    respx.request("DELETE", f"{OLLAMA}/api/delete").mock(httpx.Response(200, json={"status": "ok"}))
    engine = OllamaEngine(ollama_endpoint)
    ok, message = await engine.delete("llama3.2:3b")
    await engine.aclose()
    assert ok and "deleted" in message
    assert engine.supports("delete")


@respx.mock
async def test_ollama_delete_falls_back_to_post(ollama_endpoint):
    respx.request("DELETE", f"{OLLAMA}/api/delete").mock(httpx.Response(404))
    respx.post(f"{OLLAMA}/api/delete").mock(httpx.Response(200, json={}))
    engine = OllamaEngine(ollama_endpoint)
    ok, message = await engine.delete("old-model")
    await engine.aclose()
    assert ok


def test_neofetch_shows_tok_per_sec():
    snap = SystemSnapshot(
        engines=[
            EngineSnapshot(
                kind=EngineKind.VLLM,
                name="vLLM",
                state=EngineState.ONLINE,
                stats=InferenceStats(tokens_per_second=42.5),
            )
        ]
    )
    console = Console(record=True, width=120, force_terminal=True)
    console.print(render_neofetch(snap, show_logo=False))
    text = console.export_text()
    assert "42.5 tok/s" in text


def test_ws_frame_round_trip_header():
    frame = _ws_frame(b'{"ok":true}')
    assert frame[0] & 0x0F == 0x1  # text
    assert frame[0] & 0x80  # fin
    assert frame[1] == len(b'{"ok":true}')


async def test_serve_ui_and_ws():
    from aitop.config import Config

    class FakeCollector:
        def __init__(self):
            self.config = Config()
            self.node = "local"

        async def collect(self) -> SystemSnapshot:
            return SystemSnapshot(node="local")

        async def aclose(self) -> None:
            return None

    server = SnapshotServer(FakeCollector(), host="127.0.0.1", port=0)  # type: ignore[arg-type]
    srv = await asyncio.start_server(server._handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    server._server = srv

    async with srv:
        async with httpx.AsyncClient() as client:
            ui = await client.get(f"http://127.0.0.1:{port}/ui")
            assert ui.status_code == 200
            assert "text/html" in ui.headers["content-type"]
            assert "aitop" in ui.text

            root = await client.get(f"http://127.0.0.1:{port}/")
            body = root.json()
            assert body.get("ui") == "/ui"
            assert "/api/ws" in body["endpoints"]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        key = base64.b64encode(b"aitop-test-key-12").decode()
        req = (
            f"GET /api/ws HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        writer.write(req.encode())
        await writer.drain()

        status = await asyncio.wait_for(reader.readline(), timeout=2)
        assert b"101" in status
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=2)
            if line in (b"\r\n", b"\n", b""):
                break
            if line.lower().startswith(b"sec-websocket-accept:"):
                expected = base64.b64encode(
                    hashlib.sha1((key + _WS_GUID).encode()).digest()
                ).decode()
                assert expected.encode() in line

        header = await asyncio.wait_for(reader.readexactly(2), timeout=3)
        assert header[0] & 0x0F == 0x1
        length = header[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", await reader.readexactly(2))[0]
        payload = await asyncio.wait_for(reader.readexactly(length), timeout=3)
        assert b'"node"' in payload

        writer.write(bytes([0x88, 0x80]) + b"\x00\x00\x00\x00")
        await writer.drain()
        writer.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(writer.wait_closed(), timeout=1)

    await srv.wait_closed()


def test_models_list_cli(monkeypatch, capsys):
    from aitop import cli

    class FakeCollector:
        def __init__(self, *a, **k):
            pass

        async def collect(self):
            return SystemSnapshot(
                engines=[
                    EngineSnapshot(
                        kind=EngineKind.OLLAMA,
                        name="Ollama",
                        state=EngineState.ONLINE,
                        models=[
                            ModelInfo(
                                id="llama3.2:3b",
                                name="llama3.2:3b",
                                engine=EngineKind.OLLAMA,
                                parameter_size="3.2B",
                                quantization="Q4_K_M",
                                size_bytes=2_000_000_000,
                            )
                        ],
                        loaded=[
                            LoadedModel(
                                id="llama3.2:3b",
                                name="llama3.2:3b",
                                engine=EngineKind.OLLAMA,
                                size_bytes=2_000_000_000,
                            )
                        ],
                    )
                ]
            )

        async def aclose(self):
            return None

    monkeypatch.setattr(cli, "SnapshotCollector", FakeCollector)
    assert cli.main(["models", "list", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "llama3.2:3b" in out
    assert "Ollama" in out

    assert cli.main(["models", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["loaded"] is True


def test_delete_cli(monkeypatch, capsys):
    from aitop import cli

    class Eng:
        name = "Ollama"
        kind = EngineKind.OLLAMA

        def supports(self, cap):
            return cap == "delete"

        async def delete(self, model):
            return True, f"deleted {model}"

    class Reg:
        async def aclose(self):
            return None

    async def resolve(config, kind, host=None):
        return Eng(), Reg(), None

    monkeypatch.setattr(cli, "_resolve_engine", resolve)
    assert cli.main(["delete", "ollama", "old"]) == 0
    assert "deleted old" in capsys.readouterr().out
