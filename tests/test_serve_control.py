"""Serve control API, auth token, and docker lifecycle helpers."""

from __future__ import annotations

import asyncio

import httpx

from aitop.config import Config, EndpointConfig, FleetConfig
from aitop.engines.lifecycle import start_engine, stop_engine
from aitop.models import EngineKind, SystemSnapshot
from aitop.serve import SnapshotServer
from aitop.views.web import render_dashboard


class FakeEngine:
    kind = EngineKind.OLLAMA
    name = "Ollama"
    _caps = {"lifecycle", "load", "unload", "delete"}

    def supports(self, cap: str) -> bool:
        return cap in self._caps

    async def start(self):
        return True, "started ollama"

    async def stop(self):
        return True, "stopped ollama"

    async def restart(self):
        return True, "restarted ollama"

    async def load(self, model_id: str):
        return True, f"loaded {model_id}"

    async def unload(self, model_id: str | None = None):
        return True, f"unloaded {model_id or 'all'}"

    async def delete(self, model_id: str):
        return True, f"deleted {model_id}"


class FakeCollector:
    def __init__(self, token: str | None = None, auth_all: bool = False):
        self.config = Config(
            fleet=FleetConfig(serve_token=token, serve_auth_all=auth_all)
        )
        self.node = "local"
        self.engines = type("R", (), {"engines": [FakeEngine()]})()

    async def collect(self) -> SystemSnapshot:
        return SystemSnapshot(node="local")

    async def aclose(self) -> None:
        return None


async def _boot(server: SnapshotServer):
    srv = await asyncio.start_server(server._handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    server._server = srv
    return srv, port


def test_dashboard_has_control_surface():
    html = render_dashboard(auth_required=True).decode()
    assert "/api/models/load" in html or "data-mact" in html
    assert "catalog" in html
    assert "serve token" in html
    assert "cdn." not in html.lower()


async def test_control_api_engine_and_models():
    server = SnapshotServer(FakeCollector(), host="127.0.0.1", port=0)  # type: ignore[arg-type]
    srv, port = await _boot(server)
    base = f"http://127.0.0.1:{port}"

    async with srv, httpx.AsyncClient() as client:
        start = await client.post(f"{base}/api/engines/ollama/start")
        assert start.status_code == 200
        assert start.json()["ok"] is True

        load = await client.post(
            f"{base}/api/models/load",
            json={"engine": "ollama", "model": "llama3.2:3b"},
        )
        assert load.status_code == 200
        assert "loaded" in load.json()["message"]

        unload = await client.post(
            f"{base}/api/models/unload",
            json={"engine": "ollama", "model": "llama3.2:3b"},
        )
        assert unload.json()["ok"] is True

        delete = await client.post(
            f"{base}/api/models/delete",
            json={"engine": "ollama", "model": "old"},
        )
        assert delete.json()["ok"] is True

        bad = await client.post(f"{base}/api/models/load", json={"engine": "ollama"})
        assert bad.status_code == 400

    await srv.wait_closed()


async def test_control_api_requires_token():
    server = SnapshotServer(
        FakeCollector(token="s3cret"),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=0,
        token="s3cret",
    )
    srv, port = await _boot(server)
    base = f"http://127.0.0.1:{port}"

    async with srv, httpx.AsyncClient() as client:
        denied = await client.post(f"{base}/api/engines/ollama/stop")
        assert denied.status_code == 401

        # Reads stay open by default
        snap = await client.get(f"{base}/api/snapshot")
        assert snap.status_code == 200

        ok = await client.post(
            f"{base}/api/engines/ollama/stop",
            headers={"Authorization": "Bearer s3cret"},
        )
        assert ok.status_code == 200
        assert ok.json()["ok"] is True

        ok2 = await client.post(
            f"{base}/api/models/unload",
            headers={"X-Aitop-Token": "s3cret"},
            json={"engine": "ollama"},
        )
        assert ok2.status_code == 200

        index = await client.get(f"{base}/")
        assert index.json()["control_auth_required"] is True

    await srv.wait_closed()


async def test_auth_all_protects_snapshot():
    server = SnapshotServer(
        FakeCollector(token="lock", auth_all=True),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=0,
        token="lock",
        auth_all=True,
    )
    srv, port = await _boot(server)
    base = f"http://127.0.0.1:{port}"

    async with srv, httpx.AsyncClient() as client:
        assert (await client.get(f"{base}/healthz")).status_code == 200
        assert (await client.get(f"{base}/api/snapshot")).status_code == 401
        ok = await client.get(
            f"{base}/api/snapshot",
            headers={"Authorization": "Bearer lock"},
        )
        assert ok.status_code == 200

    await srv.wait_closed()


async def test_docker_lifecycle_needs_container(monkeypatch):
    from aitop.utils import proc

    calls: list[tuple[str, ...]] = []

    async def fake_run(*argv, **kwargs):
        calls.append(argv)
        return proc.CommandResult(argv, 0, "ok", "")

    monkeypatch.setattr("aitop.engines.lifecycle.run", fake_run)
    monkeypatch.setattr("aitop.engines.lifecycle.which", lambda _: "/usr/bin/docker")

    missing = await start_engine("ollama", managed_by="docker")
    assert not missing.ok
    assert "container" in missing.message

    started = await start_engine("ollama", managed_by="docker", container="ollama")
    assert started.ok
    assert calls[-1] == ("docker", "start", "ollama")

    stopped = await stop_engine("ollama", managed_by="docker", container="ollama")
    assert stopped.ok
    assert calls[-1] == ("docker", "stop", "ollama")


def test_endpoint_container_config():
    ep = EndpointConfig(kind=EngineKind.OLLAMA, container="my-ollama")
    assert ep.container == "my-ollama"
