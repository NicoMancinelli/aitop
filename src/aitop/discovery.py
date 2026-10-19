"""Tailscale and LAN fleet auto-discovery.

Discovers peer `aitop serve` instances running across a Tailscale tailnet or
local subnet without requiring static IP entries in config.yaml.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from aitop.net.tailscale import tailscale_binary
from aitop.serve import FleetNode
from aitop.utils.proc import run

log = logging.getLogger(__name__)

_DISCOVERY_CACHE: list[FleetNode] = []
_LAST_DISCOVERY_TIME: float = 0.0
_DISCOVERY_INTERVAL_S: float = 60.0


async def get_tailscale_peers() -> list[dict[str, Any]]:
    """Return list of active online peers from `tailscale status --json`."""
    binary = tailscale_binary()
    if binary is None:
        return []

    out = await run(binary, "status", "--json", timeout=3.0)
    if not out.ok:
        return []

    try:
        data = json.loads(out.stdout)
        if not isinstance(data, dict):
            return []
        peers = data.get("Peer") or {}
        active_peers: list[dict[str, Any]] = []
        for _key, info in peers.items():
            if not isinstance(info, dict):
                continue
            # Must be online or have IPs
            if info.get("Online") is False:
                continue
            ips = info.get("TailscaleIPs") or []
            ipv4 = next((ip for ip in ips if ":" not in ip), None)
            if ipv4:
                active_peers.append(
                    {
                        "hostname": info.get("HostName") or ipv4,
                        "ip": ipv4,
                    }
                )
        return active_peers
    except Exception as exc:
        log.debug("failed to parse tailscale peers: %s", exc)
        return []


async def probe_fleet_node(
    host: str, port: int, name: str, timeout: float = 1.0
) -> FleetNode | None:
    """Probe an endpoint's /healthz to verify if aitop serve is running."""
    url = f"http://{host}:{port}"
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.get(f"{url}/healthz")
            if resp.status_code == 200:
                body = resp.json()
                if body.get("ok"):
                    return FleetNode(name=name, url=url, timeout=3.0)
    except Exception:
        pass
    return None


async def discover_tailscale_fleet_nodes(
    port: int = 9090,
    *,
    timeout: float = 1.0,
    existing_urls: set[str] | None = None,
    force: bool = False,
) -> list[FleetNode]:
    """Find other aitop serve nodes running on the tailnet."""
    global _DISCOVERY_CACHE, _LAST_DISCOVERY_TIME

    now = time.monotonic()
    if not force and _DISCOVERY_CACHE and (now - _LAST_DISCOVERY_TIME < _DISCOVERY_INTERVAL_S):
        return _DISCOVERY_CACHE

    peers = await get_tailscale_peers()
    if not peers:
        return []

    known = existing_urls or set()
    probes = []
    for peer in peers:
        candidate_url = f"http://{peer['ip']}:{port}"
        if candidate_url in known:
            continue
        probes.append(probe_fleet_node(peer["ip"], port, peer["hostname"], timeout=timeout))

    if not probes:
        return []

    results = await asyncio.gather(*probes, return_exceptions=True)
    discovered: list[FleetNode] = [node for node in results if isinstance(node, FleetNode)]

    _DISCOVERY_CACHE = discovered
    _LAST_DISCOVERY_TIME = now
    return discovered
