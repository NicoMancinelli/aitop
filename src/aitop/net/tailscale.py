"""Tailscale detection.

`tailscale status --json` gives us the node's tailnet identity in one call and
works unprivileged. Used both for the neofetch line and, in Phase 2, to offer
the tailnet IP as a bind target for engine rebinding.
"""

from __future__ import annotations

import json
from typing import Any

from aitop.models import TailscaleStatus
from aitop.utils.proc import run, which

_CANDIDATES = (
    "tailscale",
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
    "/usr/local/bin/tailscale",
)


def tailscale_binary() -> str | None:
    """First usable tailscale CLI — PATH, then the macOS app bundle."""
    for candidate in _CANDIDATES:
        resolved = which(candidate)
        if resolved:
            return resolved
    return None


async def collect_tailscale() -> TailscaleStatus:
    binary = tailscale_binary()
    if binary is None:
        return TailscaleStatus(available=False)

    out = await run(binary, "status", "--json", timeout=4.0)
    if not out.ok:
        return TailscaleStatus(available=True, running=False)

    try:
        payload: Any = json.loads(out.stdout)
    except json.JSONDecodeError:
        return TailscaleStatus(available=True, running=False)
    if not isinstance(payload, dict):
        return TailscaleStatus(available=True, running=False)

    self_node = payload.get("Self") or {}
    ips = self_node.get("TailscaleIPs") or []
    ipv4 = next((ip for ip in ips if ":" not in ip), None)
    ipv6 = next((ip for ip in ips if ":" in ip), None)

    return TailscaleStatus(
        available=True,
        running=str(payload.get("BackendState", "")).lower() == "running",
        hostname=self_node.get("HostName") or None,
        ipv4=ipv4,
        ipv6=ipv6,
        tailnet=payload.get("MagicDNSSuffix") or (payload.get("CurrentTailnet") or {}).get("Name"),
        peer_count=len(payload.get("Peer") or {}),
    )
