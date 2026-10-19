# Findings: aitop 1.0 Production Architecture

## 1. Codebase Architecture & Invariants
- `SystemSnapshot` is the single source of truth passed across the system. It is frozen and pydantic-based.
- Telemetry collection never touches rendering directly; all updates flow through `EventBus` (`Topic.SNAPSHOT`, `Topic.LIFECYCLE`, etc.).
- Network calls are strictly asynchronous using `httpx.AsyncClient` with timeouts.
- Conftest enforces strict offline mode in tests (`AITOP_NO_UPDATE_CHECK=1`).

## 2. Fleet & Discovery Architecture
- Currently, `Config.fleet.nodes` takes explicit static `FleetNodeConfig` items.
- Tailscale already provides `snap.tailscale.peer_count` and node IPv4 addresses. We can query Tailscale peers (`tailscale status --json`) to automatically detect peer nodes running `aitop serve` on default port 9090.
- Aggregation across nodes produces cluster-level statistics:
  - `total_vram_bytes`, `used_vram_bytes`
  - `total_gpus`, `total_online_engines`, `total_loaded_models`
  - `cluster_tokens_per_second`

## 3. Alerts Architecture
- Alert rules evaluate snapshots on each polling cycle:
  - VRAM saturation (default > 90%)
  - Engine state transitions (e.g., ONLINE -> DEGRADED/OFFLINE)
  - Temperature exceeding critical threshold (> 85C)
- Alert outputs:
  - EventBus publishing (`Topic.ALERT`)
  - Webhook dispatching (HTTP POST JSON payload with title, body, severity)
  - Desktop / Terminal bell or TUI status notification

## 4. Daemon / Service Manager
- macOS: `~/Library/LaunchAgents/com.aitop.serve.plist` using `launchctl load/unload`.
- Linux: `~/.config/systemd/user/aitop.service` using `systemctl --user enable/start/status`.
