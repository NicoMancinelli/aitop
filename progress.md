# Progress Log: aitop 1.0 Release

## Session Summary: 2026-10-19
- Initialized branch `feat/v1.0.0-release`.
- Subagent researched and designed cross-platform service manager templates for macOS (`launchd`) and Linux (`systemd`).
- Implemented `aitop.service` providing `install_service`, `status_service`, `uninstall_service`.
- Implemented `aitop.discovery` providing zero-config Tailscale peer probing and caching.
- Implemented `aitop.alerts` providing threshold evaluation for VRAM, thermals, and engine state changes with cooldown deduplication and Webhook/bell dispatching.
- Implemented `ClusterSnapshot` in `aitop.models` aggregating VRAM, GPUs, engines, resident models, and cluster-wide tok/s.
- Added `/api/cluster` in `aitop.serve` and `aitop cluster` command in `aitop.cli`.
- Added `render_cluster_prometheus` in `aitop.prometheus` exporting `aitop_cluster_*` gauges.
- Enhanced web UI with interactive benchmark action and cluster endpoint link.
- Added `tests/test_v100_release.py` with 9 comprehensive tests; full test suite passes at 237 tests in 8.60s.
- Linting and formatting verified clean with `ruff`.
- Version bumped to `1.0.0` and package artifacts (`aitop-1.0.0.tar.gz` and wheel) built cleanly.
