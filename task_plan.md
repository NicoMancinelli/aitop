# Task Plan: aitop 1.0 Production Release

## Goal
Advance `aitop` from v0.3.0 to a complete, production-grade 1.0 release with cluster auto-discovery, cluster-wide metric aggregation, alerting engine, background daemon/service management, enhanced web & TUI experiences, expanded Prometheus/cluster telemetry, and full test suite verification.

## Current Phase
Phase 7: Verification, Docs, Packaging & v1.0 Release

## Phases

### Phase 1: Architecture, Discovery & Plan Initialization
- [x] Analyze codebase architecture and identify v1.0 seams
- [x] Create persistent planning files (task_plan.md, findings.md, progress.md)
- [x] Define cluster, alert, service, and UI models
- **Status:** complete

### Phase 2: Cluster Auto-Discovery & Aggregation
- [x] Design and implement `aitop.discovery` (Tailscale peer discovery + endpoint probing)
- [x] Implement `ClusterSnapshot` with aggregate VRAM, GPUs, engines, and cluster tok/s
- [x] Integrate cluster aggregation into `SnapshotCollector` and `SnapshotServer`
- **Status:** complete

### Phase 3: Alerting Engine
- [x] Implement `aitop.alerts` with rules for high VRAM, engine down, high temp
- [x] Implement dispatchers: Webhook (Slack/Discord/generic), Terminal Bell, EventBus
- [x] Wire alerting into `SnapshotCollector` background cycle
- **Status:** complete

### Phase 4: Service / Daemon Management
- [x] Implement `aitop.service` for macOS `launchd` and Linux `systemd` user service
- [x] Add CLI commands `aitop service install|status|uninstall`
- **Status:** complete

### Phase 5: TUI Polish & Web Dashboard v1.0
- [x] Add cluster summary widget to TUI Fleet pane & detail strip
- [x] Add color themes in config & TUI
- [x] Enhance web UI with cluster dashboard link and interactive benchmark trigger
- **Status:** complete

### Phase 6: Expanded Metrics & Telemetry
- [x] Add cluster-level Prometheus metrics to `aitop.prometheus` (`render_cluster_prometheus`)
- [x] Add `/api/cluster` to `aitop serve`
- **Status:** complete

### Phase 7: Verification, Docs, Packaging & v1.0 Release
- [x] Write comprehensive unit tests for all new modules (`tests/test_v100_release.py`)
- [x] Run full test suite (237 passed), linting (`ruff`), and type verification
- [x] Update `README.md`, bump version to `1.0.0` in `version.py`
- [ ] Commit, push, and open/link GitHub Pull Request
- **Status:** in_progress

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Zero external network in tests | Enforced by project conftest and respx mock fixtures |
| Modular stdlib-first service manager | Generates standard launchd plist and systemd units without external dependencies |
| Tailscale-first cluster discovery | Automatic discovery of tailnet peers running aitop serve on port 9090 |
| Pydantic immutable models for ClusterSnapshot and AlertEvent | Thread-safe, serialized across EventBus, JSON endpoints, and TUI |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| GPUSnapshot vendor name collision | 1 | Imported `Vendor` instead of `GPUVendor` |
| GPUSnapshot attribute naming | 1 | Used `vram_total_bytes` and `vram_used_bytes` instead of `memory_*` |
