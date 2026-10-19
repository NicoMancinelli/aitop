# Task Plan: aitop 1.1 Major Refactor & Feature Expansion

## Goal
Execute a major v1.1 refactor and expansion using a coordinated team of specialized subagents:
1. Modularized TUI with spotlight-style Command Palette (`Ctrl+P` / `F1`).
2. Implemented **KV Cache & VRAM Fit Advisor** (`aitop.advisor`) to calculate context capacity and quantization fit before loading.
3. Implemented **Inference Request Tracing & Latency Percentiles** (P50, P90, P95, P99) in `aitop.tracing`.
4. Implemented **Fleet WebSocket Federation** (`aitop.federation`) for real-time peer-to-peer cluster synchronization.
5. Added `GET /api/federation` and `POST /api/federation/peers` in `aitop.serve`.
6. Verified full test suite (263 passed tests in 7.0s), linting, packaging, and open Pull Request for v1.1.

## Agent Team Roles
- **Lead Orchestrator (Antigravity Main):** Planning, integration, cross-module cohesion, CommandPaletteScreen, test harness, and release PR.
- **Agent 1 (Advisor & Tracing Architect):** `aitop.advisor` (KV cache calculation, VRAM fit) and `aitop.tracing` (latency histogram & percentiles).
- **Agent 2 (Fleet Federation Engineer):** Bi-directional WebSocket event bus federation across nodes in `src/aitop/federation.py` and `aitop.serve`.

## Phases

### Phase 1: Team Launch & Modular Planning
- [x] Create branch `feat/v1.1-refactor-expansion`
- [x] Update `task_plan.md` and define team responsibilities
- [x] Launch specialized subagents with designated scopes
- **Status:** complete

### Phase 2: Core Algorithms (Advisor & Latency Tracing)
- [x] Implement `aitop.advisor` (model weight + KV cache memory estimation per context window)
- [x] Implement `aitop.tracing` (moving window latency tracker for TTFT and inter-token arrival, calculating p50/p90/p99)
- [x] Add `ModelFitResult` and `LatencySummary` in `src/aitop/models.py`
- [x] Add unit tests in `tests/test_advisor_tracing.py`
- **Status:** complete

### Phase 3: TUI Refactoring & Command Palette
- [x] Add `CommandPaletteScreen` in `src/aitop/views/tui_screens.py`
- [x] Wire `Ctrl+P` spotlight command palette into `src/aitop/views/tui.py` for actions, models, navigation, and benchmarking
- **Status:** complete

### Phase 4: Fleet WebSocket Federation
- [x] Implement `src/aitop/federation.py` with `FederationManager`, `PeerConnection`, `WebSocketClient`
- [x] Support automatic reconnect loop with exponential backoff
- [x] Relay `Topic.ALERT` and `Topic.CLUSTER` events across federated nodes
- [x] Add `/api/federation` and `/api/federation/peers` to `aitop.serve`
- [x] Add integration tests in `tests/test_federation.py`
- **Status:** complete

### Phase 5: Verification & v1.1 Release
- [x] Full regression testing (`uv run pytest -q`: 263 passed in 7.0s)
- [x] Linting & formatting (`uv run ruff check` & `uv run ruff format`)
- [x] Bump version to 1.1.0 in `src/aitop/version.py`, build sdist & wheel
- [ ] Commit, push, and open/link GitHub Pull Request
- **Status:** in_progress
