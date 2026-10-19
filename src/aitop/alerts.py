"""Alerting engine for aitop telemetry and engine lifecycles.

Evaluates snapshots against configured thresholds (VRAM, temperature, engine
downtime) and dispatches alerts via EventBus, webhooks (Slack/Discord/generic),
and terminal notifications with cooldown deduplication.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from aitop.bus import EventBus, Topic
from aitop.config import Config
from aitop.models import AlertEvent, AlertSeverity, EngineState, SystemSnapshot

log = logging.getLogger(__name__)


class AlertManager:
    """Monitors telemetry snapshots and fires AlertEvents upon rule violations."""

    def __init__(self, config: Config, bus: EventBus | None = None) -> None:
        self.config = config
        self.bus = bus
        self._last_alert_times: dict[str, float] = {}
        self._engine_states: dict[str, EngineState] = {}
        self._cooldown_s: float = 300.0  # 5 min cooldown per alert key
        self._client: httpx.AsyncClient | None = None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def evaluate(self, snapshot: SystemSnapshot) -> list[AlertEvent]:
        """Check rules against current snapshot; return fired alerts."""
        if not self.config.alerts.enabled:
            return []

        now = time.monotonic()
        alerts: list[AlertEvent] = []

        # 1. VRAM Threshold Check
        vram_limit = self.config.alerts.vram_threshold_pct
        for gpu in snapshot.hardware.gpus:
            if gpu.vram_total_bytes and gpu.vram_used_bytes:
                pct = (gpu.vram_used_bytes / gpu.vram_total_bytes) * 100.0
                if pct >= vram_limit:
                    key = f"vram:{gpu.name or str(gpu.index)}"
                    if self._should_fire(key, now):
                        alerts.append(
                            AlertEvent(
                                severity=AlertSeverity.CRITICAL
                                if pct >= 95.0
                                else AlertSeverity.WARNING,
                                title=f"High VRAM Usage ({pct:.1f}%)",
                                message=f"GPU {gpu.name or str(gpu.index)} at {pct:.1f}% VRAM ({gpu.vram_used_bytes // (1024**2)}MB / {gpu.vram_total_bytes // (1024**2)}MB)",
                                source=snapshot.node,
                                metric="vram_pct",
                                value=pct,
                                threshold=vram_limit,
                            )
                        )
                        self._last_alert_times[key] = now

        # 2. Temperature Threshold Check
        temp_limit = self.config.alerts.temp_threshold_celsius
        for gpu in snapshot.hardware.gpus:
            if gpu.temperature_c and gpu.temperature_c >= temp_limit:
                key = f"temp:{gpu.name or str(gpu.index)}"
                if self._should_fire(key, now):
                    alerts.append(
                        AlertEvent(
                            severity=AlertSeverity.CRITICAL
                            if gpu.temperature_c >= 90.0
                            else AlertSeverity.WARNING,
                            title=f"High GPU Temperature ({gpu.temperature_c:.0f}°C)",
                            message=f"GPU {gpu.name or str(gpu.index)} reached {gpu.temperature_c:.0f}°C (threshold {temp_limit:.0f}°C)",
                            source=snapshot.node,
                            metric="temperature_celsius",
                            value=gpu.temperature_c,
                            threshold=temp_limit,
                        )
                    )
                    self._last_alert_times[key] = now

        # 3. Engine Transition to Degraded/Offline
        if self.config.alerts.engine_offline:
            for eng in snapshot.engines:
                prev = self._engine_states.get(eng.name)
                self._engine_states[eng.name] = eng.state
                if prev is EngineState.ONLINE and eng.state in (
                    EngineState.OFFLINE,
                    EngineState.DEGRADED,
                ):
                    key = f"engine_down:{eng.name}"
                    if self._should_fire(key, now):
                        alerts.append(
                            AlertEvent(
                                severity=AlertSeverity.CRITICAL
                                if eng.state is EngineState.OFFLINE
                                else AlertSeverity.WARNING,
                                title=f"Runtime {eng.name} is {eng.state.value.upper()}",
                                message=f"Engine {eng.name} dropped offline or degraded: {eng.error or 'unreachable'}",
                                source=snapshot.node,
                                metric="engine_state",
                            )
                        )
                        self._last_alert_times[key] = now

        for alert in alerts:
            await self._dispatch(alert)

        return alerts

    def _should_fire(self, key: str, now: float) -> bool:
        last = self._last_alert_times.get(key, 0.0)
        return (now - last) >= self._cooldown_s

    async def _dispatch(self, alert: AlertEvent) -> None:
        # Bus
        if self.bus is not None:
            self.bus.publish(Topic.ALERT, alert, source=alert.source)

        # Terminal bell
        if self.config.alerts.terminal_bell:
            print("\a", end="", flush=True)

        # Webhook
        url = self.config.alerts.webhook_url
        if url:
            try:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=5.0)
                # Formats nicely for Slack/Discord/generic Webhooks
                payload: dict[str, Any] = {
                    "text": f"[{alert.severity.upper()}] {alert.title}\n{alert.message} (node: {alert.source})",
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "message": alert.message,
                    "node": alert.source,
                    "timestamp": alert.timestamp.isoformat(),
                }
                await self._client.post(url, json=payload)
            except Exception as exc:
                log.debug("webhook dispatch failed: %s", exc)
