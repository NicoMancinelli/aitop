"""In-process async event bus.

This is the seam that keeps telemetry collection independent of rendering.
The collector publishes typed events; the TUI, the Prometheus exporter and the
(future) WebSocket bridge each subscribe. Nothing downstream of the bus is
allowed to import `aitop.hardware` or `aitop.engines` directly.

Subscribers get a bounded queue: a slow renderer drops frames rather than
back-pressuring the collector.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

log = logging.getLogger(__name__)


class Topic(StrEnum):
    SNAPSHOT = "snapshot"
    """A full SystemSnapshot from the collector."""

    ENGINE = "engine"
    """A single EngineSnapshot refreshed out of band."""

    LIFECYCLE = "lifecycle"
    """Start/stop/restart/purge results."""

    DOWNLOAD = "download"
    """Hugging Face downloader progress."""

    LOG = "log"
    """Human-readable activity line for the TUI log pane."""

    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Event:
    topic: Topic
    payload: Any
    source: str = "local"
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


class Subscription:
    """An async iterator over events, with a bounded drop-oldest buffer."""

    def __init__(self, bus: EventBus, topics: frozenset[Topic], maxsize: int) -> None:
        self._bus = bus
        self.topics = topics
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def _offer(self, event: Event) -> None:
        if self.topics and event.topic not in self.topics:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(event)

    async def get(self) -> Event:
        return await self._queue.get()

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Event]:
        try:
            while True:
                yield await self._queue.get()
        finally:
            self.close()

    def close(self) -> None:
        self._bus._unsubscribe(self)

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class EventBus:
    def __init__(self, maxsize: int = 32) -> None:
        self._subscribers: list[Subscription] = []
        self._maxsize = maxsize

    def subscribe(self, *topics: Topic, maxsize: int | None = None) -> Subscription:
        sub = Subscription(self, frozenset(topics), maxsize or self._maxsize)
        self._subscribers.append(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subscribers:
            self._subscribers.remove(sub)

    def publish(self, topic: Topic, payload: Any, source: str = "local") -> Event:
        event = Event(topic=topic, payload=payload, source=source)
        for sub in list(self._subscribers):
            sub._offer(event)
        return event

    def log(self, message: str, source: str = "local") -> None:
        log.debug("%s: %s", source, message)
        self.publish(Topic.LOG, message, source)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
