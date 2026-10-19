"""In-memory request tracing and latency percentile tracking.

Maintains a circular buffer of recent inference request samples (duration,
time-to-first-token TTFT, tokens/sec) and computes statistical distributions
including P50, P90, P95, and P99 percentiles.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from aitop.models import LatencySummary


@dataclass(frozen=True, slots=True)
class RequestSample:
    """One captured inference request metric sample."""

    duration_ms: float
    ttft_ms: float | None = None
    tok_per_sec: float | None = None
    timestamp: float = 0.0


def compute_percentile(sorted_data: Sequence[float], p: float) -> float | None:
    """Compute percentile value using linear interpolation on sorted numbers."""
    if not sorted_data:
        return None
    n = len(sorted_data)
    if n == 1:
        return float(sorted_data[0])

    p_clamped = max(0.0, min(100.0, float(p)))
    idx = (p_clamped / 100.0) * (n - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return float(sorted_data[lower])

    weight = idx - lower
    val = float(sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight)
    return round(val, 3)


class RequestTracer:
    """Circular buffer tracer calculating latency and throughput percentiles."""

    def __init__(self, capacity: int = 1000) -> None:
        self._capacity = max(1, capacity)
        self._buffer: deque[RequestSample] = deque(maxlen=self._capacity)
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def record_sample(
        self,
        ttft_ms: float | None,
        tok_per_sec: float | None,
        duration_ms: float,
    ) -> None:
        """Record a single inference request sample into the circular buffer.

        Args:
            ttft_ms: Time to first token in milliseconds, or None if unavailable.
            tok_per_sec: Generation throughput in tokens per second, or None.
            duration_ms: Total request duration in milliseconds.
        """
        clean_duration = max(0.0, float(duration_ms))
        clean_ttft = float(ttft_ms) if ttft_ms is not None and ttft_ms >= 0 else None
        clean_tps = float(tok_per_sec) if tok_per_sec is not None and tok_per_sec >= 0 else None

        sample = RequestSample(
            duration_ms=clean_duration,
            ttft_ms=clean_ttft,
            tok_per_sec=clean_tps,
            timestamp=time.time(),
        )

        with self._lock:
            self._buffer.append(sample)

    def get_summary(self) -> LatencySummary:
        """Calculate statistical distribution and percentiles over recorded samples."""
        with self._lock:
            samples = list(self._buffer)

        if not samples:
            return LatencySummary()

        durations = sorted(s.duration_ms for s in samples)
        ttfts = sorted(s.ttft_ms for s in samples if s.ttft_ms is not None)
        tpss = sorted(s.tok_per_sec for s in samples if s.tok_per_sec is not None)

        return LatencySummary(
            sample_count=len(samples),
            # TTFT metrics
            ttft_p50=compute_percentile(ttfts, 50.0),
            ttft_p90=compute_percentile(ttfts, 90.0),
            ttft_p95=compute_percentile(ttfts, 95.0),
            ttft_p99=compute_percentile(ttfts, 99.0),
            ttft_min=round(ttfts[0], 3) if ttfts else None,
            ttft_max=round(ttfts[-1], 3) if ttfts else None,
            ttft_mean=round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
            # Tokens per second metrics
            tok_per_sec_p50=compute_percentile(tpss, 50.0),
            tok_per_sec_p90=compute_percentile(tpss, 90.0),
            tok_per_sec_p95=compute_percentile(tpss, 95.0),
            tok_per_sec_p99=compute_percentile(tpss, 99.0),
            tok_per_sec_min=round(tpss[0], 3) if tpss else None,
            tok_per_sec_max=round(tpss[-1], 3) if tpss else None,
            tok_per_sec_mean=round(sum(tpss) / len(tpss), 3) if tpss else None,
            # Duration metrics
            duration_p50=compute_percentile(durations, 50.0),
            duration_p90=compute_percentile(durations, 90.0),
            duration_p95=compute_percentile(durations, 95.0),
            duration_p99=compute_percentile(durations, 99.0),
            duration_min=round(durations[0], 3) if durations else None,
            duration_max=round(durations[-1], 3) if durations else None,
            duration_mean=round(sum(durations) / len(durations), 3) if durations else None,
        )

    def clear(self) -> None:
        """Clear all samples from the circular buffer."""
        with self._lock:
            self._buffer.clear()

    def get_samples(self) -> list[RequestSample]:
        """Return a snapshot copy of current buffer samples."""
        with self._lock:
            return list(self._buffer)


# Alias
LatencyTracer = RequestTracer

# Module-level default singleton instance
_GLOBAL_TRACER = RequestTracer()


def record_sample(
    ttft_ms: float | None,
    tok_per_sec: float | None,
    duration_ms: float,
) -> None:
    """Record sample into global tracing buffer."""
    _GLOBAL_TRACER.record_sample(ttft_ms=ttft_ms, tok_per_sec=tok_per_sec, duration_ms=duration_ms)


def get_summary() -> LatencySummary:
    """Retrieve latency summary from global tracing buffer."""
    return _GLOBAL_TRACER.get_summary()


def clear() -> None:
    """Clear global tracing buffer."""
    _GLOBAL_TRACER.clear()


def get_global_tracer() -> RequestTracer:
    """Return the global tracer instance."""
    return _GLOBAL_TRACER


__all__ = [
    "LatencySummary",
    "LatencyTracer",
    "RequestSample",
    "RequestTracer",
    "clear",
    "compute_percentile",
    "get_global_tracer",
    "get_summary",
    "record_sample",
]
