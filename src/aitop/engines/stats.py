"""Best-effort inference throughput parsers shared by engine adapters.

vLLM already exposes Prometheus gauges; Ollama and LM Studio usually only
report timing on generate/completions responses. Adapters cache the last good
sample and optionally soft-probe when models are resident.
"""

from __future__ import annotations

import re
from typing import Any

from aitop.models import InferenceStats
from aitop.utils.parse import first, to_float, to_int

_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*$"
)

# Soft-probe at most this often when a runtime has resident weights but no
# live /metrics feed (Ollama / LM Studio).
STATS_PROBE_INTERVAL_S = 30.0


def inference_stats_from_ollama(payload: dict[str, Any]) -> InferenceStats:
    """Derive tok/s from an Ollama `/api/generate` or `/api/chat` final chunk."""
    eval_count = to_int(payload.get("eval_count"))
    eval_ns = to_int(payload.get("eval_duration"))
    prompt_count = to_int(payload.get("prompt_eval_count"))
    prompt_ns = to_int(payload.get("prompt_eval_duration"))

    tps = _rate(eval_count, eval_ns)
    prompt_tps = _rate(prompt_count, prompt_ns)
    ttft_ms = (prompt_ns / 1_000_000.0) if prompt_ns and prompt_ns > 0 else None

    if tps is None and prompt_tps is None and ttft_ms is None:
        return InferenceStats()
    return InferenceStats(
        tokens_per_second=tps,
        prompt_tokens_per_second=prompt_tps,
        ttft_ms=ttft_ms,
        total_requests=1 if tps is not None or prompt_tps is not None else 0,
    )


def inference_stats_from_lmstudio(payload: dict[str, Any]) -> InferenceStats:
    """Parse LM Studio native `/api/v0/...` `stats` (+ usage) blocks."""
    raw = payload.get("stats")
    stats = raw if isinstance(raw, dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}

    tps = to_float(first(stats, "tokens_per_second", "generation_tps", "eval_tps", "tps"))
    prompt_tps = to_float(first(stats, "prompt_tokens_per_second", "prompt_tps", "prompt_eval_tps"))
    ttft = to_float(
        first(
            stats,
            "time_to_first_token",
            "time_to_first_token_seconds",
            "ttft",
            "ttft_seconds",
        )
    )
    ttft_ms = to_float(first(stats, "time_to_first_token_ms", "ttft_ms"))
    if ttft_ms is None and ttft is not None:
        # LM Studio v0 reports TTFT in seconds.
        ttft_ms = ttft * 1000.0 if ttft < 100 else ttft

    total = to_int(first(usage, "total_tokens", "completion_tokens")) or 0
    if tps is None and prompt_tps is None and ttft_ms is None:
        return InferenceStats()
    return InferenceStats(
        tokens_per_second=tps,
        prompt_tokens_per_second=prompt_tps,
        ttft_ms=ttft_ms,
        total_requests=1 if total or tps is not None else 0,
    )


def parse_prometheus_stats(text: str) -> InferenceStats:
    """Extract useful gauges/counters from a Prometheus `/metrics` dump.

    Understands vLLM, llama.cpp, and experimental Ollama metric names.
    """
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        # Keep the first sample for a metric name; labelled series overwrite
        # with the last — fine for the single-process local servers we scrape.
        values[name] = float(match.group("value"))

    tps = _first_present(
        values,
        "vllm:avg_generation_throughput_toks_per_s",
        "llamacpp:tokens_predicted_seconds",
        "llamacpp:predicted_tokens_seconds",
        "prompt_tokens_seconds",  # some llama-server builds
        "predicted_tokens_seconds",
        "ollama_tokens_per_second",
        "ollama:tokens_per_second",
    )
    prompt_tps = _first_present(
        values,
        "vllm:avg_prompt_throughput_toks_per_s",
        "llamacpp:tokens_prompt_seconds",
        "llamacpp:prompt_tokens_seconds",
        "prompt_tokens_per_second",
        "ollama_prompt_tokens_per_second",
    )
    queue = _first_present(
        values,
        "vllm:num_requests_waiting",
        "llamacpp:requests_deferred",
        "ollama_requests_waiting",
    )
    active = _first_present(
        values,
        "vllm:num_requests_running",
        "llamacpp:requests_processing",
        "ollama_requests_running",
        "http_requests_in_flight",
    )
    total_raw = _first_present(
        values,
        "vllm:request_success_total",
        "llamacpp:requests_total",
        "ollama_requests_total",
        "requests_total",
        "http_requests_total",
    )

    return InferenceStats(
        tokens_per_second=tps,
        prompt_tokens_per_second=prompt_tps,
        queue_depth=int(queue) if queue is not None else 0,
        active_requests=int(active) if active is not None else 0,
        total_requests=int(total_raw) if total_raw is not None else 0,
    )


def _rate(count: int | None, duration_ns: int | None) -> float | None:
    if not count or not duration_ns or duration_ns <= 0:
        return None
    return count / (duration_ns / 1_000_000_000.0)


def _first_present(values: dict[str, float], *names: str) -> float | None:
    for name in names:
        if name in values:
            return values[name]
    return None
