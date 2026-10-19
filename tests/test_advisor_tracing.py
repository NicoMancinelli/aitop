"""Unit tests for aitop.advisor and aitop.tracing."""

from __future__ import annotations

import concurrent.futures

import pytest
from pydantic import ValidationError

from aitop.advisor import (
    FALLBACK_BYTES_PER_1024_8B,
    calculate_fit,
    calculate_kv_cache_bytes,
    calculate_model_vram_bytes,
    fallback_kv_cache_bytes,
    get_bpw,
    parse_parameter_count,
    recommend_quantization,
)
from aitop.models import LatencySummary, ModelFitResult
from aitop.tracing import (
    LatencyTracer,
    RequestTracer,
    clear,
    compute_percentile,
    get_summary,
    record_sample,
)

# --------------------------------------------------------------------------- #
# Models Tests
# --------------------------------------------------------------------------- #


def test_model_fit_result_defaults_and_immutability():
    result = ModelFitResult()
    assert result.fits is True
    assert result.fits_in_vram is True
    assert result.gpu_offload_fraction == 1.0
    assert result.exceeds_memory is False
    assert result.max_context_length == 0
    assert result.max_context_tokens == 0

    with pytest.raises((ValidationError, TypeError)):
        result.fits = False  # type: ignore[misc]


def test_latency_summary_defaults_and_properties():
    summary = LatencySummary()
    assert summary.sample_count == 0
    assert summary.ttft_p50 is None
    assert summary.tok_per_sec_p90 is None
    assert summary.tokens_per_sec_p90 is None
    assert summary.tps_p90 is None

    populated = LatencySummary(
        sample_count=10,
        ttft_p50=120.5,
        tok_per_sec_p90=45.2,
        duration_p95=500.0,
    )
    assert populated.ttft_p50 == 120.5
    assert populated.tokens_per_sec_p90 == 45.2
    assert populated.tps_p90 == 45.2
    assert populated.duration_p95 == 500.0

    with pytest.raises((ValidationError, TypeError)):
        populated.sample_count = 20  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Advisor Tests
# --------------------------------------------------------------------------- #


def test_parse_parameter_count():
    assert parse_parameter_count(8e9) == 8_000_000_000.0
    assert parse_parameter_count(8) == 8_000_000_000.0
    assert parse_parameter_count("8B") == 8_000_000_000.0
    assert parse_parameter_count("70B") == 70_000_000_000.0
    assert parse_parameter_count("1.5B") == 1_500_000_000.0
    assert parse_parameter_count("350M") == 350_000_000.0
    assert parse_parameter_count("Llama-3-8B") == 8_000_000_000.0

    with pytest.raises(ValueError):
        parse_parameter_count("invalid-format")


def test_get_bpw():
    assert get_bpw("Q4_K_M") == 4.5
    assert get_bpw("q4_k_m") == 4.5
    assert get_bpw("Q8_0") == 8.5
    assert get_bpw("FP16") == 16.0
    assert get_bpw("BF16") == 16.0
    assert get_bpw("FP32") == 32.0
    assert get_bpw(4.5) == 4.5
    assert get_bpw("4.5") == 4.5

    with pytest.raises(ValueError):
        get_bpw("UNKNOWN_QUANT_XXX")


def test_calculate_model_vram_bytes():
    # 8B parameters at Q4_K_M (4.5 bpw) -> 8e9 * 4.5 / 8 = 4.5 GB
    vram_8b_q4 = calculate_model_vram_bytes("8B", "Q4_K_M")
    assert vram_8b_q4 == 4_500_000_000

    # 8B parameters at Q8_0 (8.5 bpw) -> 8e9 * 8.5 / 8 = 8.5 GB
    vram_8b_q8 = calculate_model_vram_bytes("8B", "Q8_0")
    assert vram_8b_q8 == 8_500_000_000

    # 8B parameters at FP16 (16 bpw) -> 8e9 * 16 / 8 = 16 GB
    vram_8b_fp16 = calculate_model_vram_bytes("8B", "FP16")
    assert vram_8b_fp16 == 16_000_000_000

    # With 10% overhead
    vram_overhead = calculate_model_vram_bytes("8B", "Q4_K_M", overhead_ratio=0.10)
    assert vram_overhead == 4_950_000_000


def test_calculate_kv_cache_bytes_explicit_architecture():
    # 2 * n_layers * n_heads_kv * d_head * context_length * bytes_per_elem
    # Example: 32 layers, 8 kv heads, d_head 128, context 2048, 2 bytes/elem (FP16)
    # = 2 * 32 * 8 * 128 * 2048 * 2 = 268,435,456 bytes (256 MB)
    kv_bytes = calculate_kv_cache_bytes(
        context_length=2048,
        n_layers=32,
        n_heads_kv=8,
        d_head=128,
        bytes_per_elem=2,
    )
    assert kv_bytes == 268_435_456

    # Context length 0 returns 0
    assert calculate_kv_cache_bytes(0, n_layers=32, n_heads_kv=8, d_head=128) == 0


def test_calculate_kv_cache_bytes_fallback():
    # Robust fallback: ~0.5 MB per 1024 context tokens for 7B-8B models
    kv_8b_1024 = fallback_kv_cache_bytes(1024, param_size="8B")
    assert kv_8b_1024 == int(FALLBACK_BYTES_PER_1024_8B)  # 524,288 bytes

    # Scaling with parameter count: 16B model should have 2x KV cache
    kv_16b_1024 = fallback_kv_cache_bytes(1024, param_size=16e9)
    assert kv_16b_1024 == int(FALLBACK_BYTES_PER_1024_8B * 2)

    # Scaling with context length: 2048 tokens should be 2x
    kv_8b_2048 = calculate_kv_cache_bytes(2048, param_size="8B")
    assert kv_8b_2048 == int(FALLBACK_BYTES_PER_1024_8B * 2)


def test_calculate_fit_entirely_fits():
    # Model 4 GB, Context KV cache ~1 GB, Available VRAM 8 GB -> fits!
    model_size = 4 * 1024 * 1024 * 1024  # 4 GiB
    vram = 8 * 1024 * 1024 * 1024  # 8 GiB
    fit = calculate_fit(
        model_size_bytes=model_size,
        context_length=2048,
        available_vram_bytes=vram,
    )
    assert fit.fits is True
    assert fit.fits_in_vram is True
    assert fit.gpu_offload_fraction == 1.0
    assert fit.exceeds_memory is False
    assert fit.headroom_bytes > 0
    assert fit.max_context_length > 2048


def test_calculate_fit_context_exceeds_vram():
    # Model 7 GB, Available VRAM 8 GB, massive context requiring 3 GB KV cache
    model_size = 7 * 1024 * 1024 * 1024
    vram = 8 * 1024 * 1024 * 1024
    kv_bytes = 3 * 1024 * 1024 * 1024  # 3 GB
    fit = calculate_fit(
        model_size_bytes=model_size,
        context_length=8192,
        available_vram_bytes=vram,
        kv_cache_bytes=kv_bytes,
    )
    assert fit.fits is False
    assert fit.exceeds_memory is True
    assert 0.0 < fit.gpu_offload_fraction < 1.0
    assert fit.headroom_bytes < 0
    # Model weights fit (7 GB < 8 GB), so some context fits in headroom:
    assert fit.max_context_length > 0


def test_calculate_fit_model_exceeds_vram():
    # Model 16 GB, Available VRAM 8 GB -> model weights alone exceed VRAM
    model_size = 16 * 1024 * 1024 * 1024
    vram = 8 * 1024 * 1024 * 1024
    fit = calculate_fit(
        model_size_bytes=model_size,
        context_length=4096,
        available_vram_bytes=vram,
    )
    assert fit.fits is False
    assert fit.exceeds_memory is True
    assert fit.gpu_offload_fraction < 0.6
    assert fit.headroom_bytes < 0
    assert fit.max_context_length == 0  # No room for context


def test_calculate_fit_zero_vram():
    fit = calculate_fit(
        model_size_bytes=4_000_000_000,
        context_length=2048,
        available_vram_bytes=0,
    )
    assert fit.fits is False
    assert fit.gpu_offload_fraction == 0.0
    assert fit.exceeds_memory is True
    assert fit.max_context_length == 0


def test_recommend_quantization():
    # 8B model with 6 GB available VRAM:
    # FP16 (16 GB) doesn't fit, Q8_0 (8.5 GB) doesn't fit, Q4_K_M (4.5 GB) fits!
    vram = 6 * 1000 * 1000 * 1000
    recommended = recommend_quantization("8B", available_vram_bytes=vram, context_length=2048)
    assert recommended in ("Q5_K_M", "Q4_K_M")

    # With only 1 GB VRAM, nothing fits:
    assert recommend_quantization("70B", available_vram_bytes=1000) is None


# --------------------------------------------------------------------------- #
# Tracing Tests
# --------------------------------------------------------------------------- #


def test_compute_percentile():
    assert compute_percentile([], 50) is None
    assert compute_percentile([42.0], 50) == 42.0
    assert compute_percentile([42.0], 99) == 42.0

    # Two items: [10.0, 20.0] -> P50 is 15.0
    assert compute_percentile([10.0, 20.0], 50.0) == 15.0

    # 100 values: 1 to 100
    data = [float(i) for i in range(1, 101)]
    assert compute_percentile(data, 50.0) == 50.5
    assert compute_percentile(data, 90.0) == 90.1
    assert compute_percentile(data, 95.0) == 95.05
    assert compute_percentile(data, 99.0) == 99.01


def test_request_tracer_lifecycle():
    tracer = RequestTracer(capacity=5)
    assert len(tracer) == 0

    # Empty summary
    empty_summary = tracer.get_summary()
    assert empty_summary.sample_count == 0
    assert empty_summary.ttft_p50 is None

    # Record 3 samples
    tracer.record_sample(ttft_ms=100.0, tok_per_sec=20.0, duration_ms=250.0)
    tracer.record_sample(ttft_ms=200.0, tok_per_sec=40.0, duration_ms=500.0)
    tracer.record_sample(ttft_ms=300.0, tok_per_sec=60.0, duration_ms=750.0)

    assert len(tracer) == 3
    summary = tracer.get_summary()
    assert summary.sample_count == 3
    assert summary.ttft_p50 == 200.0
    assert summary.ttft_min == 100.0
    assert summary.ttft_max == 300.0
    assert summary.ttft_mean == 200.0

    assert summary.tok_per_sec_p50 == 40.0
    assert summary.tok_per_sec_min == 20.0
    assert summary.tok_per_sec_max == 60.0

    assert summary.duration_p50 == 500.0

    # Clear
    tracer.clear()
    assert len(tracer) == 0


def test_request_tracer_circular_buffer_eviction():
    tracer = LatencyTracer(capacity=3)
    for i in range(1, 6):  # 1, 2, 3, 4, 5
        tracer.record_sample(
            ttft_ms=float(i * 10), tok_per_sec=float(i), duration_ms=float(i * 100)
        )

    assert len(tracer) == 3  # Capped at capacity 3
    samples = tracer.get_samples()
    # Oldest (1, 2) evicted, remaining are 3, 4, 5
    assert [s.tok_per_sec for s in samples] == [3.0, 4.0, 5.0]


def test_request_tracer_handles_none_values():
    tracer = RequestTracer(capacity=10)
    tracer.record_sample(ttft_ms=None, tok_per_sec=50.0, duration_ms=100.0)
    tracer.record_sample(ttft_ms=80.0, tok_per_sec=None, duration_ms=150.0)

    summary = tracer.get_summary()
    assert summary.sample_count == 2
    assert summary.ttft_p50 == 80.0
    assert summary.tok_per_sec_p50 == 50.0
    assert summary.duration_p50 == 125.0


def test_global_tracing_functions():
    clear()
    record_sample(ttft_ms=150.0, tok_per_sec=30.0, duration_ms=400.0)
    summary = get_summary()
    assert summary.sample_count >= 1
    assert summary.ttft_p50 == 150.0
    clear()
    assert get_summary().sample_count == 0


def test_request_tracer_thread_safety():
    tracer = RequestTracer(capacity=500)

    def worker(worker_id: int):
        for i in range(50):
            tracer.record_sample(
                ttft_ms=float(worker_id * 10 + i),
                tok_per_sec=float(i),
                duration_ms=float(i * 5),
            )
            _ = tracer.get_summary()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, w) for w in range(5)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert len(tracer) == 250
    summary = tracer.get_summary()
    assert summary.sample_count == 250
    assert summary.ttft_p50 is not None
