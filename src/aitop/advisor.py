"""VRAM requirements and KV cache capacity advisor.

Calculates model memory footprint for parameter counts and quantization depths,
evaluates KV cache growth across context windows, and determines whether
models fit in available hardware VRAM or require CPU offloading.
"""

from __future__ import annotations

import re
from typing import Final

from aitop.models import ModelFitResult

# --------------------------------------------------------------------------- #
# Quantization Bit-Depth Profiles (bits per weight)
# --------------------------------------------------------------------------- #

QUANT_BPW: Final[dict[str, float]] = {
    # Full / half precision
    "FP32": 32.0,
    "F32": 32.0,
    "FP16": 16.0,
    "F16": 16.0,
    "BF16": 16.0,
    "BFLOAT16": 16.0,
    # 8-bit
    "Q8_0": 8.5,
    "Q8_1": 9.0,
    "FP8": 8.0,
    "F8": 8.0,
    "INT8": 8.0,
    # 6-bit
    "Q6_K": 6.6,
    "Q6_K_L": 6.6,
    # 5-bit
    "Q5_K_M": 5.5,
    "Q5_K_S": 5.5,
    "Q5_0": 5.5,
    "Q5_1": 6.0,
    # 4-bit
    "Q4_K_M": 4.5,
    "Q4_K_S": 4.5,
    "Q4_0": 4.5,
    "Q4_1": 5.0,
    "IQ4_NL": 4.5,
    "IQ4_XS": 4.25,
    "INT4": 4.0,
    # 3-bit
    "Q3_K_L": 4.27,
    "Q3_K_M": 3.9,
    "Q3_K_S": 3.4,
    "IQ3_M": 3.7,
    "IQ3_S": 3.44,
    "IQ3_XXS": 3.06,
    # 2-bit
    "Q2_K": 2.6,
    "IQ2_M": 2.7,
    "IQ2_S": 2.5,
    "IQ2_XS": 2.4,
    "IQ2_XXS": 2.2,
    # 1-bit
    "IQ1_S": 1.56,
    "IQ1_M": 1.75,
}

# Baseline for robust fallback: ~0.5 MB (524,288 bytes) per 1024 context tokens for an 8B model.
FALLBACK_BASELINE_PARAMS: Final[float] = 8_000_000_000.0
FALLBACK_BYTES_PER_1024_8B: Final[float] = 512.0 * 1024.0  # 524,288 bytes (~0.5 MB)
FALLBACK_BYTES_PER_TOKEN_8B: Final[float] = FALLBACK_BYTES_PER_1024_8B / 1024.0  # 512.0 bytes/token

_PARAM_REGEX = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([kKmMgGtTbB])?$", re.IGNORECASE)


def parse_parameter_count(param_size: float | int | str) -> float:
    """Parse parameter specifications like '7B', '8B', '70B', '350M' into count."""
    if isinstance(param_size, (int, float)):
        # If passed a small scalar like 7 or 8, treat as billions
        if 0 < param_size <= 1000:
            return float(param_size) * 1e9
        return float(param_size)

    s = str(param_size).strip()
    match = _PARAM_REGEX.match(s)
    if not match:
        extracted = re.search(r"(\d+(?:\.\d+)?)\s*([bBmMkKtT])\b", s)
        if extracted:
            val = float(extracted.group(1))
            unit = extracted.group(2).upper()
            mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[unit]
            return val * mult
        raise ValueError(f"Cannot parse parameter size: {param_size}")

    val = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(unit, 1e9)
    return val * mult


def get_bpw(quantization: str | float) -> float:
    """Resolve quantization name or raw float to bits per weight (bpw)."""
    if isinstance(quantization, (int, float)):
        return float(quantization)

    cleaned = quantization.strip().upper().replace("-", "_")
    if cleaned in QUANT_BPW:
        return QUANT_BPW[cleaned]

    # Try numeric string e.g. "4.5"
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Prefix match common quantizations
    for prefix, bpw in (
        ("Q4", 4.5),
        ("Q8", 8.5),
        ("FP16", 16.0),
        ("BF16", 16.0),
        ("Q5", 5.5),
        ("Q6", 6.6),
        ("Q3", 3.9),
        ("Q2", 2.6),
        ("FP32", 32.0),
    ):
        if cleaned.startswith(prefix):
            return bpw

    raise ValueError(f"Unsupported or unknown quantization profile: {quantization}")


# --------------------------------------------------------------------------- #
# VRAM & KV Cache Sizing
# --------------------------------------------------------------------------- #


def calculate_model_vram_bytes(
    param_size: float | int | str,
    quantization: str | float = "Q4_K_M",
    overhead_ratio: float = 0.0,
    fixed_overhead_bytes: int = 0,
) -> int:
    """Calculate VRAM requirements in bytes for given parameter size and quantization.

    Args:
        param_size: Parameter count (e.g., 8e9, '8B', '70B', '350M').
        quantization: Quantization format (e.g., 'Q4_K_M', 'Q8_0', 'FP16') or raw bpw.
        overhead_ratio: Optional proportional overhead factor (e.g., 0.05 for 5% CUDA context).
        fixed_overhead_bytes: Optional fixed buffer memory in bytes.

    Returns:
        Estimated VRAM bytes required for model weights.
    """
    params = parse_parameter_count(param_size)
    bpw = get_bpw(quantization)
    weight_bytes = int((params * bpw) / 8.0)
    total = int(weight_bytes * (1.0 + overhead_ratio)) + fixed_overhead_bytes
    return max(0, total)


def calculate_kv_cache_bytes(
    context_length: int,
    n_layers: int | None = None,
    n_heads_kv: int | None = None,
    d_head: int | None = None,
    bytes_per_elem: int = 2,
    param_size: float | int | str | None = None,
) -> int:
    """Calculate KV cache memory per context length.

    Formula when architecture parameters are known:
        2 * n_layers * n_heads_kv * d_head * context_length * bytes_per_elem

    Fallback estimation:
        ~0.5 MB per 1024 context tokens for 7B-8B models, scaling with parameter size.
    """
    if context_length <= 0:
        return 0

    if n_layers is not None and n_heads_kv is not None and d_head is not None:
        return int(2 * n_layers * n_heads_kv * d_head * context_length * bytes_per_elem)

    return fallback_kv_cache_bytes(context_length=context_length, param_size=param_size)


def fallback_kv_cache_bytes(
    context_length: int,
    param_size: float | int | str | None = None,
) -> int:
    """Robust fallback estimation: ~0.5 MB per 1024 context tokens for 7B-8B models,

    scaling proportionally with parameter size.
    """
    if context_length <= 0:
        return 0

    params = (
        parse_parameter_count(param_size) if param_size is not None else FALLBACK_BASELINE_PARAMS
    )
    scale = params / FALLBACK_BASELINE_PARAMS
    bytes_per_token = FALLBACK_BYTES_PER_TOKEN_8B * scale
    return int(context_length * bytes_per_token)


# --------------------------------------------------------------------------- #
# Fit & Headroom Evaluation
# --------------------------------------------------------------------------- #


def calculate_fit(
    model_size_bytes: int,
    context_length: int,
    available_vram_bytes: int,
    *,
    kv_cache_bytes: int | None = None,
    param_size: float | int | str | None = None,
    n_layers: int | None = None,
    n_heads_kv: int | None = None,
    d_head: int | None = None,
    bytes_per_elem: int = 2,
) -> ModelFitResult:
    """Determine whether a model and its KV cache fit entirely in VRAM.

    Returns:
        ModelFitResult indicating:
        - `fits`: whether the model fits entirely in VRAM with requested context.
        - `gpu_offload_fraction`: estimated GPU offload fraction (0.0 to 1.0).
        - `exceeds_memory`: whether required memory exceeds available VRAM.
        - `max_context_length`: maximum context tokens that can fit in remaining headroom.
    """
    # 1. Determine KV cache bytes and bytes per token
    if kv_cache_bytes is not None:
        kv_bytes = max(0, kv_cache_bytes)
        bytes_per_token = (
            (kv_bytes / context_length) if context_length > 0 else FALLBACK_BYTES_PER_TOKEN_8B
        )
    elif n_layers is not None and n_heads_kv is not None and d_head is not None:
        kv_bytes = calculate_kv_cache_bytes(
            context_length=context_length,
            n_layers=n_layers,
            n_heads_kv=n_heads_kv,
            d_head=d_head,
            bytes_per_elem=bytes_per_elem,
        )
        bytes_per_token = float(2 * n_layers * n_heads_kv * d_head * bytes_per_elem)
    elif param_size is not None:
        kv_bytes = calculate_kv_cache_bytes(context_length, param_size=param_size)
        params = parse_parameter_count(param_size)
        bytes_per_token = FALLBACK_BYTES_PER_TOKEN_8B * (params / FALLBACK_BASELINE_PARAMS)
    else:
        # Estimate parameter count from model size assuming typical ~4.5 bpw (0.5625 B/param)
        if model_size_bytes > 0:
            est_params = (model_size_bytes * 8.0) / 4.5
            kv_bytes = calculate_kv_cache_bytes(context_length, param_size=est_params)
            bytes_per_token = FALLBACK_BYTES_PER_TOKEN_8B * (est_params / FALLBACK_BASELINE_PARAMS)
        else:
            kv_bytes = calculate_kv_cache_bytes(context_length)
            bytes_per_token = FALLBACK_BYTES_PER_TOKEN_8B

    total_required = model_size_bytes + kv_bytes
    headroom = available_vram_bytes - total_required

    # Calculate max context that can fit in remaining headroom after model weights
    headroom_for_context = available_vram_bytes - model_size_bytes
    if headroom_for_context > 0 and bytes_per_token > 0:
        max_context = int(headroom_for_context // bytes_per_token)
    else:
        max_context = 0

    # Fit and offload evaluation
    if available_vram_bytes <= 0:
        fits = False
        offload_fraction = 0.0
        exceeds_memory = True
    elif available_vram_bytes >= total_required:
        fits = True
        offload_fraction = 1.0
        exceeds_memory = False
    else:
        fits = False
        exceeds_memory = True
        offload_fraction = (
            round(max(0.0, min(1.0, available_vram_bytes / total_required)), 4)
            if total_required > 0
            else 0.0
        )

    return ModelFitResult(
        fits=fits,
        gpu_offload_fraction=offload_fraction,
        exceeds_memory=exceeds_memory,
        max_context_length=max_context,
        model_size_bytes=model_size_bytes,
        kv_cache_bytes=kv_bytes,
        total_required_bytes=total_required,
        available_vram_bytes=available_vram_bytes,
        headroom_bytes=headroom,
    )


def recommend_quantization(
    param_size: float | int | str,
    available_vram_bytes: int,
    context_length: int = 4096,
    candidates: list[str] | None = None,
) -> str | None:
    """Recommend the highest quality quantization level that fits entirely in VRAM."""
    search_order = candidates or ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
    for quant in search_order:
        model_bytes = calculate_model_vram_bytes(param_size, quantization=quant)
        fit = calculate_fit(
            model_size_bytes=model_bytes,
            context_length=context_length,
            available_vram_bytes=available_vram_bytes,
            param_size=param_size,
        )
        if fit.fits:
            return quant
    return None


__all__ = [
    "FALLBACK_BASELINE_PARAMS",
    "FALLBACK_BYTES_PER_1024_8B",
    "FALLBACK_BYTES_PER_TOKEN_8B",
    "QUANT_BPW",
    "calculate_fit",
    "calculate_kv_cache_bytes",
    "calculate_model_vram_bytes",
    "fallback_kv_cache_bytes",
    "get_bpw",
    "parse_parameter_count",
    "recommend_quantization",
]
