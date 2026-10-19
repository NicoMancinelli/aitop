"""Model inference latency and throughput benchmarking.

Tests an engine / model using standard chat completions or completions,
measuring time-to-first-token (TTFT) and token generation speed (tok/s).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from aitop.engines.base import BaseEngine
from aitop.models import BenchmarkResult, EngineKind


async def run_benchmark(
    engine: BaseEngine | str | EngineKind,
    model_id: str,
    *,
    prompt: str = "Explain the difference between a process and a thread in 3 sentences.",
    max_tokens: int = 64,
    timeout_s: float = 60.0,
    config: Any = None,
) -> BenchmarkResult:
    """Benchmark inference latency and throughput for a model on an engine."""
    if isinstance(engine, (str, EngineKind)):
        from aitop.config import Config
        from aitop.engines.registry import EngineRegistry

        reg = EngineRegistry(config or Config())
        adapter = None
        target_kind = engine.value if isinstance(engine, EngineKind) else str(engine).lower()
        for e in reg.engines:
            if e.kind.value == target_kind or e.name.lower() == target_kind:
                adapter = e
                break
        if adapter is None:
            return BenchmarkResult(
                model=model_id,
                engine=EngineKind.UNKNOWN,
                success=False,
                error=f"No engine adapter found for '{engine}'",
            )
        try:
            return await run_benchmark(
                adapter,
                model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
        finally:
            await reg.aclose()

    url = f"{engine.base_url}"
    client = engine.client
    start_time = time.perf_counter()
    ttft_ms: float | None = None
    completion_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        # 1. Try Ollama native endpoint if Ollama
        if engine.kind == EngineKind.OLLAMA:
            gen_url = f"{url}/api/generate"
            body = {
                "model": model_id,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": max_tokens, "temperature": 0.0},
            }
            async with client.stream(
                "POST", gen_url, json=body, timeout=httpx.Timeout(timeout_s, connect=5.0)
            ) as response:
                if response.status_code >= 400:
                    text = await response.aread()
                    return BenchmarkResult(
                        model=model_id,
                        engine=engine.kind,
                        success=False,
                        error=f"HTTP {response.status_code}: {text.decode('utf-8', errors='replace')}",
                    )
                import json

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start_time) * 1000.0
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    completion_text += chunk.get("response", "")
                    if chunk.get("done"):
                        prompt_tokens = chunk.get("prompt_eval_count") or len(prompt.split())
                        completion_tokens = chunk.get("eval_count") or len(completion_text.split())

        # 2. Try OpenAI-compatible chat completions
        else:
            chat_url = f"{url}/v1/chat/completions"
            body = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            }
            async with client.stream(
                "POST", chat_url, json=body, timeout=httpx.Timeout(timeout_s, connect=5.0)
            ) as response:
                if response.status_code in (404, 405):
                    # Fallback to /v1/completions
                    comp_url = f"{url}/v1/completions"
                    comp_body = {
                        "model": model_id,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "stream": True,
                    }
                    async with client.stream(
                        "POST",
                        comp_url,
                        json=comp_body,
                        timeout=httpx.Timeout(timeout_s, connect=5.0),
                    ) as comp_resp:
                        if comp_resp.status_code >= 400:
                            text = await comp_resp.aread()
                            return BenchmarkResult(
                                model=model_id,
                                engine=engine.kind,
                                success=False,
                                error=f"HTTP {comp_resp.status_code}: {text.decode('utf-8', errors='replace')}",
                            )
                        async for line in comp_resp.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start_time) * 1000.0
                            raw = line.removeprefix("data:").strip()
                            if raw == "[DONE]":
                                break
                            import json

                            try:
                                chunk = json.loads(raw)
                                completion_text += chunk["choices"][0].get("text", "")
                            except Exception:
                                continue
                elif response.status_code >= 400:
                    text = await response.aread()
                    return BenchmarkResult(
                        model=model_id,
                        engine=engine.kind,
                        success=False,
                        error=f"HTTP {response.status_code}: {text.decode('utf-8', errors='replace')}",
                    )
                else:
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start_time) * 1000.0
                        raw = line.removeprefix("data:").strip()
                        if raw == "[DONE]":
                            break
                        import json

                        try:
                            chunk = json.loads(raw)
                            delta = chunk["choices"][0].get("delta", {})
                            completion_text += delta.get("content", "")
                        except Exception:
                            continue

    except Exception as exc:
        return BenchmarkResult(
            model=model_id,
            engine=engine.kind,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    total_duration_s = max(time.perf_counter() - start_time, 0.001)

    # Estimate token counts if runtime did not return exact tokens
    if not prompt_tokens:
        prompt_tokens = max(1, len(prompt.split()))
    if not completion_tokens:
        # Approximate tokens if exact count isn't returned (~0.75 words per token)
        words = len(completion_text.split())
        completion_tokens = max(1, int(words * 1.3)) if words else 1

    gen_duration = total_duration_s - ((ttft_ms or 0) / 1000.0)
    gen_duration = max(gen_duration, 0.001)
    tok_per_sec = completion_tokens / gen_duration

    return BenchmarkResult(
        model=model_id,
        engine=engine.kind,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_duration_s=total_duration_s,
        ttft_ms=ttft_ms,
        tokens_per_second=tok_per_sec,
        success=True,
    )
