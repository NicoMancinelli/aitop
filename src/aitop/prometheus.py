"""Prometheus text exposition of a `SystemSnapshot`.

A pure function of the snapshot contract — same rule as the neofetch view.
Mounted at `GET /metrics` on `aitop serve`, and also available one-shot via
`aitop metrics`.
"""

from __future__ import annotations

from aitop.models import SystemSnapshot
from aitop.version import __version__


def render_prometheus(snapshot: SystemSnapshot) -> str:
    """Render a Prometheus 0.0.4 text exposition from one snapshot."""
    lines: list[str] = []
    node = _esc(snapshot.node)

    def help_type(name: str, help_text: str, kind: str = "gauge") -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")

    help_type("aitop_info", "aitop build metadata")
    lines.append(f'aitop_info{{version="{_esc(__version__)}",node="{node}"}} 1')

    help_type("aitop_scrape_duration_ms", "Wall-clock time of the last collection")
    duration = snapshot.duration_ms if snapshot.duration_ms is not None else 0
    lines.append(f'aitop_scrape_duration_ms{{node="{node}"}} {duration}')

    hw = snapshot.hardware
    cpu = hw.cpu
    help_type("aitop_cpu_load_percent", "CPU utilisation 0-100")
    lines.append(f'aitop_cpu_load_percent{{node="{node}"}} {cpu.load_percent}')
    if cpu.power_watts is not None:
        help_type("aitop_cpu_power_watts", "CPU package power draw")
        lines.append(f'aitop_cpu_power_watts{{node="{node}"}} {cpu.power_watts}')
    if cpu.temperature_c is not None:
        help_type("aitop_cpu_temperature_celsius", "CPU die temperature")
        lines.append(f'aitop_cpu_temperature_celsius{{node="{node}"}} {cpu.temperature_c}')

    mem = hw.memory
    help_type("aitop_memory_bytes", "Memory pool size and usage")
    lines.append(f'aitop_memory_bytes{{node="{node}",state="total"}} {mem.total_bytes}')
    lines.append(f'aitop_memory_bytes{{node="{node}",state="used"}} {mem.used_bytes}')
    lines.append(f'aitop_memory_bytes{{node="{node}",state="available"}} {mem.available_bytes}')
    help_type("aitop_memory_used_percent", "Memory utilisation 0-100")
    lines.append(f'aitop_memory_used_percent{{node="{node}"}} {mem.used_percent}')

    if hw.gpus:
        help_type("aitop_gpu_utilization_percent", "GPU compute utilisation 0-100")
        help_type("aitop_gpu_vram_bytes", "GPU memory pool")
        help_type("aitop_gpu_power_watts", "GPU power draw")
        help_type("aitop_gpu_temperature_celsius", "GPU temperature")
    for gpu in hw.gpus:
        labels = (
            f'node="{node}",gpu="{gpu.index}",name="{_esc(gpu.name)}",vendor="{gpu.vendor.value}"'
        )
        if gpu.utilization_percent is not None:
            lines.append(f"aitop_gpu_utilization_percent{{{labels}}} {gpu.utilization_percent}")
        if gpu.vram_total_bytes is not None:
            lines.append(f'aitop_gpu_vram_bytes{{{labels},state="total"}} {gpu.vram_total_bytes}')
        if gpu.vram_used_bytes is not None:
            lines.append(f'aitop_gpu_vram_bytes{{{labels},state="used"}} {gpu.vram_used_bytes}')
        if gpu.power_watts is not None:
            lines.append(f"aitop_gpu_power_watts{{{labels}}} {gpu.power_watts}")
        if gpu.temperature_c is not None:
            lines.append(f"aitop_gpu_temperature_celsius{{{labels}}} {gpu.temperature_c}")

    if hw.total_power_watts is not None:
        help_type("aitop_total_power_watts", "Aggregate system power draw")
        lines.append(f'aitop_total_power_watts{{node="{node}"}} {hw.total_power_watts}')

    help_type("aitop_engine_up", "1 if the engine answered a probe")
    help_type("aitop_engine_latency_ms", "Round-trip probe latency")
    help_type("aitop_engine_models_total", "Models known to the engine")
    help_type("aitop_engine_loaded_total", "Models currently resident")
    help_type("aitop_engine_resident_bytes", "Bytes of resident model weights")
    help_type("aitop_engine_tokens_per_second", "Recent generation throughput")
    help_type("aitop_engine_active_requests", "In-flight inference requests")
    help_type("aitop_engine_queue_depth", "Waiting inference requests")
    for eng in snapshot.engines:
        elabels = f'node="{node}",engine="{eng.kind.value}",name="{_esc(eng.name)}"'
        lines.append(f"aitop_engine_up{{{elabels}}} {1 if eng.online else 0}")
        if eng.latency_ms is not None:
            lines.append(f"aitop_engine_latency_ms{{{elabels}}} {eng.latency_ms}")
        lines.append(f"aitop_engine_models_total{{{elabels}}} {len(eng.models)}")
        lines.append(f"aitop_engine_loaded_total{{{elabels}}} {len(eng.loaded)}")
        lines.append(f"aitop_engine_resident_bytes{{{elabels}}} {eng.resident_bytes}")
        if eng.stats.tokens_per_second is not None:
            lines.append(
                f"aitop_engine_tokens_per_second{{{elabels}}} {eng.stats.tokens_per_second}"
            )
        lines.append(f"aitop_engine_active_requests{{{elabels}}} {eng.stats.active_requests}")
        lines.append(f"aitop_engine_queue_depth{{{elabels}}} {eng.stats.queue_depth}")

    if snapshot.tailscale.available:
        help_type("aitop_tailscale_up", "1 if the Tailscale backend is running")
        lines.append(
            f'aitop_tailscale_up{{node="{node}"}} {1 if snapshot.tailscale.running else 0}'
        )
        help_type("aitop_tailscale_peers", "Tailnet peer count")
        lines.append(f'aitop_tailscale_peers{{node="{node}"}} {snapshot.tailscale.peer_count}')

    lines.append("")  # Prometheus wants a trailing newline
    return "\n".join(lines)


def _esc(value: str) -> str:
    """Escape a label value for the Prometheus text format."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
