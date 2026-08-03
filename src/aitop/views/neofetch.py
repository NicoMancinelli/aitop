"""The "AI neofetch" view.

A pure function of a `SystemSnapshot`: hand it a snapshot from any source —
local collector, remote fleet node, a JSON file replayed from disk — and it
renders. It imports nothing from `aitop.hardware` or `aitop.engines`.
"""

from __future__ import annotations

import getpass

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from aitop.models import (
    BindScope,
    EngineSnapshot,
    EngineState,
    GPUSnapshot,
    SystemSnapshot,
    TailscaleStatus,
)
from aitop.selfupdate import UpdateStatus
from aitop.utils.fmt import (
    bytes_human,
    celsius,
    duration_human,
    heat_color,
    percent,
    ratio_bar,
    relative_time,
    truncate,
    watts,
)
from aitop.views.logo import render_logo

LABEL_STYLE = "bold bright_cyan"
VALUE_STYLE = "white"
DIM = "grey50"

STATE_MARK = {
    EngineState.ONLINE: ("●", "bold green"),
    EngineState.DEGRADED: ("◐", "bold yellow"),
    EngineState.OFFLINE: ("○", DIM),
    EngineState.UNKNOWN: ("?", DIM),
}

SCOPE_STYLE = {
    BindScope.LOOPBACK: "green",
    BindScope.LAN: "yellow",
    BindScope.TAILSCALE: "bright_magenta",
    BindScope.OTHER: DIM,
}


# Column budgets. Below each threshold the corresponding detail columns are
# dropped rather than letting Rich ellipsize every cell into uselessness.
WIDE = 112
MEDIUM = 92


class NeofetchView:
    """A width-adaptive renderable wrapping one snapshot.

    Layout decisions are deferred to render time so the view reflows when the
    terminal is resized mid-`--watch`, instead of being baked in at build time.
    """

    def __init__(
        self,
        snapshot: SystemSnapshot,
        *,
        show_logo: bool = True,
        color: bool = True,
        show_offline: bool = False,
        update: UpdateStatus | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.show_logo = show_logo
        self.color = color
        self.show_offline = show_offline
        self.update = update

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width
        snapshot = self.snapshot

        yield _header_block(snapshot, show_logo=self.show_logo and width >= 72, color=self.color)

        engines = snapshot.engines if self.show_offline else snapshot.online_engines
        yield Text()
        yield _engine_table(engines, width=width, show_offline=self.show_offline)

        if snapshot.all_loaded:
            yield Text()
            yield _loaded_table(snapshot, width=width)

        notes = _notes(snapshot)
        if notes:
            yield Text()
            yield notes

        banner = _update_banner(self.update)
        if banner is not None:
            yield Text()
            yield banner


def render_neofetch(
    snapshot: SystemSnapshot,
    *,
    show_logo: bool = True,
    color: bool = True,
    show_offline: bool = False,
    update: UpdateStatus | None = None,
) -> RenderableType:
    """Build the full renderable. Caller decides where and how wide to print it."""
    return NeofetchView(
        snapshot,
        show_logo=show_logo,
        color=color,
        show_offline=show_offline,
        update=update,
    )


def _update_banner(update: UpdateStatus | None) -> RenderableType | None:
    """One quiet line when a newer release exists. Silent otherwise."""
    if update is None or not update.available:
        return None
    text = Text()
    text.append("  ⬆ ", style="bold green")
    text.append(f"aitop {update.latest} available", style="bold white")
    text.append(f" (you have {update.current}) — run ", style=DIM)
    text.append("aitop update", style="bold bright_cyan")
    return text


# --------------------------------------------------------------------------- #
# Header: logo + facts
# --------------------------------------------------------------------------- #


def _header_block(snapshot: SystemSnapshot, *, show_logo: bool, color: bool) -> RenderableType:
    facts = _facts(snapshot)
    if not show_logo:
        return facts

    grid = Table.grid(padding=(0, 3))
    grid.add_column(vertical="top")
    grid.add_column(vertical="top")
    grid.add_row(render_logo(color=color), facts)
    return grid


def _facts(snapshot: SystemSnapshot) -> RenderableType:
    hw = snapshot.hardware
    host = hw.host

    rows = Table.grid(padding=(0, 2))
    rows.add_column(style=LABEL_STYLE, justify="left", no_wrap=True)
    rows.add_column(style=VALUE_STYLE, overflow="fold")

    def add(label: str, value: RenderableType | str | None) -> None:
        if value is None:
            return
        rows.add_row(label, value)

    title = Text(f"{getpass.getuser()}@{host.hostname}", style="bold bright_magenta")
    rule = Text("─" * max(20, len(title.plain)), style=DIM)

    add("OS", f"{host.os_name} {host.os_version}".strip())
    add("Kernel", host.kernel or None)
    add("Uptime", duration_human(host.uptime_seconds))
    add("CPU", _cpu_line(snapshot))
    add("", _meter(hw.cpu.load_percent / 100 if hw.cpu.load_percent is not None else None))
    add("Memory", _memory_line(snapshot))
    add("", _meter(hw.memory.used_percent / 100))

    for gpu in hw.gpus:
        add("GPU" if gpu.index == 0 else f"GPU {gpu.index}", _gpu_line(gpu))
        vram = _vram_line(gpu)
        if vram is not None:
            add("VRAM" if not gpu.unified_memory else "GPU mem", vram)
            fraction = gpu.vram_used_percent
            add("", _meter(fraction / 100 if fraction is not None else None))

    add("Power", _power_line(snapshot))
    add("Network", _tailscale_line(snapshot.tailscale))
    add("Runtimes", _runtime_summary(snapshot))

    return Group(title, rule, rows)


def _cpu_line(snapshot: SystemSnapshot) -> str:
    cpu = snapshot.hardware.cpu
    parts = [cpu.model]
    if cpu.performance_cores and cpu.efficiency_cores:
        parts.append(f"({cpu.performance_cores}P + {cpu.efficiency_cores}E)")
    elif cpu.logical_cores:
        physical = f"{cpu.physical_cores}c/" if cpu.physical_cores else ""
        parts.append(f"({physical}{cpu.logical_cores}t)")
    parts.append(f"@ {percent(cpu.load_percent)}")
    if cpu.temperature_c is not None:
        parts.append(f"· {celsius(cpu.temperature_c)}")
    return " ".join(parts)


def _memory_line(snapshot: SystemSnapshot) -> str:
    mem = snapshot.hardware.memory
    label = "unified" if mem.unified else "RAM"
    line = (
        f"{bytes_human(mem.used_bytes)} / {bytes_human(mem.total_bytes)} "
        f"({percent(mem.used_percent)}, {label})"
    )
    if mem.swap_used_bytes:
        line += f"  swap {bytes_human(mem.swap_used_bytes)}"
    return line


def _gpu_line(gpu: GPUSnapshot) -> str:
    parts = [gpu.name]
    if gpu.core_count:
        parts.append(f"({gpu.core_count} cores)")
    if gpu.api_version:
        parts.append(f"· {gpu.api_version}")
    if gpu.driver_version:
        parts.append(f"· driver {gpu.driver_version}")
    if gpu.utilization_percent is not None:
        parts.append(f"· {percent(gpu.utilization_percent)} busy")
    if gpu.temperature_c is not None:
        parts.append(f"· {celsius(gpu.temperature_c)}")
    return " ".join(parts)


def _vram_line(gpu: GPUSnapshot) -> str | None:
    if gpu.vram_total_bytes is None:
        return None
    used = bytes_human(gpu.vram_used_bytes) if gpu.vram_used_bytes is not None else "—"
    line = f"{used} / {bytes_human(gpu.vram_total_bytes)}"
    fraction = gpu.vram_used_percent
    if fraction is not None:
        line += f" ({percent(fraction)})"
    if gpu.unified_memory:
        line += "  shared with system"
    return line


def _power_line(snapshot: SystemSnapshot) -> str | None:
    hw = snapshot.hardware
    parts: list[str] = []
    if hw.total_power_watts is not None:
        parts.append(f"package {watts(hw.total_power_watts)}")
    if hw.cpu.power_watts is not None:
        parts.append(f"cpu {watts(hw.cpu.power_watts)}")
    gpu_power = next((g.power_watts for g in hw.gpus if g.power_watts is not None), None)
    if gpu_power is not None:
        parts.append(f"gpu {watts(gpu_power)}")
    return " · ".join(parts) if parts else None


def _tailscale_line(ts: TailscaleStatus) -> str | None:
    if not ts.available:
        return None
    if not ts.running:
        return "Tailscale installed, not connected"
    bits = [f"Tailscale {ts.ipv4 or '—'}"]
    if ts.hostname:
        bits.append(f"({ts.hostname})")
    if ts.tailnet:
        bits.append(f"· {ts.tailnet}")
    bits.append(f"· {ts.peer_count} peers")
    return " ".join(bits)


def _runtime_summary(snapshot: SystemSnapshot) -> str:
    online = snapshot.online_engines
    loaded = snapshot.all_loaded
    if not online:
        return "no AI runtimes detected"
    resident = sum(m.size_bytes or 0 for m in loaded)
    summary = f"{len(online)} online · {len(loaded)} model(s) resident"
    if resident:
        summary += f" · {bytes_human(resident)}"
    tps = next(
        (e.stats.tokens_per_second for e in online if e.stats.tokens_per_second is not None),
        None,
    )
    if tps is not None:
        summary += f" · {tps:.1f} tok/s"
    return summary


def _meter(fraction: float | None, width: int = 28) -> Text:
    return Text(ratio_bar(fraction, width=width), style=heat_color(fraction))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def _engine_table(
    engines: list[EngineSnapshot], *, width: int, show_offline: bool
) -> RenderableType:
    if not engines:
        hint = Text(
            "No AI runtimes reachable. Start Ollama or LM Studio, or add endpoints to "
            "~/.config/aitop/config.yaml",
            style="yellow",
        )
        return Group(Rule("Runtimes", style=DIM, align="left"), hint)

    detail = width >= WIDE
    medium = width >= MEDIUM

    table = Table(box=None, pad_edge=False, expand=False, padding=(0, 2))
    table.add_column("", width=1)
    table.add_column("RUNTIME", style="bold white", no_wrap=True)
    table.add_column("VERSION", style=DIM, no_wrap=True)
    table.add_column("ENDPOINT", no_wrap=True)
    if medium:
        table.add_column("SCOPE", no_wrap=True)
    table.add_column("MODELS", justify="right", no_wrap=True)
    table.add_column("LOADED", justify="right", no_wrap=True)
    if detail:
        table.add_column("TOK/S", justify="right", no_wrap=True)
        table.add_column("PID", justify="right", style=DIM, no_wrap=True)
        table.add_column("MANAGED", style=DIM, no_wrap=True)

    for engine in engines:
        mark, mark_style = STATE_MARK.get(engine.state, ("?", DIM))
        binding = engine.binding
        scope = binding.scope if binding else BindScope.OTHER
        cells: list[RenderableType] = [
            Text(mark, style=mark_style),
            engine.name,
            engine.version or "—",
            str(binding) if binding else "—",
        ]
        if medium:
            cells.append(Text(scope.value, style=SCOPE_STYLE.get(scope, DIM)))
        cells.append(str(len(engine.models)) if engine.models else "—")
        cells.append(Text(str(len(engine.loaded)), style="green" if engine.loaded else DIM))
        if detail:
            tps = engine.stats.tokens_per_second
            cells.append(f"{tps:.1f}" if tps is not None else "—")
            cells.append(str(engine.pid) if engine.pid else "—")
            cells.append(engine.managed_by or "—")
        table.add_row(*cells)

    blocks: list[RenderableType] = [Rule("Runtimes", style=DIM, align="left"), table]
    for engine in engines:
        if engine.error:
            blocks.append(Text(f"  ! {engine.name}: {engine.error}", style="yellow"))
    if not show_offline:
        blocks.append(Text("  (offline endpoints hidden — use --all)", style=DIM))
    return Group(*blocks)


def _loaded_table(snapshot: SystemSnapshot, *, width: int) -> RenderableType:
    detail = width >= WIDE
    medium = width >= MEDIUM
    name_width = 34 if detail else (26 if medium else 20)

    table = Table(box=None, pad_edge=False, expand=False, padding=(0, 2))
    table.add_column("MODEL", style="bold white", no_wrap=True)
    table.add_column("RUNTIME", style=DIM, no_wrap=True)
    if medium:
        table.add_column("PARAMS", justify="right", no_wrap=True)
    table.add_column("QUANT", no_wrap=True)
    table.add_column("SIZE", justify="right", no_wrap=True)
    table.add_column("ON GPU", justify="right", no_wrap=True)
    table.add_column("CTX", justify="right", no_wrap=True)
    if detail:
        table.add_column("EXPIRES", style=DIM, no_wrap=True)

    for model in snapshot.all_loaded:
        gpu_fraction = model.gpu_fraction
        gpu_cell = (
            Text(percent(gpu_fraction * 100), style=_offload_style(gpu_fraction))
            if gpu_fraction is not None
            else Text("—", style=DIM)
        )
        cells: list[RenderableType] = [
            truncate(model.name, name_width),
            model.engine.value,
        ]
        if medium:
            cells.append(model.parameter_size or "—")
        cells.extend(
            [
                model.quantization or "—",
                bytes_human(model.size_bytes),
                gpu_cell,
                f"{model.context_length:,}" if model.context_length else "—",
            ]
        )
        if detail:
            cells.append(relative_time(model.expires_at))
        table.add_row(*cells)

    return Group(Rule("Resident models", style=DIM, align="left"), table)


def _offload_style(fraction: float) -> str:
    if fraction >= 0.999:
        return "green"
    if fraction >= 0.5:
        return "yellow"
    return "red"


def _notes(snapshot: SystemSnapshot) -> RenderableType | None:
    notes = list(snapshot.hardware.degraded)
    if not notes:
        return None
    lines = [Text(f"  · {note}", style=DIM) for note in notes]
    return Group(Rule("Notes", style=DIM, align="left"), *lines)


def print_neofetch(
    snapshot: SystemSnapshot,
    console: Console | None = None,
    **kwargs: object,
) -> None:
    (console or Console()).print(render_neofetch(snapshot, **kwargs))  # type: ignore[arg-type]
