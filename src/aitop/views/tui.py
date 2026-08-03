"""Textual btop-style live dashboard.

Subscribes to `SnapshotCollector.stream()` via the event bus and never imports
hardware or engine adapters directly — the same rule as the neofetch view.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, RichLog, Static

from aitop.bus import EventBus, Topic
from aitop.collector import SnapshotCollector
from aitop.config import Config
from aitop.models import EngineState, SystemSnapshot
from aitop.utils.fmt import bytes_human, heat_color, percent, ratio_bar, watts


class MetricPanel(Static):
    """One labelled gauge block (CPU / memory / GPU)."""

    DEFAULT_CSS = """
    MetricPanel {
        height: auto;
        border: tall $surface;
        padding: 0 1;
        margin: 0 1 1 0;
        width: 1fr;
    }
    MetricPanel > .metric-title {
        text-style: bold;
        color: $accent;
    }
    """

    def __init__(self, title: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._body = Text("—")

    def render(self) -> Text:
        out = Text()
        out.append(f"{self._title}\n", style="bold cyan")
        out.append(self._body)
        return out

    def update_metric(self, body: Text) -> None:
        self._body = body
        self.refresh()


class AiTopApp(App[None]):
    """Live system + engine dashboard."""

    TITLE = "aitop"
    SUB_TITLE = "AI neofetch · btop"
    CSS = """
    Screen {
        layout: vertical;
    }
    #metrics {
        height: 7;
        padding: 1 0 0 1;
    }
    #engines {
        height: 1fr;
        border: tall $surface;
        margin: 0 1;
    }
    #loaded {
        height: 1fr;
        border: tall $surface;
        margin: 0 1;
    }
    #log {
        height: 6;
        border: tall $surface;
        margin: 0 1 1 1;
    }
    DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("u", "unload", "Unload"),
        Binding("j", "json_dump", "JSON", show=False),
        Binding("?", "help", "Help", show=False),
    ]

    def __init__(
        self,
        config: Config | None = None,
        *,
        allow_privileged: bool = True,
        interval: float | None = None,
    ) -> None:
        super().__init__()
        self.config = config or Config()
        self.allow_privileged = allow_privileged
        self.interval = interval or self.config.polling.hardware_interval
        self.bus = EventBus()
        self.collector = SnapshotCollector(
            self.config,
            bus=self.bus,
            allow_privileged=allow_privileged,
        )
        self.snapshot: SystemSnapshot | None = None
        self._stream_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="metrics"):
            yield MetricPanel("CPU", id="cpu")
            yield MetricPanel("Memory", id="memory")
            yield MetricPanel("GPU", id="gpu")
        with Vertical():
            yield DataTable(id="engines", cursor_type="row", zebra_stripes=True)
            yield DataTable(id="loaded", cursor_type="row", zebra_stripes=True)
            yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        engines = self.query_one("#engines", DataTable)
        engines.add_columns("State", "Runtime", "Endpoint", "Version", "Models", "Resident", "ms")
        loaded = self.query_one("#loaded", DataTable)
        loaded.add_columns("Model", "Engine", "Size", "GPU", "Context", "Expires")
        self.query_one("#log", RichLog).write("[dim]aitop tui — q quit · r refresh · u unload[/]")
        self._stream_task = asyncio.create_task(self._run_collector())
        self._consume_bus()

    async def on_unmount(self) -> None:
        if self._stream_task is not None:
            self._stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stream_task
        await self.collector.aclose()

    async def _run_collector(self) -> None:
        try:
            await self.collector.stream(interval=self.interval)
        except asyncio.CancelledError:
            raise

    @work(exclusive=True)
    async def _consume_bus(self) -> None:
        sub = self.bus.subscribe(Topic.SNAPSHOT, Topic.LOG, Topic.LIFECYCLE, Topic.ERROR)
        try:
            async for event in sub:
                if event.topic is Topic.SNAPSHOT and isinstance(event.payload, SystemSnapshot):
                    self.snapshot = event.payload
                    self._render_snapshot(event.payload)
                elif event.topic in (Topic.LOG, Topic.LIFECYCLE, Topic.ERROR):
                    self.query_one("#log", RichLog).write(f"{event.payload}")
        finally:
            sub.close()

    def _render_snapshot(self, snap: SystemSnapshot) -> None:
        hw = snap.hardware
        cpu = hw.cpu
        cpu_body = Text()
        cpu_body.append(f"{cpu.model}\n")
        frac = cpu.load_percent / 100.0
        cpu_body.append(ratio_bar(frac, width=24), style=heat_color(frac))
        cpu_body.append(f"  {percent(cpu.load_percent)}")
        if cpu.power_watts is not None:
            cpu_body.append(f"  {watts(cpu.power_watts)}", style="dim")
        self.query_one("#cpu", MetricPanel).update_metric(cpu_body)

        mem = hw.memory
        mem_body = Text()
        label = "unified" if mem.unified else "system"
        mem_body.append(
            f"{bytes_human(mem.used_bytes)} / {bytes_human(mem.total_bytes)} ({label})\n"
        )
        mfrac = mem.used_percent / 100.0
        mem_body.append(ratio_bar(mfrac, width=24), style=heat_color(mfrac))
        mem_body.append(f"  {percent(mem.used_percent)}")
        self.query_one("#memory", MetricPanel).update_metric(mem_body)

        gpu_body = Text()
        if not hw.gpus:
            gpu_body.append("no GPU detected", style="dim")
        else:
            for gpu in hw.gpus[:2]:
                gpu_body.append(f"{gpu.name}\n")
                gfrac = (gpu.utilization_percent or 0) / 100.0
                gpu_body.append(ratio_bar(gfrac, width=24), style=heat_color(gfrac))
                gpu_body.append(f"  {percent(gpu.utilization_percent)}")
                if gpu.vram_total_bytes:
                    gpu_body.append(
                        f"\nVRAM {bytes_human(gpu.vram_used_bytes)} / "
                        f"{bytes_human(gpu.vram_total_bytes)}",
                        style="dim",
                    )
                if gpu.power_watts is not None:
                    gpu_body.append(f"  {watts(gpu.power_watts)}", style="dim")
                gpu_body.append("\n")
        self.query_one("#gpu", MetricPanel).update_metric(gpu_body)

        engines = self.query_one("#engines", DataTable)
        engines.clear()
        for eng in snap.engines:
            mark, style = {
                EngineState.ONLINE: ("●", "green"),
                EngineState.DEGRADED: ("◐", "yellow"),
                EngineState.OFFLINE: ("○", "dim"),
                EngineState.UNKNOWN: ("?", "dim"),
            }.get(eng.state, ("?", "dim"))
            binding = str(eng.binding) if eng.binding else "—"
            engines.add_row(
                Text(mark, style=style),
                eng.name,
                binding,
                eng.version or "—",
                str(len(eng.models)),
                bytes_human(eng.resident_bytes) if eng.loaded else "—",
                f"{eng.latency_ms:.0f}" if eng.latency_ms is not None else "—",
                key=f"{eng.kind}:{binding}",
            )

        loaded = self.query_one("#loaded", DataTable)
        loaded.clear()
        for model in snap.all_loaded:
            gpu = percent(model.gpu_fraction * 100 if model.gpu_fraction is not None else None)
            ctx = (
                f"{model.context_used or '—'}/{model.context_length}"
                if model.context_length
                else "—"
            )
            loaded.add_row(
                model.name,
                model.engine.value,
                bytes_human(model.size_bytes),
                gpu,
                ctx,
                "—",
                key=f"{model.engine}:{model.id}",
            )

        host = hw.host.hostname
        online = len(snap.online_engines)
        self.sub_title = f"{host} · {online} runtime(s) online · {snap.duration_ms or 0:.0f} ms"

    def action_refresh(self) -> None:
        self.run_worker(self._force_refresh(), exclusive=True)

    async def _force_refresh(self) -> None:
        snap = await self.collector.collect()
        self.bus.publish(Topic.SNAPSHOT, snap, source=self.collector.node)
        self.query_one("#log", RichLog).write("[dim]refreshed[/]")

    def action_unload(self) -> None:
        self.run_worker(self._unload_selected(), exclusive=True)

    async def _unload_selected(self) -> None:
        table = self.query_one("#loaded", DataTable)
        if table.row_count == 0:
            self.query_one("#log", RichLog).write("[yellow]no resident models to unload[/]")
            return
        row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
        if row_key is None or row_key.value is None:
            return
        key = str(row_key.value)
        engine_kind, _, model_id = key.partition(":")
        engines = self.collector.engines.engines
        target = next((e for e in engines if e.kind.value == engine_kind), None)
        if target is None:
            self.query_one("#log", RichLog).write(f"[yellow]no adapter for {engine_kind}[/]")
            return
        if not target.supports("unload"):
            self.query_one("#log", RichLog).write(f"[yellow]{target.name} cannot unload[/]")
            return
        self.query_one("#log", RichLog).write(f"[cyan]unloading {model_id}…[/]")
        ok, message = await target.unload(model_id)
        colour = "green" if ok else "yellow"
        self.query_one("#log", RichLog).write(f"[{colour}]{message}[/]")
        self.bus.publish(Topic.LIFECYCLE, message, source=target.name)
        await self._force_refresh()

    def action_json_dump(self) -> None:
        if self.snapshot is None:
            return
        self.query_one("#log", RichLog).write(self.snapshot.model_dump_json()[:500] + "…")

    def action_help(self) -> None:
        self.query_one("#log", RichLog).write(
            "[bold]keys[/]  q quit · r refresh · u unload selected model · j dump json"
        )


def run_tui(
    config: Config | None = None,
    *,
    allow_privileged: bool = True,
    interval: float | None = None,
) -> int:
    app = AiTopApp(config, allow_privileged=allow_privileged, interval=interval)
    app.run()
    return 0
