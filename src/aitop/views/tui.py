"""Textual btop-style live dashboard.

Subscribes to `SnapshotCollector.stream()` via the event bus and never imports
hardware or engine adapters directly — the same rule as the neofetch view.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, RichLog, Static, TabbedContent, TabPane

from aitop.bus import EventBus, Topic
from aitop.collector import SnapshotCollector
from aitop.config import Config
from aitop.models import BindScope, EngineSnapshot, EngineState, SystemSnapshot
from aitop.utils.fmt import (
    bytes_human,
    celsius,
    core_bars,
    heat_color,
    percent,
    ratio_bar,
    relative_time,
    sparkline,
    watts,
)
from aitop.views.tui_screens import (
    ConfirmScreen,
    FilterScreen,
    HelpScreen,
    ModelPickerScreen,
    PullScreen,
)

_HISTORY = 56
_EMPTY = "__empty__"
_SORT_MODES = ("name", "size", "state")

_STATE = {
    EngineState.ONLINE: ("●", "green"),
    EngineState.DEGRADED: ("◐", "yellow"),
    EngineState.OFFLINE: ("○", "dim"),
    EngineState.UNKNOWN: ("?", "dim"),
}

_SCOPE_STYLE = {
    BindScope.LOOPBACK: "green",
    BindScope.LAN: "yellow",
    BindScope.TAILSCALE: "#d2a8ff",
    BindScope.OTHER: "dim",
}


class MetricPanel(Static):
    """One labelled gauge block (CPU / memory / GPU) with history."""

    # Gauges are display-only — keep them out of the Tab focus cycle.
    can_focus = False

    def __init__(self, title: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._body = Text("—")

    def render(self) -> Text:
        out = Text()
        out.append(f"{self._title}\n", style="bold #39d2c0")
        out.append(self._body)
        return out

    def update_metric(self, body: Text) -> None:
        self._body = body
        self.refresh()


class StatusBar(Static):
    """Status strip: pause, filters, network, degraded probes."""

    can_focus = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._text = Text("")

    def render(self) -> Text:
        return self._text

    def update_status(self, text: Text) -> None:
        self._text = text
        self.refresh()


class DetailBar(Static):
    """Selection inspector under the tables — pid, errors, format, etc."""

    can_focus = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._text = Text("select a row for details", style="dim")

    def render(self) -> Text:
        return self._text

    def update_detail(self, text: Text) -> None:
        self._text = text
        self.refresh()


def _row_key(table: DataTable) -> str | None:
    if table.row_count == 0:
        return None
    with contextlib.suppress(Exception):
        cell = table.coordinate_to_cell_key(table.cursor_coordinate)
        if cell.row_key is not None and cell.row_key.value is not None:
            key = str(cell.row_key.value)
            return None if key == _EMPTY else key
    return None


def _restore_cursor(table: DataTable, key: str | None) -> None:
    if key is None or table.row_count == 0:
        return
    with contextlib.suppress(Exception):
        table.move_cursor(row=table.get_row_index(key), animate=False)


def _engine_key(eng: EngineSnapshot) -> str:
    binding = str(eng.binding) if eng.binding else ""
    return f"{eng.kind.value}|{binding}|{eng.name}"


def _age_label(when: datetime | None) -> str:
    if when is None:
        return "—"
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    secs = max(0, int((datetime.now(UTC) - when).total_seconds()))
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    return f"{secs // 3600}h ago"


class AiTopApp(App[None]):
    """Live system + engine dashboard."""

    TITLE = "aitop"
    SUB_TITLE = "AI neofetch · btop"
    CSS = """
    Screen {
        background: #0e1116;
    }
    Header {
        background: #161b22;
        color: #e6edf3;
    }
    Footer {
        background: #161b22;
    }
    #metrics {
        height: 11;
        padding: 1 0 0 1;
        background: #0e1116;
    }
    MetricPanel {
        height: 1fr;
        border: tall #30363d;
        background: #161b22;
        padding: 0 1;
        margin: 0 1 0 0;
        width: 1fr;
    }
    #status {
        height: 1;
        padding: 0 2;
        color: #8b949e;
        background: #0e1116;
    }
    #tabs {
        height: 1fr;
        margin: 0 1;
    }
    TabbedContent {
        height: 1fr;
    }
    TabPane {
        padding: 0;
    }
    DataTable {
        height: 1fr;
        background: #161b22;
    }
    DataTable > .datatable--cursor {
        background: #1f6feb;
        color: #ffffff;
    }
    DataTable > .datatable--hover {
        background: #21262d;
    }
    #detail {
        height: 2;
        padding: 0 2;
        color: #8b949e;
        background: #0e1116;
    }
    #log {
        height: 6;
        border: tall #30363d;
        background: #161b22;
        margin: 0 1 1 1;
        scrollbar-background: #161b22;
        scrollbar-color: #30363d;
    }
    #log:focus {
        border: tall #39d2c0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("question_mark", "help", "Help"),
        Binding("r", "refresh", "Refresh"),
        Binding("space", "toggle_pause", "Pause"),
        Binding("a", "toggle_offline", "All"),
        Binding("slash", "filter_catalog", "Filter"),
        Binding("o", "cycle_sort", "Sort"),
        Binding("p", "pull_model", "Pull"),
        Binding("u", "unload", "Unload"),
        Binding("l", "load", "Load"),
        Binding("d", "delete_model", "Delete"),
        Binding("s", "start_engine", "Start"),
        Binding("x", "stop_engine", "Stop"),
        Binding("e", "restart_engine", "Restart"),
        Binding("1", "tab_engines", "Engines", show=False),
        Binding("2", "tab_catalog", "Catalog", show=False),
        Binding("3", "tab_loaded", "Loaded", show=False),
        Binding("4", "tab_fleet", "Fleet", show=False),
        Binding("n", "next_node", "Node"),
        Binding("N", "prev_node", "Prev node", show=False),
        Binding("left_square_bracket", "prev_node", "Prev node", show=False),
        Binding("right_square_bracket", "next_node", "Next node", show=False),
        Binding("j", "json_dump", "JSON", show=False),
        Binding("escape", "clear_filter", "Clear filter", show=False),
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
        self.fleet_snapshots: list[SystemSnapshot] = []
        self.node_index = 0
        self._stream_task: asyncio.Task[None] | None = None
        self._cpu_hist: deque[float] = deque(maxlen=_HISTORY)
        self._mem_hist: deque[float] = deque(maxlen=_HISTORY)
        self._gpu_hist: deque[float] = deque(maxlen=_HISTORY)
        self._vram_hist: deque[float] = deque(maxlen=_HISTORY)
        self.paused = False
        self.show_offline = False
        self.catalog_filter = ""
        self.catalog_sort = "name"
        self._pending: SystemSnapshot | None = None
        self._logged_errors: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="metrics"):
            yield MetricPanel("CPU", id="cpu")
            yield MetricPanel("Memory", id="memory")
            yield MetricPanel("GPU", id="gpu")
        yield StatusBar(id="status")
        with TabbedContent(id="tabs"):
            with TabPane("Engines", id="tab-engines"):
                yield DataTable(id="engines", cursor_type="row", zebra_stripes=True)
            with TabPane("Catalog", id="tab-catalog"):
                yield DataTable(id="catalog", cursor_type="row", zebra_stripes=True)
            with TabPane("Loaded", id="tab-loaded"):
                yield DataTable(id="loaded", cursor_type="row", zebra_stripes=True)
            with TabPane("Fleet", id="tab-fleet"):
                yield DataTable(id="fleet", cursor_type="row", zebra_stripes=True)
        yield DetailBar(id="detail")
        yield RichLog(id="log", highlight=True, markup=True, max_lines=500)
        yield Footer()

    def on_mount(self) -> None:
        engines = self.query_one("#engines", DataTable)
        engines.add_columns(
            "State",
            "Runtime",
            "Endpoint",
            "Scope",
            "Ver",
            "Models",
            "Resident",
            "tok/s",
            "Q",
            "ms",
        )
        catalog = self.query_one("#catalog", DataTable)
        catalog.add_columns("Model", "Runtime", "Params", "Quant", "Size", "Ctx", "State")
        loaded = self.query_one("#loaded", DataTable)
        loaded.add_columns(
            "Model", "Runtime", "Params", "Quant", "Size", "GPU", "Context", "Expires"
        )
        fleet = self.query_one("#fleet", DataTable)
        fleet.add_columns("Node", "Host", "Engines", "Online", "GPUs", "VRAM", "Age")
        self.query_one("#log", RichLog).write(
            "[dim]aitop tui — [bold]?[/] help · space pause · / filter · n node · 1/2/3/4 panes[/]"
        )
        self._stream_task = asyncio.create_task(self._run_collector())
        self._consume_bus()
        self.set_focus(engines)

    async def on_unmount(self) -> None:
        if self._stream_task is not None:
            self._stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stream_task
        await self.collector.aclose()

    async def _run_collector(self) -> None:
        try:
            if self.config.fleet.nodes:
                await self._stream_fleet()
            else:
                await self.collector.stream(interval=self.interval)
        except asyncio.CancelledError:
            raise

    async def _stream_fleet(self) -> None:
        """Local + remote snapshots when fleet.nodes is configured."""
        period = self.interval
        while True:
            cycle_start = time.perf_counter()
            try:
                snaps = await self.collector.collect_fleet()
                self.fleet_snapshots = snaps
                if snaps:
                    idx = self.node_index % len(snaps)
                    self.node_index = idx
                    snap = snaps[idx]
                    self.bus.publish(Topic.SNAPSHOT, snap, source=snap.node)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.bus.publish(Topic.ERROR, str(exc), source="fleet")
            elapsed = time.perf_counter() - cycle_start
            await asyncio.sleep(max(0.1, period - elapsed))

    @work(exclusive=True)
    async def _consume_bus(self) -> None:
        sub = self.bus.subscribe(Topic.SNAPSHOT, Topic.LOG, Topic.LIFECYCLE, Topic.ERROR)
        try:
            async for event in sub:
                if event.topic is Topic.SNAPSHOT and isinstance(event.payload, SystemSnapshot):
                    if self.paused:
                        self._pending = event.payload
                        self._render_status(event.payload)
                        continue
                    self.snapshot = event.payload
                    self._render_snapshot(event.payload)
                elif event.topic in (Topic.LOG, Topic.LIFECYCLE, Topic.ERROR):
                    self.query_one("#log", RichLog).write(f"{event.payload}")
        finally:
            sub.close()

    def on_data_table_row_highlighted(self, _event: DataTable.RowHighlighted) -> None:
        self._render_detail()

    def on_tabbed_content_tab_activated(self, _event: TabbedContent.TabActivated) -> None:
        self._render_detail()

    # -- rendering ---------------------------------------------------------- #

    def _render_snapshot(self, snap: SystemSnapshot) -> None:
        self._render_metrics(snap)
        self._render_engines(snap)
        self._render_catalog(snap)
        self._render_loaded(snap)
        self._render_fleet()
        self._render_status(snap)
        self._render_detail()
        self._log_new_errors(snap)

        host = snap.hardware.host.hostname
        online = len(snap.online_engines)
        pause = " · PAUSED" if self.paused else ""
        filt = f" · /{self.catalog_filter}" if self.catalog_filter else ""
        node = ""
        if self.fleet_snapshots:
            node = f" · node {snap.node} ({self.node_index + 1}/{len(self.fleet_snapshots)})"
        self.sub_title = (
            f"{host}{node} · {online} online · {snap.duration_ms or 0:.0f} ms{pause}{filt}"
        )

    def _render_metrics(self, snap: SystemSnapshot) -> None:
        hw = snap.hardware
        cpu = hw.cpu
        self._cpu_hist.append(cpu.load_percent)
        cpu_body = Text()
        cores = ""
        if cpu.logical_cores:
            cores = f" · {cpu.logical_cores}t"
            if cpu.physical_cores and cpu.physical_cores != cpu.logical_cores:
                cores = f" · {cpu.physical_cores}c/{cpu.logical_cores}t"
        cpu_body.append(f"{cpu.model}{cores}\n")
        frac = cpu.load_percent / 100.0
        cpu_body.append(ratio_bar(frac, width=22), style=heat_color(frac))
        cpu_body.append(f"  {percent(cpu.load_percent)}")
        extras: list[str] = []
        if cpu.power_watts is not None:
            extras.append(watts(cpu.power_watts))
        if cpu.temperature_c is not None:
            extras.append(celsius(cpu.temperature_c))
        if extras:
            cpu_body.append(f"  {' · '.join(extras)}", style="dim")
        cpu_body.append(f"\n{sparkline(list(self._cpu_hist), width=28)}", style="#39d2c0")
        if cpu.per_core_percent:
            cpu_body.append("\n")
            cpu_body.append(core_bars(cpu.per_core_percent, width=3, cols=8))
        self.query_one("#cpu", MetricPanel).update_metric(cpu_body)

        mem = hw.memory
        self._mem_hist.append(mem.used_percent)
        mem_body = Text()
        label = "unified" if mem.unified else "system"
        mem_body.append(
            f"{bytes_human(mem.used_bytes)} / {bytes_human(mem.total_bytes)} ({label})\n"
        )
        mfrac = mem.used_percent / 100.0
        mem_body.append(ratio_bar(mfrac, width=22), style=heat_color(mfrac))
        mem_body.append(f"  {percent(mem.used_percent)}")
        if mem.swap_total_bytes:
            mem_body.append(
                f"\nswap {bytes_human(mem.swap_used_bytes)} / {bytes_human(mem.swap_total_bytes)}",
                style="dim",
            )
        mem_body.append(f"\n{sparkline(list(self._mem_hist), width=28)}", style="#39d2c0")
        self.query_one("#memory", MetricPanel).update_metric(mem_body)

        gpu_body = Text()
        if not hw.gpus:
            self._gpu_hist.append(0.0)
            self._vram_hist.append(0.0)
            gpu_body.append("no GPU detected\n", style="dim")
            gpu_body.append(sparkline(list(self._gpu_hist), width=28), style="dim")
        else:
            util = hw.gpus[0].utilization_percent or 0.0
            self._gpu_hist.append(util)
            vram_pct = hw.gpus[0].vram_used_percent or 0.0
            self._vram_hist.append(vram_pct)
            for gpu in hw.gpus[:2]:
                driver = f" · {gpu.driver_version}" if gpu.driver_version else ""
                gpu_body.append(f"{gpu.name}{driver}\n")
                gfrac = (gpu.utilization_percent or 0) / 100.0
                gpu_body.append(ratio_bar(gfrac, width=22), style=heat_color(gfrac))
                gpu_body.append(f"  {percent(gpu.utilization_percent)}")
                bits: list[str] = []
                if gpu.power_watts is not None:
                    bits.append(watts(gpu.power_watts))
                if gpu.temperature_c is not None:
                    bits.append(celsius(gpu.temperature_c))
                if bits:
                    gpu_body.append(f"  {' · '.join(bits)}", style="dim")
                if gpu.vram_total_bytes:
                    gpu_body.append(
                        f"\nVRAM {bytes_human(gpu.vram_used_bytes)} / "
                        f"{bytes_human(gpu.vram_total_bytes)}",
                        style="dim",
                    )
                gpu_body.append("\n")
            gpu_body.append("util ", style="dim")
            gpu_body.append(sparkline(list(self._gpu_hist), width=24), style="#39d2c0")
            gpu_body.append("\nvram ", style="dim")
            gpu_body.append(sparkline(list(self._vram_hist), width=24), style="#d2a8ff")
        self.query_one("#gpu", MetricPanel).update_metric(gpu_body)

    def _render_engines(self, snap: SystemSnapshot) -> None:
        table = self.query_one("#engines", DataTable)
        prev = _row_key(table)
        table.clear()
        engines = snap.engines if self.show_offline else snap.online_engines
        if not engines:
            hint = (
                "no AI runtimes reachable — start Ollama / LM Studio, or press a for offline"
                if not self.show_offline
                else "no engines configured — add endpoints in ~/.config/aitop/config.yaml"
            )
            table.add_row(Text("○", style="dim"), Text(hint, style="dim"), key=_EMPTY)
            return
        for eng in engines:
            mark, style = _STATE.get(eng.state, ("?", "dim"))
            binding = eng.binding
            endpoint = str(binding) if binding else "—"
            scope = binding.scope if binding else BindScope.OTHER
            tps = (
                f"{eng.stats.tokens_per_second:.1f}"
                if eng.stats.tokens_per_second is not None
                else "—"
            )
            queue = str(eng.stats.queue_depth) if eng.stats.queue_depth else "—"
            name = eng.name
            if eng.error:
                name = Text(eng.name, style="yellow")
            table.add_row(
                Text(mark, style=style),
                name,
                endpoint,
                Text(scope.value, style=_SCOPE_STYLE.get(scope, "dim")),
                (eng.version or "—")[:12],
                str(len(eng.models)),
                bytes_human(eng.resident_bytes) if eng.loaded else "—",
                tps,
                queue,
                f"{eng.latency_ms:.0f}" if eng.latency_ms is not None else "—",
                key=_engine_key(eng),
            )
        _restore_cursor(table, prev)

    def _render_catalog(self, snap: SystemSnapshot) -> None:
        table = self.query_one("#catalog", DataTable)
        prev = _row_key(table)
        table.clear()
        engines = snap.engines if self.show_offline else snap.online_engines
        rows: list[tuple[str, str, str, str, int | None, str, Text, str]] = []
        needle = self.catalog_filter.lower()
        for eng in engines:
            loaded_ids = {m.id for m in eng.loaded}
            for model in eng.models:
                hay = " ".join(
                    filter(
                        None,
                        [
                            model.name,
                            model.id,
                            eng.name,
                            model.parameter_size,
                            model.quantization,
                            model.family,
                            model.format,
                        ],
                    )
                ).lower()
                if needle and needle not in hay:
                    continue
                state = "● resident" if model.id in loaded_ids else "○ disk"
                style = "green" if model.id in loaded_ids else "dim"
                rows.append(
                    (
                        model.name,
                        eng.name,
                        model.parameter_size or "—",
                        model.quantization or "—",
                        model.size_bytes,
                        str(model.max_context) if model.max_context else "—",
                        Text(state, style=style),
                        f"{eng.kind.value}:{model.id}",
                    )
                )

        if self.catalog_sort == "size":
            rows.sort(key=lambda r: r[4] or 0, reverse=True)
        elif self.catalog_sort == "state":
            rows.sort(key=lambda r: (0 if "resident" in r[6].plain else 1, r[0].lower()))
        else:
            rows.sort(key=lambda r: r[0].lower())

        if not rows:
            msg = (
                f"no models match /{self.catalog_filter}"
                if self.catalog_filter
                else "no models on disk — pull one or wait for engines to come online"
            )
            table.add_row(Text(msg, style="dim"), key=_EMPTY)
            return

        for name, runtime, params, quant, size, ctx, state, key in rows:
            table.add_row(
                name,
                runtime,
                params,
                quant,
                bytes_human(size),
                ctx,
                state,
                key=key,
            )
        _restore_cursor(table, prev)

    def _render_loaded(self, snap: SystemSnapshot) -> None:
        table = self.query_one("#loaded", DataTable)
        prev = _row_key(table)
        table.clear()
        models = snap.all_loaded
        if not models:
            table.add_row(
                Text("nothing resident — press l to load, or 2 for catalog", style="dim"),
                key=_EMPTY,
            )
            return
        for model in models:
            gpu = percent(model.gpu_fraction * 100 if model.gpu_fraction is not None else None)
            if model.context_fill is not None and model.context_length:
                bar = ratio_bar(model.context_fill, width=8)
                ctx = Text()
                ctx.append(bar, style=heat_color(model.context_fill))
                ctx.append(
                    f" {model.context_used or 0}/{model.context_length}",
                    style="dim",
                )
            elif model.context_length:
                ctx = Text(f"{model.context_used or '—'}/{model.context_length}")
            else:
                ctx = Text("—")
            table.add_row(
                model.name,
                model.engine.value,
                model.parameter_size or "—",
                model.quantization or "—",
                bytes_human(model.size_bytes),
                gpu,
                ctx,
                relative_time(model.expires_at),
                key=f"{model.engine.value}:{model.id}",
            )
        _restore_cursor(table, prev)

    def _render_fleet(self) -> None:
        table = self.query_one("#fleet", DataTable)
        prev = _row_key(table)
        table.clear()
        snaps = self.fleet_snapshots
        if not snaps:
            if self.config.fleet.nodes:
                table.add_row(
                    Text("waiting for fleet snapshots…", style="dim"),
                    key=_EMPTY,
                )
            else:
                table.add_row(
                    Text(
                        "no fleet.nodes — add peers in config, or use aitop fleet",
                        style="dim",
                    ),
                    key=_EMPTY,
                )
            return
        for i, snap in enumerate(snaps):
            mark = "▸ " if i == self.node_index else "  "
            host = snap.hardware.host.hostname or snap.node
            online = len(snap.online_engines)
            gpus = len(snap.hardware.gpus)
            vram = sum((g.vram_used_bytes or 0) for g in snap.hardware.gpus)
            table.add_row(
                f"{mark}{snap.node}",
                host,
                str(len(snap.engines)),
                str(online),
                str(gpus) if gpus else "—",
                bytes_human(vram) if vram else "—",
                _age_label(snap.collected_at),
                key=snap.node,
            )
        _restore_cursor(table, prev)

    def _render_status(self, snap: SystemSnapshot) -> None:
        text = Text()
        if self.paused:
            text.append(" PAUSED ", style="bold reverse #d29922")
            text.append(f" {_age_label(snap.collected_at)}  ", style="#d29922")
        if self.fleet_snapshots:
            text.append(f"node {snap.node}  ·  ")
        text.append(f"engines {len(snap.online_engines)}/{len(snap.engines)}")
        text.append("  ·  ")
        text.append(f"loaded {len(snap.all_loaded)}")
        if self.show_offline:
            text.append("  ·  ")
            text.append("showing offline", style="#d29922")
        if self.catalog_filter:
            text.append("  ·  ")
            text.append(f"/{self.catalog_filter}", style="#39d2c0")
        if self.catalog_sort != "name":
            text.append("  ·  ")
            text.append(f"sort:{self.catalog_sort}", style="dim")
        ts = snap.tailscale
        if ts.available and ts.running:
            text.append("  ·  ")
            parts = ["tailscale"]
            if ts.ipv4:
                parts.append(ts.ipv4)
            if ts.hostname:
                parts.append(ts.hostname)
            if ts.tailnet:
                parts.append(ts.tailnet)
            if ts.peer_count:
                parts.append(f"{ts.peer_count} peers")
            text.append(" ".join(parts), style="#39d2c0")
        if snap.hardware.total_power_watts is not None:
            text.append("  ·  ")
            text.append(watts(snap.hardware.total_power_watts), style="dim")
        if snap.hardware.degraded:
            text.append("  ·  ")
            text.append(f"! {snap.hardware.degraded[0]}", style="yellow")
        self.query_one("#status", StatusBar).update_status(text)

    def _render_detail(self) -> None:
        snap = self.snapshot
        bar = self.query_one("#detail", DetailBar)
        if snap is None:
            bar.update_detail(Text("waiting for first snapshot…", style="dim"))
            return

        tabs = self.query_one("#tabs", TabbedContent)
        active = tabs.active

        if active == "tab-engines":
            key = _row_key(self.query_one("#engines", DataTable))
            if key is None:
                bar.update_detail(
                    Text("no engine selected — start a runtime or press a", style="dim")
                )
                return
            eng = next((e for e in snap.engines if _engine_key(e) == key), None)
            if eng is None:
                bar.update_detail(Text("—", style="dim"))
                return
            parts: list[str] = [eng.name, eng.state.value]
            if eng.pid:
                parts.append(f"pid {eng.pid}")
            if eng.managed_by:
                parts.append(f"via {eng.managed_by}")
            if eng.remote:
                parts.append("remote")
            if eng.binding:
                parts.append(eng.binding.scope.value)
            if eng.stats.active_requests:
                parts.append(f"{eng.stats.active_requests} active")
            if eng.stats.ttft_ms is not None:
                parts.append(f"ttft {eng.stats.ttft_ms:.0f}ms")
            text = Text(" · ".join(parts))
            if eng.error:
                text.append(f"  ! {eng.error}", style="yellow")
            bar.update_detail(text)
            return

        if active == "tab-catalog":
            key = _row_key(self.query_one("#catalog", DataTable))
            if key is None:
                bar.update_detail(Text("no model selected — / to filter", style="dim"))
                return
            kind, _, model_id = key.partition(":")
            for eng in snap.engines:
                if eng.kind.value != kind:
                    continue
                for model in eng.models:
                    if model.id != model_id:
                        continue
                    bits = [model.name, eng.name]
                    if model.family:
                        bits.append(model.family)
                    if model.format:
                        bits.append(model.format)
                    if model.max_context:
                        bits.append(f"ctx {model.max_context}")
                    if model.modified_at:
                        bits.append(f"updated {_age_label(model.modified_at)}")
                    bar.update_detail(Text(" · ".join(bits)))
                    return
            bar.update_detail(Text(model_id, style="dim"))
            return

        if active == "tab-fleet":
            key = _row_key(self.query_one("#fleet", DataTable))
            if key is None:
                bar.update_detail(
                    Text("configure fleet.nodes to monitor remote aitop serve peers", style="dim")
                )
                return
            peer = next((s for s in self.fleet_snapshots if s.node == key), None)
            if peer is None:
                bar.update_detail(Text(key, style="dim"))
                return
            bits = [
                peer.node,
                peer.hardware.host.hostname or "—",
                f"{len(peer.online_engines)} online",
                f"{len(peer.all_loaded)} loaded",
            ]
            if peer.duration_ms is not None:
                bits.append(f"{peer.duration_ms:.0f} ms")
            bar.update_detail(Text(" · ".join(bits)))
            return

        # Loaded
        key = _row_key(self.query_one("#loaded", DataTable))
        if key is None:
            bar.update_detail(Text("nothing resident", style="dim"))
            return
        kind, _, model_id = key.partition(":")
        for model in snap.all_loaded:
            if model.engine.value == kind and model.id == model_id:
                bits = [model.name, model.engine.value]
                if model.family:
                    bits.append(model.family)
                if model.vram_bytes is not None:
                    bits.append(f"vram {bytes_human(model.vram_bytes)}")
                if model.context_fill is not None:
                    bits.append(f"ctx {model.context_fill:.0%}")
                bar.update_detail(Text(" · ".join(bits)))
                return
        bar.update_detail(Text(model_id, style="dim"))

    def _log_new_errors(self, snap: SystemSnapshot) -> None:
        log = self.query_one("#log", RichLog)
        for eng in snap.engines:
            if not eng.error:
                continue
            marker = f"{eng.kind.value}:{eng.error}"
            if marker in self._logged_errors:
                continue
            self._logged_errors.add(marker)
            log.write(f"[yellow]! {eng.name}: {eng.error}[/]")
        for note in snap.hardware.degraded:
            marker = f"hw:{note}"
            if marker in self._logged_errors:
                continue
            self._logged_errors.add(marker)
            log.write(f"[dim]! hardware: {note}[/]")

    # -- actions ------------------------------------------------------------ #

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        if not self.paused and self._pending is not None:
            self.snapshot = self._pending
            self._render_snapshot(self._pending)
            self._pending = None
            self.query_one("#log", RichLog).write("[dim]resumed[/]")
        elif self.paused:
            self.query_one("#log", RichLog).write("[yellow]paused — space to resume[/]")
            if self.snapshot is not None:
                self._render_status(self.snapshot)
                host = self.snapshot.hardware.host.hostname
                self.sub_title = f"{host} · PAUSED · {_age_label(self.snapshot.collected_at)}"

    def action_toggle_offline(self) -> None:
        self.show_offline = not self.show_offline
        state = "all endpoints" if self.show_offline else "online only"
        self.query_one("#log", RichLog).write(f"[dim]filter: {state}[/]")
        if self.snapshot is not None:
            self._render_snapshot(self.snapshot)

    def action_filter_catalog(self) -> None:
        self.run_worker(self._filter_catalog(), exclusive=True)

    async def _filter_catalog(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-catalog"
        result = await self.push_screen_wait(FilterScreen(self.catalog_filter))
        if result is None:
            return
        self.catalog_filter = result
        msg = f"filter: /{result}" if result else "filter cleared"
        self.query_one("#log", RichLog).write(f"[dim]{msg}[/]")
        if self.snapshot is not None:
            self._render_catalog(self.snapshot)
            self._render_status(self.snapshot)
            filt = f" · /{self.catalog_filter}" if self.catalog_filter else ""
            host = self.snapshot.hardware.host.hostname
            online = len(self.snapshot.online_engines)
            pause = " · PAUSED" if self.paused else ""
            self.sub_title = (
                f"{host} · {online} online · {self.snapshot.duration_ms or 0:.0f} ms{pause}{filt}"
            )
        self.query_one("#catalog", DataTable).focus()

    def action_clear_filter(self) -> None:
        if not self.catalog_filter:
            return
        self.catalog_filter = ""
        self.query_one("#log", RichLog).write("[dim]filter cleared[/]")
        if self.snapshot is not None:
            self._render_catalog(self.snapshot)
            self._render_status(self.snapshot)

    def action_cycle_sort(self) -> None:
        idx = _SORT_MODES.index(self.catalog_sort)
        self.catalog_sort = _SORT_MODES[(idx + 1) % len(_SORT_MODES)]
        self.query_one("#log", RichLog).write(f"[dim]catalog sort: {self.catalog_sort}[/]")
        self.query_one("#tabs", TabbedContent).active = "tab-catalog"
        if self.snapshot is not None:
            self._render_catalog(self.snapshot)
            self._render_status(self.snapshot)
        self.query_one("#catalog", DataTable).focus()

    def action_tab_engines(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-engines"
        self.query_one("#engines", DataTable).focus()
        self._render_detail()

    def action_tab_catalog(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-catalog"
        self.query_one("#catalog", DataTable).focus()
        self._render_detail()

    def action_tab_loaded(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-loaded"
        self.query_one("#loaded", DataTable).focus()
        self._render_detail()

    def action_tab_fleet(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-fleet"
        self.query_one("#fleet", DataTable).focus()
        self._render_detail()

    def action_next_node(self) -> None:
        self._cycle_node(+1)

    def action_prev_node(self) -> None:
        self._cycle_node(-1)

    def _cycle_node(self, delta: int) -> None:
        if not self.fleet_snapshots:
            self.query_one("#log", RichLog).write(
                "[dim]no fleet nodes — configure fleet.nodes or stay on local[/]"
            )
            return
        self.node_index = (self.node_index + delta) % len(self.fleet_snapshots)
        snap = self.fleet_snapshots[self.node_index]
        self.snapshot = snap
        # Reset sparklines when jumping nodes so history isn't mixed.
        self._cpu_hist.clear()
        self._mem_hist.clear()
        self._gpu_hist.clear()
        self._vram_hist.clear()
        self._render_snapshot(snap)
        self.query_one("#log", RichLog).write(
            f"[cyan]node → {snap.node} ({self.node_index + 1}/{len(self.fleet_snapshots)})[/]"
        )

    def action_refresh(self) -> None:
        self.run_worker(self._force_refresh(), exclusive=True)

    async def _force_refresh(self) -> None:
        if self.config.fleet.nodes:
            snaps = await self.collector.collect_fleet()
            self.fleet_snapshots = snaps
            if snaps:
                self.node_index %= len(snaps)
                snap = snaps[self.node_index]
                self.bus.publish(Topic.SNAPSHOT, snap, source=snap.node)
                return
        snap = await self.collector.collect()
        self.bus.publish(Topic.SNAPSHOT, snap, source=self.collector.node)
        if not self.paused:
            self.snapshot = snap
            self._render_snapshot(snap)
        self.query_one("#log", RichLog).write("[dim]refreshed[/]")

    def action_unload(self) -> None:
        self.run_worker(self._unload_selected(), exclusive=True)

    def action_load(self) -> None:
        self.run_worker(self._load_with_picker(), exclusive=True)

    def action_pull_model(self) -> None:
        self.run_worker(self._pull_model(), exclusive=True)

    def action_delete_model(self) -> None:
        self.run_worker(self._delete_selected(), exclusive=True)

    def action_start_engine(self) -> None:
        self.run_worker(self._lifecycle_selected("start"), exclusive=True)

    def action_stop_engine(self) -> None:
        self.run_worker(self._lifecycle_selected("stop", confirm=True), exclusive=True)

    def action_restart_engine(self) -> None:
        self.run_worker(self._lifecycle_selected("restart", confirm=True), exclusive=True)

    async def _confirm(self, title: str, body: str) -> bool:
        return bool(await self.push_screen_wait(ConfirmScreen(title, body)))

    async def _unload_selected(self) -> None:
        table = self.query_one("#loaded", DataTable)
        key = _row_key(table)
        if key is None:
            self.query_one("#tabs", TabbedContent).active = "tab-loaded"
            table = self.query_one("#loaded", DataTable)
            table.focus()
            key = _row_key(table)
        if key is None:
            self.query_one("#log", RichLog).write("[yellow]select a resident model first[/]")
            return
        engine_kind, _, model_id = key.partition(":")
        target = self._engine_by_kind(engine_kind)
        if target is None:
            return
        if not target.supports("unload"):
            self.query_one("#log", RichLog).write(f"[yellow]{target.name} cannot unload[/]")
            return
        if not await self._confirm(
            "Unload model?", f"Evict [bold]{model_id}[/] from {target.name}"
        ):
            self.query_one("#loaded", DataTable).focus()
            return
        self.query_one("#log", RichLog).write(f"[cyan]unloading {model_id}…[/]")
        ok, message = await target.unload(model_id)
        self._log_result(ok, message, source=target.name)
        await self._force_refresh()
        self.query_one("#loaded", DataTable).focus()

    async def _load_with_picker(self) -> None:
        engine = self._selected_engine()
        tabs = self.query_one("#tabs", TabbedContent)
        if tabs.active == "tab-catalog":
            key = _row_key(self.query_one("#catalog", DataTable))
            if key is not None:
                engine_kind, _, model_id = key.partition(":")
                target = self._engine_by_kind(engine_kind)
                if target is not None and target.supports("load"):
                    if not await self._confirm(
                        "Load model?", f"Warm [bold]{model_id}[/] into {target.name}"
                    ):
                        self.query_one("#catalog", DataTable).focus()
                        return
                    self.query_one("#log", RichLog).write(f"[cyan]loading {model_id}…[/]")
                    ok, message = await target.load(model_id)
                    self._log_result(ok, message, source=target.name)
                    await self._force_refresh()
                    self.query_one("#catalog", DataTable).focus()
                    return

        if engine is None:
            self.query_one("#log", RichLog).write("[yellow]select an engine (pane 1) first[/]")
            return
        if not engine.supports("load"):
            self.query_one("#log", RichLog).write(f"[yellow]{engine.name} cannot load[/]")
            return
        snap = self.snapshot
        if snap is None:
            return
        eng_snap = next((e for e in snap.engines if e.kind == engine.kind), None)
        if eng_snap is None or not eng_snap.models:
            self.query_one("#log", RichLog).write("[yellow]no models available to load[/]")
            return
        loaded_ids = {m.id for m in eng_snap.loaded}
        options = [
            (
                m.id,
                f"{m.name}  {m.parameter_size or ''}  {m.quantization or ''}  "
                f"{bytes_human(m.size_bytes)}",
            )
            for m in eng_snap.models
            if m.id not in loaded_ids
        ]
        if not options:
            self.query_one("#log", RichLog).write("[dim]all known models already resident[/]")
            return
        chosen = await self.push_screen_wait(ModelPickerScreen(engine.name, options))
        if not chosen:
            self.query_one("#engines", DataTable).focus()
            return
        self.query_one("#log", RichLog).write(f"[cyan]loading {chosen}…[/]")
        ok, message = await engine.load(chosen)
        self._log_result(ok, message, source=engine.name)
        await self._force_refresh()
        self.query_one("#engines", DataTable).focus()

    async def _pull_model(self) -> None:
        engine = self._selected_engine()
        if engine is None:
            self.query_one("#log", RichLog).write("[yellow]select an engine first[/]")
            return
        if not engine.supports("pull"):
            self.query_one("#log", RichLog).write(f"[yellow]{engine.name} cannot pull[/]")
            return
        model = await self.push_screen_wait(PullScreen(engine.name))
        if not model:
            return
        log = self.query_one("#log", RichLog)
        log.write(f"[cyan]pulling {model} into {engine.name}…[/]")

        def on_progress(tick) -> None:
            status = tick.status or model
            if tick.total_bytes and tick.completed_bytes is not None:
                frac = tick.completed_bytes / tick.total_bytes
                log.write(
                    f"[dim]{status}  {bytes_human(tick.completed_bytes)} / "
                    f"{bytes_human(tick.total_bytes)} ({frac:.0%})[/]"
                )
            else:
                log.write(f"[dim]{status}[/]")

        ok, message = await engine.pull(model, on_progress=on_progress)
        self._log_result(ok, message, source=engine.name)
        await self._force_refresh()
        self.query_one("#catalog", DataTable).focus()

    async def _delete_selected(self) -> None:
        tabs = self.query_one("#tabs", TabbedContent)
        tabs.active = "tab-catalog"
        table = self.query_one("#catalog", DataTable)
        table.focus()
        key = _row_key(table)
        if key is None:
            self.query_one("#log", RichLog).write("[yellow]select a catalog model first[/]")
            return
        engine_kind, _, model_id = key.partition(":")
        target = self._engine_by_kind(engine_kind)
        if target is None:
            return
        if not target.supports("delete"):
            self.query_one("#log", RichLog).write(f"[yellow]{target.name} cannot delete[/]")
            return
        if not await self._confirm(
            "Delete model from disk?",
            f"Permanently remove [bold]{model_id}[/] from {target.name}",
        ):
            table.focus()
            return
        self.query_one("#log", RichLog).write(f"[cyan]deleting {model_id}…[/]")
        ok, message = await target.delete(model_id)
        self._log_result(ok, message, source=target.name)
        await self._force_refresh()
        table.focus()

    async def _lifecycle_selected(self, action: str, *, confirm: bool = False) -> None:
        engine = self._selected_engine()
        if engine is None:
            self.query_one("#log", RichLog).write("[yellow]select an engine row first[/]")
            return
        if not engine.supports("lifecycle"):
            self.query_one("#log", RichLog).write(
                f"[yellow]{engine.name} has no lifecycle control[/]"
            )
            return
        if confirm and not await self._confirm(
            f"{action.title()} engine?", f"{action.title()} [bold]{engine.name}[/]"
        ):
            self.query_one("#engines", DataTable).focus()
            return
        self.query_one("#log", RichLog).write(f"[cyan]{action} {engine.name}…[/]")
        ok, message = await getattr(engine, action)()
        self._log_result(ok, message, source=engine.name)
        await self._force_refresh()
        self.query_one("#engines", DataTable).focus()

    def _selected_engine(self):
        tabs = self.query_one("#tabs", TabbedContent)
        if tabs.active == "tab-engines":
            key = _row_key(self.query_one("#engines", DataTable))
            if key is not None:
                kind = key.split("|", 1)[0]
                return self._engine_by_kind(kind)
        for table_id in ("catalog", "loaded"):
            key = _row_key(self.query_one(f"#{table_id}", DataTable))
            if key is not None:
                kind = key.split(":", 1)[0]
                return self._engine_by_kind(kind)
        key = _row_key(self.query_one("#engines", DataTable))
        if key is not None:
            return self._engine_by_kind(key.split("|", 1)[0])
        return None

    def _engine_by_kind(self, kind: str):
        engines = self.collector.engines.engines
        target = next((e for e in engines if e.kind.value == kind), None)
        if target is None:
            self.query_one("#log", RichLog).write(f"[yellow]no adapter for {kind}[/]")
        return target

    def _log_result(self, ok: bool, message: str, *, source: str) -> None:
        colour = "green" if ok else "yellow"
        self.query_one("#log", RichLog).write(f"[{colour}]{message}[/]")
        self.bus.publish(Topic.LIFECYCLE, message, source=source)

    def action_json_dump(self) -> None:
        if self.snapshot is None:
            return
        self.query_one("#log", RichLog).write(self.snapshot.model_dump_json()[:500] + "…")


def run_tui(
    config: Config | None = None,
    *,
    allow_privileged: bool = True,
    interval: float | None = None,
) -> int:
    app = AiTopApp(config, allow_privileged=allow_privileged, interval=interval)
    app.run()
    return 0
