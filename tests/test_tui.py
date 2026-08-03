"""TUI helpers and formatting used by the live dashboard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aitop.models import (
    Binding,
    BindScope,
    EngineKind,
    EngineSnapshot,
    EngineState,
)
from aitop.utils.fmt import core_bars, sparkline
from aitop.views.tui import _age_label, _engine_key, _row_key


def test_core_bars_empty():
    text = core_bars([])
    assert "—" in text.plain


def test_core_bars_layout():
    text = core_bars([10, 50, 90, 100], width=3, cols=2)
    # 2 columns => newline between pairs
    assert "\n" in text.plain
    assert text.plain.count("\n") == 1


def test_sparkline_width():
    assert len(sparkline([0, 50, 100], width=12)) == 12


def test_engine_key_is_stable():
    a = EngineSnapshot(
        kind=EngineKind.OLLAMA,
        name="Ollama",
        state=EngineState.ONLINE,
        binding=Binding(host="127.0.0.1", port=11434, scope=BindScope.LOOPBACK),
    )
    b = EngineSnapshot(
        kind=EngineKind.OLLAMA,
        name="Ollama",
        state=EngineState.ONLINE,
        binding=Binding(host="127.0.0.1", port=11434, scope=BindScope.LOOPBACK),
    )
    assert _engine_key(a) == _engine_key(b)
    assert "ollama" in _engine_key(a)


def test_age_label():
    assert _age_label(None) == "—"
    recent = datetime.now(UTC) - timedelta(seconds=12)
    assert _age_label(recent).endswith("s ago")
    older = datetime.now(UTC) - timedelta(minutes=5)
    assert _age_label(older).endswith("m ago")


def _make_snap(*, offline: bool = False, error: str | None = None):
    from aitop.models import (
        CPUSnapshot,
        HardwareSnapshot,
        HostSnapshot,
        InferenceStats,
        LoadedModel,
        MemorySnapshot,
        ModelInfo,
        SystemSnapshot,
        TailscaleStatus,
    )

    engines = [
        EngineSnapshot(
            kind=EngineKind.OLLAMA,
            name="Ollama",
            state=EngineState.OFFLINE if offline else EngineState.ONLINE,
            binding=Binding(host="127.0.0.1", port=11434, scope=BindScope.LOOPBACK),
            version="0.5.7",
            pid=4242,
            managed_by="systemd",
            error=error,
            models=[
                ModelInfo(
                    id="llama3.2:3b",
                    name="llama3.2:3b",
                    engine=EngineKind.OLLAMA,
                    parameter_size="3B",
                    quantization="Q4_K_M",
                    format="gguf",
                    size_bytes=2_000_000_000,
                    max_context=8192,
                    family="llama",
                ),
                ModelInfo(
                    id="tinyllama",
                    name="tinyllama",
                    engine=EngineKind.OLLAMA,
                    size_bytes=600_000_000,
                ),
            ],
            loaded=[
                LoadedModel(
                    id="llama3.2:3b",
                    name="llama3.2:3b",
                    engine=EngineKind.OLLAMA,
                    parameter_size="3B",
                    quantization="Q4_K_M",
                    size_bytes=2_000_000_000,
                    vram_bytes=1_500_000_000,
                    context_length=8192,
                    context_used=1024,
                )
            ]
            if not offline
            else [],
            stats=InferenceStats(tokens_per_second=33.0, queue_depth=1, ttft_ms=40.0),
            latency_ms=4.0,
        )
    ]
    return SystemSnapshot(
        hardware=HardwareSnapshot(
            host=HostSnapshot(hostname="testbox"),
            cpu=CPUSnapshot(
                model="Test CPU",
                load_percent=42.0,
                logical_cores=4,
                per_core_percent=[10, 20, 30, 40],
            ),
            memory=MemorySnapshot(
                total_bytes=8 * 1024**3,
                used_bytes=4 * 1024**3,
                available_bytes=4 * 1024**3,
            ),
            degraded=["nvml unavailable"],
        ),
        engines=engines,
        tailscale=TailscaleStatus(
            available=True,
            running=True,
            ipv4="100.64.0.1",
            hostname="testbox",
            tailnet="example.ts.net",
            peer_count=3,
        ),
        duration_ms=12.0,
    )


async def test_tui_smoke_mount_and_render():
    """Mount the app under Textual's test harness and feed one snapshot."""
    from aitop.config import Config
    from aitop.views.tui import AiTopApp

    snap = _make_snap()
    app = AiTopApp(Config(), allow_privileged=False, interval=60.0)

    async def noop_stream(interval=None):
        return

    app.collector.stream = noop_stream  # type: ignore[method-assign]

    async with app.run_test(size=(140, 40)) as pilot:
        app.snapshot = snap
        app._render_snapshot(snap)
        await pilot.pause()

        engines = app.query_one("#engines")
        assert engines.row_count == 1
        catalog = app.query_one("#catalog")
        assert catalog.row_count == 2
        loaded = app.query_one("#loaded")
        assert loaded.row_count == 1

        # Cursor preservation across re-render
        engines.move_cursor(row=0)
        key = _row_key(engines)
        app._render_engines(snap)
        assert _row_key(engines) == key

        # Detail strip picks up pid / managed_by
        detail = app.query_one("#detail").render().plain
        assert "pid 4242" in detail
        assert "systemd" in detail

        # Status shows Tailscale peers + degraded probe
        status = app.query_one("#status").render().plain
        assert "100.64.0.1" in status
        assert "peers" in status
        assert "nvml" in status

        # Pause toggles
        await pilot.press("space")
        assert app.paused is True
        await pilot.press("space")
        assert app.paused is False

        # Offline filter
        await pilot.press("a")
        assert app.show_offline is True

        # Tab jumps
        await pilot.press("2")
        assert app.query_one("#tabs").active == "tab-catalog"
        await pilot.press("3")
        assert app.query_one("#tabs").active == "tab-loaded"
        await pilot.press("1")
        assert app.query_one("#tabs").active == "tab-engines"

        # Sort cycles
        await pilot.press("o")
        assert app.catalog_sort == "size"
        await pilot.press("o")
        assert app.catalog_sort == "state"
        await pilot.press("o")
        assert app.catalog_sort == "name"

        # Help modal opens and closes
        await pilot.press("?")
        assert len(app.screen_stack) > 1
        await pilot.press("escape")
        await pilot.pause()


async def test_tui_empty_states_and_filter():
    from aitop.config import Config
    from aitop.views.tui import AiTopApp

    snap = _make_snap(offline=True)
    app = AiTopApp(Config(), allow_privileged=False, interval=60.0)

    async def noop_stream(interval=None):
        return

    app.collector.stream = noop_stream  # type: ignore[method-assign]

    async with app.run_test(size=(120, 40)) as pilot:
        app.snapshot = snap
        app._render_snapshot(snap)
        await pilot.pause()

        engines = app.query_one("#engines")
        # Offline hidden by default → empty placeholder row
        assert engines.row_count == 1
        assert _row_key(engines) is None

        await pilot.press("a")
        app._render_snapshot(snap)
        assert engines.row_count == 1
        assert _row_key(engines) is not None

        # Catalog filter via state (modal Input is awkward in pilot)
        app.catalog_filter = "tiny"
        app._render_catalog(snap)
        catalog = app.query_one("#catalog")
        assert catalog.row_count == 1

        app.catalog_filter = "nope-no-match"
        app._render_catalog(snap)
        assert catalog.row_count == 1
        # Empty placeholder uses _EMPTY key → _row_key returns None
        assert _row_key(catalog) is None

        # Loaded empty
        loaded = app.query_one("#loaded")
        app._render_loaded(snap)
        assert loaded.row_count == 1
        assert _row_key(loaded) is None


async def test_confirm_and_filter_screens():
    # Smoke-compose each modal inside a tiny host app.
    from textual.app import App, ComposeResult
    from textual.widgets import Static

    from aitop.views.tui_screens import ConfirmScreen, FilterScreen, HelpScreen

    class Host(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = Host()
    async with app.run_test() as pilot:
        app.push_screen(HelpScreen())
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        result_box: list[bool | None] = []

        def _capture(value: bool) -> None:
            result_box.append(value)

        app.push_screen(ConfirmScreen("Stop?", "Stop [bold]ollama[/]"), _capture)
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
        assert result_box == [True]

        filt: list[str | None] = []
        app.push_screen(FilterScreen("llama"), lambda v: filt.append(v))
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert filt == ["llama"]
