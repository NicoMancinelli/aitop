"""aitop command-line entry point.

aitop                  # one-shot AI neofetch dashboard (default)
aitop --watch          # refresh it in place until Ctrl-C
aitop tui              # btop-style live Textual dashboard
aitop --json | jq .    # the raw SystemSnapshot
aitop start|stop|…     # lifecycle control
aitop pull MODEL       # pull a model into Ollama
aitop models search q  # search the Hugging Face hub
aitop serve            # expose /api/snapshot for the fleet
aitop update           # check for and install a newer release
aitop doctor           # what aitop can and cannot see on this machine
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn
from rich.table import Table

from aitop import __version__
from aitop.collector import SnapshotCollector
from aitop.config import Config, config_path
from aitop.engines.registry import EngineRegistry
from aitop.hub import search_hub
from aitop.models import DownloadProgress, EngineKind, SystemSnapshot
from aitop.selfupdate import (
    REPO_URL,
    UpdateStatus,
    apply_update,
    check_for_update,
    detect_install_method,
    update_command,
)
from aitop.serve import SnapshotServer, fleet_nodes_from_config, merge_fleet
from aitop.views.neofetch import render_neofetch

log = logging.getLogger("aitop")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aitop",
        description="Telemetry and lifecycle management for local AI runtimes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  aitop                  one-shot dashboard\n"
            "  aitop --watch          refresh every 2s\n"
            "  aitop tui              live btop-style TUI\n"
            "  aitop --json | jq .    machine-readable snapshot\n"
            "  aitop unload ollama    evict resident Ollama weights\n"
            "  aitop load ollama m     warm a model into memory\n"
            "  aitop pull llama3.2    pull a model into Ollama\n"
            "  aitop metrics           Prometheus text exposition\n"
            "  aitop serve            expose this node to the fleet\n"
            "  aitop config init       write a starter config.yaml\n"
            "  aitop update           upgrade to the latest release\n"
            "  aitop doctor           show what telemetry is available here\n"
        ),
    )
    _add_common(parser)
    _add_dashboard_flags(parser)

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    update = subparsers.add_parser(
        "update",
        help="check for and install a newer release",
        description=f"Upgrade aitop in place from {REPO_URL}.",
    )
    _add_common(update)
    update.add_argument(
        "--check",
        action="store_true",
        help="only report whether an update exists; change nothing",
    )
    update.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact command that would run, then stop",
    )
    update.add_argument(
        "--ref",
        default=None,
        metavar="TAG",
        help="install a specific tag or branch instead of the latest release",
    )

    doctor = subparsers.add_parser(
        "doctor",
        help="report install method, config, and telemetry availability",
    )
    _add_common(doctor)

    tui = subparsers.add_parser("tui", help="btop-style live Textual dashboard")
    _add_common(tui)
    tui.add_argument(
        "--interval",
        "-i",
        type=float,
        default=None,
        metavar="SECONDS",
        help="refresh interval (default: from config)",
    )
    tui.add_argument(
        "--no-privileged",
        action="store_true",
        help="never invoke sudo helpers such as powermetrics",
    )

    serve = subparsers.add_parser(
        "serve",
        help="expose /api/snapshot and /api/stream for the fleet",
    )
    _add_common(serve)
    serve.add_argument("--host", default=None, help="bind address (default: from config)")
    serve.add_argument("--port", "-p", type=int, default=None, help="bind port (default: 9090)")
    serve.add_argument(
        "--interval",
        "-i",
        type=float,
        default=None,
        metavar="SECONDS",
        help="snapshot refresh interval",
    )
    serve.add_argument(
        "--no-privileged",
        action="store_true",
        help="never invoke sudo helpers such as powermetrics",
    )

    fleet = subparsers.add_parser(
        "fleet",
        help="show local + remote fleet snapshots",
    )
    _add_common(fleet)
    fleet.add_argument("--json", action="store_true", help="emit JSON array of snapshots")
    fleet.add_argument(
        "--no-privileged",
        action="store_true",
        help="never invoke sudo helpers such as powermetrics",
    )

    for action in ("start", "stop", "restart"):
        sp = subparsers.add_parser(action, help=f"{action} a local AI runtime")
        _add_common(sp)
        sp.add_argument(
            "engine",
            help="runtime kind: ollama, lmstudio, vllm, llama-server, mlx",
        )
        sp.add_argument(
            "--host",
            default=None,
            help="optional host:port override when selecting the endpoint",
        )

    unload = subparsers.add_parser("unload", help="evict resident model weights")
    _add_common(unload)
    unload.add_argument("engine", help="runtime kind or endpoint name")
    unload.add_argument("model", nargs="?", default=None, help="model id (default: all)")

    load = subparsers.add_parser("load", help="warm a model into memory/VRAM")
    _add_common(load)
    load.add_argument("engine", help="runtime kind or endpoint name")
    load.add_argument("model", help="model id to load")

    rebind = subparsers.add_parser(
        "rebind",
        help="rebind an engine to a new host (e.g. Tailscale IP)",
    )
    _add_common(rebind)
    rebind.add_argument("engine", help="runtime kind (currently: ollama)")
    rebind.add_argument("host", help="new bind address, e.g. 0.0.0.0 or 100.x.y.z")

    pull = subparsers.add_parser("pull", help="pull a model into a local engine")
    _add_common(pull)
    pull.add_argument("model", help="model name, e.g. llama3.2:3b")
    pull.add_argument(
        "--engine",
        default="ollama",
        help="runtime to pull into (default: ollama)",
    )

    metrics = subparsers.add_parser(
        "metrics",
        help="emit a Prometheus text exposition of the current snapshot",
    )
    _add_common(metrics)
    metrics.add_argument(
        "--no-privileged",
        action="store_true",
        help="never invoke sudo helpers such as powermetrics",
    )

    cfg = subparsers.add_parser("config", help="config file helpers")
    _add_common(cfg)
    cfg_sub = cfg.add_subparsers(dest="config_command", required=True)
    init_cfg = cfg_sub.add_parser("init", help="write a starter config.yaml")
    _add_common(init_cfg)
    init_cfg.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing config file",
    )
    init_cfg.add_argument(
        "--path",
        type=Path,
        default=None,
        help=f"destination path (default: {config_path()})",
    )

    models = subparsers.add_parser("models", help="model hub helpers")
    _add_common(models)
    models_sub = models.add_subparsers(dest="models_command", required=True)
    search = models_sub.add_parser("search", help="search the Hugging Face hub")
    _add_common(search)
    search.add_argument("query", help="search string")
    search.add_argument("--limit", type=int, default=15, help="max results (default: 15)")
    search.add_argument(
        "--tag",
        default="gguf",
        help="HF filter tag (default: gguf; pass '' for none)",
    )

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"config file to load (default: {config_path()})",
    )
    parser.add_argument("--no-color", action="store_true", help="disable colour output")
    parser.add_argument("--debug", action="store_true", help="verbose logging to stderr")


def _add_dashboard_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--version", action="version", version=f"aitop {__version__}")
    parser.add_argument(
        "--neofetch",
        action="store_true",
        help="render the startup dashboard (the default; accepted for symmetry)",
    )
    parser.add_argument(
        "--watch",
        "-w",
        action="store_true",
        help="refresh the dashboard in place until interrupted",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=None,
        metavar="SECONDS",
        help="refresh interval for --watch (default: from config, 2.0s)",
    )
    parser.add_argument("--json", action="store_true", help="emit the raw snapshot as JSON")
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="include endpoints that were probed but are offline",
    )
    parser.add_argument("--no-logo", action="store_true", help="skip the ASCII header")
    parser.add_argument(
        "--no-privileged",
        action="store_true",
        help="never invoke sudo helpers such as powermetrics",
    )
    parser.add_argument(
        "--no-update-check",
        action="store_true",
        help="skip the cached check for a newer release",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.debug)

    config = Config.load(args.config)
    console = Console(no_color=args.no_color, stderr=False)

    try:
        match args.command:
            case "update":
                return asyncio.run(_run_update(args, config, console))
            case "doctor":
                return asyncio.run(_run_doctor(args, config, console))
            case "tui":
                return _run_tui(args, config)
            case "serve":
                return asyncio.run(_run_serve(args, config, console))
            case "fleet":
                return asyncio.run(_run_fleet(args, config, console))
            case "start" | "stop" | "restart":
                return asyncio.run(_run_lifecycle(args, config, console))
            case "unload":
                return asyncio.run(_run_unload(args, config, console))
            case "load":
                return asyncio.run(_run_load(args, config, console))
            case "rebind":
                return asyncio.run(_run_rebind(args, config, console))
            case "pull":
                return asyncio.run(_run_pull(args, config, console))
            case "metrics":
                return asyncio.run(_run_metrics(args, config, console))
            case "config":
                return _run_config(args, console)
            case "models":
                return asyncio.run(_run_models(args, config, console))
            case _:
                return asyncio.run(_run_dashboard(args, config, console))
    except KeyboardInterrupt:
        console.print()
        return 130


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #


async def _run_dashboard(args: argparse.Namespace, config: Config, console: Console) -> int:
    collector = SnapshotCollector(config, allow_privileged=not args.no_privileged)
    try:
        if args.json:
            snapshot = await collector.collect()
            print(snapshot.model_dump_json(indent=2))
            return 0

        if args.watch:
            await _watch(collector, args, config, console)
            return 0

        snapshot, update = await _collect_with_status(collector, args, config, console)
        console.print(_render(snapshot, args, update))
        if update.available and config.updates.auto_apply:
            await _auto_apply(console)
        return 0
    finally:
        await collector.aclose()


async def _collect_with_status(
    collector: SnapshotCollector,
    args: argparse.Namespace,
    config: Config,
    console: Console,
) -> tuple[SystemSnapshot, UpdateStatus]:
    """Collect telemetry and check for updates concurrently.

    The first collection takes ~1s (system_profiler, sudo probing), so the
    update check rides along for free and never extends the wall clock.
    """

    async def both() -> tuple[SystemSnapshot, UpdateStatus]:
        return await asyncio.gather(  # type: ignore[return-value]
            collector.collect(),
            check_for_update(
                enabled=config.updates.check and not args.no_update_check,
                ttl=config.updates.ttl_seconds,
                timeout=config.updates.timeout,
            ),
        )

    if args.no_color or not console.is_terminal:
        return await both()
    with console.status("[cyan]probing hardware and AI runtimes…", spinner="dots"):
        return await both()


async def _watch(
    collector: SnapshotCollector,
    args: argparse.Namespace,
    config: Config,
    console: Console,
) -> None:
    interval = args.interval or config.polling.hardware_interval
    snapshot, update = await _collect_with_status(collector, args, config, console)
    with Live(
        _render(snapshot, args, update),
        console=console,
        refresh_per_second=4,
        screen=False,
        transient=False,
    ) as live:
        while True:
            await asyncio.sleep(max(0.25, interval))
            snapshot = await collector.collect()
            live.update(_render(snapshot, args, update))


def _render(snapshot: SystemSnapshot, args: argparse.Namespace, update: UpdateStatus | None):
    return render_neofetch(
        snapshot,
        show_logo=not args.no_logo,
        color=not args.no_color,
        show_offline=args.all,
        update=update,
    )


async def _auto_apply(console: Console) -> None:
    console.print("[cyan]updates.auto_apply is on — installing…")
    ok, message, _ = await apply_update()
    console.print(f"[{'green' if ok else 'yellow'}]{message}")


# --------------------------------------------------------------------------- #
# TUI / serve / fleet
# --------------------------------------------------------------------------- #


def _run_tui(args: argparse.Namespace, config: Config) -> int:
    from aitop.views.tui import run_tui

    return run_tui(
        config,
        allow_privileged=not args.no_privileged,
        interval=args.interval,
    )


async def _run_serve(args: argparse.Namespace, config: Config, console: Console) -> int:
    host = args.host or config.fleet.serve_host
    port = args.port or config.fleet.serve_port
    collector = SnapshotCollector(config, allow_privileged=not args.no_privileged)
    server = SnapshotServer(
        collector,
        host=host,
        port=port,
        interval=args.interval,
    )
    console.print(f"[bold]aitop serve[/] http://{host}:{port}")
    console.print("[dim]/healthz  /api/snapshot  /api/stream  /metrics[/]")
    try:
        await server.run()
    finally:
        await collector.aclose()
    return 0


async def _run_fleet(args: argparse.Namespace, config: Config, console: Console) -> int:
    collector = SnapshotCollector(config, allow_privileged=not args.no_privileged)
    try:
        local = await collector.collect()
        snapshots = await merge_fleet(local, fleet_nodes_from_config(config))
        if args.json:
            import json

            print(json.dumps([s.model_dump(mode="json") for s in snapshots], indent=2))
            return 0

        for snap in snapshots:
            console.print(f"[bold cyan]▸ node[/] {snap.node}")
            console.print(
                render_neofetch(snap, show_logo=False, color=not args.no_color, show_offline=False)
            )
            console.print()
        if len(snapshots) == 1 and not config.fleet.nodes:
            console.print(
                "[dim]No fleet.nodes configured. Add peers under fleet.nodes in "
                f"{config_path()}, or point another host's aitop serve here.[/]"
            )
        return 0
    finally:
        await collector.aclose()


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


async def _resolve_engine(config: Config, kind_or_name: str, host: str | None = None):
    from aitop.config import EndpointConfig
    from aitop.engines.registry import ADAPTERS
    from aitop.utils.parse import split_host_port

    registry = EngineRegistry(config)
    engines = registry.build()
    needle = kind_or_name.lower()
    matches = [
        e
        for e in engines
        if e.kind.value == needle or e.name.lower() == needle or e.display_name.lower() == needle
    ]
    if host:
        matches = [e for e in matches if e.host == host or f"{e.host}:{e.port}" == host]
    if matches:
        return matches[0], registry, None

    try:
        kind = EngineKind(needle)
    except ValueError:
        await registry.aclose()
        return None, registry, f"unknown engine {kind_or_name!r}"

    adapter_cls = ADAPTERS.get(kind)
    if adapter_cls is None:
        await registry.aclose()
        return None, registry, f"no adapter for {kind.value}"

    if host:
        h, p = split_host_port(host, EndpointConfig(kind=kind).resolved_port())
        ep = EndpointConfig(kind=kind, host=h, port=p)
    else:
        ep = EndpointConfig(kind=kind)

    engine = adapter_cls(ep, client=registry._client)
    return engine, registry, None


async def _run_lifecycle(args: argparse.Namespace, config: Config, console: Console) -> int:
    engine, registry, err = await _resolve_engine(config, args.engine, getattr(args, "host", None))
    try:
        if err or engine is None:
            console.print(f"[yellow]{err}")
            return 1
        action = args.command
        console.print(f"[cyan]{action} {engine.name}…")
        ok, message = await getattr(engine, action)()
        console.print(f"[{'green' if ok else 'yellow'}]{message}")
        return 0 if ok else 1
    finally:
        await registry.aclose()


async def _run_unload(args: argparse.Namespace, config: Config, console: Console) -> int:
    engine, registry, err = await _resolve_engine(config, args.engine)
    try:
        if err or engine is None:
            console.print(f"[yellow]{err}")
            return 1
        if not engine.supports("unload"):
            console.print(f"[yellow]{engine.name} does not support unload")
            return 1
        ok, message = await engine.unload(args.model)
        console.print(f"[{'green' if ok else 'yellow'}]{message}")
        return 0 if ok else 1
    finally:
        await registry.aclose()


async def _run_load(args: argparse.Namespace, config: Config, console: Console) -> int:
    engine, registry, err = await _resolve_engine(config, args.engine)
    try:
        if err or engine is None:
            console.print(f"[yellow]{err}")
            return 1
        if not engine.supports("load"):
            console.print(f"[yellow]{engine.name} does not support load")
            return 1
        console.print(f"[cyan]loading {args.model} on {engine.name}…")
        ok, message = await engine.load(args.model)
        console.print(f"[{'green' if ok else 'yellow'}]{message}")
        return 0 if ok else 1
    finally:
        await registry.aclose()


async def _run_metrics(args: argparse.Namespace, config: Config, console: Console) -> int:
    from aitop.prometheus import render_prometheus

    collector = SnapshotCollector(config, allow_privileged=not args.no_privileged)
    try:
        snapshot = await collector.collect()
        # Prometheus scrapers expect plain text on stdout — no Rich markup.
        sys.stdout.write(render_prometheus(snapshot))
        return 0
    finally:
        await collector.aclose()


_SAMPLE_CONFIG = """\
# aitop config — generated by `aitop config init`
# Docs: https://github.com/NicoMancinelli/aitop

polling:
  hardware_interval: 2.0
  engine_interval: 2.0

ui:
  show_per_core: true

updates:
  check: true
  auto_apply: false

fleet:
  serve_host: 127.0.0.1
  serve_port: 9090
  # nodes:
  #   - name: pveclaw
  #     url: http://100.100.1.7:9090

# endpoints:
#   - kind: ollama
#     host: 100.100.1.7
#     name: pveclaw-ollama
#     remote: true
#   - kind: lmstudio
#     enabled: false
"""


def _run_config(args: argparse.Namespace, console: Console) -> int:
    if args.config_command != "init":
        console.print("[yellow]unknown config subcommand")
        return 1
    path = args.path or config_path()
    if path.exists() and not args.force:
        console.print(f"[yellow]{path} already exists — pass --force to overwrite")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_SAMPLE_CONFIG)
    console.print(f"[green]wrote {path}")
    return 0


async def _run_rebind(args: argparse.Namespace, config: Config, console: Console) -> int:
    engine, registry, err = await _resolve_engine(config, args.engine)
    try:
        if err or engine is None:
            console.print(f"[yellow]{err}")
            return 1
        if not engine.supports("rebind"):
            console.print(f"[yellow]{engine.name} does not support rebind")
            return 1
        # Offer Tailscale IP as a hint when the user asked for "tailscale".
        host = args.host
        if host.lower() in {"tailscale", "tailnet", "ts"}:
            from aitop.net.tailscale import collect_tailscale

            ts = await collect_tailscale()
            if not ts.ipv4:
                console.print("[yellow]no Tailscale IPv4 available on this node")
                return 1
            host = ts.ipv4
            console.print(f"[dim]using Tailscale IP {host}[/]")
        ok, message = await engine.rebind(host)
        console.print(f"[{'green' if ok else 'yellow'}]{message}")
        return 0 if ok else 1
    finally:
        await registry.aclose()


async def _run_pull(args: argparse.Namespace, config: Config, console: Console) -> int:
    engine, registry, err = await _resolve_engine(config, args.engine)
    try:
        if err or engine is None:
            console.print(f"[yellow]{err}")
            return 1
        if not engine.supports("pull"):
            console.print(f"[yellow]{engine.name} does not support pull")
            return 1

        progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            console=console,
            transient=True,
        )
        task_id = None

        def on_progress(tick: DownloadProgress) -> None:
            nonlocal task_id
            total = tick.total_bytes or 0
            completed = tick.completed_bytes or 0
            if task_id is None:
                task_id = progress.add_task(tick.status or args.model, total=total or None)
            progress.update(
                task_id,
                description=tick.status or args.model,
                completed=completed,
                total=total or None,
            )

        with progress:
            ok, message = await engine.pull(args.model, on_progress=on_progress)
        console.print(f"[{'green' if ok else 'yellow'}]{message}")
        return 0 if ok else 1
    finally:
        await registry.aclose()


async def _run_models(args: argparse.Namespace, config: Config, console: Console) -> int:
    if args.models_command != "search":
        console.print("[yellow]unknown models subcommand")
        return 1
    tag = args.tag or None
    results = await search_hub(args.query, limit=args.limit, filter_tag=tag)
    if not results:
        console.print("[yellow]no results (or hub unreachable)")
        return 1

    table = Table(box=None, pad_edge=False, padding=(0, 2))
    table.add_column("MODEL", style="bold")
    table.add_column("DOWNLOADS", justify="right")
    table.add_column("LIKES", justify="right")
    table.add_column("TAGS")
    for model in results:
        tags = ", ".join(model.tags[:4])
        table.add_row(
            model.id,
            f"{model.downloads:,}" if model.downloads is not None else "—",
            f"{model.likes:,}" if model.likes is not None else "—",
            tags,
        )
    console.print(table)
    console.print(
        "\n[dim]Pull into Ollama when a matching library tag exists:[/] "
        f"aitop pull {results[0].id.split('/')[-1]}"
    )
    return 0


# --------------------------------------------------------------------------- #
# aitop update
# --------------------------------------------------------------------------- #


async def _run_update(args: argparse.Namespace, config: Config, console: Console) -> int:
    method = detect_install_method()
    status = await check_for_update(force=True, timeout=max(config.updates.timeout, 10.0))

    console.print(f"[bold]installed via[/] {method.value}")
    console.print(f"[bold]current[/]      {status.current}")

    if not status.checked:
        console.print(f"[yellow]{status.describe()}")
        # A missing release feed is not a failure of the user's command.
        return 0 if args.check else 1

    console.print(f"[bold]latest[/]       {status.latest}")

    if not status.available and not args.ref:
        console.print("[green]already up to date")
        return 0

    if args.check:
        console.print(f"[bold green]{status.describe()}")
        return 0

    ref = args.ref or (f"v{status.latest}" if status.latest else None)
    ok, message, argv = await apply_update(ref=ref, dry_run=args.dry_run)

    if argv and not ok:
        console.print(f"[yellow]{message}")
        console.print(f"[dim]tried: {' '.join(argv)}")
        return 1
    if not ok:
        console.print(f"[yellow]{message}")
        _print_manual_instructions(console, method)
        return 1

    console.print(f"[green]{message}")
    return 0


def _print_manual_instructions(console: Console, method) -> None:
    argv = update_command(method, None)
    console.print("\n[bold]To update manually:")
    if argv:
        console.print(f"  {' '.join(argv)}")
    else:
        console.print(f"  uv tool install --force git+{REPO_URL}")


# --------------------------------------------------------------------------- #
# aitop doctor
# --------------------------------------------------------------------------- #


async def _run_doctor(args: argparse.Namespace, config: Config, console: Console) -> int:
    from aitop.hardware.collector import HardwareCollector
    from aitop.selfupdate import cache_path, repo_checkout

    table = Table(box=None, pad_edge=False, padding=(0, 2))
    table.add_column("", style="bold bright_cyan", no_wrap=True)
    table.add_column("")

    method = detect_install_method()
    checkout = repo_checkout()
    table.add_row("version", __version__)
    table.add_row("installed via", method.value + ("" if method.self_updatable else "  (manual)"))
    if checkout:
        table.add_row("checkout", str(checkout))
    table.add_row("python", sys.version.split()[0])
    table.add_row("config", str(args.config or config_path()))
    table.add_row("config loaded", "yes" if config.source else "no — using defaults")
    table.add_row("update cache", str(cache_path()))
    table.add_row(
        "fleet nodes",
        ", ".join(n.name for n in config.fleet.nodes) or "none",
    )

    collector = HardwareCollector(allow_privileged=True)
    probes = await collector.probes()
    table.add_row("hardware probes", ", ".join(p.name for p in probes) or "none")

    snapshot = await collector.collect()
    table.add_row("gpus", ", ".join(g.name for g in snapshot.gpus) or "none detected")

    endpoints = config.all_endpoints()
    table.add_row(
        "endpoints probed",
        ", ".join(f"{e.kind.value}@{e.host}:{e.resolved_port()}" for e in endpoints),
    )
    table.add_row(
        "adapters",
        ", ".join(k.value for k in EngineRegistry.supported),
    )

    console.print(table)

    if snapshot.degraded:
        console.print("\n[bold]Unavailable telemetry:")
        for note in snapshot.degraded:
            console.print(f"  [dim]· {note}")
    return 0


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
