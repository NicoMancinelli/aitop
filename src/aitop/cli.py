"""aitop command-line entry point.

    aitop              # one-shot AI neofetch dashboard (default)
    aitop --watch      # refresh it in place until Ctrl-C
    aitop --json       # the raw SystemSnapshot
    aitop update       # check for and install a newer release
    aitop doctor       # what aitop can and cannot see on this machine

`--json` emits the same payload the future web UI, Prometheus exporter and
fleet stream will consume.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live

from aitop import __version__
from aitop.collector import SnapshotCollector
from aitop.config import Config, config_path
from aitop.models import SystemSnapshot
from aitop.selfupdate import (
    REPO_URL,
    UpdateStatus,
    apply_update,
    check_for_update,
    detect_install_method,
    update_command,
)
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
            "  aitop --json | jq .    machine-readable snapshot\n"
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
    from rich.table import Table

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
