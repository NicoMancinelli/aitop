"""CLI argument handling and the JSON / one-shot output paths."""

from __future__ import annotations

import json

import pytest

from aitop import cli
from aitop.models import EngineKind, EngineSnapshot, EngineState, SystemSnapshot
from aitop.selfupdate import InstallMethod, UpdateStatus


class FakeCollector:
    """Stands in for SnapshotCollector so the CLI tests touch no hardware."""

    instances: list[FakeCollector] = []

    def __init__(self, config=None, bus=None, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.closed = False
        self.collect_calls = 0
        FakeCollector.instances.append(self)

    async def collect(self) -> SystemSnapshot:
        self.collect_calls += 1
        return SystemSnapshot(
            engines=[
                EngineSnapshot(
                    kind=EngineKind.OLLAMA,
                    name="Ollama",
                    state=EngineState.ONLINE,
                    version="0.5.7",
                )
            ]
        )

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_collector(monkeypatch):
    FakeCollector.instances.clear()
    monkeypatch.setattr(cli, "SnapshotCollector", FakeCollector)
    return FakeCollector


def test_json_mode_emits_a_valid_snapshot(capsys):
    assert cli.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engines"][0]["kind"] == "ollama"
    assert payload["engines"][0]["version"] == "0.5.7"
    assert "hardware" in payload and "tailscale" in payload


def test_default_mode_renders_the_dashboard(capsys):
    assert cli.main(["--no-color"]) == 0
    out = capsys.readouterr().out
    assert "Ollama" in out
    assert "Runtimes" in out


def test_collector_is_always_closed():
    cli.main(["--json"])
    assert FakeCollector.instances[0].closed is True


def test_no_privileged_flag_reaches_the_collector():
    cli.main(["--json", "--no-privileged"])
    assert FakeCollector.instances[0].kwargs["allow_privileged"] is False
    cli.main(["--json"])
    assert FakeCollector.instances[1].kwargs["allow_privileged"] is True


def test_version_flag_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert "aitop" in capsys.readouterr().out


def test_config_flag_is_honoured(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("polling:\n  hardware_interval: 9.0\n")
    cli.main(["--json", "--config", str(path)])
    assert FakeCollector.instances[0].config.polling.hardware_interval == 9.0


def test_missing_config_file_is_not_fatal(tmp_path):
    assert cli.main(["--json", "--config", str(tmp_path / "absent.yaml")]) == 0


def test_keyboard_interrupt_returns_the_conventional_code(monkeypatch):
    async def interrupted(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(FakeCollector, "collect", interrupted)

    assert cli.main(["--no-color"]) == 130
    # Ctrl-C must still release the HTTP client.
    assert FakeCollector.instances[0].closed is True


def test_parser_accepts_the_documented_flags():
    args = cli.build_parser().parse_args(
        ["--watch", "--interval", "0.5", "--all", "--no-logo", "--no-color", "--neofetch"]
    )
    assert args.watch and args.all and args.no_logo and args.no_color and args.neofetch
    assert args.interval == 0.5
    assert args.command is None  # bare invocation = dashboard


# --------------------------------------------------------------------------- #
# aitop update
# --------------------------------------------------------------------------- #


@pytest.fixture
def stub_update(monkeypatch):
    """Control what `aitop update` sees without touching the network."""

    state = {
        "status": UpdateStatus(current="0.1.0", latest="0.2.0", checked=True),
        "method": InstallMethod.UV_TOOL,
        "applied": [],
        "result": (True, "updated via uv tool — restart aitop to use the new version", ["uv"]),
    }

    async def fake_check(**kwargs):
        return state["status"]

    async def fake_apply(*, ref=None, dry_run=False):
        state["applied"].append((ref, dry_run))
        return state["result"]

    monkeypatch.setattr(cli, "check_for_update", fake_check)
    monkeypatch.setattr(cli, "apply_update", fake_apply)
    monkeypatch.setattr(cli, "detect_install_method", lambda: state["method"])
    return state


def test_update_installs_the_latest_release(stub_update, capsys):
    assert cli.main(["update", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "0.2.0" in out
    assert "restart aitop" in out
    # The newest release tag is what gets pinned, not the default branch.
    assert stub_update["applied"] == [("v0.2.0", False)]


def test_update_check_changes_nothing(stub_update, capsys):
    assert cli.main(["update", "--check", "--no-color"]) == 0
    assert "0.2.0 is available" in capsys.readouterr().out
    assert stub_update["applied"] == []


def test_update_is_a_noop_when_current(stub_update, capsys):
    stub_update["status"] = UpdateStatus(current="0.2.0", latest="0.2.0", checked=True)
    assert cli.main(["update", "--no-color"]) == 0
    assert "already up to date" in capsys.readouterr().out
    assert stub_update["applied"] == []


def test_update_dry_run_is_forwarded(stub_update):
    cli.main(["update", "--dry-run", "--no-color"])
    assert stub_update["applied"] == [("v0.2.0", True)]


def test_update_honours_an_explicit_ref(stub_update):
    cli.main(["update", "--ref", "main", "--no-color"])
    assert stub_update["applied"] == [("main", False)]


def test_update_explains_a_dev_checkout(stub_update, capsys):
    stub_update["method"] = InstallMethod.GIT_CHECKOUT
    stub_update["result"] = (False, "this is a development checkout — run `git pull`", None)

    assert cli.main(["update", "--no-color"]) == 1
    out = capsys.readouterr().out
    assert "git pull" in out
    assert "To update manually" in out


def test_update_survives_a_repo_with_no_releases(stub_update, capsys):
    stub_update["status"] = UpdateStatus(current="0.1.0", error="no releases published yet")
    assert cli.main(["update", "--check", "--no-color"]) == 0
    assert "no releases published yet" in capsys.readouterr().out


def test_update_reports_a_failed_install(stub_update, capsys):
    stub_update["result"] = (False, "uv: exit 1 (network unreachable)", ["uv", "tool", "install"])
    assert cli.main(["update", "--no-color"]) == 1
    out = capsys.readouterr().out
    assert "network unreachable" in out
    assert "tried:" in out


# --------------------------------------------------------------------------- #
# aitop doctor
# --------------------------------------------------------------------------- #


def test_doctor_reports_the_environment(capsys):
    assert cli.main(["doctor", "--no-color"]) == 0
    out = capsys.readouterr().out
    for expected in ("version", "installed via", "python", "config", "hardware probes"):
        assert expected in out


def test_dashboard_skips_the_update_check_when_asked(monkeypatch, capsys):
    called = []

    async def fake_check(**kwargs):
        called.append(kwargs)
        return UpdateStatus(current="0.1.0")

    monkeypatch.setattr(cli, "check_for_update", fake_check)
    cli.main(["--no-color", "--no-update-check"])
    assert called and called[0]["enabled"] is False
