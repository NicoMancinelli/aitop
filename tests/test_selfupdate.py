"""Version comparison, install-method detection, and the update flow."""

from __future__ import annotations

import json
import sys
import time

import httpx
import pytest
import respx

from aitop import selfupdate
from aitop.selfupdate import (
    GIT_SPEC,
    RELEASES_API,
    InstallMethod,
    UpdateStatus,
    apply_update,
    check_for_update,
    detect_install_method,
    parse_version,
    update_command,
)
from aitop.utils.proc import CommandResult


@pytest.fixture(autouse=True)
def allow_update_checks(monkeypatch):
    """This module tests the checker itself, so re-enable it.

    The cache is still redirected to tmp_path by the session-wide
    `no_network_update_checks` fixture in conftest.py; respx handles the HTTP.
    """
    monkeypatch.delenv("AITOP_NO_UPDATE_CHECK", raising=False)


# --------------------------------------------------------------------------- #
# Version comparison
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("0.1.0", "0.2.0"),
        ("0.1.0", "0.1.1"),
        ("0.9.9", "1.0.0"),
        ("1.0.0rc1", "1.0.0"),  # pre-release sorts below the final
        ("1.2", "1.2.1"),
        ("v0.1.0", "v0.10.0"),  # numeric, not lexicographic
        ("garbage", "0.0.1"),
    ],
)
def test_version_ordering(lower, higher):
    assert parse_version(lower) < parse_version(higher)


@pytest.mark.parametrize("pair", [("1.2.3", "v1.2.3"), ("0.1.0", "0.1.0")])
def test_version_equality_ignores_the_v_prefix(pair):
    assert parse_version(pair[0]) == parse_version(pair[1])


def test_update_status_availability():
    assert UpdateStatus(current="0.1.0", latest="0.2.0", checked=True).available
    assert not UpdateStatus(current="0.2.0", latest="0.2.0", checked=True).available
    assert not UpdateStatus(current="0.3.0", latest="0.2.0", checked=True).available
    assert not UpdateStatus(current="0.1.0").available  # never checked


# --------------------------------------------------------------------------- #
# Install method detection
# --------------------------------------------------------------------------- #


def test_detects_a_git_checkout(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='aitop'\n")
    pkg = tmp_path / "src" / "aitop"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(selfupdate, "package_root", lambda: pkg)

    assert detect_install_method() is InstallMethod.GIT_CHECKOUT
    assert selfupdate.repo_checkout() == tmp_path


def test_detects_uv_tool_install(monkeypatch, tmp_path):
    pkg = tmp_path / "uv" / "tools" / "aitop" / "lib" / "aitop"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(selfupdate, "package_root", lambda: pkg)
    monkeypatch.setattr(selfupdate, "repo_checkout", lambda: None)

    assert detect_install_method() is InstallMethod.UV_TOOL


def test_detects_pipx_install(monkeypatch, tmp_path):
    pkg = tmp_path / "pipx" / "venvs" / "aitop" / "lib" / "aitop"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(selfupdate, "package_root", lambda: pkg)
    monkeypatch.setattr(selfupdate, "repo_checkout", lambda: None)

    assert detect_install_method() is InstallMethod.PIPX


def test_detects_plain_pip_install(monkeypatch, tmp_path):
    pkg = tmp_path / "lib" / "python3.13" / "site-packages" / "aitop"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(selfupdate, "package_root", lambda: pkg)
    monkeypatch.setattr(selfupdate, "repo_checkout", lambda: None)

    assert detect_install_method() is InstallMethod.PIP


def test_only_managed_installs_are_self_updatable():
    assert InstallMethod.UV_TOOL.self_updatable
    assert InstallMethod.PIPX.self_updatable
    assert InstallMethod.PIP.self_updatable
    assert not InstallMethod.GIT_CHECKOUT.self_updatable
    assert not InstallMethod.UNKNOWN.self_updatable


@pytest.mark.parametrize(
    ("method", "head"),
    [
        (InstallMethod.UV_TOOL, ["uv", "tool", "install", "--force"]),
        (InstallMethod.PIPX, ["pipx", "install", "--force"]),
        (InstallMethod.PIP, [sys.executable, "-m", "pip", "install", "--upgrade"]),
    ],
)
def test_update_command_pins_the_requested_tag(method, head):
    argv = update_command(method, "v0.2.0")
    assert argv is not None
    assert argv[:-1] == head
    assert argv[-1] == f"{GIT_SPEC}@v0.2.0"


def test_update_command_without_a_ref_tracks_the_default_branch():
    assert update_command(InstallMethod.UV_TOOL, None)[-1] == GIT_SPEC


def test_no_update_command_for_unmanaged_installs():
    assert update_command(InstallMethod.GIT_CHECKOUT, "v1.0.0") is None
    assert update_command(InstallMethod.UNKNOWN, None) is None


# --------------------------------------------------------------------------- #
# Release lookup
# --------------------------------------------------------------------------- #


@respx.mock
async def test_check_reports_a_newer_release(monkeypatch):
    monkeypatch.setattr(selfupdate, "__version__", "0.1.0")
    respx.get(RELEASES_API).mock(httpx.Response(200, json={"tag_name": "v0.4.0"}))

    status = await check_for_update()
    assert status.checked
    assert status.latest == "0.4.0"
    assert status.available
    assert "0.4.0 is available" in status.describe()


@respx.mock
async def test_check_is_quiet_when_current(monkeypatch):
    monkeypatch.setattr(selfupdate, "__version__", "9.9.9")
    respx.get(RELEASES_API).mock(httpx.Response(200, json={"tag_name": "v0.1.0"}))

    status = await check_for_update()
    assert status.checked and not status.available
    assert "up to date" in status.describe()


@respx.mock
async def test_repo_without_releases_is_not_an_error():
    respx.get(RELEASES_API).mock(httpx.Response(404))
    status = await check_for_update()
    assert not status.checked
    assert status.error == "no releases published yet"
    assert not status.available


@respx.mock
async def test_network_failure_degrades_silently():
    respx.get(RELEASES_API).mock(side_effect=httpx.ConnectTimeout("nope"))
    status = await check_for_update()
    assert not status.checked
    assert "update check failed" in (status.error or "")


@respx.mock
async def test_malformed_release_payload_is_survivable():
    respx.get(RELEASES_API).mock(httpx.Response(200, json={"unexpected": True}))
    status = await check_for_update()
    assert not status.checked


async def test_disabled_check_makes_no_request():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(RELEASES_API).mock(httpx.Response(200, json={"tag_name": "v9.0.0"}))
        status = await check_for_update(enabled=False)
    assert not route.called
    assert not status.checked


async def test_env_var_disables_the_check(monkeypatch):
    monkeypatch.setenv("AITOP_NO_UPDATE_CHECK", "1")
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(RELEASES_API).mock(httpx.Response(200, json={"tag_name": "v9.0.0"}))
        status = await check_for_update()
    assert not route.called
    assert not status.checked


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


@respx.mock
async def test_fresh_cache_avoids_a_second_request(monkeypatch):
    monkeypatch.setattr(selfupdate, "__version__", "0.1.0")
    route = respx.get(RELEASES_API).mock(httpx.Response(200, json={"tag_name": "v0.5.0"}))

    first = await check_for_update()
    second = await check_for_update()

    assert route.call_count == 1  # the second read came from disk
    assert first.latest == second.latest == "0.5.0"


@respx.mock
async def test_stale_cache_triggers_a_refetch(monkeypatch, tmp_path):
    monkeypatch.setattr(selfupdate, "__version__", "0.1.0")
    (tmp_path / "update-check.json").write_text(
        json.dumps({"checked_at": time.time() - 90000, "latest": "0.2.0"})
    )
    respx.get(RELEASES_API).mock(httpx.Response(200, json={"tag_name": "v0.6.0"}))

    status = await check_for_update(ttl=3600)
    assert status.latest == "0.6.0"


@respx.mock
async def test_force_bypasses_a_fresh_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(selfupdate, "__version__", "0.1.0")
    (tmp_path / "update-check.json").write_text(
        json.dumps({"checked_at": time.time(), "latest": "0.2.0"})
    )
    respx.get(RELEASES_API).mock(httpx.Response(200, json={"tag_name": "v0.7.0"}))

    assert (await check_for_update(force=True)).latest == "0.7.0"


@respx.mock
async def test_corrupt_cache_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(selfupdate, "__version__", "0.1.0")
    (tmp_path / "update-check.json").write_text("{not json")
    respx.get(RELEASES_API).mock(httpx.Response(200, json={"tag_name": "v0.8.0"}))

    assert (await check_for_update()).latest == "0.8.0"


@respx.mock
async def test_unwritable_cache_does_not_break_the_check(monkeypatch, tmp_path):
    """A read-only cache directory degrades to "check every time", not a crash."""
    monkeypatch.setattr(selfupdate, "__version__", "0.1.0")
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory")
    monkeypatch.setattr(selfupdate, "cache_path", lambda: blocked / "update-check.json")
    respx.get(RELEASES_API).mock(httpx.Response(200, json={"tag_name": "v0.9.0"}))

    status = await check_for_update()
    assert status.checked and status.latest == "0.9.0"


# --------------------------------------------------------------------------- #
# Applying updates
# --------------------------------------------------------------------------- #


async def test_apply_update_runs_the_right_command(monkeypatch):
    monkeypatch.setattr(selfupdate, "detect_install_method", lambda: InstallMethod.UV_TOOL)
    seen: dict[str, tuple] = {}

    async def fake_run(*argv, **kwargs):
        seen["argv"] = argv
        return CommandResult(argv, 0, "installed", "")

    monkeypatch.setattr("aitop.utils.proc.run", fake_run)

    ok, message, argv = await apply_update(ref="v0.3.0")
    assert ok
    assert seen["argv"] == ("uv", "tool", "install", "--force", f"{GIT_SPEC}@v0.3.0")
    assert "restart aitop" in message


async def test_apply_update_reports_command_failure(monkeypatch):
    monkeypatch.setattr(selfupdate, "detect_install_method", lambda: InstallMethod.UV_TOOL)

    async def fake_run(*argv, **kwargs):
        return CommandResult(argv, 1, "", "network unreachable")

    monkeypatch.setattr("aitop.utils.proc.run", fake_run)

    ok, message, argv = await apply_update(ref="v0.3.0")
    assert not ok
    assert argv is not None
    assert "exit 1" in message


async def test_apply_update_refuses_to_touch_a_dev_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(selfupdate, "detect_install_method", lambda: InstallMethod.GIT_CHECKOUT)
    monkeypatch.setattr(selfupdate, "repo_checkout", lambda: tmp_path)

    ok, message, argv = await apply_update()
    assert not ok
    assert argv is None
    assert "git pull" in message


async def test_apply_update_dry_run_executes_nothing(monkeypatch):
    monkeypatch.setattr(selfupdate, "detect_install_method", lambda: InstallMethod.PIPX)

    async def explode(*argv, **kwargs):
        raise AssertionError("dry run must not execute anything")

    monkeypatch.setattr("aitop.utils.proc.run", explode)

    ok, message, argv = await apply_update(ref="v1.0.0", dry_run=True)
    assert ok
    assert message.startswith("would run: pipx install --force")
    assert argv is not None
