"""Zero-config defaults with optional `~/.config/aitop/config.yaml` overrides.

aitop must be useful with no config file at all: the defaults below describe
where local engines normally listen, and discovery probes those. A config file
only ever *adds* endpoints (remote fleet nodes, non-standard ports) or tunes
intervals.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from aitop.models import EngineKind

log = logging.getLogger(__name__)

DEFAULT_PORTS: dict[EngineKind, int] = {
    EngineKind.OLLAMA: 11434,
    EngineKind.LMSTUDIO: 1234,
    EngineKind.VLLM: 8000,
    EngineKind.LLAMA_SERVER: 8080,
    EngineKind.MLX: 8080,
}


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "aitop" / "config.yaml"


class EndpointConfig(BaseModel):
    """An engine endpoint to probe. Everything but `kind` has a sane default."""

    model_config = ConfigDict(extra="forbid")

    kind: EngineKind
    host: str = "127.0.0.1"
    port: int | None = None
    name: str | None = None
    enabled: bool = True
    remote: bool = False
    api_key: str | None = None
    timeout: float = 2.0

    def resolved_port(self) -> int:
        return self.port or DEFAULT_PORTS.get(self.kind, 8080)

    def label(self) -> str:
        return self.name or self.kind.value


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto: bool = True
    """Probe the well-known local ports for every supported engine."""

    scan_processes: bool = True
    """Also match running processes by name, to catch non-default ports."""

    respect_env: bool = True
    """Honour OLLAMA_HOST / LMSTUDIO_HOST style environment variables."""


class PollingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hardware_interval: float = 2.0
    engine_interval: float = 2.0
    process_timeout: float = 4.0


class UIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str = "auto"
    ascii_logo: bool = True
    color: bool = True
    show_per_core: bool = True


class UpdateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: bool = True
    """Look for a newer release on startup. Cached, so it is usually free."""

    interval_hours: float = 24.0
    """How long a check result stays fresh before we ask GitHub again."""

    auto_apply: bool = False
    """Install updates automatically instead of just mentioning them.

    Off by default: silently replacing the binary a user is running is a
    surprise, and this tool is often run against production homelab nodes.
    """

    timeout: float = 3.0

    @property
    def ttl_seconds(self) -> float:
        return max(0.0, self.interval_hours) * 3600


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    polling: PollingConfig = Field(default_factory=PollingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    updates: UpdateConfig = Field(default_factory=UpdateConfig)
    endpoints: list[EndpointConfig] = Field(default_factory=list)
    """Extra endpoints, merged with auto-discovered ones."""

    source: Path | None = None
    """Where this config was loaded from; None means built-in defaults."""

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Load config, falling back to defaults if the file is absent or bad."""
        target = path or config_path()
        if not target.is_file():
            return cls()
        try:
            raw: Any = yaml.safe_load(target.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            log.warning("ignoring unreadable config %s: %s", target, exc)
            return cls()
        if not isinstance(raw, dict):
            log.warning("ignoring config %s: expected a mapping", target)
            return cls()
        try:
            return cls(**raw, source=target)
        except Exception as exc:  # pydantic ValidationError and friends
            log.warning("ignoring invalid config %s: %s", target, exc)
            return cls()

    def default_endpoints(self) -> list[EndpointConfig]:
        """Well-known local endpoints, honouring engine env vars."""
        if not self.discovery.auto:
            return []
        out: list[EndpointConfig] = []
        for kind, port in DEFAULT_PORTS.items():
            host, resolved = "127.0.0.1", port
            if self.discovery.respect_env:
                host, resolved = _env_override(kind, host, port)
            out.append(EndpointConfig(kind=kind, host=host, port=resolved))
        return out

    def all_endpoints(self) -> list[EndpointConfig]:
        """Auto-discovery defaults plus user endpoints, de-duplicated."""
        merged: dict[tuple[EngineKind, str, int], EndpointConfig] = {}
        for ep in [*self.default_endpoints(), *self.endpoints]:
            if not ep.enabled:
                merged.pop((ep.kind, ep.host, ep.resolved_port()), None)
                continue
            merged[(ep.kind, ep.host, ep.resolved_port())] = ep
        return list(merged.values())


_ENV_VARS: dict[EngineKind, tuple[str, ...]] = {
    EngineKind.OLLAMA: ("OLLAMA_HOST",),
    EngineKind.LMSTUDIO: ("LMSTUDIO_HOST", "LM_STUDIO_HOST"),
    EngineKind.VLLM: ("VLLM_HOST",),
}


def _env_override(kind: EngineKind, host: str, port: int) -> tuple[str, int]:
    """Parse `HOST[:PORT]` / `http://HOST:PORT` engine env vars."""
    for var in _ENV_VARS.get(kind, ()):
        value = os.environ.get(var)
        if not value:
            continue
        value = value.removeprefix("http://").removeprefix("https://").rstrip("/")
        if ":" in value and not value.startswith("["):
            head, _, tail = value.rpartition(":")
            if tail.isdigit():
                return (head or host), int(tail)
            return value, port
        return value, port
    return host, port
