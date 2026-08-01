# aitop

**AI-centric neofetch meets btop.** A lightweight, modular terminal dashboard for
local AI runtimes — hardware thermals, VRAM overhead, model residency, and (soon)
live inference telemetry, plus a full lifecycle console for local and remote engines.

```
       ▄▄▄▄▄▄▄▄▄          neek@nM1
    ▄██▀▀     ▀▀██▄       ──────────────────────
   ██▀   ▄▄▄▄▄   ▀██      OS        macOS 15.6
  ██   ▄██▀ ▀██▄   ██     CPU       Apple M1 (4P + 4E) @ 11%
  ██   ██     ██   ██     Memory    9.8 GB / 16.0 GB (61%, unified)
  ██   ▀██▄ ▄██▀   ██     GPU       Apple M1 (8 cores) · Metal 3
   ██▄   ▀▀▀▀▀   ▄██      GPU mem   6.1 GB / 10.7 GB (57%)
    ▀██▄▄     ▄▄██▀       Network   Tailscale 100.100.1.2 (nM1)
       ▀▀▀▀▀▀▀▀▀          Runtimes  1 online · 1 model resident
   ▄▀█ █ ▀█▀ █▀█ █▀█
   █▀█ █  █  █▄█ █▀▀
```

> **Status: Phase 1.** Engine abstraction, hardware telemetry, and the neofetch
> view are implemented. Lifecycle control, the HF model hub, the btop-style live
> TUI, and the fleet/gateway features are Phases 2–5 — the seams for all of them
> exist in the current code.

---

## Install

One line — installs `uv` first if you don't have it, then puts `aitop` on your PATH
in its own isolated environment:

```bash
curl -LsSf https://raw.githubusercontent.com/NicoMancinelli/aitop/main/install.sh | bash
```

Already have `uv` or `pipx`? Skip the script:

```bash
uv tool install git+https://github.com/NicoMancinelli/aitop
```

```bash
pipx install git+https://github.com/NicoMancinelli/aitop
```

Requires Python 3.11+, which `uv` will fetch for you if needed.

<details>
<summary>Pinning a version, or installing from a branch</summary>

```bash
AITOP_VERSION=v0.2.0 curl -LsSf https://raw.githubusercontent.com/NicoMancinelli/aitop/main/install.sh | bash
AITOP_REF=main       curl -LsSf https://raw.githubusercontent.com/NicoMancinelli/aitop/main/install.sh | bash
AITOP_METHOD=pipx    curl -LsSf https://raw.githubusercontent.com/NicoMancinelli/aitop/main/install.sh | bash
```

</details>

## Update

```bash
aitop update           # fetch the newest release and install it
aitop update --check   # just tell me if there's a new version
aitop update --dry-run # print the exact command that would run
```

`aitop update` works out how you installed it (`uv tool`, `pipx`, `pip`) and runs the
matching upgrade, pinned to the latest release tag. If you're running from a git
checkout it refuses to touch your working tree and tells you to `git pull` instead.

### Auto-update

By default aitop **checks** for new releases and **never installs** them without being
asked. The check is cached for 24 hours, runs concurrently with telemetry collection so
it costs no wall-clock time, and fails silently when offline. When a newer release
exists you get one line at the bottom of the dashboard:

```
  ⬆ aitop 0.2.0 available (you have 0.1.0) — run aitop update
```

To make it fully automatic, or to turn it off entirely:

```yaml
# ~/.config/aitop/config.yaml
updates:
  check: true          # look for new releases at all
  interval_hours: 24   # how long a check result stays fresh
  auto_apply: false    # set true to install updates without asking
```

Per-invocation and environment overrides:

```bash
aitop --no-update-check      # skip the check this once
export AITOP_NO_UPDATE_CHECK=1   # skip it everywhere (good for CI and scripts)
```

`auto_apply` is off by default deliberately: aitop is often pointed at production
homelab nodes, and silently swapping the binary mid-session is a surprise.

## Usage

```bash
aitop                  # one-shot dashboard (the "AI neofetch")
aitop --watch          # refresh in place until Ctrl-C
aitop --all            # also show endpoints that were probed but are offline
aitop --json | jq .    # raw SystemSnapshot — the same payload every consumer sees
aitop --no-logo        # facts only, no ASCII header
aitop doctor           # what telemetry is available on this machine, and what isn't
aitop update           # upgrade in place
```

`aitop doctor` is the first thing to run when something looks wrong — it reports the
install method, which config file was loaded, which hardware probes activated, and
exactly which telemetry is unavailable and why.

On macOS, watts and die temperature come from `powermetrics`, which is root-only.
aitop probes it with `sudo -n` and silently degrades if that would prompt — it will
never block on a password. To get those numbers, either run `sudo aitop` or add a
NOPASSWD sudoers entry for `/usr/bin/powermetrics`.

## Architecture

The rule the codebase enforces: **telemetry collection never touches rendering,
and rendering never touches a subprocess.**

```
aitop/
├── models.py          Pydantic snapshot contract — the ONLY thing renderers see
├── bus.py             Async pub/sub; bounded, drop-oldest queues per subscriber
├── collector.py       Fans out to hardware + engines + tailscale, emits a snapshot
├── config.py          Zero-config defaults, optional ~/.config/aitop/config.yaml
├── engines/
│   ├── base.py        BaseEngine ABC: detect / poll / start / stop / unload / rebind
│   ├── registry.py    Endpoint probing + psutil process scan, all concurrent
│   ├── ollama.py      /api/version, /api/tags, /api/ps
│   └── lmstudio.py    /api/v0/models, /v1/models fallback, `lms` CLI hooks
├── hardware/
│   ├── base.py        HardwareProbe ABC + ProbeResult (degrades, never raises)
│   ├── system.py      psutil CPU/memory/host, sysctl P+E core split
│   ├── apple.py       ioreg (no root) + system_profiler + powermetrics (root)
│   ├── nvidia.py      nvidia-smi CSV query
│   ├── amd.py         rocm-smi --json
│   └── collector.py   Probe selection and concurrent assembly
├── net/tailscale.py   tailscale status --json
├── selfupdate.py      Install-method detection, release lookup, in-place upgrade
├── views/neofetch.py  Pure function of a SystemSnapshot → Rich renderable
└── cli.py             argparse entrypoint
```

`SystemSnapshot` is fully JSON-serializable by design. A web UI, the Prometheus
exporter, and remote fleet streaming are all "add a bus subscriber" — no changes
to the collectors.

### Graceful degradation

Every hardware probe reports what it *couldn't* read instead of failing:

- `nvidia-smi` / `rocm-smi` missing → the probe is never activated
- `powermetrics` would prompt for a password → skipped, noted under **Notes**
- Apple `iogpu.wired_limit_mb` unset → the GPU memory ceiling is estimated from
  Metal's `recommendedMaxWorkingSetSize` ratios, and labelled as an estimate
- An engine's telemetry endpoint 404s → `DEGRADED`, not a crash

## Configuration

Entirely optional. Auto-discovery probes the well-known ports (Ollama 11434,
LM Studio 1234, vLLM 8000, llama-server 8080) and honours `OLLAMA_HOST` and
friends. Write `~/.config/aitop/config.yaml` only to add endpoints or tune it:

```yaml
polling:
  hardware_interval: 2.0
  engine_interval: 2.0

ui:
  show_per_core: true

endpoints:
  - kind: ollama
    host: 100.100.1.7      # a tailnet node
    port: 11434
    name: pveclaw-ollama
    remote: true
  - kind: lmstudio
    enabled: false         # never probe LM Studio on this box
```

## Supported today

| Runtime   | Detect | Models | Residency | Unload |
|-----------|:------:|:------:|:---------:|:------:|
| Ollama    | ✅ | ✅ | ✅ (incl. GPU offload %) | ✅ |
| LM Studio | ✅ | ✅ | ✅ (native API or `lms`)  | ✅ (`lms`) |
| MLX / vLLM / llama-server | port reserved | — | — | — |

| Platform | CPU | Memory | GPU | Power | Thermals |
|----------|:---:|:------:|:---:|:-----:|:--------:|
| Apple Silicon | ✅ (P/E split) | ✅ unified | ✅ util + wired limit | root only | root only |
| Linux + NVIDIA | ✅ | ✅ | ✅ | ✅ | ✅ |
| Linux + AMD ROCm | ✅ | ✅ | ✅ | ✅ | ✅ |

## Development

```bash
git clone https://github.com/NicoMancinelli/aitop
cd aitop
make bootstrap
```

That is the only setup step. Afterwards the ordinary commands work as you'd expect —
no environment variables to remember:

```bash
uv run pytest
uv run ruff check src tests
uv run aitop
```

`make` on its own lists the shortcuts (`make test`, `make lint`, `make fmt`, `make run`,
`make watch`, `make doctor`, `make build`, `make clean`).

<details>
<summary>Why bootstrap exists: the iCloud Drive trap</summary>

If the checkout lives under `~/Library/Mobile Documents/…`, iCloud syncs and rewrites
files inside `.venv` and silently corrupts the editable install. The failure is
`ModuleNotFoundError: No module named 'aitop'` even though `_editable_impl_aitop.pth`
is present, readable, and points at a directory that exists. Nothing in the traceback
implicates iCloud, and re-running `uv sync` fixes it only until the next sync.

`scripts/bootstrap.sh` detects this and puts the environment at `~/.venvs/<repo>`,
leaving `.venv` as a symlink. The venv's thousands of files then never enter the synced
tree at all, while `uv sync` and `uv run` keep working unchanged — `uv` follows and
preserves the symlink. On Linux, or outside iCloud, the script is a plain
`uv sync --extra dev`.

</details>

### Releasing

The tag is the trigger and `aitop.__version__` is the single source of truth —
`pyproject.toml` reads it dynamically, and CI refuses to publish a tag that disagrees
with it.

```bash
# 1. bump the version
vim src/aitop/__init__.py       # __version__ = "0.2.0"

# 2. tag and push
git commit -am "release: v0.2.0"
git tag v0.2.0
git push origin main --tags
```

That runs tests on macOS and Linux, builds the sdist and wheel, and publishes a GitHub
release with generated notes. Every installed copy picks it up on its next update
check. Publishing to PyPI is wired up but dormant — register the project, configure a
Trusted Publisher, then set the repository variable `PUBLISH_TO_PYPI=true`.

## Licence

MIT — see [LICENSE](LICENSE).
