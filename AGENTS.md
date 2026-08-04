# AGENTS.md

## Cursor Cloud specific instructions

`aitop` is a single Python package (a CLI + Textual TUI dashboard for local AI
runtimes). Tooling is [`uv`](https://docs.astral.sh/uv/); Python 3.13 is pinned
via `.python-version` and provisioned by `uv`. There is no web frontend, no
database, and no backend server of its own beyond the optional `aitop serve`
fleet gateway.

Standard commands are already documented — use the `Makefile` targets (`make
test`, `make lint`, `make fmt`, `make build`, `make run`, `make tui`, `make
serve`) or the `uv run ...` forms in `README.md`. `uv` is already on `PATH` (the
installer added it to `~/.bashrc`/`~/.profile`), and the update script keeps
`.venv` in sync, so you can run `make`/`uv run` commands directly.

Non-obvious things worth knowing:

- **Graceful degradation is expected, not a failure.** On the cloud Linux VM
  there is no GPU and no local AI runtime (Ollama/LM Studio/vLLM/etc.), so the
  dashboard shows "no AI runtimes detected" and "no GPU telemetry source found".
  This is the designed behavior — every probe reports what it could not read
  instead of crashing. CPU/memory/host telemetry still populate for real.
- **Tests need no network or services.** `tests/conftest.py` has an autouse
  fixture that forbids outbound network access and stubs runtime/Hugging Face
  HTTP calls (via `respx`). Do not add tests that hit the network.
- **Silence update checks** in scripts/CI with `AITOP_NO_UPDATE_CHECK=1` (or the
  `--no-update-check` flag); otherwise the CLI tries to reach the GitHub
  releases API.
- **`aitop serve`** exposes `/healthz`, `/api/snapshot`, and an SSE
  `/api/stream` on port 9090 by default — a good end-to-end check of the
  collector → bus → HTTP pipeline without any AI runtime present.
- **`aitop doctor`** is the fastest way to see which probes/telemetry are active
  and why others are unavailable on the current machine.
