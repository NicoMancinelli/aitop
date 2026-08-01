#!/usr/bin/env bash
#
# Set up the development environment.
#
#   ./scripts/bootstrap.sh
#
# On most machines this is just `uv sync --extra dev`.
#
# On macOS, when the checkout lives inside iCloud Drive, it does one extra
# thing first: it puts the virtualenv *outside* the synced tree and leaves a
# symlink behind. iCloud rewrites files under a synced `.venv` and silently
# corrupts the editable install — the package stops importing even though the
# `.pth` file is present and its target directory exists. Nothing in the
# traceback points at iCloud, so it costs an hour to diagnose the first time.
#
# The symlink means every ordinary command (`uv sync`, `uv run pytest`) keeps
# working with no environment variables to remember, while the thousands of
# files that make up the venv never enter the synced tree at all.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33m    warning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\n\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || die "uv is not installed — see https://docs.astral.sh/uv/"

# --------------------------------------------------------------------------- #
# Is this checkout inside a cloud-synced directory?
# --------------------------------------------------------------------------- #

in_icloud() {
    [ "$(uname -s)" = "Darwin" ] || return 1
    case "$REPO_ROOT" in
        *"/Library/Mobile Documents/"*) return 0 ;;
        *) return 1 ;;
    esac
}

external_venv_path() {
    # Keep the name unique per checkout so two clones can't share one venv.
    printf '%s/.venvs/%s' "$HOME" "$(basename "$REPO_ROOT")"
}

# --------------------------------------------------------------------------- #
# Relocate the venv out of the synced tree
# --------------------------------------------------------------------------- #

if in_icloud; then
    TARGET="$(external_venv_path)"
    log "Checkout is inside iCloud Drive"
    info "Keeping the virtualenv outside it, at ${TARGET}"

    if [ -e "$VENV" ] && [ ! -L "$VENV" ]; then
        # A real directory here is the broken state we are fixing.
        warn "removing the in-tree .venv (iCloud corrupts it)"
        rm -rf "$VENV"
    fi

    # The target must exist before we link: uv creates the environment with a
    # plain mkdir, which fails outright on a dangling symlink. An empty
    # directory is fine — uv populates it in place.
    mkdir -p "$TARGET"

    if [ -L "$VENV" ]; then
        CURRENT="$(readlink "$VENV")"
        if [ "$CURRENT" != "$TARGET" ]; then
            info "repointing .venv: ${CURRENT} -> ${TARGET}"
            rm -f "$VENV"
            ln -s "$TARGET" "$VENV"
        else
            info ".venv already points outside iCloud"
        fi
    else
        ln -s "$TARGET" "$VENV"
        info "linked .venv -> ${TARGET}"
    fi

    # Belt and braces: ask the File Provider to leave the link itself alone.
    xattr -w 'com.apple.fileprovider.ignore#P' 1 "$VENV" 2>/dev/null || true
else
    log "Setting up the development environment"
fi

# --------------------------------------------------------------------------- #
# Sync
# --------------------------------------------------------------------------- #

log "Installing dependencies"
uv sync --extra dev

log "Verifying"
uv run python -c 'import aitop; print(f"    aitop {aitop.__version__} imports cleanly")'
uv run pytest -q

log "Ready"
cat <<EOF
    uv run aitop            run the dashboard
    uv run pytest           run the tests
    uv run ruff check src   lint

EOF
