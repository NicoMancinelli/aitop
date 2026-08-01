#!/usr/bin/env bash
#
# aitop installer.
#
#   curl -LsSf https://raw.githubusercontent.com/NicoMancinelli/aitop/main/install.sh | bash
#
# Installs the latest release as an isolated CLI tool. Re-running upgrades in
# place, so this doubles as the update path for anyone who prefers not to use
# `aitop update`.
#
# Environment:
#   AITOP_VERSION=v0.2.0   pin a specific tag (default: latest release)
#   AITOP_REF=main         install a branch instead of a release
#   AITOP_METHOD=uv|pipx|pip   force an installer (default: autodetect)
#   AITOP_NO_MODIFY_PATH=1 don't offer to add the bin dir to PATH

set -euo pipefail

# Remember the PATH we inherited, so we can tell whether the install dir will
# still be visible in the user's *next* shell, not just this one.
ORIGINAL_PATH="${PATH}"

REPO="NicoMancinelli/aitop"
REPO_URL="https://github.com/${REPO}"
API_LATEST="https://api.github.com/repos/${REPO}/releases/latest"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33m    warning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\n\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #

case "$(uname -s)" in
    Darwin|Linux) ;;
    *) die "unsupported platform: $(uname -s). aitop targets macOS and Linux." ;;
esac

have curl || have wget || die "need curl or wget to talk to GitHub"

fetch() {
    if have curl; then
        curl -LsSf "$1"
    else
        wget -qO- "$1"
    fi
}

# --------------------------------------------------------------------------- #
# Resolve the version to install
# --------------------------------------------------------------------------- #

resolve_ref() {
    if [ -n "${AITOP_REF:-}" ]; then
        printf '%s' "$AITOP_REF"
        return
    fi
    if [ -n "${AITOP_VERSION:-}" ]; then
        printf '%s' "$AITOP_VERSION"
        return
    fi
    # tag_name from the releases API; empty if the repo has no releases yet.
    fetch "$API_LATEST" 2>/dev/null \
        | grep -m1 '"tag_name"' \
        | sed -E 's/.*"tag_name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/' \
        || true
}

log "Resolving version"
REF="$(resolve_ref || true)"
if [ -z "$REF" ]; then
    warn "no published release found — installing from the default branch"
    SPEC="git+${REPO_URL}"
    info "version: main (unreleased)"
else
    SPEC="git+${REPO_URL}@${REF}"
    info "version: ${REF}"
fi

# --------------------------------------------------------------------------- #
# Pick an installer
# --------------------------------------------------------------------------- #

pick_method() {
    if [ -n "${AITOP_METHOD:-}" ]; then
        printf '%s' "$AITOP_METHOD"
        return
    fi
    if have uv;   then printf 'uv';   return; fi
    if have pipx; then printf 'pipx'; return; fi
    printf 'uv-bootstrap'
}

METHOD="$(pick_method)"

if [ "$METHOD" = "uv-bootstrap" ]; then
    log "Installing uv (no uv or pipx found)"
    info "uv keeps aitop in its own environment, so it can't collide with your"
    info "system Python packages."
    fetch https://astral.sh/uv/install.sh | sh
    # The uv installer drops binaries here; make them visible to this shell.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    have uv || die "uv installed but is not on PATH — open a new shell and re-run"
    METHOD="uv"
fi

# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #

log "Installing aitop via ${METHOD}"
case "$METHOD" in
    uv)
        uv tool install --force "$SPEC"
        BIN_DIR="$(uv tool dir --bin 2>/dev/null || printf '%s' "$HOME/.local/bin")"
        ;;
    pipx)
        pipx install --force "$SPEC"
        BIN_DIR="$HOME/.local/bin"
        ;;
    pip)
        python3 -m pip install --user --upgrade "$SPEC"
        BIN_DIR="$(python3 -m site --user-base)/bin"
        ;;
    *)
        die "unknown AITOP_METHOD '${METHOD}' (expected uv, pipx or pip)"
        ;;
esac

# --------------------------------------------------------------------------- #
# Verify and report
# --------------------------------------------------------------------------- #

export PATH="${BIN_DIR}:$PATH"

if ! have aitop; then
    die "aitop was installed to ${BIN_DIR} but is not on PATH — add it and re-run"
fi

log "Installed"
info "$(aitop --version)"
info "binary: $(command -v aitop)"

# We prepended BIN_DIR above so the verification above could run. Warn only if
# it was absent from the PATH we started with — that's what new shells inherit.
case ":${ORIGINAL_PATH}:" in
    *":${BIN_DIR}:"*) ;;
    *)
        printf '\n'
        warn "${BIN_DIR} is not on your PATH, so new shells won't find aitop."
        info "Add this to your ~/.zshrc or ~/.bashrc:"
        printf '\n      export PATH="%s:$PATH"\n' "$BIN_DIR"
        ;;
esac

cat <<EOF

  Next steps:

    aitop              one-shot dashboard
    aitop --watch      live refresh
    aitop doctor       what telemetry is available on this machine
    aitop update       upgrade to the newest release

  Docs: ${REPO_URL}

EOF
