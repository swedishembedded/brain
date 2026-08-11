#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Run Claude Code against a LOCAL brain server instead of the hosted Anthropic API.
#
# It launches `brain serve --anthropic` (serving MODEL, a fully-qualified
# `<vendor>/<repo>` reference -- Qwen/Qwen3-0.6B by default), captures the
# freshly-generated per-launch API key, points Claude Code at the local endpoint
# with that key, and starts `claude` normally (interactive). The brain server is
# stopped automatically when you quit claude.
#
#   examples/api/claude-with-brain.sh              # interactive claude on the local model
#   examples/api/claude-with-brain.sh -p "hi"      # or pass any claude flags through
#
# Nothing to pre-fetch or pre-import: MODEL doesn't have to be resident yet.
# brain's transparent auto-fetch (see docs/using/models-and-weights.md) downloads +
# converts it on the first request that names it -- the first message you send
# in claude takes as long as that cold fetch (progress streams to Claude Code
# as it happens); every one after is instant. Point MODEL at any
# `<vendor>/<repo>[-<QUANT>]` your machine can serve (`docs/using/models-and-weights.md`
# lists the supported architectures) to use a different model, or set
# `BRAIN_AUTO_FETCH=0` to require it already be resident (the pre-auto-fetch
# behavior) and fail fast instead of fetching.
#
# --check runs the same preflight -> launch -> key-capture sequence, makes ONE
# authenticated GET /v1/models, prints OK, and exits WITHOUT exec'ing `claude` — the
# non-interactive mode `tests/e2e/examples.bats` drives. Under BRAIN_MOCK=1 it also
# skips the `claude`-installed check (the mock model needs no real `claude` binary
# to prove the server + key path works), so `--check` is runnable in CI with no
# weights and no `claude` install. Without BRAIN_MOCK, `GET /v1/models` alone never
# triggers a fetch (discovery routes are deliberately fetch-free — see
# `.agents/rules/api-security.md`), so `--check` stays fast and offline-safe even
# though MODEL is a real, not-yet-fetched reference; see `tests/e2e/claude_code.bats`
# for a full interactive run against the deterministic mock.

set -uo pipefail

CHECK=0
if [ "${1:-}" = "--check" ]; then
  CHECK=1
  shift
fi

# ---- config (override via env) ----------------------------------------------
PORT="${PORT:-8787}"
MOCK="${BRAIN_MOCK:-0}"
MODEL="${MODEL:-$([ "$MOCK" = "0" ] && echo Qwen/Qwen3-0.6B || echo brain/mock)}"
BRAIN="${BRAIN:-./target/release/brain}"

# ---- preflight --------------------------------------------------------------
[ -x "$BRAIN" ] || { echo "error: brain binary not found at '$BRAIN' (build: make release)" >&2; exit 1; }
if [ "$CHECK" = "0" ]; then
  command -v claude >/dev/null 2>&1 || { echo "error: the 'claude' CLI is not installed" >&2; exit 1; }
fi

# ---- launch the brain Anthropic surface -------------------------------------
LOG="$(mktemp)"
RUNDIR="$(mktemp -d)"
READY="$RUNDIR/ready"
if [ "$MOCK" = "0" ]; then
  "$BRAIN" serve --anthropic "$PORT" --ready-file "$READY" >"$LOG" 2>&1 &
else
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --anthropic "$PORT" --ready-file "$READY" >"$LOG" 2>&1 &
fi
BRAIN_PID=$!
echo "brain serve --anthropic on http://127.0.0.1:$PORT  (pid $BRAIN_PID)"

# Always stop the server (and clean up the log/ready dir) when this script exits.
cleanup() { kill "$BRAIN_PID" 2>/dev/null || true; rm -f "$LOG"; rm -rf "$RUNDIR"; }
trap cleanup EXIT INT TERM

# Wait for the ready file: it is touched only once the listener is actually
# bound AND (since it's written first) the APIKEY line below is already on
# disk -- so once this loop exits, no retry is needed for either.
for _ in $(seq 1 60); do
  [ -e "$READY" ] && break
  kill -0 "$BRAIN_PID" 2>/dev/null || { echo "error: brain server exited on startup:" >&2; cat "$LOG" >&2; exit 1; }
  sleep 0.5
done
[ -e "$READY" ] || { echo "error: brain server never became ready:" >&2; cat "$LOG" >&2; exit 1; }

# brain generates a fresh key each launch and prints it as: `APIKEY anthropic <key>`.
API_KEY="$(grep -m1 '^APIKEY anthropic ' "$LOG" | awk '{print $3}')"
[ -n "$API_KEY" ] || { echo "error: could not read the generated API key:" >&2; cat "$LOG" >&2; exit 1; }
echo "generated Anthropic key: ${API_KEY:0:14}…  (model: $MODEL)"

if [ "$CHECK" = "1" ]; then
  if curl -fsS --max-time 5 -H "x-api-key: $API_KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    echo "OK: authenticated GET /v1/models succeeded"
    exit 0
  else
    echo "error: authenticated GET /v1/models failed" >&2
    cat "$LOG" >&2
    exit 1
  fi
fi

# ---- point Claude Code at the local backend ---------------------------------
# ANTHROPIC_API_KEY overrides any logged-in subscription/system key; routing EVERY model
# alias (incl. the haiku-class background model) to the local MODEL means nothing ever
# reaches the hosted API.
export ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT"
export ANTHROPIC_API_KEY="$API_KEY"
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

# ---- start Claude Code against brain ----------------------------------------
echo "launching claude (Ctrl-D or /exit to quit; brain stops automatically)…"
claude "$@"
