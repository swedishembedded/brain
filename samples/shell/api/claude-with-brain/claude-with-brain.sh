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
#   samples/shell/api/claude-with-brain/claude-with-brain.sh              # interactive claude on the local model
#   samples/shell/api/claude-with-brain/claude-with-brain.sh -p "hi"      # or pass any claude flags through
#   samples/shell/api/claude-with-brain/claude-with-brain.sh --model Qwen/Qwen3-8B --port 9000 -- -p "hi"
#
# Options (before any claude flags; `--` ends them): --model REF, --port N,
# --brain PATH (default ./target/release/brain), --models-dir DIR (default the
# model store brain serve resolves), --max-output-tokens N, --ready-timeout
# SECONDS (default 600: startup scans the whole model store), --no-autofetch,
# --mock, --check.
#
# --max-output-tokens (default 8192) is the reply budget Claude Code asks for.
# Claude Code otherwise asks for 32000 output tokens on top of a prompt of
# about 20000 (its system prompt and tool definitions), sized for Claude's own
# context; a request that cannot fit the local model's context is refused, so
# the budget is set to fit.
#
# Nothing to pre-fetch or pre-import: MODEL doesn't have to be resident yet.
# The server runs with `--autofetch`, so brain downloads +
# converts it on the first request that names it -- the first message you send
# in claude takes as long as that cold fetch (progress streams to Claude Code
# as it happens); every one after is instant. Point MODEL at any
# `<vendor>/<repo>[-<QUANT>]` your machine can serve to use a different model, or pass
# `--no-autofetch` to require it already be resident (`brain pull MODEL`) and
# fail fast instead of fetching.
#
# --check runs the same preflight -> launch -> key-capture sequence, makes ONE
# authenticated GET /v1/models, prints OK, and exits WITHOUT exec'ing `claude` - the
# non-interactive mode `tests/e2e/examples.bats` drives. With --mock it also
# skips the `claude`-installed check (the mock model needs no real `claude` binary
# to prove the server + key path works), so `--check` is runnable in CI with no
# weights and no `claude` install. Without --mock, `GET /v1/models` alone never
# triggers a fetch (discovery routes are deliberately fetch-free), so `--check`
# stays fast and offline-safe even
# though MODEL is a real, not-yet-fetched reference; see `tests/e2e/claude_code.bats`
# for a full interactive run against the deterministic mock.

set -uo pipefail

CHECK=0
MOCK=0
AUTOFETCH=(--autofetch)
PORT=8787
MODEL=""
BRAIN=./target/release/brain
MODELS_DIR=()
MAX_OUTPUT_TOKENS=8192
READY_TIMEOUT=600
need_value() { [ $# -ge 2 ] || { echo "error: $1 needs a value" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1 ;;
    --mock) MOCK=1 ;;   # serve the weight-free mock model instead
    --no-autofetch) AUTOFETCH=() ;;   # require MODEL already pulled
    --model) need_value "$@"; MODEL="$2"; shift ;;
    --port) need_value "$@"; PORT="$2"; shift ;;
    --brain) need_value "$@"; BRAIN="$2"; shift ;;
    --models-dir) need_value "$@"; MODELS_DIR=(--models-dir "$2"); shift ;;
    --max-output-tokens) need_value "$@"; MAX_OUTPUT_TOKENS="$2"; shift ;;
    --ready-timeout) need_value "$@"; READY_TIMEOUT="$2"; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
  shift
done
[ -n "$MODEL" ] || MODEL="$([ "$MOCK" = "0" ] && echo Qwen/Qwen3-0.6B || echo brain/mock)"
case "$PORT$MAX_OUTPUT_TOKENS$READY_TIMEOUT" in *[!0-9]*) echo "error: --port, --max-output-tokens and --ready-timeout take whole numbers" >&2; exit 2 ;; esac

# ---- preflight --------------------------------------------------------------
[ -x "$BRAIN" ] || { echo "error: brain binary not found at '$BRAIN' (build: make build/release)" >&2; exit 1; }
if [ "$CHECK" = "0" ]; then
  command -v claude >/dev/null 2>&1 || { echo "error: the 'claude' CLI is not installed" >&2; exit 1; }
fi

# ---- launch the brain Anthropic surface -------------------------------------
LOG="$(mktemp)"
RUNDIR="$(mktemp -d)"
READY="$RUNDIR/ready"
KEYS="$RUNDIR/keys.json"   # 0600; the key is kept out of the server log
if [ "$MOCK" = "0" ]; then
  "$BRAIN" serve "${AUTOFETCH[@]}" "${MODELS_DIR[@]}" --anthropic "$PORT" --api-keys-out "$KEYS" --ready-file "$READY" >"$LOG" 2>&1 &
else
  # The mock models need no weights, so the server gets an empty store of its
  # own rather than scanning the user's.
  mkdir -p "$RUNDIR/models"
  "$BRAIN" serve --device cpu --mock --models-dir "$RUNDIR/models" --anthropic "$PORT" --api-keys-out "$KEYS" --ready-file "$READY" >"$LOG" 2>&1 &
fi
BRAIN_PID=$!
echo "brain serve --anthropic on http://127.0.0.1:$PORT  (pid $BRAIN_PID)"

# Always stop the server (and clean up the log/ready dir) when this script exits.
cleanup() { kill "$BRAIN_PID" 2>/dev/null || true; rm -f "$LOG"; rm -rf "$RUNDIR"; }
trap cleanup EXIT INT TERM

# Wait for the ready file: it is touched only once the listener is actually
# bound AND (since it's written first) the keys file below is already on
# disk -- so once this loop exits, no retry is needed for either. A server
# that fails exits, which ends the wait at once; --ready-timeout bounds the
# rest (the startup scan of a large model store takes minutes).
for _ in $(seq 1 $((READY_TIMEOUT * 2))); do
  [ -e "$READY" ] && break
  kill -0 "$BRAIN_PID" 2>/dev/null || { echo "error: brain server exited on startup:" >&2; cat "$LOG" >&2; exit 1; }
  sleep 0.5
done
[ -e "$READY" ] || { echo "error: brain server not ready after ${READY_TIMEOUT}s (--ready-timeout):" >&2; cat "$LOG" >&2; exit 1; }

# brain generates a fresh key each launch and writes it to $KEYS.
API_KEY="$(sed -n 's/.*"anthropic": *"\([^"]*\)".*/\1/p' "$KEYS" 2>/dev/null | head -1)"
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
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS"
export DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

# ---- start Claude Code against brain ----------------------------------------
echo "launching claude (Ctrl-D or /exit to quit; brain stops automatically)…"
claude "$@"
