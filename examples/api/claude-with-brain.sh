#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Run Claude Code against a LOCAL brain server instead of the hosted Anthropic API.
#
# It launches `brain serve --anthropic` (serving a local qwen3 model), captures the
# freshly-generated per-launch API key, points Claude Code at the local endpoint with
# that key, and starts `claude` normally (interactive). The brain server is stopped
# automatically when you quit claude.
#
#   examples/api/claude-with-brain.sh              # interactive claude on your local qwen3
#   examples/api/claude-with-brain.sh -p "hi"      # or pass any claude flags through
#
# First time: build brain and import a Qwen3 checkpoint into brain's format, e.g.
#   make release
#   ./target/release/brain qwen import --hf /path/to/Qwen3-0.6B --out qwen3.safetensors
# (download a Qwen3 dir from HuggingFace: config.json + model.safetensors + tokenizer.json)

set -uo pipefail

# ---- config (override via env) ----------------------------------------------
PORT="${PORT:-8787}"
MODEL="${MODEL:-qwen}"                                   # brain's qwen resident id
BRAIN="${BRAIN:-./target/release/brain}"
# A brain-native qwen3 checkpoint + its tokenizer (see the import note above).
QWEN_WEIGHTS="${BRAIN_QWEN_WEIGHTS:-qwen3.safetensors}"
QWEN_TOKENIZER="${BRAIN_QWEN_TOKENIZER:-tokenizer.json}"

# ---- preflight --------------------------------------------------------------
[ -x "$BRAIN" ] || { echo "error: brain binary not found at '$BRAIN' (build: make release)" >&2; exit 1; }
command -v claude >/dev/null 2>&1 || { echo "error: the 'claude' CLI is not installed" >&2; exit 1; }
if [ ! -f "$QWEN_WEIGHTS" ]; then
  echo "error: qwen weights not found: '$QWEN_WEIGHTS'" >&2
  echo "  import one:  $BRAIN qwen import --hf /path/to/Qwen3-0.6B --out $QWEN_WEIGHTS" >&2
  echo "  or set BRAIN_QWEN_WEIGHTS / BRAIN_QWEN_TOKENIZER to your files." >&2
  exit 1
fi
[ -f "$QWEN_TOKENIZER" ] || { echo "error: tokenizer not found: '$QWEN_TOKENIZER'" >&2; exit 1; }

# ---- launch the brain Anthropic surface -------------------------------------
LOG="$(mktemp)"
BRAIN_QWEN_WEIGHTS="$QWEN_WEIGHTS" BRAIN_QWEN_TOKENIZER="$QWEN_TOKENIZER" \
  "$BRAIN" serve --anthropic "$PORT" >"$LOG" 2>&1 &
BRAIN_PID=$!
echo "brain serve --anthropic on http://127.0.0.1:$PORT  (pid $BRAIN_PID)"

# Always stop the server (and clean the log) when this script exits.
cleanup() { kill "$BRAIN_PID" 2>/dev/null || true; rm -f "$LOG"; }
trap cleanup EXIT INT TERM

# Wait for it to bind (it prints "apiserve: anthropic on ..." when ready).
for _ in $(seq 1 60); do
  grep -q 'apiserve: anthropic on' "$LOG" 2>/dev/null && break
  kill -0 "$BRAIN_PID" 2>/dev/null || { echo "error: brain server exited on startup:" >&2; cat "$LOG" >&2; exit 1; }
  sleep 0.5
done

# brain generates a fresh key each launch and prints it as: `APIKEY anthropic <key>`.
API_KEY="$(grep -m1 '^APIKEY anthropic ' "$LOG" | awk '{print $3}')"
[ -n "$API_KEY" ] || { echo "error: could not read the generated API key:" >&2; cat "$LOG" >&2; exit 1; }
echo "generated Anthropic key: ${API_KEY:0:14}…  (model: $MODEL)"

# ---- point Claude Code at the local backend ---------------------------------
# ANTHROPIC_API_KEY overrides any logged-in subscription/system key; routing EVERY model
# alias (incl. the haiku-class background model) to the local qwen means nothing ever
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
