#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# End-to-end proof that the REAL `claude` CLI works against brain's Anthropic Messages
# surface — the core "brain as a Claude Code backend" claim. Runs against the built-in
# BRAIN_MOCK model (deterministic, instant), so it needs no weights and never hangs on
# a slow model; every claude call has a hard timeout.
#
# Env isolation: claude uses ONLY our locally-generated key and talks ONLY to the local
# backend — the system/subscription key can never be used. ALL model aliases (incl. the
# haiku-class background model) point at the local mock.
#
# Skips only if `claude`/`jq`/`timeout` are missing or no brain binary exists.
# Run: make test/e2e/claude-code   (or: BRAIN_BIN=./target/debug/brain bats tests/e2e/claude_code.bats)
#
# NOTE: never `pkill claude` — this repo's own dev session may itself be a claude
# process. Teardown kills only the recorded brain-serve PID.

CLAUDE_TIMEOUT="${CLAUDE_TIMEOUT:-60}"

setup_file() {
  command -v claude >/dev/null 2>&1 || skip "claude CLI not installed"
  command -v jq >/dev/null 2>&1 || skip "jq not installed"
  command -v timeout >/dev/null 2>&1 || skip "timeout not available"

  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  BRAIN="${BRAIN_BIN:-$REPO/target/release/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/debug/brain"
  [ -x "$BRAIN" ] || skip "no brain binary (run: make build)"

  export E2E_DIR="$(mktemp -d)"
  export E2E_PORT="${E2E_PORT:-8792}"
  export E2E_KEYS="$E2E_DIR/keys.json"
  export E2E_LOG="$E2E_DIR/serve.log"

  # Local Anthropic surface backed by the deterministic mock model.
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --anthropic "$E2E_PORT" \
    --api-keys-out "$E2E_KEYS" >"$E2E_LOG" 2>&1 &
  echo "$!" > "$E2E_DIR/pid"

  for _ in $(seq 1 60); do
    if [ -f "$E2E_KEYS" ]; then
      local k; k="$(jq -r .anthropic "$E2E_KEYS" 2>/dev/null)"
      if [ -n "$k" ] && [ "$k" != "null" ] && \
         curl -fsS --max-time 3 -o /dev/null -H "x-api-key: $k" \
           "http://127.0.0.1:$E2E_PORT/v1/models" 2>/dev/null; then
        break
      fi
    fi
    sleep 0.5
  done
  export E2E_KEY="$(jq -r .anthropic "$E2E_KEYS" 2>/dev/null)"
  [ -n "$E2E_KEY" ] && [ "$E2E_KEY" != "null" ] || { cat "$E2E_LOG"; skip "brain serve did not start"; }
}

teardown_file() {
  # Kill ONLY the recorded brain-serve PID. NEVER pkill claude.
  [ -f "$E2E_DIR/pid" ] && kill -9 "$(cat "$E2E_DIR/pid")" 2>/dev/null || true
  [ -n "$E2E_DIR" ] && rm -rf "$E2E_DIR"
}

# Drive `claude` in a clean env: our key only, all model aliases -> the local mock,
# telemetry off, a throwaway HOME (no logged-in account), and a hard timeout so it can
# never hang.
run_claude() {
  timeout "$CLAUDE_TIMEOUT" env -u ANTHROPIC_AUTH_TOKEN \
    HOME="$E2E_DIR/home" \
    ANTHROPIC_BASE_URL="http://127.0.0.1:$E2E_PORT" \
    ANTHROPIC_API_KEY="$E2E_KEY" \
    ANTHROPIC_MODEL="mock" \
    ANTHROPIC_DEFAULT_HAIKU_MODEL="mock" \
    ANTHROPIC_DEFAULT_SONNET_MODEL="mock" \
    ANTHROPIC_DEFAULT_OPUS_MODEL="mock" \
    DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    PATH="$PATH" \
    claude "$@"
}

@test "the local Anthropic surface lists the mock model" {
  run curl -fsS --max-time 5 -H "x-api-key: $E2E_KEY" "http://127.0.0.1:$E2E_PORT/v1/models"
  [ "$status" -eq 0 ]
  echo "$output" | jq -e '.data | map(.id) | index("mock")' >/dev/null
}

@test "missing key is rejected (401) — never open" {
  run curl -s --max-time 5 -o /dev/null -w "%{http_code}" -X POST \
    -H 'content-type: application/json' \
    -d '{"model":"mock","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
    "http://127.0.0.1:$E2E_PORT/v1/messages"
  [ "$output" = "401" ]
}

@test "claude uses OUR key (not the subscription) and reaches the local backend" {
  mkdir -p "$E2E_DIR/home"
  run run_claude -p "hello" --output-format stream-json --verbose
  [ "$status" -eq 0 ]
  # The init event proves the key source + that it routed to our local mock model.
  echo "$output" | grep -E '"type":"system"' | head -1 | jq -e '.apiKeySource == "ANTHROPIC_API_KEY"' >/dev/null
  echo "$output" | grep -E '"type":"system"' | head -1 | jq -e '.model == "mock"' >/dev/null
}

@test "claude -p (json) gets a non-error result from brain" {
  mkdir -p "$E2E_DIR/home"
  run run_claude -p "Say pong" --output-format json
  [ "$status" -eq 0 ]
  echo "$output" | jq -e '.is_error == false' >/dev/null
  echo "$output" | jq -e '.result | length > 0' >/dev/null
}

@test "claude -p (stream-json) streams a terminal success result" {
  mkdir -p "$E2E_DIR/home"
  run run_claude -p "Say hello in one word." --output-format stream-json --verbose
  [ "$status" -eq 0 ]
  # The final NDJSON result event has is_error=false.
  echo "$output" | grep -E '"type":"result"' | tail -1 | jq -e '.is_error == false' >/dev/null
}
