#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# DeepSeek-R1-Distill-Qwen-1.5B served end to end from its checkpoint as
# downloaded: `brain serve` over a models directory holding only that repo
# (a symlink to the real store's copy, so nothing is written there) lists it
# under its own id, and an OpenAI chat completion comes back with the R1
# reasoning split out as `reasoning_content`, a final answer, and
# finish_reason "stop" - generation ended on the checkpoint's own
# end-of-sentence token, not on the length cap.
#
# Needs a GPU and the checkpoint under $BRAIN_MODELS_DIR (default
# ~/.local/share/brain/models); skips otherwise.
# Run: BRAIN_BIN=./target/debug/brain bats tests/e2e/deepseek_chat.bats

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

setup_file() {
  command -v jq >/dev/null 2>&1 || skip "jq not installed"
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no brain binary"
  local store="${BRAIN_MODELS_DIR:-$HOME/.local/share/brain/models}"
  [ -f "$store/$MODEL/config.json" ] || skip "$MODEL is not downloaded"

  export CONF_DIR="$(mktemp -d)"
  mkdir -p "$CONF_DIR/models/deepseek-ai"
  ln -s "$store/$MODEL" "$CONF_DIR/models/$MODEL"
  export PORT="${DEEPSEEK_CHAT_PORT:-8906}"
  "$BRAIN" serve --models-dir "$CONF_DIR/models" --openai "$PORT" \
    --api-keys-out "$CONF_DIR/keys.json" --ready-file "$CONF_DIR/ready" >"$CONF_DIR/serve.log" 2>&1 &
  export SERVER_PID=$!
  for _ in $(seq 1 120); do
    [ -e "$CONF_DIR/ready" ] && break
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 0.5
  done
  [ -e "$CONF_DIR/ready" ] || { cat "$CONF_DIR/serve.log" >&3; skip "brain serve did not become ready"; }
  export KEY="$(jq -r .openai "$CONF_DIR/keys.json")"
}

teardown_file() {
  [ -n "${SERVER_PID:-}" ] && kill -9 "$SERVER_PID" 2>/dev/null || true
  [ -n "${CONF_DIR:-}" ] && rm -rf "$CONF_DIR"
}

@test "the checkpoint is listed under its own id" {
  curl -fsS -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" >"$CONF_DIR/models.json"
  jq -e --arg m "$MODEL" '.data | map(.id) | index($m) != null' "$CONF_DIR/models.json"
}

@test "a chat completion reasons, answers and stops on its own end token" {
  local body
  body=$(jq -n --arg m "$MODEL" '{model: $m, messages: [{role: "user", content: "What is 17 + 25? Reply with just the number."}], max_tokens: 2048, temperature: 0}')
  local status
  status=$(curl -sS --max-time 900 -o "$CONF_DIR/chat.json" -w '%{http_code}' \
    -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
    -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -d "$body")
  [ "$status" -eq 200 ] || { cat "$CONF_DIR/chat.json" >&3; false; }
  jq -e '.choices[0].message.reasoning_content | length > 0' "$CONF_DIR/chat.json"
  jq -e '.choices[0].message.content | test("42")' "$CONF_DIR/chat.json"
  jq -e '.choices[0].finish_reason == "stop"' "$CONF_DIR/chat.json"
}
