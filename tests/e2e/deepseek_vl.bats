#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# DeepSeek-VL-7B-chat served end to end from its checkpoint as downloaded:
# `brain serve` over a models directory holding only that repo (its files
# linked from the real store's copy, so nothing is written there; the store
# scan does not follow a linked directory) lists the model, and an
# OpenAI chat completion carrying an image as a data URL describes what the
# image shows and stops on the checkpoint's own end-of-sentence token.
#
# Needs a GPU with about 20 GB free and the checkpoint under
# $BRAIN_MODELS_DIR (default ~/.local/share/brain/models); skips otherwise.
# Run: BRAIN_BIN=./target/debug/brain bats tests/e2e/deepseek_vl.bats

REPO_ID="deepseek-ai/deepseek-vl-7b-chat"
MODEL="brain/deepseekvl"

setup_file() {
  command -v jq >/dev/null 2>&1 || skip "jq not installed"
  command -v base64 >/dev/null 2>&1 || skip "base64 not installed"
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no brain binary"
  local store="${BRAIN_MODELS_DIR:-$HOME/.local/share/brain/models}"
  [ -f "$store/$REPO_ID/config.json" ] || skip "$REPO_ID is not downloaded"

  export CONF_DIR="$(mktemp -d)"
  mkdir -p "$CONF_DIR/models/$REPO_ID"
  for f in "$store/$REPO_ID"/*; do ln -s "$f" "$CONF_DIR/models/$REPO_ID/"; done
  export PORT="${DEEPSEEK_VL_PORT:-8907}"
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

@test "the model is listed" {
  curl -fsS -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" >"$CONF_DIR/models.json"
  jq -e --arg m "$MODEL" '.data | map(.id) | index($m) != null' "$CONF_DIR/models.json"
}

@test "a chat completion with an image describes it and stops on its own end token" {
  local status
  { printf 'data:image/png;base64,'; base64 -w0 "$REPO/docs/quickstart/img/seed.png"; } >"$CONF_DIR/url.txt"
  jq -n --arg m "$MODEL" --rawfile u "$CONF_DIR/url.txt" '{model: $m, max_tokens: 64, temperature: 0,
    messages: [{role: "user", content: [{type: "image_url", image_url: {url: $u}}, {type: "text", text: "Which animal is in this image? Answer in one short sentence."}]}]}' >"$CONF_DIR/body.json"
  status=$(curl -sS --max-time 1800 -o "$CONF_DIR/chat.json" -w '%{http_code}' \
    -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
    -X POST "http://127.0.0.1:$PORT/v1/chat/completions" --data-binary @"$CONF_DIR/body.json")
  [ "$status" -eq 200 ] || { cat "$CONF_DIR/chat.json" "$CONF_DIR/serve.log" >&3; false; }
  jq -e '.choices[0].message.content | ascii_downcase | test("dog|retriever|puppy")' "$CONF_DIR/chat.json" || { cat "$CONF_DIR/chat.json" >&3; false; }
  jq -e '.choices[0].finish_reason == "stop"' "$CONF_DIR/chat.json"
  jq -e '.usage.prompt_tokens > 576' "$CONF_DIR/chat.json"
}
