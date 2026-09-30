#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Janus-Pro-7B served end to end from its checkpoint as downloaded: `brain
# serve` over a models directory holding only that repo (its files linked
# from the real store's copy, so nothing is written there) lists the model;
# an OpenAI chat completion with an image describes it; and
# /v1/images/generations draws a 384x384 PNG, the one size the model
# generates, while refusing any other size by name. The two requests run on
# two different builds of the checkpoint, which the scheduler swaps.
#
# Needs a GPU with about 20 GB free and the checkpoint under
# $BRAIN_MODELS_DIR (default ~/.local/share/brain/models); skips otherwise.
# Run: BRAIN_BIN=./target/debug/brain bats tests/e2e/januspro.bats

REPO_ID="deepseek-ai/Janus-Pro-7B"
MODEL="brain/januspro"

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
  export PORT="${JANUSPRO_PORT:-8908}"
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

post() {
  curl -sS --max-time 1800 -o "$CONF_DIR/$1.json" -w '%{http_code}' \
    -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
    -X POST "http://127.0.0.1:$PORT$2" --data-binary @"$CONF_DIR/$1.body"
}

@test "the model is listed" {
  curl -fsS -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" >"$CONF_DIR/models.json"
  jq -e --arg m "$MODEL" '.data | map(.id) | index($m) != null' "$CONF_DIR/models.json"
}

@test "a chat completion with an image describes it" {
  { printf 'data:image/png;base64,'; base64 -w0 "$REPO/docs/quickstart/img/seed.png"; } >"$CONF_DIR/url.txt"
  jq -n --arg m "$MODEL" --rawfile u "$CONF_DIR/url.txt" '{model: $m, max_tokens: 64, temperature: 0,
    messages: [{role: "user", content: [{type: "image_url", image_url: {url: $u}}, {type: "text", text: "Which animal is in this image? Answer in one short sentence."}]}]}' >"$CONF_DIR/chat.body"
  local status
  status=$(post chat /v1/chat/completions)
  [ "$status" -eq 200 ] || { cat "$CONF_DIR/chat.json" "$CONF_DIR/serve.log" >&3; false; }
  jq -e '.choices[0].message.content | ascii_downcase | test("dog|retriever|puppy")' "$CONF_DIR/chat.json" || { cat "$CONF_DIR/chat.json" >&3; false; }
}

@test "an image generation draws a 384x384 PNG and refuses other sizes" {
  jq -n --arg m "$MODEL" '{model: $m, prompt: "A red apple on a wooden table.", size: "512x512", seed: 7}' >"$CONF_DIR/wrong.body"
  local status
  status=$(post wrong /v1/images/generations)
  [ "$status" -eq 400 ] || { cat "$CONF_DIR/wrong.json" >&3; false; }
  jq -e '.error.message | test("384")' "$CONF_DIR/wrong.json"

  jq -n --arg m "$MODEL" '{model: $m, prompt: "A red apple on a wooden table.", size: "384x384", seed: 7}' >"$CONF_DIR/image.body"
  status=$(post image /v1/images/generations)
  [ "$status" -eq 200 ] || { cat "$CONF_DIR/image.json" "$CONF_DIR/serve.log" >&3; false; }
  jq -r '.data[0].b64_json' "$CONF_DIR/image.json" | base64 -d >"$CONF_DIR/apple.png"
  [ "$(od -An -tx1 -N8 "$CONF_DIR/apple.png" | tr -d ' \n')" = "89504e470d0a1a0a" ]
  # The PNG header's width and height (bytes 16..24, big-endian).
  [ "$(od -An -tu1 -j16 -N8 "$CONF_DIR/apple.png" | tr -s ' ' | sed 's/^ //')" = "0 0 1 128 0 0 1 128" ]
}
