#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# DeepSeek-VL-7B fine-tuned from its checkpoint as downloaded: `brain
# deepseekvl finetune` extracts an image's tower features, releases the towers,
# trains the aligner and a LoRA over the bf16 decoder for a few steps on one
# card, and writes the adapter and the aligner to --out.
#
# Needs a GPU with about 20 GB free and the checkpoint under
# $BRAIN_MODELS_DIR (default ~/.local/share/brain/models); skips otherwise.
# Run: BRAIN_BIN=./target/release/brain bats tests/e2e/deepseek_vl_finetune.bats

REPO_ID="deepseek-ai/deepseek-vl-7b-chat"

setup_file() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no brain binary"
  local store="${BRAIN_MODELS_DIR:-$HOME/.local/share/brain/models}"
  [ -f "$store/$REPO_ID/config.json" ] || skip "$REPO_ID is not downloaded"
  export BRAIN STORE="$store"
  export WORK="$(mktemp -d)"
  mkdir -p "$WORK/data"
  cp "$REPO/docs/quickstart/img/seed.png" "$WORK/data/dog.png"
  printf '%s\n' \
    '{"image":"dog.png","messages":[{"role":"user","content":"<image_placeholder>\nWhat animal is this?"},{"role":"assistant","content":"This is a golden retriever puppy."}]}' \
    '{"image":"dog.png","messages":[{"role":"user","content":"<image_placeholder>\nDescribe the picture."},{"role":"assistant","content":"A golden retriever puppy looking at the camera."}]}' \
    >"$WORK/data/train.jsonl"
}

teardown_file() {
  [ -n "${WORK:-}" ] && rm -rf "$WORK"
}

@test "the aligner and an adapter train on a bf16 decoder read from the checkpoint as downloaded" {
  run "$BRAIN" deepseekvl finetune --weights "$STORE/$REPO_ID" --dataset "$WORK/data" \
    --out "$WORK/out" --steps 4 --lr 1e-4 --seed 1
  [ "$status" -eq 0 ] || { echo "$output" >&3; false; }
  [[ "$output" == *"trained on 2 example(s)"* ]]
  [ -s "$WORK/out/adapter.safetensors" ]
  [ -s "$WORK/out/aligner.safetensors" ]
  # the loss moved down: "loss A -> B" with B < A
  echo "$output" | awk '/^trained on/ { for (i = 1; i <= NF; i++) if ($i == "loss") { a = $(i+1); b = $(i+3) } } END { exit !(b + 0 < a + 0) }'
}

# An example may carry several images, one placeholder per image in its
# messages, and a step may average several examples.
@test "an example with two images trains, several examples to a step" {
  mkdir -p "$WORK/two"
  cp "$REPO/docs/quickstart/img/seed.png" "$WORK/two/a.png"
  cp "$REPO/docs/quickstart/img/depth.png" "$WORK/two/b.png"
  printf '%s\n' \
    '{"images":["a.png","b.png"],"messages":[{"role":"user","content":"<image_placeholder><image_placeholder>\nWhich of the two is a photograph?"},{"role":"assistant","content":"The first one."}]}' \
    '{"images":["b.png","a.png"],"messages":[{"role":"user","content":"<image_placeholder><image_placeholder>\nWhich of the two is a photograph?"},{"role":"assistant","content":"The second one."}]}' \
    >"$WORK/two/train.jsonl"
  run "$BRAIN" deepseekvl finetune --weights "$STORE/$REPO_ID" --dataset "$WORK/two" \
    --out "$WORK/out2" --steps 3 --batch 2 --lr 1e-4 --seed 1
  [ "$status" -eq 0 ] || { echo "$output" >&3; false; }
  [[ "$output" == *"trained on 2 example(s)"* ]]
  [ -s "$WORK/out2/adapter.safetensors" ]
}

# The fine-tune is served: BRAIN_DEEPSEEKVL_TUNED names the directory `finetune`
# wrote, and the served model answers with what it learned.
@test "a fine-tune is served and answers with what it learned" {
  mkdir -p "$WORK/models/$REPO_ID" "$WORK/learn"
  for f in "$STORE/$REPO_ID"/*; do ln -s "$f" "$WORK/models/$REPO_ID/"; done
  cp "$WORK/data/dog.png" "$WORK/learn/dog.png"
  printf '%s\n' '{"image":"dog.png","messages":[{"role":"user","content":"<image_placeholder>\nWhat animal is this?"},{"role":"assistant","content":"That is a Blorbville puppy."}]}' >"$WORK/learn/train.jsonl"
  run "$BRAIN" deepseekvl finetune --weights "$REPO_ID" --models-dir "$WORK/models" --dataset "$WORK/learn" \
    --out "$WORK/tuned" --steps 14 --lr 3e-4 --seed 1
  [ "$status" -eq 0 ] || { echo "$output" >&3; false; }

  local port="${DEEPSEEK_VL_FT_PORT:-8942}"
  BRAIN_DEEPSEEKVL_TUNED="$WORK/tuned" "$BRAIN" serve --models-dir "$WORK/models" --openai "$port" \
    --api-keys-out "$WORK/keys.json" --ready-file "$WORK/ready" >"$WORK/serve.log" 2>&1 &
  local server=$!
  for _ in $(seq 1 120); do [ -e "$WORK/ready" ] && break; sleep 0.5; done
  [ -e "$WORK/ready" ] || { kill -9 "$server" 2>/dev/null; cat "$WORK/serve.log" >&3; false; }
  local key; key="$(jq -r .openai "$WORK/keys.json")"
  { printf 'data:image/png;base64,'; base64 -w0 "$WORK/learn/dog.png"; } >"$WORK/url.txt"
  jq -n --rawfile u "$WORK/url.txt" '{model: "brain/deepseekvl", max_tokens: 30, temperature: 0, messages: [{role: "user", content: [{type: "image_url", image_url: {url: $u}}, {type: "text", text: "What animal is this?"}]}]}' >"$WORK/ask.json"
  curl -fsS --max-time 900 -H "Authorization: Bearer $key" -H 'content-type: application/json' \
    -X POST "http://127.0.0.1:$port/v1/chat/completions" --data-binary @"$WORK/ask.json" >"$WORK/answer.json" || { kill -9 "$server" 2>/dev/null; cat "$WORK/serve.log" >&3; false; }
  kill -9 "$server" 2>/dev/null || true
  jq -e '.choices[0].message.content | test("Blorbville")' "$WORK/answer.json" || { cat "$WORK/answer.json" >&3; false; }
}

# The same fine-tune kept beside the checkpoint is a model of its own: no
# environment variable, the id carries its owner, name and tag, and the base
# model beside it still answers as the base.
@test "a stored fine-tune is listed as its own model and the base is unchanged" {
  [ -d "$WORK/tuned" ] || skip "the fine-tune test did not run"
  mkdir -p "$WORK/models/$REPO_ID/adapters/local/blorb"
  cp -r "$WORK/tuned" "$WORK/models/$REPO_ID/adapters/local/blorb/v1"
  local port="${DEEPSEEK_VL_FT_PORT:-8942}"
  "$BRAIN" serve --models-dir "$WORK/models" --openai "$port" \
    --api-keys-out "$WORK/keys2.json" --ready-file "$WORK/ready2" >"$WORK/serve2.log" 2>&1 &
  local server=$!
  for _ in $(seq 1 120); do [ -e "$WORK/ready2" ] && break; sleep 0.5; done
  [ -e "$WORK/ready2" ] || { kill -9 "$server" 2>/dev/null; cat "$WORK/serve2.log" >&3; false; }
  local key; key="$(jq -r .openai "$WORK/keys2.json")"
  curl -fsS -H "Authorization: Bearer $key" "http://127.0.0.1:$port/v1/models" >"$WORK/models.json"
  jq -e '[.data[].id] | index("brain/deepseekvl:local:blorb:v1") != null and index("brain/deepseekvl") != null' "$WORK/models.json" \
    || { kill -9 "$server" 2>/dev/null; cat "$WORK/models.json" >&3; false; }
  { printf 'data:image/png;base64,'; base64 -w0 "$WORK/learn/dog.png"; } >"$WORK/url2.txt"
  for model in brain/deepseekvl:local:blorb:v1 brain/deepseekvl; do
    jq -n --rawfile u "$WORK/url2.txt" --arg m "$model" '{model: $m, max_tokens: 30, temperature: 0, messages: [{role: "user", content: [{type: "image_url", image_url: {url: $u}}, {type: "text", text: "What animal is this?"}]}]}' >"$WORK/ask2.json"
    curl -fsS --max-time 900 -H "Authorization: Bearer $key" -H 'content-type: application/json' \
      -X POST "http://127.0.0.1:$port/v1/chat/completions" --data-binary @"$WORK/ask2.json" >"$WORK/answer-${model//[:\/]/_}.json" || { kill -9 "$server" 2>/dev/null; cat "$WORK/serve2.log" >&3; false; }
  done
  kill -9 "$server" 2>/dev/null || true
  jq -e '.choices[0].message.content | test("Blorbville")' "$WORK/answer-brain_deepseekvl_local_blorb_v1.json"
  ! jq -e '.choices[0].message.content | test("Blorbville")' "$WORK/answer-brain_deepseekvl.json"
}
