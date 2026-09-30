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
