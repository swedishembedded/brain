#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Janus-Pro-7B fine-tuned from its checkpoint as downloaded, in both modes:
# `brain januspro finetune --mode understanding` trains the aligner and a LoRA
# over the bf16 decoder on an image and its reply; `--mode generation` encodes
# the image once with the frozen VQ-16 and trains the adapter, the generation
# head, the aligner and the code embedding on its codes. Each writes its
# trained parameters to --out.
#
# Needs a GPU with about 20 GB free and the checkpoint under
# $BRAIN_MODELS_DIR (default ~/.local/share/brain/models); skips otherwise.
# Run: BRAIN_BIN=./target/release/brain bats tests/e2e/januspro_finetune.bats

REPO_ID="deepseek-ai/Janus-Pro-7B"

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
  mkdir -p "$WORK/und" "$WORK/gen"
  cp "$REPO/docs/quickstart/img/seed.png" "$WORK/und/dog.png"
  cp "$REPO/docs/quickstart/img/seed.png" "$WORK/gen/dog.png"
  printf '%s\n' \
    '{"image":"dog.png","messages":[{"role":"user","content":"<image_placeholder>\nWhat animal is this?"},{"role":"assistant","content":"This is a golden retriever puppy."}]}' \
    >"$WORK/und/train.jsonl"
  printf '%s\n' '{"prompt":"A golden retriever puppy.","image":"dog.png"}' >"$WORK/gen/train.jsonl"
}

teardown_file() {
  [ -n "${WORK:-}" ] && rm -rf "$WORK"
}

# "loss A -> B" on the summary line, with B below A.
loss_fell() {
  echo "$1" | awk '/^trained on/ { for (i = 1; i <= NF; i++) if ($i == "loss") { a = $(i+1); b = $(i+3) } } END { exit !(b + 0 < a + 0) }'
}

@test "understanding: the aligner and an adapter train on the checkpoint as downloaded" {
  run "$BRAIN" januspro finetune --mode understanding --weights "$STORE/$REPO_ID" --dataset "$WORK/und" \
    --out "$WORK/out_und" --steps 4 --lr 1e-4 --seed 1
  [ "$status" -eq 0 ] || { echo "$output" >&3; false; }
  [ -s "$WORK/out_und/adapter.safetensors" ]
  [ -s "$WORK/out_und/aligner.safetensors" ]
  loss_fell "$output"
}

@test "generation: the adapter and the generation heads train on the image's VQ codes" {
  run "$BRAIN" januspro finetune --mode generation --weights "$STORE/$REPO_ID" --dataset "$WORK/gen" \
    --out "$WORK/out_gen" --steps 4 --lr 1e-4 --aligner-lr 1e-4 --seed 1
  [ "$status" -eq 0 ] || { echo "$output" >&3; false; }
  [ -s "$WORK/out_gen/adapter.safetensors" ]
  [ -s "$WORK/out_gen/generation.safetensors" ]
  loss_fell "$output"
}
