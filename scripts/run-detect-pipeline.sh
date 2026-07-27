#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# End-to-end demo: serve z-image + yolo over D-Bus, then run the
# generate -> detect -> draw pipeline, saving an image at every step.
#
# Weight locations come from the environment (no hardcoded paths in code). Point
# these at your checkout; the defaults match this box's resource tree and can be
# overridden by exporting them before calling this script.
#
#   scripts/run-detect-pipeline.sh
#   PROMPT="a red bicycle" SIZE=512 scripts/run-detect-pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RES="${BRAIN_RESOURCES:-/data/workspace/resources}"
export BRAIN_ZIMAGE_DIT="${BRAIN_ZIMAGE_DIT:-$RES/image-models/z-image/weights-bf16/split_files/diffusion_models/z_image_turbo_bf16.safetensors}"
export BRAIN_ZIMAGE_VAE="${BRAIN_ZIMAGE_VAE:-$RES/image-models/z-image/weights/vae/diffusion_pytorch_model.safetensors}"
export BRAIN_ZIMAGE_QWEN="${BRAIN_ZIMAGE_QWEN:-$RES/image-models/common/qwen3-4b-text-encoder/split_files/text_encoders/qwen_3_4b.safetensors}"
export BRAIN_ZIMAGE_TOKENIZER="${BRAIN_ZIMAGE_TOKENIZER:-$RES/image-models/z-image/weights/tokenizer/tokenizer.json}"
export BRAIN_YOLO="${BRAIN_YOLO:-/data/resources/yolo/yolov8n.brain.weights}"

export OUT="${OUT:-/tmp/brain_pipeline}"
export SIZE="${SIZE:-512}"
export STEPS="${STEPS:-8}"

BIN="${BRAIN_BIN:-$(pwd)/target/release/brain}"
[ -x "$BIN" ] || BIN="$(pwd)/target/debug/brain"
[ -x "$BIN" ] || { echo "build brain first: cargo build --release -p brain-cli" >&2; exit 1; }

echo "serving z-image + yolo over a private D-Bus session; output -> $OUT"
exec dbus-run-session -- bash -c '
  set -e
  "'"$BIN"'" serve --dbus --reserve-gb 4 2>'"$OUT"'/serve.log &
  SRV=$!
  trap "kill $SRV 2>/dev/null || true" EXIT
  # Wait for the bus name to appear (models loading can take a while).
  for i in $(seq 1 60); do
    busctl --user list 2>/dev/null | grep -q com.swedishembedded.Brain1 && break
    sleep 1
  done
  python3 examples/dbus/detect_pipeline.py
'
