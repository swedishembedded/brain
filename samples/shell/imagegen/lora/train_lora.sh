#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# A folder of images in, a LoRA adapter out. Two steps, both brain.
#
#   samples/shell/imagegen/lora/train_lora.sh [weight flags] <image-dir> [adapter.brain] [trigger phrase]
#
# Step 1 captions every image with a vision-language model and writes
# <dir>/captions.yaml -- editable YAML with block scalars, so you can read
# what the model saw and fix it before spending GPU time on it. Captions ARE
# the training signal; a vague caption set caps the adapter's ceiling, so
# reviewing them is the highest-leverage minute in this script.
#
# Step 2 trains the adapter on those captions. Use the result with
# `brain flux2 text2image --adapter <adapter.brain>`.
#
# The trigger phrase is what you type later to invoke the style. A rare token
# binds cleanly but reads oddly; a natural phrase composes well but shifts a
# concept the base model already owns, so it leaks into every prompt that
# mentions it. Pick knowing that.
#
# Optional, all env: MODEL (qwen3vl|fastvlm), SEED, STEPS, RANK, LR, SIZE, VARIANT,
# TRAINER (device|host -- device is the GPU trainer and the default), BRAIN.
#
# Weights come from the model store; `--dit`, `--vae`, `--text-encoder` and
# `--tokenizer` (before the positional arguments) name FLUX.2 weights outright,
# and `--captioner DIR` the captioner's checkpoint (the captioner needs its own
# model; the store's, or fetched on demand, when not named).

set -euo pipefail
FLUX2_ROLES=(); CAPTIONER=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dit|--vae|--text-encoder|--tokenizer) FLUX2_ROLES+=("$1" "${2:?$1 needs a path}"); shift 2 ;;
    --captioner) CAPTIONER=(--weights "${2:?--captioner needs a directory}"); shift 2 ;;
    --) shift; break ;;
    -*) echo "unknown flag $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

DIR="${1:?usage: train_lora.sh [--dit P] [--vae P] [--text-encoder P] [--tokenizer P] [--captioner DIR] <image-dir> [adapter.brain] [trigger phrase]}"; DIR="${DIR%/}"
OUT="${2:-$DIR/adapter.brain}"
TRIGGER="${3:-my style}"
BRAIN="${BRAIN:-./target/release/brain}"

"$BRAIN" label images "$DIR" --model "${MODEL:-fastvlm}" "${CAPTIONER[@]}" --trigger "$TRIGGER"

echo "captions written to $DIR/captions.yaml - review them, then press enter" >&2
[ -t 0 ] && read -r _

# One seed per run: unset, a random one is drawn and printed.
SEED="${SEED:-$RANDOM$RANDOM}"
echo "train_lora: seed $SEED (SEED=$SEED reproduces this run)" >&2

exec "$BRAIN" flux2 lora_train "${FLUX2_ROLES[@]}" --data "$DIR" --save "$OUT" --seed "$SEED" \
  --variant "${VARIANT:-klein-4b}" --trainer "${TRAINER:-device}" \
  --steps "${STEPS:-1500}" --rank "${RANK:-16}" --lr "${LR:-1e-4}" \
  --size "${SIZE:-512}" --ckpt-every "${CKPT_EVERY:-100}"
