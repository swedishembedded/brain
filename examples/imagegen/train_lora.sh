#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# A folder of images in, a LoRA adapter out. Two steps, both brain.
#
#   examples/imagegen/train_lora.sh <image-dir> [adapter.brain] [trigger phrase]
#
# Step 1 captions every image with a vision-language model and writes
# <dir>/captions.yaml -- editable YAML with block scalars, so you can read
# what the model saw and fix it before spending GPU time on it. Captions ARE
# the training signal; a vague caption set caps the adapter's ceiling, so
# reviewing them is the highest-leverage minute in this script.
#
# Step 2 trains the adapter on those captions. Use the result with
# `brain flux2 generate --adapter <adapter.brain>`.
#
# The trigger phrase is what you type later to invoke the style. A rare token
# binds cleanly but reads oddly; a natural phrase composes well but shifts a
# concept the base model already owns, so it leaks into every prompt that
# mentions it. Pick knowing that.
#
# Optional, all env: MODEL (qwen3vl|fastvlm), STEPS, RANK, LR, SIZE, VARIANT,
# TRAINER (device|host -- device is the GPU trainer and the default), BRAIN.
#
# Weights come from BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}; the captioner needs
# its own model, fetched on demand or pointed at with --weights.

set -euo pipefail

DIR="${1:?usage: train_lora.sh <image-dir> [adapter.brain] [trigger phrase]}"; DIR="${DIR%/}"
OUT="${2:-$DIR/adapter.brain}"
TRIGGER="${3:-my style}"
BRAIN="${BRAIN:-./target/release/brain}"

"$BRAIN" label images "$DIR" --model "${MODEL:-fastvlm}" --trigger "$TRIGGER"

echo "captions written to $DIR/captions.yaml - review them, then press enter" >&2
[ -t 0 ] && read -r _

exec "$BRAIN" flux2 finetune "$DIR" --out "$OUT" \
  --variant "${VARIANT:-klein-4b}" --trainer "${TRAINER:-device}" \
  --steps "${STEPS:-1500}" --rank "${RANK:-16}" --lr "${LR:-1e-4}" \
  --size "${SIZE:-512}" --ckpt-every "${CKPT_EVERY:-100}"
