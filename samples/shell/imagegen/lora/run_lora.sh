#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Generate with an adapter that `train_lora.sh` produced.
#
#   samples/shell/imagegen/lora/run_lora.sh [weight flags] <adapter|folder> "<prompt>" [count]
#
# The first argument is either the adapter file or the folder you trained on
# -- `train_lora.sh` writes `adapter.brain` inside it, so passing the folder
# is enough and you do not have to remember where training put things.
#
#   samples/shell/imagegen/lora/run_lora.sh ~/photos/bohemian "a bohemian style bedroom"
#
# Whatever trigger phrase you trained with has to appear in the prompt, or the
# adapter has little to fire on. Results are `<name>-01.jpg` ... one per seed,
# beside the adapter.
#
# REF restyles an existing photograph instead of generating from nothing, and
# STRENGTH is the dial: 1.0 generates a new image conditioned on it, lower
# values keep progressively more of it, 0 returns it. Below 1.0 the reference
# IS the starting latent and so must be at the output size; the script
# resizes it for you.
#
#   REF=room.jpg STRENGTH=0.96 samples/shell/imagegen/lora/run_lora.sh ~/photos/boho "..."
#
# Optional, all env: REF, STRENGTH, SEED (first; each result adds 1), STEPS,
# SIZE (WxH), SCALE (adapter strength, 0 reproduces the base model -- the way
# to see what the adapter is actually contributing), VARIANT, PRECISION, BRAIN.
#
# Weights come from the model store; `--dit`, `--vae`, `--text-encoder` and
# `--tokenizer` (before the positional arguments) name one outright. brain picks
# a card with room unless DEVICE (gpu0/gpu1/cpu, passed as --device) says
# otherwise.

set -euo pipefail
FLUX2_ROLES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dit|--vae|--text-encoder|--tokenizer) FLUX2_ROLES+=("$1" "${2:?$1 needs a path}"); shift 2 ;;
    --) shift; break ;;
    -*) echo "unknown flag $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

A="${1:?usage: run_lora.sh [--dit P] [--vae P] [--text-encoder P] [--tokenizer P] <adapter|folder> \"<prompt>\" [count]}"
[ -d "$A" ] && A="$A/adapter.brain"
[ -f "$A" ] || { echo "no adapter at $A (train_lora.sh writes adapter.brain)" >&2; exit 1; }
PROMPT="${2:?a prompt is required -- include the trigger phrase you trained with}"
N="${3:-4}"
OUT="${A%.brain}"; SIZE="${SIZE:-1024x768}"

ACTION=text2image
ARGS=()
if [ -n "${REF:-}" ]; then
  ACTION=edit
  R="$(dirname "$A")/.ref.png"
  python3 -c "from PIL import Image;import sys;Image.open(sys.argv[1]).convert('RGB').resize((${SIZE%x*},${SIZE#*x}),Image.LANCZOS).save(sys.argv[2])" "$REF" "$R"
  ARGS+=(--image "$R" --strength "${STRENGTH:-1.0}")
  [ "${STRENGTH:-1.0}" = "1.0" ] && ARGS+=(--ref-size 768)
fi

# One seed per run: unset, a random one is drawn; each result adds 1 to it.
SEED="${SEED:-$(( (RANDOM << 15) ^ RANDOM ))}"
echo "run_lora: seeds $SEED.. (SEED=$SEED reproduces this run)" >&2

for i in $(seq 1 "$N"); do
  "${BRAIN:-./target/release/brain}" ${DEVICE:+--device "$DEVICE"} flux2 "$ACTION" "${FLUX2_ROLES[@]}" \
    --variant "${VARIANT:-klein-4b}" --precision "${PRECISION:-int8}" \
    --prompt "$PROMPT" --adapter "$A@${SCALE:-1.0}" \
    --width "${SIZE%x*}" --height "${SIZE#*x}" "${ARGS[@]}" \
    --steps "${STEPS:-12}" --seed "$(( SEED + i - 1 ))" \
    --out "image=$OUT-$(printf '%02d' "$i").jpg"
done
