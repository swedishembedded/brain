#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Each image becomes its OWN video - point this at one still, or at a
# folder of them, and get one clip per image, named after it.
#
#   samples/shell/videogen/ltxv/images_to_video.sh <image-or-folder> ["prompt"] [seconds]
#
# <image-or-folder> is either:
#   a single PNG/JPEG file    -> one clip: photo.png makes photo.mp4
#   a folder of PNG/JPEG files -> one clip PER image, same rule, so
#                                 shots/a.png and shots/b.jpg make
#                                 shots/a.mp4 and shots/b.mp4
#
# Each image conditions ONLY its own clip's opening frame
# (`brain ltxv t2v --start-frame`) - this is independent image-to-video,
# not a sequence: N images make N unrelated clips, not one clip that
# passes through all of them. For a single continuous clip that opens on
# one still, passes through others, and closes on a last one, see
# samples/shell/videogen/ltxv/chain_images_to_video.sh instead.
#
# The prompt is, in order: the second argument if given; else
# <folder>/prompt.txt if the input is a folder and that file exists (used
# for every image in it); else a generic placeholder describing plain
# image-to-video motion, which is a wiring convenience, NOT something to
# rely on for quality - a real prompt describing what should actually
# happen in the shot is always better than the default.
#
# Optional, all env: WIDTH, HEIGHT, STEPS, SEED, FPS, BRAIN_DEVICE, BRAIN.
#
# Audio is ON by default: LTX-2.5 is natively audio-visual and generates
# sound from the SAME forwards as the picture, over the same prompt - it
# only reads as a description of sound if the prompt actually describes
# one (the default placeholder above does; a real prompt should too, or
# the track comes out close to silent/mono). LTX_AUDIO=0 turns it off
# (a real video-only DiT forward, cheaper, no audio VAE needed).
#
# LTX_TRACE=<0-5> (default 4) controls how much of the load/generate is
# printed as it happens (`brain --trace-ltxv`, an existing, tested
# instrumentation family - not something this script adds): 4 gives each
# host stage's duration and the per-generation weight cache's hit/miss
# split; 5 adds every transformer block individually (GGUF-read/quantize/
# GPU milliseconds), useful when the model's own weight streaming - not
# generation itself - is what's taking a while. 0 is silent.
#
# Weights: every role is resolved from brain's model store (default root
# ~/.local/share/brain/models, repo Lightricks/LTX-2.5); a missing role makes
# brain stop with its own message. To use files outside the store, put
# brain's role flags (--dit, --vae, --audio-vae, --text-encoder, ...) after a
# literal "--" and they are passed through unchanged. LTX_TINY=1 runs the
# tiny random-weight DiT instead - a real end-to-end wiring test (the
# still genuinely conditions the noise), but not a quality claim.

set -euo pipefail

DEFAULT_PROMPT="the scene comes to life with smooth, natural motion and a gently moving camera, with ambient sound matching the setting"

# Arguments after a literal "--" go to brain unchanged (see the Weights note).
BRAIN_FLAGS=()
POSITIONAL=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; BRAIN_FLAGS=("$@"); break; fi
  POSITIONAL+=("$1"); shift
done
set -- ${POSITIONAL[@]+"${POSITIONAL[@]}"}

SRC="${1:?usage: images_to_video.sh <image-or-folder> [\"prompt\"] [seconds]}"
CLI_PROMPT="${2:-}"
FPS="${FPS:-24}"
FRAMES=$(( ${3:-5} * FPS / 8 * 8 + 1 ))   # must be 1 + 8k

if [ "${LTX_TINY:-0}" = "1" ]; then
  DIT_CONFIG=tiny
  AUDIO_FLAG=()
else
  DIT_CONFIG=ltx25_22b
  AUDIO_FLAG=()
  [ "${LTX_AUDIO:-1}" = "1" ] && AUDIO_FLAG=(--audio)
fi

IMAGES=()
FOLDER_PROMPT_FILE=""
if [ -f "$SRC" ]; then
  IMAGES=("$SRC")
elif [ -d "$SRC" ]; then
  while IFS= read -r -d '' f; do IMAGES+=("$f"); done < <(
    find "$SRC" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print0 | sort -z
  )
  [ "${#IMAGES[@]}" -gt 0 ] || { echo "images_to_video: $SRC has no PNG/JPEG images" >&2; exit 1; }
  FOLDER_PROMPT_FILE="$SRC/prompt.txt"
else
  echo "images_to_video: no such file or folder: $SRC" >&2
  exit 1
fi

if [ -n "$CLI_PROMPT" ]; then
  PROMPT="$CLI_PROMPT"
elif [ -n "$FOLDER_PROMPT_FILE" ] && [ -s "$FOLDER_PROMPT_FILE" ]; then
  PROMPT="$(cat "$FOLDER_PROMPT_FILE")"
else
  PROMPT="$DEFAULT_PROMPT"
fi
echo "images_to_video: prompt: $PROMPT" >&2

for IMG in "${IMAGES[@]}"; do
  OUT="${IMG%.*}.mp4"
  echo "images_to_video: $IMG -> $OUT"
  "${BRAIN:-./target/release/brain}" --trace-ltxv "${LTX_TRACE:-4}" ltxv t2v \
    --prompt "$PROMPT" \
    --frames "$FRAMES" --width "${WIDTH:-1280}" --height "${HEIGHT:-704}" \
    --steps "${STEPS:-8}" ${SEED:+--seed "$SEED"} --fps "$FPS" \
    --dit-config "$DIT_CONFIG" ${AUDIO_FLAG[@]+"${AUDIO_FLAG[@]}"} \
    --start-frame "$IMG" \
    --output-path "$OUT" \
    ${BRAIN_FLAGS[@]+"${BRAIN_FLAGS[@]}"}
done
