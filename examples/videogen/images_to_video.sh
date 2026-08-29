#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Each image becomes its OWN video - point this at one still, or at a
# folder of them, and get one clip per image, named after it.
#
#   examples/videogen/images_to_video.sh <image-or-folder> ["prompt"] [seconds]
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
# examples/videogen/chain_images_to_video.sh instead.
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
# Weights: point LTX_MODEL_DIR at a folder holding LTX-2.5's files FLAT (no
# vae/text_encoders/diffusion_models subfolders - see
# _resolve_ltxv_weights.sh's own header for exactly which filenames and why
# the DiT is the one file that folder alone will not contain) and this finds
# BRAIN_LTXV_{DIT,VAE,TEXT_ENCODER} for you; unset, it defaults to
# $BRAIN_MODELS_DIR/Lightricks/LTX-2.5, brain's own standard model
# directory (BRAIN_MODELS_DIR, else XDG_DATA_HOME/brain/models, else
# ~/.local/share/brain/models). Anything not found there is asked for
# interactively, once. LTX_TINY=1 skips all of this and runs the
# tiny random-weight DiT instead - a real end-to-end wiring test (the
# still genuinely conditions the noise), but not a quality claim.

set -euo pipefail

DEFAULT_PROMPT="the scene comes to life with smooth, natural motion and a gently moving camera"

SRC="${1:?usage: images_to_video.sh <image-or-folder> [\"prompt\"] [seconds]}"
CLI_PROMPT="${2:-}"
FPS="${FPS:-24}"
FRAMES=$(( ${3:-5} * FPS / 8 * 8 + 1 ))   # must be 1 + 8k

if [ "${LTX_TINY:-0}" = "1" ]; then
  DIT_CONFIG=tiny
else
  # shellcheck source=_resolve_ltxv_weights.sh
  source "$(dirname "${BASH_SOURCE[0]}")/_resolve_ltxv_weights.sh"
  DIT_CONFIG=ltx25_22b
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
  "${BRAIN:-./target/release/brain}" ltxv t2v \
    --prompt "$PROMPT" \
    --frames "$FRAMES" --width "${WIDTH:-1280}" --height "${HEIGHT:-704}" \
    --steps "${STEPS:-8}" --seed "${SEED:-7}" --fps "$FPS" \
    --dit-config "$DIT_CONFIG" \
    --start-frame "$IMG" \
    --output-path "$OUT"
done
