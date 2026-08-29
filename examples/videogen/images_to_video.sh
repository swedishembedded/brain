#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# A folder of stills and a prompt in, one clip out.
#
#   examples/videogen/images_to_video.sh <folder/> [out.mp4] [seconds]
#
# <folder/> holds:
#   prompt.txt          the whole clip's prompt (required)
#   1-3 PNG/JPEG stills, named so sorting gives the order you want them in
#                        (01.png, 02.png, ... is the simplest scheme)
#
# The stills are anchor frames, not a slideshow: LTX-2.5 denoises everything
# BETWEEN them from the prompt, so what you get is one continuous shot that
# starts at (and, with two or three stills, passes through and ends at)
# exactly those pixels. How many stills decides which of the model's
# conditioning slots get used:
#
#   1 image  -> --start-frame                 opens on that still
#   2 images -> --start-frame + --end-frame   opens and closes on them
#   3 images -> --start-frame + --mid-frame + --end-frame
#               opens, passes through the middle one at the clip's
#               midpoint, and closes on the last - the way to keep a
#               longer or moving-camera clip on course throughout, not
#               only at its ends
#
# Passing the SAME image twice (e.g. as both --start-frame and --end-frame)
# asks for a clip that returns to where it started, which some prompts read
# as "loop" and others as "the same picture, unmoving" - describe motion
# explicitly in the prompt if you want the former.
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
# stills genuinely condition the noise), but not a quality claim.

set -euo pipefail

DIR="${1:?usage: images_to_video.sh <folder/> [out.mp4] [seconds]}"
OUT="${2:-clip.mp4}"
FPS="${FPS:-24}"
FRAMES=$(( ${3:-5} * FPS / 8 * 8 + 1 ))   # must be 1 + 8k

if [ "${LTX_TINY:-0}" = "1" ]; then
  DIT_CONFIG=tiny
else
  # shellcheck source=_resolve_ltxv_weights.sh
  source "$(dirname "${BASH_SOURCE[0]}")/_resolve_ltxv_weights.sh"
  DIT_CONFIG=ltx25_22b
fi

[ -d "$DIR" ] || { echo "images_to_video: no such folder: $DIR" >&2; exit 1; }

PROMPT_FILE="$DIR/prompt.txt"
[ -f "$PROMPT_FILE" ] || { echo "images_to_video: $DIR has no prompt.txt (the clip's prompt goes there)" >&2; exit 1; }
PROMPT="$(cat "$PROMPT_FILE")"
[ -n "$PROMPT" ] || { echo "images_to_video: $PROMPT_FILE is empty" >&2; exit 1; }

IMAGES=()
while IFS= read -r -d '' f; do IMAGES+=("$f"); done < <(
  find "$DIR" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print0 | sort -z
)

case "${#IMAGES[@]}" in
  0) echo "images_to_video: $DIR has no PNG/JPEG stills (need 1-3)" >&2; exit 1 ;;
  1) ANCHORS=(--start-frame "${IMAGES[0]}") ;;
  2) ANCHORS=(--start-frame "${IMAGES[0]}" --end-frame "${IMAGES[1]}") ;;
  3) ANCHORS=(--start-frame "${IMAGES[0]}" --mid-frame "${IMAGES[1]}" --end-frame "${IMAGES[2]}") ;;
  *) echo "images_to_video: $DIR has ${#IMAGES[@]} stills, only 1-3 are supported (start/mid/end)" >&2; exit 1 ;;
esac

exec "${BRAIN:-./target/release/brain}" ltxv t2v \
  --prompt "$PROMPT" \
  --frames "$FRAMES" --width "${WIDTH:-1280}" --height "${HEIGHT:-704}" \
  --steps "${STEPS:-8}" --seed "${SEED:-7}" --fps "$FPS" \
  --dit-config "$DIT_CONFIG" \
  "${ANCHORS[@]}" \
  --output-path "$OUT"
