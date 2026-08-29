#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# A numbered sequence of stills and a prompt in, one long chained clip out.
#
#   examples/videogen/chain_images_to_video.sh <folder/> [out.mp4] [seconds-per-clip]
#
# <folder/> holds:
#   prompt.txt                the prompt for every segment (required)
#   image-00.*, image-01.*, image-02.*, ...
#                              PNG/JPEG stills, ZERO-PADDED so plain sort
#                              gives the order you want (image-00 before
#                              image-01 before ... before image-10)
#
# Each PAIR of consecutive stills becomes one clip, start-frame on the
# first and end-frame on the second - image-00 -> image-01, then
# image-01 -> image-02, and so on. N stills make N-1 clips. Because each
# clip's END is the next clip's START (the same still, held fixed), the
# clips are generated to already agree at every seam, and this script
# concatenates them into one file with `ffmpeg -c copy` (no re-encode) -
# the seam itself repeats that one still for two frames rather than one
# (the end of clip i and the start of clip i+1 are both it), a duplicate
# frame at 24fps, not a visible cut.
#
#   MID=1 examples/videogen/chain_images_to_video.sh <folder/> ...
#
# uses the mid-frame conditioning slot as well, which needs an ODD number
# of stills (an even count leaves a still with no partner): stills are
# grouped in NON-overlapping triples, image-00/01/02 as start/mid/end,
# then image-02/03/04, and so on - still N-1's clip and still N+1's clip
# meet at the shared still exactly as in the default mode, just three
# stills per clip instead of two, which is the way to keep a longer or
# moving-camera SEGMENT on course through its own middle, not only at
# its ends. Without MID, only start/end are used, which is the answer if
# your build's `brain ltxv t2v` does not support --mid-frame at all.
#
# Optional, all env: WIDTH, HEIGHT, STEPS, SEED, FPS, MID, BRAIN_DEVICE,
# BRAIN. Segment N is written to <out>.segments/clip-N.mp4 - inspect them
# individually if the final concatenation is not what you expected.
#
# Weights: point LTX_MODEL_DIR at a folder holding LTX-2.5's files FLAT (no
# vae/text_encoders/diffusion_models subfolders); unset, it defaults to
# $BRAIN_MODELS_DIR/Lightricks/LTX-2.5. Anything missing is asked for
# interactively, once - see _resolve_ltxv_weights.sh's header for exactly
# which filenames and why the DiT is the one file that folder alone will
# not contain. LTX_TINY=1 skips all of this and runs the tiny
# random-weight DiT instead - see examples/videogen/images_to_video.sh's
# header for what that does and does not prove.

set -euo pipefail

DIR="${1:?usage: chain_images_to_video.sh <folder/> [out.mp4] [seconds-per-clip]}"
OUT="${2:-chain.mp4}"
FPS="${FPS:-24}"
FRAMES=$(( ${3:-5} * FPS / 8 * 8 + 1 ))   # must be 1 + 8k
MID="${MID:-0}"
BRAIN="${BRAIN:-./target/release/brain}"

if [ "${LTX_TINY:-0}" = "1" ]; then
  DIT_CONFIG=tiny
else
  # shellcheck source=_resolve_ltxv_weights.sh
  source "$(dirname "${BASH_SOURCE[0]}")/_resolve_ltxv_weights.sh"
  DIT_CONFIG=ltx25_22b
fi

command -v ffmpeg >/dev/null || { echo "chain_images_to_video: ffmpeg is required (concatenates the segments)" >&2; exit 1; }

[ -d "$DIR" ] || { echo "chain_images_to_video: no such folder: $DIR" >&2; exit 1; }

PROMPT_FILE="$DIR/prompt.txt"
[ -f "$PROMPT_FILE" ] || { echo "chain_images_to_video: $DIR has no prompt.txt (used for every segment)" >&2; exit 1; }
PROMPT="$(cat "$PROMPT_FILE")"
[ -n "$PROMPT" ] || { echo "chain_images_to_video: $PROMPT_FILE is empty" >&2; exit 1; }
echo "chain_images_to_video: prompt: $PROMPT" >&2

IMAGES=()
while IFS= read -r -d '' f; do IMAGES+=("$f"); done < <(
  find "$DIR" -maxdepth 1 -type f \( -iname 'image-*.png' -o -iname 'image-*.jpg' -o -iname 'image-*.jpeg' \) -print0 | sort -z
)
N=${#IMAGES[@]}

if [ "$MID" = "1" ]; then
  [ "$N" -ge 3 ] || { echo "chain_images_to_video: MID=1 needs at least 3 image-NN.* stills, found $N" >&2; exit 1; }
  [ $(( (N - 1) % 2 )) -eq 0 ] || {
    echo "chain_images_to_video: MID=1 groups stills in non-overlapping triples (image-00/01/02, image-02/03/04, ...)," >&2
    echo "  which needs an ODD count so the last triple has an end still; found $N stills." >&2
    exit 1
  }
  STEP=2
else
  [ "$N" -ge 2 ] || { echo "chain_images_to_video: need at least 2 image-NN.* stills, found $N" >&2; exit 1; }
  STEP=1
fi

SEGDIR="${OUT%.*}.segments"
mkdir -p "$SEGDIR"
LIST="$SEGDIR/concat.txt"
: > "$LIST"

i=0
n=0
while [ $(( i + STEP )) -lt "$N" ]; do
  SEG="$SEGDIR/clip-$(printf '%02d' "$n").mp4"
  if [ "$MID" = "1" ]; then
    ANCHORS=(--start-frame "${IMAGES[$i]}" --mid-frame "${IMAGES[$((i+1))]}" --end-frame "${IMAGES[$((i+2))]}")
  else
    ANCHORS=(--start-frame "${IMAGES[$i]}" --end-frame "${IMAGES[$((i+1))]}")
  fi

  echo "chain_images_to_video: segment $n: ${ANCHORS[*]}"
  "$BRAIN" ltxv t2v \
    --prompt "$PROMPT" \
    --frames "$FRAMES" --width "${WIDTH:-1280}" --height "${HEIGHT:-704}" \
    --steps "${STEPS:-8}" --seed "${SEED:-7}" --fps "$FPS" \
    --dit-config "$DIT_CONFIG" \
    "${ANCHORS[@]}" \
    --output-path "$SEG"

  printf "file '%s'\n" "$(realpath "$SEG")" >> "$LIST"
  i=$(( i + STEP ))
  n=$(( n + 1 ))
done

ffmpeg -y -v error -f concat -safe 0 -i "$LIST" -c copy "$OUT"
echo "chain_images_to_video: wrote $OUT from $n segment(s) (kept in $SEGDIR/)"
