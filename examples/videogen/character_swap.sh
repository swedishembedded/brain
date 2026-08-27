#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Swap a character in an existing clip, keeping choreography, camera and set.
#
#   examples/videogen/character_swap.sh <clip.mp4> <character-image-dir> [out.mp4]
#
# READ THIS FIRST -- the swap CANNOT be completed today, and this script stops
# short of it on purpose rather than pretending. What blocks it is not effort,
# it is that the required trained weights do not exist:
#
#   * The mechanism that locks an existing clip's motion is an IC-LoRA -- an
#     "In-Context LoRA" (NOT "Identity & Composition"). It appends the control
#     video's latent tokens to the sequence as frozen reference tokens, and the
#     ADAPTER is what teaches the model to attend across them. See
#     `ltxv::refcond` for the ported mechanism and its parity gate.
#   * Lightricks has published exactly ONE IC-LoRA for LTX-2.5, and it is a
#     pixel spatial upscaler. There is no Canny, depth, pose or union control
#     adapter for LTX-2.5. Those exist only for the older LTX-2.3-22b
#     (Union-Control = Canny+Depth+Pose) and LTX-2-19b checkpoints, and a LoRA
#     only ever works with the model it was trained on.
#   * The reference slot holds ONE thing, and that is what kills the plan. An
#     IC-LoRA is trained for one reference SEMANTICS: Union-Control reads it as
#     a structure signal, while LTX-2.3's Ingredients IC-LoRA reads it as a
#     character reference sheet. So "lock the choreography AND inject the
#     actor" needs both meanings in one slot, from two adapters trained
#     separately. Ingredients also generates FROM its sheet rather than
#     preserving an input clip, and there is no LTX-2.5 build of it either.
#     Nothing in this path takes a face crop or a per-subject embedding, so on
#     LTX-2.5 today "who" the new character is comes from the prompt alone.
#   * The diffusion video decoder does not help either: it maps a latent to
#     pixels (`decode_video(latent, tiling, generator)`) and has no identity
#     input of any kind. It cannot paint a face onto a body.
#
# So what this script DOES do is produce, from your inputs, the two control
# signals such a swap needs, both of which are real and both of which are
# exactly what you would feed an IC-LoRA once you have one:
#
#   <out>.control.mp4  the structure reference: a Canny edge video of the
#                      source. This is what pins body position, physics, camera
#                      move and background, frame for frame.
#   <out>.mask.mp4     the character pin: which region the reference is allowed
#                      to govern. White = keep the source structure, black =
#                      regenerate freely from the prompt. Feeding this as the
#                      conditioning attention mask is how you say WHICH of
#                      several characters gets replaced.
#
# PIN picks the character by a point on a frame, segmented with SAM 2.1.
# brain's SAM 2.1 is the IMAGE path only -- the video memory bank is out of
# scope there -- so a single point on frame 0 cannot be propagated through the
# clip. Instead give a point per keyframe and the mask is held between them:
#
#   PIN="640,300"                 one point, frame 0, held for the whole clip
#   PIN="640,300@0;700,320@48"    re-pin at frame 48 (a point per keyframe)
#
# Only the first form is honest for a locked-off shot; anything with real
# camera or subject movement wants several. This is the nearest true thing to
# "click the stuntman once" that the parts on hand support.
#
# The character images are checked and passed through for the prompt you will
# write, but note again that they do NOT bind identity anywhere in this path.
#
# Optional, all env: PIN, PROMPT, FPS, WIDTH, HEIGHT, EDGE_LOW, EDGE_HIGH,
# KEEP_FRAMES, BRAIN_DEVICE, BRAIN.
#
# Weights: BRAIN_SAM2_WEIGHTS for the mask. ffmpeg does the rest.

set -euo pipefail

CLIP="${1:?usage: character_swap.sh <clip.mp4> <character-image-dir> [out.mp4]}"
CHARS="${2:?a directory of images of the target character is required}"
OUT="${3:-swapped.mp4}"
BRAIN="${BRAIN:-./target/release/brain}"
BASE="${OUT%.*}"
WORK="$BASE.work"

[ -f "$CLIP" ] || { echo "no clip at $CLIP" >&2; exit 1; }
[ -d "$CHARS" ] || { echo "no character image directory at $CHARS" >&2; exit 1; }
NCHAR=$(find "$CHARS" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
[ "$NCHAR" -gt 0 ] || { echo "$CHARS holds no .png/.jpg images" >&2; exit 1; }

FPS="${FPS:-$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
  -of default=nw=1:nk=1 "$CLIP" | awk -F/ '{printf "%d", ($2?$1/$2:$1)}')}"
IFS=, read -r W H <<<"$(ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height -of csv=p=0 "$CLIP")"
W="${WIDTH:-$W}"; H="${HEIGHT:-$H}"

rm -rf "$WORK"; mkdir -p "$WORK/frames"
ffmpeg -v error -y -i "$CLIP" -vf "scale=$W:$H" "$WORK/frames/%06d.ppm"
NF=$(find "$WORK/frames" -name '*.ppm' | wc -l)
echo "character_swap: $NF frames at ${W}x${H} @ ${FPS}fps, $NCHAR character image(s)" >&2

# ---- 1. structure reference: Canny edges, the real control signal ----------
ffmpeg -v error -y -i "$CLIP" \
  -vf "scale=$W:$H,edgedetect=low=${EDGE_LOW:-0.1}:high=${EDGE_HIGH:-0.4}" \
  -r "$FPS" "$BASE.control.mp4"

# ---- 2. character pin: SAM 2.1 masks at the pinned keyframes ---------------
PIN="${PIN:-}"
if [ -z "$PIN" ]; then
  echo "character_swap: no PIN given, so the whole frame is pinned (every" >&2
  echo "  character would be replaced). Pass PIN=\"x,y\" to pick one." >&2
  ffmpeg -v error -y -f lavfi -i "color=white:s=${W}x${H}:r=$FPS" \
    -frames:v "$NF" "$BASE.mask.mp4"
else
  mkdir -p "$WORK/mask"
  # Parse "x,y@frame;..." into a sorted keyframe list; a bare "x,y" is frame 0.
  KFS=$(echo "$PIN" | tr ';' '\n' | awk -F@ '{f=($2==""?0:$2); print f"\t"$1}' | sort -n)
  echo "$KFS" | while IFS=$'\t' read -r f pt; do
    [ -n "$pt" ] || continue
    SRC=$(printf "$WORK/frames/%06d.ppm" $((f + 1)))
    [ -f "$SRC" ] || { echo "PIN frame $f is past the end of the clip" >&2; exit 1; }
    "$BRAIN" sam2 segment --in image="$SRC" points="$pt" \
      --out mask="$WORK/mask/$(printf '%06d' "$f").ppm" >/dev/null
  done
  # Hold each keyframe's mask until the next one, so the pin covers every frame.
  python3 - "$WORK" "$NF" <<'PY'
import sys, pathlib, shutil
work, nf = pathlib.Path(sys.argv[1]), int(sys.argv[2])
md = work / "mask"
keys = sorted(int(p.stem) for p in md.glob("*.ppm"))
if not keys:
    raise SystemExit("character_swap: SAM 2.1 produced no masks")
out = work / "maskseq"; out.mkdir(exist_ok=True)
cur = keys[0]
for i in range(nf):
    while keys and i >= keys[0]:
        cur = keys.pop(0)
    shutil.copyfile(md / f"{cur:06d}.ppm", out / f"{i:06d}.ppm")
PY
  ffmpeg -v error -y -framerate "$FPS" -i "$WORK/maskseq/%06d.ppm" \
    -vf "scale=$W:$H,format=gray" "$BASE.mask.mp4"
fi

[ -n "${KEEP_FRAMES:-}" ] || rm -rf "$WORK"

cat >&2 <<EOF

character_swap: control signals written.
  $BASE.control.mp4   Canny structure reference (locks motion, camera, set)
  $BASE.mask.mp4      character pin (white = hold structure, black = regenerate)

NOT written: $OUT. Generating it needs an LTX-2.5 IC-LoRA trained for edge
control, and none is published -- see this script's header. Two honest routes:

  1. Train one. packages/ltx-trainer/configs/v2v_ic_lora.yaml is the config and
     wants a reference_latents/ dataset; the pair above is one sample of it.
  2. Run the control video through LTX-2.3-22b with its published
     IC-LoRA-Union-Control (Canny+Depth+Pose) in the reference pipeline. That
     preserves choreography -- but the new character comes from the PROMPT, so
     it is a structure-preserving restyle, not an identity swap.
EOF
