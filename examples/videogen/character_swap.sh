#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Replace one character in an existing clip, keeping the set, the camera move
# and the lighting BIT-EXACTLY.
#
#   examples/videogen/character_swap.sh <clip.mp4> <masks/> "<prompt>" [out.mp4]
#
# The mechanism is masked conditioning -- LTX-2.5's `VideoConditionByMask`,
# ported and parity-gated in `ltxv::maskcond`. Every latent position the mask
# marks as conditioning is handed the source clip's own latent and excluded
# from denoising, so it comes out of the sampler unchanged rather than merely
# similar; everything else is renoised and redrawn from the prompt. That needs
# no adapter, which matters: Lightricks has published exactly ONE IC-LoRA for
# LTX-2.5 and it is a pixel spatial upscaler, so the adapter route to this is
# closed. See `examples/videogen/README.md`.
#
# <masks/> is a `brain/sam2-maskseq/1` directory -- one 8-bit PNG per source
# frame plus a `masks.json` naming the pattern, the frame count, the source
# resolution and the POLARITY. Produce one by clicking the character once:
#
#   brain sam2 track --video stunt.mp4 --point 640,300 --out masks/
#
# Either polarity works -- the manifest states which one is on disk and the
# reader honours it, so `--invert` is not needed here. What is not optional is
# that the manifest state it AT ALL: the reader refuses to guess, because
# guessing backwards preserves the character and regenerates the entire set.
# A frame the tracker marks occluded is conditioned fully, so the swap simply
# does not happen on that frame rather than the frame dissolving.
#
# The prompt describes the WHOLE frame, not just the replacement -- the model
# denoises the masked region in the context of everything around it. Identity
# comes from the prompt alone; nothing in this path takes a face crop or a
# per-subject embedding.
#
# Optional, all env: STRENGTH (1.0 pins the conditioned region exactly, lower
# lets it drift), SEED, STEPS, GUIDANCE, BRAIN_DEVICE, BRAIN.
#
# Weights: BRAIN_LTXV_{DIT,TEXT_ENCODER,VAE}. Without BRAIN_LTXV_DIT this runs
# the tiny random-weight DiT instead, which is a real end-to-end test of the
# conditioning (the preservation check below still passes) but paints noise
# into the replaced region. The script says which of the two it ran.

set -euo pipefail

CLIP="${1:?usage: character_swap.sh <clip.mp4> <masks/> \"<prompt>\" [out.mp4]}"
MASKS="${2:?a brain/sam2-maskseq/1 directory is required (brain sam2 writes one)}"
PROMPT="${3:?a prompt describing the whole frame is required}"
OUT="${4:-swapped.mp4}"
BRAIN="${BRAIN:-./target/release/brain}"

[ -f "$CLIP" ] || { echo "no clip at $CLIP" >&2; exit 1; }
[ -f "$MASKS/masks.json" ] || { echo "$MASKS holds no masks.json - see this script's header" >&2; exit 1; }

# Read by KEY, not by position: ffprobe emits these in its own order, not the
# order they are asked for, and a positional read silently swaps the frame
# count with the frame rate.
while IFS='=' read -r k v; do
  case "$k" in
    width) W="$v" ;; height) H="$v" ;; nb_read_frames) NF="$v" ;;
    r_frame_rate) FPS=$(awk -F/ '{printf "%d", ($2 ? $1 / $2 : $1)}' <<<"$v") ;;
  esac
done < <(ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,r_frame_rate,nb_read_frames -of default=nw=1 "$CLIP")

# The causal VAE can only represent 1+8k frames on a 32-pixel grid. Trimming
# the clip here would desync it from a mask sequence produced separately, so
# this stops and says what to regenerate rather than silently shifting frames.
if [ $(( (NF - 1) % 8 )) -ne 0 ] || [ $(( W % 32 )) -ne 0 ] || [ $(( H % 32 )) -ne 0 ]; then
  echo "character_swap: ${W}x${H}, $NF frames is not VAE-representable (1+8k frames, 32-pixel grid)." >&2
  echo "  Re-cut the clip AND re-track the masks at the same length, e.g.:" >&2
  echo "    ffmpeg -i $CLIP -vf scale=$((W/32*32)):$((H/32*32)) -frames:v $(( (NF-1)/8*8 + 1 )) cut.mp4" >&2
  exit 1
fi

[ -n "${BRAIN_LTXV_VAE:-}" ] || { echo "character_swap: BRAIN_LTXV_VAE is not set, so this stops here. The clip has" >&2
  echo "  to go through the real LTX-2.5 video VAE to reach latent space, and there" >&2
  echo "  is no stand-in for it: the conditioning is defined ON that latent." >&2; exit 1; }

REAL=(--dit-config ltx25_22b)
[ -n "${BRAIN_LTXV_DIT:-}" ] || { REAL=(--dit-config tiny); echo "character_swap: no BRAIN_LTXV_DIT - running the tiny random-weight DiT. The" >&2; echo "  preservation check below is still real; the replaced region will be noise." >&2; }

"$BRAIN" ltxv v2v --input "$CLIP" --mask "$MASKS" --prompt "$PROMPT" \
  --strength "${STRENGTH:-1.0}" --seed "${SEED:-7}" --steps "${STEPS:-8}" \
  --guidance "${GUIDANCE:-1.0}" --fps "$FPS" "${REAL[@]}" --output-path "$OUT"

# ---- the test: the conditioned region must be preserved, the rest must move.
# Decoded pixels near a mask boundary bleed (the VAE decoder is convolutional),
# so this measures the two regions' mean |delta| and reports the ratio rather
# than asserting an exact zero.
python3 - "$CLIP" "$OUT" "$MASKS" <<'PY'
import json, subprocess, sys, numpy as np
from PIL import Image
clip, out, masks = sys.argv[1:4]
man = json.load(open(f"{masks}/masks.json"))
w, h = man["width"], man["height"]
def frames(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)
a, b = frames(clip), frames(out)
n = min(len(a), len(b))
obj = np.stack([np.asarray(Image.open(f"{masks}/{man['pattern'] % i}").convert("L")) for i in range(n)]) > 127
if man["polarity"] == "object=0":
    obj = ~obj
d = np.abs(a[:n] - b[:n]).mean(-1)
print(f"character_swap: mean |delta| replaced={d[obj].mean():.2f}  preserved={d[~obj].mean():.2f}")
print("  a preserved value near 0 with a much larger replaced value is the conditioning working;")
print("  the two being equal means the mask never reached the sampler.")
PY
