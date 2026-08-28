#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Portraits of a person, from a folder of their photographs.
#
#   examples/imagegen/portrait_from_refs.sh <dir> [count]
#
# The folder holds numbered photographs of the person -- `ref-01.jpeg`,
# `02.png`, any name that is (optionally `ref-`) a number, in any format brain
# decodes. Results are written as `result-01.jpg` ... one per seed.
#
# The photographs are passed to brain exactly as they are: brain bounds each
# reference's encoded size itself (`--ref-size`, 512 px by default), because a
# reference costs tokens quadratically and that is brain's policy to set, not
# this script's. Nothing is resampled on the way in.
#
# The references supply the PERSON. Pose, framing and lighting come from the
# prompt: edit POSE, it is the knob that matters here.
#
#   POSE='Light from the left, dark background, three-quarter view.' \
#     examples/imagegen/portrait_from_refs.sh ~/photos 4
#
# ADAPTER points at a per-identity LoRA trained on the same folder (see
# train_identity_lora.sh). The references condition on the person's APPEARANCE
# for one generation; an adapter has learned them, and the two compose -- the
# adapter carries identity even in poses no reference shows, which is what the
# references alone cannot do. Together they measure far better than either
# alone; use both.
#
# LORA_SCALE dials the adapter, and 0 reproduces the base model exactly, so it
# is also how you see what the adapter contributes. Start LOW, around 0.5:
# identity plateaus well before image quality gives out, so the highest scale
# that still fits is not the right one -- past the plateau the skin goes waxy
# and the skull inflates while the measured identity stops improving.
#
# Grade the result with a number, not an opinion:
#
#   examples/imagegen/identity_score.sh <dir> <dir>
#
# MASK=1 switches on inpainting instead: the folder must then hold a
# `target.*` whose WHITE region is the hole to fill, and that region is
# regenerated while everything else is preserved exactly. The output takes the
# target's own size. It keeps the original photograph's body and background, at
# the cost of a visible seam where the two meet -- the blend happens in latent
# space, which preserves content faithfully but does not make the halves agree
# on lighting. Prefer the default unless you need the rest of the frame kept
# bit-for-bit. To swap a face INTO another photograph, use face_swap.sh, which
# is built for exactly that and masks the head for you.
#
# The LoRA and the text encoder are FLAGS, because they are the two things
# worth changing between one run and the next:
#
#   examples/imagegen/portrait_from_refs.sh ~/photos 4 \\
#     --adapter out/adapter.brain --lora-scale 0.5
#
#   --adapter <path>        a per-identity LoRA (see train_identity_lora.sh)
#   --lora-scale <s>        its strength, default 0.5 - see above, start low
#   --text-encoder <path>   swap the encoder: an HF directory, or a single
#                           .safetensors/.gguf file
#   --                      end of flags (a count may follow the directory)
#
# Every flag also has an environment variable of the same name in caps
# (ADAPTER, LORA_SCALE, TEXT_ENCODER), and the flag wins when both are given.
#
# Other options, env only: MASK, SEED (first seed; each result adds 1),
# STRENGTH (mask mode only), STEPS, REF_PX (bound on each reference's encoded
# long edge; 0 = each at its own resolution), SIZE (WxH, default mode only),
# POSE, VARIANT, PRECISION, BRAIN.
#
# Weights come from BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}; brain picks a card
# with room unless BRAIN_DEVICE says otherwise.

set -euo pipefail

USAGE='usage: portrait_from_refs.sh <dir-of-images> [count] [--adapter P] [--lora-scale S] [--text-encoder P]'
D=""; N=""
while [ $# -gt 0 ]; do
  case "$1" in
    --adapter)      ADAPTER="${2:?--adapter needs a path}"; shift 2 ;;
    --lora-scale)   LORA_SCALE="${2:?--lora-scale needs a number}"; shift 2 ;;
    --text-encoder) TEXT_ENCODER="${2:?--text-encoder needs a path}"; shift 2 ;;
    -h|--help)      echo "$USAGE"; exit 0 ;;
    --)             shift; continue ;;
    -*)             echo "unknown flag $1" >&2; echo "$USAGE" >&2; exit 1 ;;
    *)              if [ -z "$D" ]; then D="$1"; elif [ -z "$N" ]; then N="$1";
                    else echo "unexpected argument $1" >&2; exit 1; fi; shift ;;
  esac
done
[ -n "$D" ] || { echo "$USAGE" >&2; exit 1; }
D="${D%/}"; N="${N:-4}"
BRAIN="${BRAIN:-./target/release/brain}"

mapfile -t REFS < <(
  find "$D" -maxdepth 1 -type f \
    | grep -Ei '/(ref[-_]?)?[0-9]+\.(jpe?g|png|ppm|webp|bmp)$' | sort
)

ARGS=(--ref-size "${REF_PX:-512}")
if [ "${MASK:-0}" = 1 ]; then
  TARGET="$(find "$D" -maxdepth 1 -type f | grep -Ei '/target\.(jpe?g|png|ppm|webp|bmp)$' | sort | head -1 || true)"
  [ -n "$TARGET" ] || { echo "MASK=1 but no target.* image in $D" >&2; exit 1; }
  W="$D/.work"; mkdir -p "$W"
  # The only real image DERIVATION here: the hole to fill is wherever the
  # author painted the target white. Brain takes the output size from the
  # target itself, so nothing needs resampling to make the two agree.
  python3 - "$TARGET" "$W/mask.png" <<'PY'
import sys
import numpy as np
from PIL import Image, ImageFilter
src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
hole = (np.asarray(im, dtype=int) > 245).all(axis=2)
if not hole.any():
    raise SystemExit(f"{src} has no white region to fill")
Image.fromarray((hole * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(6)).save(dst)
print(f"mask: {hole.mean():.1%} of the frame regenerates", file=sys.stderr)
PY
  # The target is the init latent, so it must come FIRST.
  ARGS+=(--mask "$W/mask.png" --ref "$TARGET" --strength "${STRENGTH:-0.99}" --ref-cond-scale 0)
else
  SIZE="${SIZE:-768x1024}"
  ARGS+=(--width "${SIZE%x*}" --height "${SIZE#*x}")
fi

for f in "${REFS[@]}"; do ARGS+=(--ref "$f"); done
[ "${#REFS[@]}" -gt 0 ] || [ -n "${ADAPTER:-}" ] || { echo "$D: no numbered reference images" >&2; exit 1; }
if [ -n "${ADAPTER:-}" ]; then
  [ -e "$ADAPTER" ] || { echo "adapter not found: $ADAPTER" >&2; exit 1; }
  ARGS+=(--adapter "$ADAPTER" --lora-scale "${LORA_SCALE:-0.5}")
fi
[ -n "${TEXT_ENCODER:-}" ] && ARGS+=(--text-encoder "$TEXT_ENCODER")

for i in $(seq 1 "$N"); do
  "$BRAIN" flux2 generate \
    --variant "${VARIANT:-klein-4b}" --precision "${PRECISION:-int8}" \
    --prompt "A photorealistic portrait photograph of the person in the reference images, keeping their exact facial identity, head shape, eye colour and skin texture. ${POSE:-Soft key light from the left, dark plain background, plain dark crew-neck top, head level and facing camera, calm closed-mouth expression, sharp focus on the eyes, 85mm lens, shallow depth of field.}" \
    "${ARGS[@]}" \
    --steps "${STEPS:-12}" --seed "$(( ${SEED:-101} + i - 1 ))" \
    --out "$D/result-$(printf '%02d' "$i").jpg"
done
