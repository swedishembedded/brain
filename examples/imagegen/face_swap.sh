#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Put someone's face into someone else's photograph.
#
#   examples/imagegen/face_swap.sh <dir> [count]
#
# The folder holds numbered images -- `ref-01.jpg`, `ref-02.png`, `03.jpeg`,
# any name that is (optionally `ref-`) a number, in any format brain decodes.
#
#   THE FIRST ONE IS THE TARGET: the photograph whose pose, framing, clothing,
#   background and lighting are kept. Every one after it is a photograph of
#   the person whose face goes into it.
#
# Results are written as `result-01.jpg` ... one per seed.
#
# # Why the target has to be first, and why that is the model's rule
#
# Under `--strength` and under `--mask`, brain treats the FIRST `--ref` as the
# init latent: it is VAE-encoded and the denoise starts from it, which is the
# only reason the pose survives at all. Every later reference is conditioning.
# So the ordering is not a convention this script invented, it is what the
# pipeline means, and numbering the files puts it where you can see it.
#
# # The two settings that decide whether this works
#
# **`--ref-cond-scale 0`.** The target does double duty: it seeds the latent
# AND it conditions the model. But it contains a face -- the one being replaced
# -- so conditioning on it feeds the model the wrong identity, which then
# competes with the face references and wins, because it is also the thing
# every pixel is being pulled back toward. Measured on a deliberately hard
# swap, switching this off moved ArcFace identity from 0.018 to 0.198 with pose
# preservation unchanged. Leave it on and the script appears to run correctly
# and quietly returns the original person.
#
# **The mask.** A `mask.*` in the folder is used as-is -- white regenerates,
# black keeps the target bit-for-bit, greys blend. That is the escape hatch for
# a target the detector cannot read, and the way to choose the region by hand.
# Otherwise one is derived from the detector's five facial landmarks: an
# ellipse over the head, rotated to the face axis and feathered, so the blend
# has no hard edge. MASK_GROW scales it, and large enough to include the
# hairline matters -- whatever the mask does not cover is kept, so a mask that
# stops at the forehead leaves the target's hair and skull outline on a face
# that is no longer theirs.
#
# Set MASK=0 for the plain `--strength` route instead: no seam, but the pose is
# only approximately preserved and identity transfer measured weaker.
#
# # Grade it, do not eyeball it
#
#   examples/imagegen/identity_score.sh <dir> <dir>
#
# A face swap is exactly the case where the eye is easiest to fool: the picture
# is a real photograph almost everywhere, so it looks convincing whether or not
# the face actually changed.
#
# The LoRA and the text encoder are FLAGS, matching portrait_from_refs.sh:
#
#   examples/imagegen/face_swap.sh ~/shot 2 --adapter out/adapter.brain
#
#   --adapter <path>        a per-identity LoRA from train_identity_lora.sh --
#                           it COMPOSES with the face references, and the two
#                           together measure far better than either alone
#   --lora-scale <s>        its strength, default 0.5
#   --text-encoder <path>   swap the encoder: an HF directory, or a single
#                           .safetensors/.gguf file
#
# Each also has an environment variable of the same name in caps, and the flag
# wins when both are given.
#
# Other options, env only: MASK (0 disables), MASK_GROW, STRENGTH, SEED, STEPS,
# PROMPT, VARIANT, PRECISION, BRAIN.
#
# Weights: BRAIN_SCRFD_DIR for the landmarks, BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}
# for generation. References are sized by brain and the output takes the
# target's own size, so pass photographs exactly as they are.

set -euo pipefail

USAGE='usage: face_swap.sh <dir-of-images> [count] [--adapter P] [--lora-scale S] [--text-encoder P]'
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
D="${D%/}"; N="${N:-2}"
BRAIN="${BRAIN:-./target/release/brain}"
W="$D/.swap"; mkdir -p "$W"

mapfile -t REFS < <(
  find "$D" -maxdepth 1 -type f \
    | grep -Ei '/(ref[-_]?)?[0-9]+\.(jpe?g|png|ppm|webp|bmp)$' | sort
)
[ "${#REFS[@]}" -ge 2 ] || {
  echo "$D: need at least two numbered images - the first is the target, the rest are the face" >&2
  exit 1
}
TARGET="${REFS[0]}"
FACES=("${REFS[@]:1}")
echo "target: $(basename "$TARGET")   face: ${#FACES[@]} image(s)" >&2

ARGS=(--ref "$TARGET" --ref-cond-scale 0)
if [ "${MASK:-1}" != 0 ]; then
  SUPPLIED="$(find "$D" -maxdepth 1 -type f \
    | grep -Ei '/mask\.(jpe?g|png|ppm|webp|bmp)$' | sort | head -1 || true)"
  if [ -n "$SUPPLIED" ]; then
    echo "mask: using $(basename "$SUPPLIED") from the folder" >&2
    cp "$SUPPLIED" "$W/mask.png"
  else
    BRAIN="$BRAIN" python3 "$(dirname "$0")/head_mask.py" \
      "$TARGET" "$W/mask.png" "${MASK_GROW:-2.0}"
  fi
  ARGS+=(--mask "$W/mask.png" --strength "${STRENGTH:-0.99}")
else
  ARGS+=(--strength "${STRENGTH:-0.9}")
fi

for f in "${FACES[@]}"; do ARGS+=(--ref "$f"); done
if [ -n "${ADAPTER:-}" ]; then
  [ -e "$ADAPTER" ] || { echo "adapter not found: $ADAPTER" >&2; exit 1; }
  ARGS+=(--adapter "$ADAPTER" --lora-scale "${LORA_SCALE:-0.5}")
fi
[ -n "${TEXT_ENCODER:-}" ] && ARGS+=(--text-encoder "$TEXT_ENCODER")

for i in $(seq 1 "$N"); do
  "$BRAIN" flux2 generate \
    --variant "${VARIANT:-klein-4b}" --precision "${PRECISION:-int8}" \
    --prompt "${PROMPT:-A photograph of the person in the face reference images, keeping their exact facial identity, head shape, eye colour and skin texture, in the pose, framing, clothing, background and lighting of the first reference photograph.}" \
    "${ARGS[@]}" \
    --steps "${STEPS:-12}" --seed "$(( ${SEED:-101} + i - 1 ))" \
    --out "$D/result-$(printf '%02d' "$i").jpg"
done
