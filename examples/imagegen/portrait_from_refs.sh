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
# The references supply the PERSON. Pose, framing and lighting come from the
# prompt: edit POSE, it is the knob that matters here.
#
#   POSE='Light from the left, dark background, three-quarter view.' \
#     examples/imagegen/portrait_from_refs.sh ~/photos 4
#
# MASK=1 switches on inpainting instead: the folder must then hold a
# `target.*` whose WHITE region is the hole to fill, and that region is
# regenerated while everything else is preserved exactly. It keeps the
# original photograph's body and background, at the cost of a visible seam
# where the two meet -- the blend happens in latent space, which preserves
# content faithfully but does not make the halves agree on lighting. Prefer
# the default unless you need the rest of the frame kept bit-for-bit.
#
# ADAPTER points at a per-identity LoRA trained on the same folder (see
# train_identity_lora.sh). The references condition on the person's APPEARANCE
# for one generation; an adapter has learned them, and the two compose -- the
# adapter carries identity even in poses no reference shows, which is what the
# references alone cannot do. Together they measure far better than either
# alone -- use both.
#
# LORA_SCALE dials the adapter, and 0 reproduces the base model exactly, so it
# is also how you see what the adapter contributes. Start LOW, around 0.5:
# identity plateaus well before image quality gives out, so the highest scale
# that still fits is not the right one -- past the plateau the skin goes waxy
# and the skull inflates while the measured identity stops improving. Sweep it
# and score each step rather than trusting one setting.
#
# Grade the result with a number, not an opinion:
#
#   examples/imagegen/identity_score.sh <dir> <dir>
#
# Optional, all env: ADAPTER, LORA_SCALE, MASK, SEED (first seed; each result
# adds 1), STRENGTH (mask mode only), STEPS, REF_PX (long edge of each
# reference), SIZE (WxH, default mode only), POSE, VARIANT, PRECISION, BRAIN.
#
# Weights come from BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}; brain picks a card
# with room unless BRAIN_DEVICE says otherwise.

set -euo pipefail

D="${1:?usage: portrait_from_refs.sh <dir-of-images> [count]}"; D="${D%/}"
N="${2:-4}"
W="$D/.work"; mkdir -p "$W"

read -r CW CH < <(python3 - "$D" "$W" "${REF_PX:-384}" "${MASK:-0}" "${SIZE:-768x1024}" <<'PY'
import sys, os, glob, re
import numpy as np
from PIL import Image, ImageFilter
d, work, refpx, want_mask, size = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
for i, p in enumerate(sorted(glob.glob(f"{d}/*")), 1):
    if re.match(r"^(ref[-_]?)?\d+\.[a-z0-9]+$", os.path.basename(p), re.I):
        r = Image.open(p).convert("RGB")
        s = refpx / max(r.size)
        r.resize((max(16, round(r.width * s / 16) * 16), max(16, round(r.height * s / 16) * 16)), Image.LANCZOS).save(f"{work}/ref{i}.png")
if not want_mask:
    print(*(int(v) for v in size.split("x")))
    raise SystemExit
tgt = next((p for p in glob.glob(f"{d}/target.*")), None)
if not tgt:
    raise SystemExit("MASK=1 but no target.* image")
im = Image.open(tgt).convert("RGB")
w, h = (max(16, round(x / 16) * 16) for x in im.size)
im = im.resize((w, h), Image.LANCZOS)
im.save(f"{work}/target.png")
hole = (np.asarray(im, dtype=int) > 245).all(axis=2)
if not hole.any():
    raise SystemExit("target has no white region to fill")
Image.fromarray((hole * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(6)).save(f"{work}/mask.png")
print(w, h)
PY
)

ARGS=()
[ -n "${ADAPTER:-}" ] && ARGS+=(--adapter "$ADAPTER" --lora-scale "${LORA_SCALE:-1.0}")
[ "${MASK:-0}" = 1 ] && ARGS+=(--mask "$W/mask.png" --ref "$W/target.png" --strength "${STRENGTH:-0.99}")
REFS=0
for f in "$W"/ref*.png; do [ -e "$f" ] && { ARGS+=(--ref "$f"); REFS=$((REFS + 1)); }; done
[ "$REFS" -gt 0 ] || [ -n "${ADAPTER:-}" ] || { echo "$D: no numbered reference images" >&2; exit 1; }

for i in $(seq 1 "$N"); do
  "${BRAIN:-./target/release/brain}" flux2 generate \
    --variant "${VARIANT:-klein-4b}" --precision "${PRECISION:-int8}" \
    --prompt "A photorealistic portrait photograph of the person in the reference images, keeping their exact facial identity, head shape, eye colour and skin texture. ${POSE:-Soft key light from the left, dark plain background, plain dark crew-neck top, head level and facing camera, calm closed-mouth expression, sharp focus on the eyes, 85mm lens, shallow depth of field.}" \
    --width "$CW" --height "$CH" "${ARGS[@]}" \
    --steps "${STEPS:-12}" --seed "$(( ${SEED:-101} + i - 1 ))" \
    --out "$D/result-$(printf '%02d' "$i").jpg"
done
