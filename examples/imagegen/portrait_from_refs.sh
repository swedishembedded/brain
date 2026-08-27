#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Put a person into a target pose. Point it at ONE folder, get a portrait.
#
#   examples/imagegen/portrait_from_refs.sh <dir> [out.jpg]
#
# The folder holds numbered photographs of the person -- `ref-01.jpeg`,
# `02.png`, `3.webp`, any name that is (optionally `ref-`) a number, in any
# format brain decodes -- plus one `target.*`, the photograph whose pose you
# want. There is nothing else to work out: the references are downscaled to a
# size that fits the card (`--ref-size`), so a folder of full-resolution phone
# photos is a valid input.
#
# By DEFAULT the target is passed as a reference too, so every image in the
# folder is used and pose, framing and lighting arrive as pixels rather than as
# adjectives. The cost is that reference conditioning carries identity hard: the
# target person's face can bleed into the result, and the numbered references
# can just as easily win and bring their own backgrounds with them.
#
# The other way round is `TARGET_REF=0` plus a `POSE` that says in words what
# the target looks like. The target image is then not conditioned on at all, so
# identity comes from the numbered references alone and the pose is exactly as
# specific as your sentence is. Try this one when the default reproduces a
# reference photo's setting instead of the target's.
#
# Optional, all env: POSE (what to reproduce from the target, in words -- edit
# this first, and always when TARGET_REF=0), TARGET_REF, SIZE (`WxH`), SEED,
# STEPS, REF_SIZE, VARIANT, PRECISION, BRAIN.
#
# Weights come from BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}. Placement does not:
# with no --device, brain places the DiT, the text encoder and the VAE itself,
# on whichever cards can hold them, and prints what it chose. An int8 9B DiT
# and its text encoder do not share one 24 GB card, so on a two-card box they
# land on separate cards without you saying so, and on a one-card box the
# whole encoder stays beside the DiT exactly as it always did. Set BRAIN_DEVICE
# or BRAIN_FLUX2_TE_DEVICE=gpu<i>[:i8] only to override that decision.

set -euo pipefail

D="${1:?usage: portrait_from_refs.sh <dir-of-images> [out.jpg]}"; D="${D%/}"
OUT="${2:-$D/portrait.jpg}"
SIZE="${SIZE:-768x1024}"

REFS=()
while IFS= read -r f; do REFS+=(--ref "$D/$f"); done \
  < <(ls -1 "$D" | grep -Ei '^(ref[-_]?)?[0-9]+\.[a-z0-9]+$' | sort)
[ "${#REFS[@]}" -gt 0 ] || { echo "$D: no numbered reference images" >&2; exit 1; }

TARGET="$(ls -1 "$D" | grep -Ei '^target\.' | head -1)"
[ -n "$TARGET" ] || { echo "$D: no target.* image" >&2; exit 1; }
[ "${TARGET_REF:-1}" != 1 ] || REFS+=(--ref "$D/$TARGET")

exec "${BRAIN:-./target/release/brain}" flux2 generate \
  --variant "${VARIANT:-klein-9b}" --precision "${PRECISION:-int8}" \
  --prompt "A photorealistic portrait photograph of the person in the reference images, ${POSE:-in the same pose, framing and lighting as the final reference image}." \
  --width "${SIZE%x*}" --height "${SIZE#*x}" \
  --steps "${STEPS:-12}" --seed "${SEED:-1}" \
  --ref-size "${REF_SIZE:-384}" "${REFS[@]}" --out "$OUT"
