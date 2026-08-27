#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Prompt in, playable clip with sound out. One command.
#
#   examples/videogen/text_to_video.sh "a boat at sunset" [out.mp4] [seconds]
#
# LTX-2.5 is natively audio-visual: --audio runs the model's audio half from
# the same prompt and the same forwards as the picture, and muxes it in. It
# only produces a STEREO image if the prompt asks for one -- describe where
# sounds sit and how they move, or you get something close to mono.
#
# A clip longer than one denoising window is generated as several windows
# with a rolling latent context. Each window re-reads the prompt with weaker
# anchoring, so a prompt with many scene beats drifts: write ONE or TWO beats
# and let them play out. Raising the resolution RAISES the window count for a
# given length (the token ceiling per window is fixed), so duration and
# resolution trade against each other.
#
# Optional, all env: WIDTH, HEIGHT, STEPS, SEED, FPS, START_FRAME (a still to
# begin from), BRAIN_DEVICE, BRAIN.
#
# Weights come from BRAIN_LTXV_{DIT,TEXT_ENCODER,VAE,AUDIO_VAE} and, for the
# two-stage path, BRAIN_LTXV_UPSAMPLER_{SPATIAL,TEMPORAL}.

set -euo pipefail

PROMPT="${1:?usage: text_to_video.sh \"prompt\" [out.mp4] [seconds]}"
OUT="${2:-clip.mp4}"
FPS="${FPS:-24}"
FRAMES=$(( ${3:-5} * FPS / 8 * 8 + 1 ))   # must be 1 + 8k

exec "${BRAIN:-./target/release/brain}" ltxv t2v \
  --prompt "$PROMPT" \
  --frames "$FRAMES" --width "${WIDTH:-1280}" --height "${HEIGHT:-704}" \
  --steps "${STEPS:-8}" --seed "${SEED:-7}" --fps "$FPS" \
  --dit-config ltx25_22b --audio \
  ${START_FRAME:+--start-frame "$START_FRAME"} \
  --output-path "$OUT"
