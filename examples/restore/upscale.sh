#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Upscale an image 4x with Real-ESRGAN.
#
#   examples/restore/upscale.sh <image> [out.jpg]
#
# Any format brain decodes goes in; the output format follows the extension
# you type (.png, .jpg/.jpeg, .ppm) -- an extension nobody supports is an
# error rather than a file that lies about itself.
#
# `--tile` bounds device memory: the image is upscaled in tiles of that size
# rather than whole, so a large input does not need a proportionally large
# card. Lower it if you run out of memory; it does not change the result.
#
# Generation models cap out well below print resolution, so the usual shape
# is: generate at what the model does well, then upscale here.
#
# Optional, all env: TILE, BRAIN_DEVICE (gpu0/gpu1/cpu -- the CPU backend
# handles this fine if the cards are busy), BRAIN.
#
# Needs BRAIN_ESRGAN_WEIGHTS pointing at a RealESRGAN_x4plus.pth.

set -euo pipefail

IN="${1:?usage: upscale.sh <image> [out.jpg]}"
OUT="${2:-${IN%.*}-4x.jpg}"

exec "${BRAIN:-./target/release/brain}" rrdbnet upscale \
  --in "image=$IN" --tile "${TILE:-128}" --out "image=$OUT"
