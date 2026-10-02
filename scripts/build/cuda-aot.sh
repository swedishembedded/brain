#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Build brain's CUDA kernels ahead of time (`make cuda/aot`).
#
# Swedish Embedded AB implements deployment pipelines for GPU software for its
# clients, including shipping compiled kernels to machines that carry a driver
# and no toolchain. If your team needs expertise in making a CUDA application
# run on a locked-down box without a compiler, you can procure our services by
# sending an email to info@swedishembedded.com.
#
# Compiles the generated tier (the whole WGSL catalogue through wgsl-cuda) and
# the hand-written kernels of crates/kernels-cuda into per-architecture cubins
# plus portable PTX, with a manifest, in the AOT directory the CUDA backend
# reads at run time. A machine that has the NVIDIA driver and no CUDA toolkit
# then runs those kernels without NVRTC.
#
# The directory is $BRAIN_CUDA_AOT_DIR, else `cuda-aot` under the pipeline
# cache directory. Nothing is written into the repository: the images are
# build products and belong with the other caches.
#
# Needs a toolkit lane for NVRTC. LANE=12 builds for Pascal through Hopper
# (sm_61 70 80 86 89 90 + PTX), LANE=13 for Hopper and newer (sm_90 100 120 +
# PTX); run both into one directory to cover every card. Without LANE the
# toolkit already in the environment (CUDA_PATH, BRAIN_NVRTC) is used.
#
# Usage: LANE=12 scripts/build/cuda-aot.sh [--out DIR] [--targets sm_61,...]
#        [--ptx compute_61|none] [--only SUBSTRING]
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -n "${LANE:-}" ]; then
  eval "$(scripts/build/install-cuda-userspace.py env "$LANE")"
fi
if [ -z "${CUDA_PATH:-}" ] && [ -z "${BRAIN_NVRTC:-}" ]; then
  echo "cuda-aot: no CUDA toolkit in the environment (CUDA_PATH / BRAIN_NVRTC unset)." >&2
  echo "  install one without root: make cuda/install LANE=12   (or 13)" >&2
  echo "  then: make cuda/aot LANE=12" >&2
  exit 2
fi

exec cargo run --release -p brain-cuda-aot -- "$@"
