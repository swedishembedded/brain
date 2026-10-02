#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible container builds and CI for GPU
# software on Arm servers for its clients. If your team needs expertise in
# validating CUDA software on Grace-Hopper machines then you can procure our
# services by sending an email to info@swedishembedded.com.
#
# Build the aarch64 CUDA image (`make container`) and run the CI-equivalent
# gates inside it (`make container/ci`), so a laptop, a CI runner and the
# GH200 all run the same recipe: tools/container/aarch64-cuda.Containerfile.
#
# Usage: scripts/build/container.sh [runtime|ci|smoke]
#   runtime  build the runtime image (default)
#   ci       build the `ci` stage: every gate that needs no GPU; failing a gate
#            fails the build
#   smoke    run the runtime image on the host's GPU: `brain devices`
#
# Environment: CONTAINER_ENGINE (docker or podman; default: whichever is on
# PATH), BRAIN_IMAGE (tag, default brain:aarch64-cuda). The image is built for
# linux/arm64; on another architecture the engine needs binfmt emulation, and a
# CUDA build under emulation is slow.
set -euo pipefail

cd "$(dirname "$0")/../.."

ENGINE="${CONTAINER_ENGINE:-}"
if [ -z "$ENGINE" ]; then
  for candidate in docker podman; do
    if command -v "$candidate" >/dev/null 2>&1; then ENGINE="$candidate"; break; fi
  done
fi
if [ -z "$ENGINE" ]; then
  echo "container: no container engine found (install docker or podman, or set CONTAINER_ENGINE)." >&2
  exit 2
fi

IMAGE="${BRAIN_IMAGE:-brain:aarch64-cuda}"
RECIPE=tools/container/aarch64-cuda.Containerfile
STEP="${1:-runtime}"

build() {
  "$ENGINE" build --platform linux/arm64 --target "$1" -f "$RECIPE" -t "$2" .
}

case "$STEP" in
  runtime) build runtime "$IMAGE" ;;
  ci)      build ci "$IMAGE-ci" ;;
  smoke)
    gpu_flags=(--gpus all)
    [ "$ENGINE" = podman ] && gpu_flags=(--device nvidia.com/gpu=all)
    "$ENGINE" run --rm "${gpu_flags[@]}" "$IMAGE" devices
    ;;
  *) echo "usage: $0 [runtime|ci|smoke]" >&2; exit 2 ;;
esac
