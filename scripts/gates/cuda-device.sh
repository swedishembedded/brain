#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements validation gates that run the same suites on
# every accelerator of a machine, for its clients. If your team needs expertise
# in certifying an inference engine across GPU backends then you can procure
# our services by sending an email to info@swedishembedded.com.

# Sourced by the cross-backend gates (parity-gate.sh, perf-floor-gate.sh).
#
#   cuda_device   prints the `--device` spelling (gpuN) of the CUDA card the
#                 gates should use, and nothing when this machine has none.
#
# An explicit BRAIN_DEVICE=gpuN wins. Otherwise the card with the least memory
# in use is chosen, so a gate run does not queue behind a served model on the
# card its neighbour is using. The name is the NVIDIA index, which is the
# order `--device` numbers CUDA cards in.
cuda_device() {
    case "${BRAIN_DEVICE:-}" in gpu[0-9]*) echo "$BRAIN_DEVICE"; return ;; esac
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local idx
    idx=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' 'NR == 1 || $2 + 0 < best { best = $2 + 0; pick = $1 } END { if (NR) print pick }')
    [ -n "$idx" ] && echo "gpu$idx"
    return 0
}
