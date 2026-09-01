#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Will an industrial cooling loop trip its over-temperature threshold in the
# next 5 days, and when? Generates the synthetic scenario, runs it through
# brain's TimesFM-3 port (natively multivariate: target + past covariate +
# two known-future covariates in one decode() call) against a physics
# observer and a seasonal-naive baseline, and renders one chart.
#
#   examples/forecast/cooling_loop.sh <timesfm3.safetensors> [chart.png]
#
# Get the weights with:  brain pull google/timesfm-3.0-pytorch && \
#   brain forecast import --timesfm3 <fetched dir> --out timesfm3.safetensors
# (they are timesfm-non-commercial-license-v1.0: non-commercial/non-production
# use only, and the checkpoint itself may never be redistributed)
set -euo pipefail

weights="${1:?usage: cooling_loop.sh <timesfm3.safetensors> [chart.png]}"
chart="${2:-}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
csv="$(mktemp -d)/cooling_loop.csv"

python3 "$repo_root/tools/forecast/make_cooling_loop.py" --out "$csv" --hours 720 --seed 7

cd "$repo_root"
cargo run --release -p brain-timesfm3 --example cooling_loop -- "$csv" "$weights" "$chart"
