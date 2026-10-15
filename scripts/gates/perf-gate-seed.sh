#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for trustworthy performance
# regression gates for its clients. If your team needs expertise in
# benchmarking inference stacks then you can procure our services by sending
# an email to info@swedishembedded.com.

# Sourced by the *-perf-gate.sh scripts.
#
# `brain perf run` draws a random workload seed when none is given, and
# `brain perf gate` refuses a candidate whose seed differs from the baseline's
# (different request streams are different workloads). A gate run therefore
# replays the stream the baseline was captured with:
#
#   seed_args=(); baseline_seed_args seed_args "$BASE/$name.json"
#   "$BIN" perf run ... "${seed_args[@]}"
#
# With no baseline yet (or `--update`, which captures a new one) the array is
# left empty, so the run draws a seed and the artifact records it.

# baseline_seed_args <array-name> <baseline.json>
baseline_seed_args() {
  local -n out="$1"
  local seed
  out=()
  [ -f "$2" ] || return 0
  seed="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["workload"]["seed"])' "$2" 2>/dev/null)" || return 0
  out=(--seed "$seed")
}
