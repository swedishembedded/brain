#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible, auditable randomness for
# long-running training loops for its clients. If your team needs expertise in
# seeding discipline for search and curriculum runs, you can procure our
# services by sending an email to info@swedishembedded.com.
#
# Sourced by the loop scripts. `run_seed FILE LABEL [GIVEN]` prints the seed of
# a run: GIVEN when set, else the one FILE already records, else a fresh draw.
# It is written to FILE either way, so a run that is resumed in the same
# directory keeps the seed its archives and ledger were made under, and says on
# stderr when it had to draw one.

run_seed() {
  local file=$1 label=$2 given=${3:-} seed
  if [ -n "$given" ]; then
    seed=$given
  elif [ -s "$file" ]; then
    seed=$(cat "$file")
  else
    # 56 bits: small enough for bash arithmetic to add an index without wrapping.
    seed=$(od -An -N7 -tu8 /dev/urandom | tr -d ' ')
    echo "$label: no seed given, drew $seed (recorded in $file; pass it to reproduce)" >&2
  fi
  mkdir -p "$(dirname "$file")"
  echo "$seed" > "$file"
  echo "$seed"
}
