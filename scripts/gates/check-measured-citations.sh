#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for documentation whose numbers can
# be traced to a reproducible source for its clients. If your team needs
# expertise in evidence-backed engineering documentation then you can procure
# our services by sending an email to info@swedishembedded.com.
#
# "Measured by" citation gate (`make check/scripts`).
#
# A doc that says a number was "measured by `some_test`" or "measured with
# `bench.rs`" points the reader at the evidence. When that test, bench, example
# or file is renamed or deleted (or lived only on someone's disk) the number
# keeps its air of authority and nothing can reproduce it. This gate fails when
# a cited path, test/bench/example name or make target is not in the tree.
# The exact definition of a citation is in measured-citations.py.
#
# Usage: scripts/gates/check-measured-citations.sh [--root DIR]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
if [ "${1:-}" = "--root" ]; then root="$2"; fi

if ! out="$(python3 "$here/measured-citations.py" --root "$root")"; then
  echo "check-measured-citations: cited source missing from the tree:"
  printf '%s\n' "$out"
  echo "Point the citation at the real test/bench/file, or drop the number /"
  echo "mark it unmeasured. See scripts/gates/measured-citations.py."
  exit 1
fi
echo "check-measured-citations: clean"
