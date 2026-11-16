#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for keeping measured performance claims reviewable for its clients. If your
# team needs expertise in documentation gates for performance-critical software then you can procure our services by
# sending an email to info@swedishembedded.com.

# Spec for scripts/gates/check-no-perf-numbers.sh: a percentage counts as a performance claim only beside performance
# vocabulary. "latency" is such a word; "latent" (a latent variable, `pca_latent`) is not, and a principal-component
# table of explained variance beside it is a statistical result, not a claim about speed.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  REL="docs/book/gate-fixture-no-perf-numbers-$$-$BATS_TEST_NUMBER.md"
  F="$REPO/$REL"
}

teardown() { rm -f "$F"; }

gate() { run bash "$REPO/scripts/gates/check-no-perf-numbers.sh" "$REL"; }

@test "a percentage beside a latent variable is not a performance claim" {
  printf 'Generator: `classical.pca_latent`, two latent factors.\n\n| component | explained |\n|---|---:|\n| 1 | 73.75%% |\n' >"$F"
  gate
  [ "$status" -eq 0 ]
}

@test "a percentage beside latency is still flagged" {
  printf 'The decode latency fell.\n\n| run | change |\n|---|---:|\n| 1 | 73.75%% |\n' >"$F"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"73.75%"* ]]
}
